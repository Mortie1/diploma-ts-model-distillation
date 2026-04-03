import torch
from torch import nn


class Normalize1D(nn.Module):
    """
    Batch-version of Normalize for 1D Input.
    Used as an example of a batch transform.
    """

    def __init__(
        self,
        mean=None,
        std=None,
        mode: str = "channel",
        auto_fill_from_train: bool = False,
        eps: float = 1e-6,
    ):
        """
        Args:
            mean (float | list | Tensor | None): mean used in normalization.
            std (float | list | Tensor | None): std used in normalization.
            mode (str): normalization mode: "channel", "sample", or "global".
            auto_fill_from_train (bool): expects train stats to be injected later.
            eps (float): epsilon to avoid division by zero.
        """
        super().__init__()

        if mode not in {"channel", "sample", "global"}:
            raise ValueError(f"Unknown mode `{mode}`. Use channel/sample/global.")

        self.mode = mode
        self.auto_fill_from_train = auto_fill_from_train
        self.eps = eps

        if mean is None:
            mean_t = torch.empty(0, dtype=torch.float32)
            self._has_mean = False
        else:
            mean_t = torch.as_tensor(mean, dtype=torch.float32)
            self._has_mean = True
        if std is None:
            std_t = torch.empty(0, dtype=torch.float32)
            self._has_std = False
        else:
            std_t = torch.as_tensor(std, dtype=torch.float32)
            self._has_std = True
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    @property
    def needs_train_stats(self) -> bool:
        if self.mode == "sample":
            return False
        return self.auto_fill_from_train and (not self._has_mean or not self._has_std)

    def set_stats(self, mean, std):
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.mean.device)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=self.std.device)
        self.mean = mean_t
        self.std = std_t
        self._has_mean = True
        self._has_std = True

    def _batch_stats(self, x):
        if self.mode == "sample":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            return mean, std
        if self.mode == "global":
            mean = x.mean()
            std = x.std()
            return mean, std
        mean = x.mean(dim=(0, 2), keepdim=True)
        std = x.std(dim=(0, 2), keepdim=True)
        return mean, std

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        if self._has_mean and self._has_std:
            mean = self.mean
            std = self.std
        else:
            mean, std = self._batch_stats(x)
        x = (x - mean) / (std + self.eps)
        return x
