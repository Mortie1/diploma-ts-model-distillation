import logging
from itertools import repeat

import numpy as np
import torch
from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.transforms import Normalize1D
from src.utils.init_utils import set_worker_seed

logger = logging.getLogger(__name__)


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    if batch_transforms is None:
        return

    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def _iter_normalize_modules(module):
    if isinstance(module, Normalize1D):
        yield module
        return
    if hasattr(module, "children"):
        for child in module.children():
            yield from _iter_normalize_modules(child)


def _compute_train_stats(train_values: np.ndarray, mode: str, eps: float):
    if mode == "global":
        mean = np.float32(train_values.mean())
        std = np.float32(max(train_values.std(), eps))
        return mean, std
    mean = train_values.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_values.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    mean = mean[..., None]
    std = std[..., None]
    return mean, std


def _compute_train_stats_from_dataset(train_dataset, mode: str, eps: float):
    """
    Compute normalization stats directly from train dataset samples.
    Expects each sample to contain `inputs` shaped [C, T] (or [T]).
    """
    n = len(train_dataset)
    if n <= 0:
        raise ValueError("Cannot compute train stats from an empty dataset.")

    sum_global = 0.0
    sumsq_global = 0.0
    count_global = 0

    sum_c = None
    sumsq_c = None
    count_c = 0

    for idx in range(n):
        sample = train_dataset[idx]
        if not isinstance(sample, dict) or "inputs" not in sample:
            raise KeyError(
                "Expected train dataset sample to be dict with key `inputs` "
                "for Normalize1D auto-fill."
            )

        x = sample["inputs"]
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        x_np = x_np.astype(np.float64, copy=False)

        if x_np.ndim == 1:
            x_np = x_np[None, :]
        elif x_np.ndim != 2:
            raise ValueError(
                f"Expected inputs with ndim 1 or 2, got shape {x_np.shape}"
            )

        if mode == "global":
            sum_global += float(x_np.sum())
            sumsq_global += float((x_np * x_np).sum())
            count_global += int(x_np.size)
        else:
            if sum_c is None:
                c = x_np.shape[0]
                sum_c = np.zeros((c,), dtype=np.float64)
                sumsq_c = np.zeros((c,), dtype=np.float64)
            if x_np.shape[0] != sum_c.shape[0]:
                raise ValueError(
                    "Inconsistent channel count while computing train stats: "
                    f"expected {sum_c.shape[0]}, got {x_np.shape[0]} at idx={idx}"
                )
            sum_c += x_np.sum(axis=1)
            sumsq_c += (x_np * x_np).sum(axis=1)
            count_c += int(x_np.shape[1])

    if mode == "global":
        if count_global <= 0:
            raise ValueError("Failed to compute global stats: zero sample count.")
        mean = np.float32(sum_global / float(count_global))
        var = max((sumsq_global / float(count_global)) - float(mean) ** 2, 0.0)
        std = np.float32(max(np.sqrt(var), eps))
        return mean, std

    if count_c <= 0 or sum_c is None or sumsq_c is None:
        raise ValueError("Failed to compute channel stats: zero sample count.")
    mean_c = (sum_c / float(count_c)).astype(np.float32, copy=False)
    var_c = (sumsq_c / float(count_c)) - mean_c.astype(np.float64) ** 2
    var_c = np.maximum(var_c, 0.0)
    std_c = np.sqrt(var_c).astype(np.float32, copy=False)
    std_c = np.maximum(std_c, np.float32(eps))
    mean = mean_c[None, :, None]
    std = std_c[None, :, None]
    return mean, std


def _autofill_normalize1d_train_stats(datasets, batch_transforms):
    if batch_transforms is None:
        return

    train_dataset = datasets.get("train")
    if train_dataset is None:
        return

    train_values = None
    if hasattr(train_dataset, "values"):
        values_attr = getattr(train_dataset, "values")
        if values_attr is not None:
            values_arr = np.asarray(values_attr, dtype=np.float32)
            if values_arr.ndim == 2:
                train_values = values_arr

    stats_cache = {}
    for transforms in batch_transforms.values():
        if transforms is None:
            continue
        for transform_module in transforms.values():
            if transform_module is None:
                continue
            for norm_module in _iter_normalize_modules(transform_module):
                if not norm_module.needs_train_stats:
                    continue
                cache_key = (norm_module.mode, float(norm_module.eps))
                if cache_key not in stats_cache:
                    if train_values is not None:
                        stats_cache[cache_key] = _compute_train_stats(
                            train_values=train_values,
                            mode=norm_module.mode,
                            eps=norm_module.eps,
                        )
                    else:
                        logger.info(
                            "Normalize1D auto-fill: computing %s stats from train dataset samples.",
                            norm_module.mode,
                        )
                        stats_cache[cache_key] = _compute_train_stats_from_dataset(
                            train_dataset=train_dataset,
                            mode=norm_module.mode,
                            eps=norm_module.eps,
                        )
                mean, std = stats_cache[cache_key]
                norm_module.set_stats(mean=mean, std=std)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    _autofill_normalize1d_train_stats(datasets, batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        if dataset_partition == "train":
            assert config.dataloader.batch_size <= len(dataset), (
                f"The batch size ({config.dataloader.batch_size}) cannot "
                f"be larger than the dataset length ({len(dataset)})"
            )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
