from __future__ import annotations

from typing import Optional

import torch

from src.model.adapters.classification.base import BaseClassificationAdapter


class MomentClassificationAdapter(BaseClassificationAdapter):
    provider_name = "moment"
    model_size_to_id = {
        "small": "AutonLab/MOMENT-1-small",
        "base": "AutonLab/MOMENT-1-base",
        "large": "AutonLab/MOMENT-1-large",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unfreeze_classification_head()

    def _init_provider_model(self):
        try:
            from momentfm import MOMENTPipeline

            model = MOMENTPipeline.from_pretrained(
                self.model_id,
                model_kwargs={
                    "task_name": "classification",
                    "n_channels": self.in_channels,
                    "num_class": self.n_classes,
                },
            )
            if hasattr(model, "init"):
                model.init()
            return model
        except Exception:
            return None

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None:
            return None
        mask = torch.ones(x.size(0), x.size(-1), dtype=torch.long, device=x.device)
        calls = (
            lambda: self.provider_model(x_enc=x, input_mask=mask),
            lambda: self.provider_model(x_enc=x),
            lambda: self.provider_model(x),
        )
        for fn in calls:
            try:
                out = fn()
                if isinstance(out, dict) and out.get("logits") is not None:
                    return out["logits"]
                if hasattr(out, "logits") and out.logits is not None:
                    return out.logits
            except Exception:
                continue
        return None

    def _unfreeze_classification_head(self):
        if self.provider_model is None:
            return
        head = getattr(self.provider_model, "head", None)
        if head is None:
            return
        for p in head.parameters():
            p.requires_grad = True
