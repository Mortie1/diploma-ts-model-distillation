from __future__ import annotations

from typing import Optional

import torch

from src.model.adapters.classification.base import BaseClassificationAdapter


class MomentClassificationAdapter(BaseClassificationAdapter):
    provider_name = "moment"

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
