from __future__ import annotations

from typing import Optional

import torch

from src.model.adapters.classification.base import BaseClassificationAdapter


class TSPulseClassificationAdapter(BaseClassificationAdapter):
    provider_name = "tspulse"

    def _init_provider_model(self):
        # IBM Granite TSPulse may expose custom APIs; try HF auto loading with remote code.
        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception:
            return None

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None:
            return None
        calls = (
            lambda: self.provider_model(x),
            lambda: self.provider_model(input_values=x),
            lambda: self.provider_model(inputs=x),
            lambda: self.provider_model(x_enc=x),
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
