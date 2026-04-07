from __future__ import annotations

from typing import Optional

import torch

from src.model.adapters.classification.base import BaseClassificationAdapter


class TSPulseClassificationAdapter(BaseClassificationAdapter):
    provider_name = "tspulse"
    model_size_to_id = {
        "r1": "ibm-granite/granite-timeseries-tspulse-r1",
    }

    def _init_provider_model(self):
        # IBM Granite TSPulse may expose custom APIs; try HF auto loading with remote code.
        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize TSPulse provider for model_id `{self.model_id}`."
            ) from e

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
            except TypeError:
                # Try next supported call signature.
                continue
            except Exception as e:
                raise RuntimeError(
                    "TSPulse forward failed while producing classification logits."
                ) from e
        raise RuntimeError(
            "TSPulse provider forward completed, but no logits were returned for any known call signature."
        )
