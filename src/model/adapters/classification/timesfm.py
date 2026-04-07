from __future__ import annotations

from src.model.adapters.classification.base import BaseClassificationAdapter


class TimesFMClassificationAdapter(BaseClassificationAdapter):
    provider_name = "timesfm"
    model_size_to_id = {
        "500m": "google/timesfm-2.0-500m-pytorch",
    }

    def _init_provider_model(self):
        raise RuntimeError(
            "TimesFMClassificationAdapter is not implemented: "
            "TimesFM 2.0 checkpoint is not wired to a trainable classification backend in this codebase."
        )
