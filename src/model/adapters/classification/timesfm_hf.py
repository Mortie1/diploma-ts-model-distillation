from __future__ import annotations

from src.model.adapters.classification.base import BaseClassificationAdapter


class TimesFMHFClassificationAdapter(BaseClassificationAdapter):
    provider_name = "timesfm_hf"
    model_size_to_id = {
        "200m": "google/timesfm-2.5-200m-pytorch",
        "500m": "google/timesfm-2.5-500m-pytorch",
    }

    def _init_provider_model(self):
        # TimesFM-HF checkpoint is used as a frozen/external backbone for classification.
        return None
