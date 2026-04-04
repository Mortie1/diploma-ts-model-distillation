from __future__ import annotations

from src.model.adapters.classification.base import BaseClassificationAdapter


class TimesFMHFClassificationAdapter(BaseClassificationAdapter):
    provider_name = "timesfm_hf"

    def _init_provider_model(self):
        # TimesFM-HF checkpoint is used as a frozen/external backbone for classification.
        return None
