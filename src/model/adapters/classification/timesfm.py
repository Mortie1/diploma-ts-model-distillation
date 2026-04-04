from __future__ import annotations

from src.model.adapters.classification.base import BaseClassificationAdapter


class TimesFMClassificationAdapter(BaseClassificationAdapter):
    provider_name = "timesfm"

    def _init_provider_model(self):
        # TimesFM is initialized for representation transfer into classification.
        return None
