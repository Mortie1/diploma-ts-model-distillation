from __future__ import annotations

from src.model.adapters.classification.base import BaseClassificationAdapter


class ChronosClassificationAdapter(BaseClassificationAdapter):
    provider_name = "chronos"

    def _init_provider_model(self):
        # Chronos is initialized for representation transfer into classification.
        return None
