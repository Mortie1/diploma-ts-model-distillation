from __future__ import annotations

from src.model.adapters.classification.base import BaseClassificationAdapter


class ChronosClassificationAdapter(BaseClassificationAdapter):
    provider_name = "chronos"
    model_size_to_id = {
        "tiny": "amazon/chronos-t5-tiny",
        "mini": "amazon/chronos-t5-mini",
        "small": "amazon/chronos-t5-small",
        "base": "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
    }

    def _init_provider_model(self):
        raise RuntimeError(
            "ChronosClassificationAdapter is not implemented: "
            "Chronos T5 pipeline does not provide a trainable classification backend in this codebase."
        )
