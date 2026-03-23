from __future__ import annotations

import re
from typing import Any

from omegaconf import OmegaConf


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return slug[:180] if slug else "run"


def _short_model_id(model_id: str | None) -> str:
    if not model_id:
        return "model"
    return model_id.split("/")[-1]


def _build_short_name(cfg_dict: dict) -> str:
    task = _as_dict(cfg_dict.get("task")).get("name", "task")
    model = _as_dict(cfg_dict.get("model"))
    provider = model.get("provider", "model")
    horizon = model.get("horizon")
    finetune_mode = model.get("finetune_mode", "none")
    distill_enabled = _as_dict(cfg_dict.get("distillation")).get("enabled", False)

    parts = [str(task), str(provider)]
    if horizon is not None:
        parts.append(f"h{horizon}")
    if finetune_mode and finetune_mode != "none":
        parts.append(finetune_mode)
    if distill_enabled:
        parts.append("distill")
    return "-".join(parts)


def _detect_main_change(cfg_dict: dict) -> str:
    model = _as_dict(cfg_dict.get("model"))
    optimizer = _as_dict(cfg_dict.get("optimizer"))
    dataloader = _as_dict(cfg_dict.get("dataloader"))

    finetune_mode = model.get("finetune_mode", "none")
    if finetune_mode != "none":
        return f"ft-{finetune_mode}"

    residual_scale = model.get("residual_scale")
    if isinstance(residual_scale, (int, float)) and float(residual_scale) != 1.0:
        return f"res{residual_scale}"

    lr = optimizer.get("lr")
    if isinstance(lr, (int, float)) and float(lr) != 1e-3:
        return f"lr{lr}"

    bs = dataloader.get("batch_size")
    if isinstance(bs, int) and bs != 64:
        return f"bs{bs}"

    provider = model.get("provider")
    if provider and provider != "chronos":
        return str(provider)

    return "base"


def _build_expected_effect(cfg_dict: dict) -> str:
    model = _as_dict(cfg_dict.get("model"))
    optimizer = _as_dict(cfg_dict.get("optimizer"))
    dataloader = _as_dict(cfg_dict.get("dataloader"))

    lr = optimizer.get("lr")
    bs = dataloader.get("batch_size")
    provider = model.get("provider", "model")
    finetune_mode = model.get("finetune_mode", "none")

    lr_msg = "default convergence behavior"
    if isinstance(lr, (int, float)):
        if lr >= 1e-3:
            lr_msg = "faster convergence of trainable head, with higher instability risk"
        elif lr <= 1e-4:
            lr_msg = "more stable but slower convergence"

    bs_msg = "standard throughput"
    if isinstance(bs, int):
        if bs >= 128:
            bs_msg = "higher throughput and fewer eval steps, with possible quality trade-offs"
        elif bs <= 32:
            bs_msg = "lower throughput, potentially better gradient noise regularization"

    return (
        f"Expected effect: {lr_msg}; {bs_msg}. "
        f"Rationale: this run uses provider={provider}, finetune_mode={finetune_mode}."
    )


def apply_auto_run_metadata(config) -> None:
    """
    Populate run metadata for trackers.

    - display_name: human-readable name for trackers
    - run_name: filesystem-safe name for local save_dir
    - description: short rationale for the run
    """
    cfg_dict = _as_dict(config)
    writer = _as_dict(cfg_dict.get("writer"))
    auto_name = writer.get("auto_name", False)

    if not auto_name:
        return

    main_change = writer.get("main_change") or _detect_main_change(cfg_dict)
    base_short_name = writer.get("short_name") or _build_short_name(cfg_dict)
    short_name = f"{base_short_name}-{main_change}"
    lr = _as_dict(cfg_dict.get("optimizer")).get("lr", "na")
    bs = _as_dict(cfg_dict.get("dataloader")).get("batch_size", "na")
    model_id = _short_model_id(_as_dict(cfg_dict.get("model")).get("model_id"))

    display_name = f"{short_name} / lr={lr} / bs={bs}"
    run_name = _slugify(f"{short_name}__{model_id}__lr={lr}__bs={bs}")
    description = (
        f"Main change: {main_change}. "
        + _build_expected_effect(cfg_dict)
    )

    OmegaConf.set_struct(config, False)
    config.writer.main_change = main_change
    config.writer.display_name = display_name
    config.writer.run_name = run_name
    config.writer.description = description
    OmegaConf.set_struct(config, True)
