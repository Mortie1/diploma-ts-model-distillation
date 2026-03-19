from __future__ import annotations

import logging
from typing import Optional

import torch


def _build_cuda_error_message(context: str, err: Optional[Exception] = None) -> str:
    lines = [
        f"CUDA is not usable in {context}.",
        f"PyTorch: {torch.__version__}, CUDA build: {torch.version.cuda}",
        f"torch.cuda.device_count(): {torch.cuda.device_count()}",
    ]
    if err is not None:
        lines.append(f"Runtime probe error: {type(err).__name__}: {err}")
    lines.extend(
        [
            "Checks:",
            "1) Run `nvidia-smi` in the same shell/session.",
            "2) Ensure PyTorch CUDA build is installed (not CPU-only).",
            "3) In WSL2, update Windows NVIDIA driver and `wsl --update`, then restart WSL.",
            "4) Retry with a clean shell where CUDA-related env vars are minimal.",
        ]
    )
    return "\n".join(lines)


def _can_use_cuda() -> tuple[bool, Optional[Exception]]:
    try:
        if not torch.cuda.is_available():
            return False, None
        # Runtime probe: catches cases where device_count>0 but actual init fails.
        x = torch.tensor([0.0], device="cuda:0")
        _ = x.item()
        return True, None
    except Exception as err:  # noqa: BLE001
        return False, err


def resolve_torch_device(
    requested_device: str,
    *,
    require_cuda: bool = False,
    logger: Optional[logging.Logger] = None,
    context: str = "runtime",
) -> str:
    """
    Resolve a training/inference device with CUDA runtime probing.
    """
    if requested_device != "auto":
        if requested_device.startswith("cuda"):
            ok, err = _can_use_cuda()
            if not ok:
                message = _build_cuda_error_message(context=context, err=err)
                if require_cuda:
                    raise RuntimeError(message)
                if logger is not None:
                    logger.warning(message)
                return "cpu"
        return requested_device

    ok, err = _can_use_cuda()
    if ok:
        return "cuda"

    message = _build_cuda_error_message(context=context, err=err)
    if require_cuda:
        raise RuntimeError(message)
    if logger is not None:
        logger.warning(message)
    else:
        print(message)
    return "cpu"

