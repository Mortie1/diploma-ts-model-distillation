#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.data_utils import get_dataloaders  # noqa: E402
from src.model.modules import TransformerEncoder  # noqa: E402
from src.trainer.base_trainer import BaseTrainer  # noqa: E402
from src.utils.device import resolve_torch_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Recompute fold test metrics from saved model_best checkpoints using "
            "global (full-dataset) classification scores."
        )
    )
    p.add_argument("--results-tsv", type=Path, required=True)
    p.add_argument(
        "--output-tsv",
        type=Path,
        default=None,
        help="Output TSV path. Default: <results-dir>/results_recomputed.tsv",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for dataloader.batch_size during recompute.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for dataloader.num_workers during recompute.",
    )
    return p.parse_args()


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _resolve_checkpoint_path(row: dict, ckpt_cfg=None) -> Path:
    explicit_ckpt = str(row.get("checkpoint_path", "")).strip()
    if explicit_ckpt:
        return Path(explicit_ckpt)

    run_name = str(row.get("run_name", "")).strip()
    if not run_name:
        raise ValueError("Missing run_name in results row.")
    save_dir = "saved"
    if ckpt_cfg is not None:
        save_dir = str(ckpt_cfg.get("trainer", {}).get("save_dir", "saved"))
    return Path(save_dir) / run_name / "model_best.pth"


def _load_model_for_checkpoint(ckpt_path: Path, device: str):
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config")
    if cfg is None:
        raise RuntimeError(f"Checkpoint {ckpt_path} has no `config`.")

    cfg = OmegaConf.create(cfg)
    state_dict = checkpoint["state_dict"]
    state_kind = str(checkpoint.get("state_dict_kind", "full")).lower()
    model = instantiate(cfg.model).to(device)
    try:
        if state_kind == "trainable_only":
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    except RuntimeError as e:
        target = str(cfg.model.get("_target_", ""))
        is_student = (
            target.endswith("StudentClassifier")
            or target == "src.model.StudentClassifier"
        )
        if not is_student:
            raise
        print(
            "State dict load failed for StudentClassifier; switching to legacy compatibility model. "
            f"reason={type(e).__name__}: {e}"
        )
        model = _build_legacy_student_from_state_dict(cfg, state_dict, device)
        if state_kind == "trainable_only":
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

    model.eval()
    return checkpoint, cfg, model


class _LegacyStudentClassifier(nn.Module):
    """
    Compatibility model for older StudentClassifier checkpoints where frontend
    consumed all channels directly (no channel-independent reshape).
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hidden_dim: int,
        output_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        ff_mult: int,
        use_swiglu: bool,
        use_rope: bool,
        rope_base: int,
        k1: int,
        k2: int,
    ):
        super().__init__()
        p1 = k1 // 2
        p2 = k2 // 2
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=k1, padding=p1, stride=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k2, padding=p2, stride=1),
            nn.GELU(),
        )
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ff_mult=ff_mult,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.up = nn.Linear(hidden_dim, output_dim)
        self.head = nn.Linear(output_dim, n_classes)

    def forward(self, inputs: torch.Tensor, **batch):
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3:
            raise ValueError(
                f"LegacyStudentClassifier expects [B, C, T] or [B, T], got {tuple(inputs.shape)}."
            )
        feats = self.feature(inputs).transpose(1, 2)
        encoded = self.encoder(feats)
        pooled = encoded.mean(dim=1)
        student_hidden = self.up(pooled)
        logits = self.head(torch.nn.functional.gelu(student_hidden))
        return {
            "logits": logits,
            "student_hidden": student_hidden,
            "student_pred": logits,
        }


def _infer_encoder_layers_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    max_idx = -1
    for key in state_dict.keys():
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                max_idx = max(max_idx, int(parts[2]))
    return max_idx + 1 if max_idx >= 0 else 1


def _build_legacy_student_from_state_dict(cfg, state_dict, device: str):
    w1 = state_dict.get("feature.0.weight")
    w2 = state_dict.get("feature.2.weight")
    up_w = state_dict.get("up.weight")
    head_w = state_dict.get("head.weight")
    if w1 is None or w2 is None or up_w is None or head_w is None:
        raise RuntimeError(
            "Cannot build legacy student: missing expected state_dict keys."
        )

    in_channels = int(w1.shape[1])
    hidden_dim = int(w1.shape[0])
    k1 = int(w1.shape[2])
    k2 = int(w2.shape[2])
    output_dim = int(up_w.shape[0])
    n_classes = int(head_w.shape[0])
    n_layers = _infer_encoder_layers_from_state_dict(state_dict)
    n_heads = int(cfg.model.get("n_heads", 4))
    dropout = float(cfg.model.get("dropout", 0.1))
    ff_mult = int(cfg.model.get("ff_mult", 4))
    use_swiglu = bool(cfg.model.get("use_swiglu", False))
    use_rope = bool(cfg.model.get("use_rope", True))
    rope_base = int(cfg.model.get("rope_base", 10_000))

    model = _LegacyStudentClassifier(
        in_channels=in_channels,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        ff_mult=ff_mult,
        use_swiglu=use_swiglu,
        use_rope=use_rope,
        rope_base=rope_base,
        k1=k1,
        k2=k2,
    ).to(device)
    return model


def _recompute_test_metrics(cfg, model, device: str):
    dataloaders, batch_transforms = get_dataloaders(cfg, device)
    test_loader = dataloaders["test"]
    transforms = None
    if batch_transforms is not None:
        transforms = batch_transforms.get("inference")

    logits_all = []
    targets_all = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="recompute-test", leave=False):
            for k in cfg.trainer.device_tensors:
                batch[k] = batch[k].to(device)
            if transforms is not None:
                for name, transform in transforms.items():
                    if transform is not None and name in batch:
                        batch[name] = transform(batch[name])

            out = model(**batch)
            logits = out["logits"]
            targets = batch["targets"]
            if not BaseTrainer._is_classification_logits_targets(logits, targets):
                raise RuntimeError("Invalid logits/targets shape for classification")
            logits_all.append(logits.detach().cpu())
            targets_all.append(targets.detach().cpu())

    if not logits_all:
        return {
            "test_Accuracy": float("nan"),
            "test_MacroF1": float("nan"),
            "test_MicroF1": float("nan"),
        }

    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    s = BaseTrainer._classification_scores_from_logits_targets(logits, targets)
    return {
        "test_Accuracy": float(s["Accuracy"]),
        "test_MacroF1": float(s["MacroF1"]),
        "test_MicroF1": float(s["MicroF1"]),
    }


def _mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return float(vals[0]), 0.0, 1
    return float(mean(vals)), float(stdev(vals)), len(vals)


def main():
    args = parse_args()

    rows = _load_rows(args.results_tsv)
    if len(rows) == 0:
        raise RuntimeError(f"No rows in {args.results_tsv}")
    run_name_counts = Counter(str(r.get("run_name", "")).strip() for r in rows)
    duplicate_run_names = [k for k, v in run_name_counts.items() if k and v > 1]
    if duplicate_run_names:
        print(
            "WARNING: Duplicate run_name values found in results TSV. "
            "Rows with same run_name will point to the same checkpoint unless "
            "`checkpoint_path` column is provided."
        )
        for rn in duplicate_run_names:
            print(f"  duplicate run_name: {rn} (count={run_name_counts[rn]})")

    out_tsv = args.output_tsv
    if out_tsv is None:
        out_tsv = args.results_tsv.parent / "results_recomputed.tsv"

    device = resolve_torch_device(
        args.device,
        require_cuda=args.require_cuda,
        logger=None,
        context="recompute",
    )
    print(f"Using device={device}")

    out_rows = []
    for row in rows:
        fold = row.get("fold", "")
        run_name = row.get("run_name", "")
        ckpt_path = None
        model_target = ""
        model_id = ""
        model_size = ""
        try:
            ckpt_path = _resolve_checkpoint_path(row)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            checkpoint, cfg, model = _load_model_for_checkpoint(ckpt_path, device)
            model_target = str(cfg.model.get("_target_", ""))
            model_id = str(cfg.model.get("model_id", ""))
            model_size = str(cfg.model.get("model_size", ""))

            # Optional dataloader overrides.
            if args.batch_size is not None:
                cfg.dataloader.batch_size = int(args.batch_size)
            if args.num_workers is not None:
                cfg.dataloader.num_workers = int(args.num_workers)

            metrics = _recompute_test_metrics(cfg, model, str(device))
            status = "ok_recomputed"
        except Exception as e:
            metrics = {
                "test_Accuracy": float("nan"),
                "test_MacroF1": float("nan"),
                "test_MicroF1": float("nan"),
            }
            status = f"fail_recompute: {type(e).__name__}: {e}"

        out_rows.append(
            {
                **row,
                "status_recomputed": status,
                "checkpoint_path_recomputed": str(ckpt_path) if ckpt_path else "",
                "model_target_recomputed": model_target,
                "model_id_recomputed": model_id,
                "model_size_recomputed": model_size,
                "test_Accuracy_recomputed": metrics["test_Accuracy"],
                "test_MacroF1_recomputed": metrics["test_MacroF1"],
                "test_MicroF1_recomputed": metrics["test_MicroF1"],
            }
        )
        print(
            f"fold={fold} run={run_name} -> {status} "
            f"model={model_target} model_id={model_id or '-'} model_size={model_size or '-'} "
            f"ckpt={ckpt_path} macro={metrics['test_MacroF1']}"
        )

    fieldnames = list(out_rows[0].keys())
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    ok = [
        r
        for r in out_rows
        if str(r.get("status_recomputed", "")).strip() == "ok_recomputed"
    ]
    acc_vals = [float(r["test_Accuracy_recomputed"]) for r in ok]
    macro_vals = [float(r["test_MacroF1_recomputed"]) for r in ok]
    micro_vals = [float(r["test_MicroF1_recomputed"]) for r in ok]

    acc_m, acc_s, acc_n = _mean_std(acc_vals)
    mac_m, mac_s, mac_n = _mean_std(macro_vals)
    mic_m, mic_s, mic_n = _mean_std(micro_vals)

    print("\n=== Recomputed Aggregated (ok_recomputed folds) ===")
    print(f"total_folds={len(out_rows)} ok_folds={len(ok)}")
    print(f"test_Accuracy: n={acc_n} mean={acc_m} std={acc_s}")
    print(f"test_MacroF1: n={mac_n} mean={mac_m} std={mac_s}")
    print(f"test_MicroF1: n={mic_n} mean={mic_m} std={mic_s}")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
