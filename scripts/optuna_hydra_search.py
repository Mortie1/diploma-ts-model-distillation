#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple

import optuna

METRIC_RE_TEMPLATE = r"^\s*{metric}\s*:\s*([-+eE0-9\.]+)\s*$"
OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cuda error: out of memory",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Optuna tuner for Hydra training entrypoint (train.py). "
            "Runs one subprocess per trial and optimizes by parsing metric from stdout."
        )
    )
    p.add_argument("--config-name", default="distill_train")
    p.add_argument(
        "--preset", default="distill_pamap", choices=["distill_pamap", "fm_pamap"]
    )
    p.add_argument("--study-name", default="optuna_ts")
    p.add_argument("--storage", default=None)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout-sec", type=int, default=None)
    p.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    p.add_argument("--metric-name", default="val_MacroF1")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--train-script", default="train.py")
    p.add_argument("--workdir", default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", default="tpe", choices=["tpe", "random"])
    p.add_argument("--pruner", default="median", choices=["none", "median"])
    p.add_argument(
        "--continue-on-trial-error",
        action="store_true",
        help="Do not stop study on RuntimeError in a trial; mark trial as FAIL and continue.",
    )
    p.add_argument("--writer", default="cometml")
    p.add_argument("--comet-mode", default="online", choices=["online", "offline"])
    p.add_argument("--base-overrides", nargs="*", default=[])
    p.add_argument("--mode", default="single", choices=["single", "cv"])
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds per trial when --mode=cv.",
    )
    p.add_argument(
        "--cv-seed",
        type=int,
        default=42,
        help="Base seed for fold sampling when --mode=cv.",
    )
    p.add_argument(
        "--cv-subjects",
        default="101,102,103,104,105,106,107,108",
        help="Comma-separated subject ids for PAMAP SSL folds.",
    )
    p.add_argument(
        "--enqueue-best-from",
        default=None,
        help="Optional JSON file with previous best params.",
    )
    return p.parse_args()


def default_storage(study_name: str, workdir: Path) -> str:
    db_dir = workdir / "outputs" / "optuna"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = (db_dir / (study_name + ".db")).resolve()
    return "sqlite:///" + str(db_path)


def preset_distill_pamap(trial: optuna.Trial) -> List[str]:
    hs = trial.suggest_categorical("model.hidden_dim", [256, 384, 512])
    n_heads = trial.suggest_categorical("model.n_heads", [4, 8])
    n_layers = trial.suggest_int("model.n_layers", 2, 8)
    dropout = trial.suggest_float("model.dropout", 0.05, 0.30)
    lr = trial.suggest_float("optimizer.lr", 1e-5, 5e-4, log=True)
    wd = trial.suggest_float("optimizer.weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("dataloader.batch_size", [32, 64, 128, 256])
    lambda_feat = trial.suggest_float("loss_function.lambda_feat", 1e-2, 10.0, log=True)
    warmup_ratio = trial.suggest_float("trainer.warmup_ratio", 0.02, 0.20)

    return [
        "model.hidden_dim={}".format(hs),
        "model.n_heads={}".format(n_heads),
        "model.n_layers={}".format(n_layers),
        "model.dropout={}".format(dropout),
        "optimizer.lr={}".format(lr),
        "optimizer.weight_decay={}".format(wd),
        "dataloader.batch_size={}".format(bs),
        "loss_function.lambda_feat={}".format(lambda_feat),
        "loss_function.lambda_logit=0.0",
        "trainer.warmup_ratio={}".format(warmup_ratio),
    ]


def preset_fm_pamap(trial: optuna.Trial) -> List[str]:
    lr = trial.suggest_float("optimizer.lr", 1e-5, 5e-3, log=True)
    wd = trial.suggest_float("optimizer.weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("dataloader.batch_size", [32, 64, 128, 256])
    warmup_ratio = trial.suggest_float("trainer.warmup_ratio", 0.02, 0.20)

    return [
        "optimizer.lr={}".format(lr),
        "optimizer.weight_decay={}".format(wd),
        "dataloader.batch_size={}".format(bs),
        "trainer.warmup_ratio={}".format(warmup_ratio),
    ]


PRESETS: Dict[str, Callable[[optuna.Trial], List[str]]] = {
    "distill_pamap": preset_distill_pamap,
    "fm_pamap": preset_fm_pamap,
}


def try_parse_metric(line: str, metric_name: str) -> Optional[float]:
    pat = re.compile(METRIC_RE_TEMPLATE.format(metric=re.escape(metric_name)))
    m = pat.match(line.rstrip("\n"))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def is_oom_line(line: str) -> bool:
    low = line.lower()
    return any(pat in low for pat in OOM_PATTERNS)


def make_objective(args, study_out_dir: Path):
    metric_name = args.metric_name
    metric_sign = 1.0 if args.direction == "maximize" else -1.0
    train_script = str((Path(args.workdir) / args.train_script).resolve())
    preset_fn = PRESETS[args.preset]

    def _run_one_train(
        trial: optuna.Trial,
        run_name: str,
        trial_log,
        extra_overrides: Optional[List[str]] = None,
    ) -> float:
        extra_overrides = extra_overrides or []
        trial_overrides = preset_fn(trial)
        tags = "[optuna,{study},trial_{trial},{preset}]".format(
            study=args.study_name,
            trial=trial.number,
            preset=args.preset,
        )
        cmd = [
            args.python_bin,
            train_script,
            "-cn={}".format(args.config_name),
            "writer={}".format(args.writer),
            "writer.mode={}".format(args.comet_mode),
            "writer.run_name={}".format(run_name),
            "writer.tags={}".format(tags),
            *args.base_overrides,
            *trial_overrides,
            *extra_overrides,
        ]

        best_metric: Optional[float] = None
        metric_step = 0
        oom_detected = False
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        trial_log.write("CMD: " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=args.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            trial_log.write(line)
            if is_oom_line(line):
                oom_detected = True
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=30)
                raise optuna.TrialPruned(
                    "Pruned due to OOM for run `{}`.".format(run_name)
                )
            value = try_parse_metric(line, metric_name)
            if value is not None:
                if best_metric is None:
                    best_metric = value
                else:
                    if args.direction == "maximize":
                        best_metric = max(best_metric, value)
                    else:
                        best_metric = min(best_metric, value)
                trial.report(value, step=metric_step)
                metric_step += 1
                if trial.should_prune():
                    proc.terminate()
                    proc.wait(timeout=30)
                    raise optuna.TrialPruned(
                        "Pruned at step={} with {}={}".format(
                            metric_step, metric_name, value
                        )
                    )
        ret = proc.wait()

        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("run_log_file", str(trial_log))
        trial.set_user_attr("cmd", " ".join(cmd))

        if ret != 0:
            if oom_detected:
                raise optuna.TrialPruned(
                    "Pruned due to OOM for run `{}` (non-zero exit).".format(run_name)
                )
            raise RuntimeError(
                "Trial process failed with exit code {}. See log.".format(ret)
            )
        if best_metric is None:
            raise RuntimeError("Metric `{}` not found in logs.".format(metric_name))
        return best_metric

    def _sample_cv_folds(trial_number: int) -> List[Tuple[int, int]]:
        subjects = [int(s.strip()) for s in args.cv_subjects.split(",") if s.strip()]
        pairs = [(v, t) for v, t in itertools.permutations(subjects, 2)]
        rng = random.Random(args.cv_seed + trial_number)
        rng.shuffle(pairs)
        return pairs[: max(1, int(args.cv_folds))]

    def objective(trial: optuna.Trial) -> float:
        trial_log_path = (
            study_out_dir / "trials" / ("trial_{:04d}.log".format(trial.number))
        )
        trial_log_path.parent.mkdir(parents=True, exist_ok=True)

        with trial_log_path.open("w", encoding="utf-8") as trial_log:
            if args.mode == "single":
                run_name = "opt-{}-t{:04d}".format(args.study_name, trial.number)
                best_metric = _run_one_train(
                    trial=trial,
                    run_name=run_name,
                    trial_log=trial_log,
                )
                trial.set_user_attr("log_file", str(trial_log_path))
                trial.set_user_attr("best_{}".format(metric_name), best_metric)
                trial.set_user_attr(
                    "signed_best_{}".format(metric_name), metric_sign * best_metric
                )
                return best_metric

            folds = _sample_cv_folds(trial.number)
            fold_metrics = []
            for fold_idx, (val_subj, test_subj) in enumerate(folds, start=1):
                run_name = "opt-{}-t{:04d}-f{:02d}-v{}-t{}".format(
                    args.study_name, trial.number, fold_idx, val_subj, test_subj
                )
                extra = [
                    "datasets.train.protocol_mode=ssl_wearables",
                    "datasets.val.protocol_mode=ssl_wearables",
                    "datasets.test.protocol_mode=ssl_wearables",
                    "datasets.train.val_subjects=[{}]".format(val_subj),
                    "datasets.val.val_subjects=[{}]".format(val_subj),
                    "datasets.test.val_subjects=[{}]".format(val_subj),
                    "datasets.train.test_subjects=[{}]".format(test_subj),
                    "datasets.val.test_subjects=[{}]".format(test_subj),
                    "datasets.test.test_subjects=[{}]".format(test_subj),
                ]
                trial_log.write(
                    "\n=== CV fold {}/{} (val={}, test={}) ===\n".format(
                        fold_idx, len(folds), val_subj, test_subj
                    )
                )
                fold_best = _run_one_train(
                    trial=trial,
                    run_name=run_name,
                    trial_log=trial_log,
                    extra_overrides=extra,
                )
                fold_metrics.append(float(fold_best))

                fold_mean = float(mean(fold_metrics))
                trial.report(fold_mean, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        "Pruned at cv fold {} with mean {}={}".format(
                            fold_idx, metric_name, fold_mean
                        )
                    )

            cv_value = float(mean(fold_metrics))
            trial.set_user_attr("log_file", str(trial_log_path))
            trial.set_user_attr("cv_fold_metrics", fold_metrics)
            trial.set_user_attr("best_{}".format(metric_name), cv_value)
            trial.set_user_attr(
                "signed_best_{}".format(metric_name), metric_sign * cv_value
            )
            return cv_value

    return objective


def maybe_enqueue_from_json(study: optuna.Study, path: Optional[str]):
    if not path:
        return
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("--enqueue-best-from not found: {}".format(p))
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--enqueue-best-from JSON must be an object of param->value.")
    study.enqueue_trial(payload)


def main():
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    storage = args.storage or default_storage(args.study_name, workdir)

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
    )
    maybe_enqueue_from_json(study, args.enqueue_best_from)

    study_out_dir = workdir / "outputs" / "optuna" / args.study_name
    study_out_dir.mkdir(parents=True, exist_ok=True)

    print("study_name={}".format(args.study_name))
    print("storage={}".format(storage))
    print("n_trials={} timeout_sec={}".format(args.n_trials, args.timeout_sec))
    print("direction={} metric={}".format(args.direction, args.metric_name))
    print("preset={} config={}".format(args.preset, args.config_name))
    print("study_out_dir={}".format(study_out_dir))

    objective = make_objective(args, study_out_dir)
    catch_exceptions = (RuntimeError,) if args.continue_on_trial_error else ()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_sec,
        catch=catch_exceptions,
    )

    best = study.best_trial
    print("\n=== Best Trial ===")
    print("number={}".format(best.number))
    print("value={}".format(best.value))
    print("params={}".format(best.params))
    print("user_attrs={}".format(best.user_attrs))

    best_json = {
        "number": best.number,
        "value": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    best_path = study_out_dir / "best_trial.json"
    best_path.write_text(
        json.dumps(best_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("saved_best={}".format(best_path))


if __name__ == "__main__":
    main()
