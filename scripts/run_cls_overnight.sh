#!/usr/bin/env bash
set -euo pipefail

# Night runner for classification FM experiments.
# Modes:
#   MODE=debug : offline smoke + batch autotune
#   MODE=main  : online experiments using tuned batch sizes
#   MODE=all   : debug then main

source .venv/bin/activate

MODE="${MODE:-all}"           # debug|main|all
OUT_DIR="${OUT_DIR:-/tmp/cls_overnight}"
mkdir -p "$OUT_DIR"

DEBUG_EPOCHS="${DEBUG_EPOCHS:-1}"
DEBUG_EPOCH_LEN="${DEBUG_EPOCH_LEN:-50}"
MAIN_EPOCHS="${MAIN_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"

# Start batch sizes (will be tuned in debug mode).
START_BS_MOMENT="${START_BS_MOMENT:-64}"
START_BS_CHRONOS2="${START_BS_CHRONOS2:-16}"
START_BS_TIREX="${START_BS_TIREX:-32}"
START_BS_TSPULSE="${START_BS_TSPULSE:-16}"
MAX_BS_CAP="${MAX_BS_CAP:-1024}"

TUNED_TSV="$OUT_DIR/tuned_batch_sizes.tsv"
RESULTS_TSV="$OUT_DIR/results.tsv"

# family|target|size|start_bs
MODELS=(
  "moment|src.model.MomentClassificationAdapter|small|${START_BS_MOMENT}"
  "moment|src.model.MomentClassificationAdapter|base|${START_BS_MOMENT}"
  "chronos2|src.model.Chronos2ClassificationAdapter|base|${START_BS_CHRONOS2}"
  "tirex|src.model.TiRexClassificationAdapter|base|${START_BS_TIREX}"
  "tspulse|src.model.TSPulseClassificationAdapter|r1|${START_BS_TSPULSE}"
)

DATASETS=(
  "pamap2"
  "ptbxl"
  "insect_wingbeat"
  "cwru_bearing"
)

is_oom_log() {
  local log_file="$1"
  rg -qi "out of memory|cuda out of memory|oom on batch|memoryallocation" "$log_file"
}

# returns: train_len n_classes in_channels
dataset_meta() {
  local ds="$1"
  python - "$ds" <<'PY'
import ast
import csv
import numpy as np
from pathlib import Path
import sys

from src.datasets.download import maybe_download_ucr
from src.datasets.classification.cwru_bearing_dataset import CWRUBearingDataset


def meta_insect():
    root = Path("data/raw/ucr")
    name = "InsectWingbeatSound"
    candidates = [
        root / name / f"{name}_TRAIN.tsv",
        root / name / f"{name}_TRAIN.txt",
        root / f"{name}_TRAIN.tsv",
        root / f"{name}_TRAIN.txt",
    ]
    train_file = next((p for p in candidates if p.exists()), None)
    if train_file is None:
        maybe_download_ucr(dataset_name=name, root=root)
        train_file = next((p for p in candidates if p.exists()), None)
    if train_file is None:
        checked = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find InsectWingbeat train split. Checked: {checked}")
    arr = np.loadtxt(train_file, delimiter=None)
    y = arr[:, 0].astype(np.int64)
    return len(y), len(np.unique(y)), 1


def meta_cwru():
    ds = CWRUBearingDataset(root="data/raw/cwru_bearing", split="train", normalize=True)
    return len(ds), int(ds.labels.max()) + 1, 1


def meta_pamap2():
    # defaults used in src/configs/datasets/pamap2.yaml (feature_set=all)
    return 0, 12, 52


def meta_ptbxl():
    root = Path("data/raw/ptbxl")
    db = root / "ptbxl_database.csv"
    scp = root / "scp_statements.csv"
    if not (db.exists() and scp.exists()):
        # Download happens in dataset class at runtime; for metadata we only need class count estimate.
        # Conservative defaults for model head:
        return 0, 5, 12

    code_to_super = {}
    with scp.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if str(r.get("diagnostic", "")).strip() not in {"1", "1.0", "True", "true"}:
                continue
            code = str(r.get("scp_code") or r.get("Unnamed: 0") or r.get("") or "").strip()
            superc = str(r.get("diagnostic_class", "")).strip()
            if code and superc:
                code_to_super[code] = superc

    labels = set()
    n_train = 0
    with db.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                fold = int(float(r["strat_fold"]))
            except Exception:
                continue
            if fold not in set(range(1, 9)):
                continue
            n_train += 1
            try:
                scp_codes = ast.literal_eval(r["scp_codes"])
            except Exception:
                continue
            for code in scp_codes.keys():
                s = code_to_super.get(str(code))
                if s:
                    labels.add(s)
    n_classes = len(labels) if labels else 5
    return n_train, n_classes, 12


ds = sys.argv[1]
if ds == "insect_wingbeat":
    tl, nc, ic = meta_insect()
elif ds == "cwru_bearing":
    tl, nc, ic = meta_cwru()
elif ds == "pamap2":
    tl, nc, ic = meta_pamap2()
elif ds == "ptbxl":
    tl, nc, ic = meta_ptbxl()
else:
    raise SystemExit(f"Unknown dataset {ds}")

print(tl, nc, ic)
PY
}

make_run_name() {
  local family="$1" size="$2" ds="$3" stage="$4" change="$5"
  local ds_short="$ds"
  ds_short="${ds_short/pamap2/pamap}"
  ds_short="${ds_short/insect_wingbeat/insect}"
  ds_short="${ds_short/cwru_bearing/cwru}"
  ds_short="${ds_short/ptbxl/ptbxl}"
  local size_short="$size"
  size_short="${size_short/small/s}"
  size_short="${size_short/base/b}"
  size_short="${size_short/large/l}"
  local name="${family}-${size_short}-${ds_short}-${stage}-${change}"
  echo "${name:0:40}"
}

run_once() {
  local ds="$1" family="$2" target="$3" size="$4" bs="$5" stage="$6" n_epochs="$7" epoch_len="$8" writer_mode="$9"
  local train_len="${10}" n_classes="${11}" in_channels="${12}"

  local run_name
  run_name="$(make_run_name "$family" "$size" "$ds" "$stage" "bs${bs}")"
  local tags="[classification,${family},${family}-${size},${ds}]"
  local log_file="$OUT_DIR/${stage}_${family}_${size}_${ds}_bs${bs}.log"

  local writer_overrides=()
  if [[ "$writer_mode" == "local" ]]; then
    writer_overrides+=("writer=local")
  else
    writer_overrides+=("writer=cometml" "writer.mode=online" "writer.run_name=${run_name}" "writer.tags=${tags}")
  fi

  local bs_eff="$bs"
  if [[ "$train_len" != "0" ]] && (( bs_eff > train_len )); then
    bs_eff="$train_len"
  fi
  if (( bs_eff < 1 )); then
    bs_eff=1
  fi

  local cmd=(
    python train.py
    -cn=fm_train
    "datasets=${ds}"
    "model._target_=${target}"
    "model.model_size=${size}"
    "model.n_classes=${n_classes}"
    "model.in_channels=${in_channels}"
    "model.require_provider_model=true"
    "model.fail_on_provider_fallback=true"
    "trainer.require_cuda=${REQUIRE_CUDA}"
    "trainer.n_epochs=${n_epochs}"
    "trainer.epoch_len=${epoch_len}"
    "trainer.log_step=1"
    "trainer.monitor=max val_MacroF1"
    "dataloader.batch_size=${bs_eff}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "dataloader.pin_memory=true"
    "trainer.amp_enabled=true"
    "trainer.amp_dtype=bf16"
    "trainer.compile_enabled=false"
  )

  cmd+=("${writer_overrides[@]}")

  set +e
  "${cmd[@]}" >"$log_file" 2>&1
  local rc=$?
  set -e

  local status="ok"
  if [[ $rc -ne 0 ]]; then
    status="fail"
  fi
  echo "$status|$log_file|$run_name|$bs_eff"
}

run_debug_tune() {
  echo -e "dataset\tfamily\tsize\ttarget\tn_classes\tin_channels\tstart_bs\ttuned_bs\tstatus\tlog_file" > "$TUNED_TSV"

  for ds in "${DATASETS[@]}"; do
    read -r train_len n_classes in_channels <<< "$(dataset_meta "$ds")"

    for row in "${MODELS[@]}"; do
      IFS='|' read -r family target size start_bs <<< "$row"

      local_bs="$start_bs"
      best_bs=0
      status="fail"
      last_log=""

      while (( local_bs >= 1 )); do
        out="$(run_once "$ds" "$family" "$target" "$size" "$local_bs" "dbg" "$DEBUG_EPOCHS" "$DEBUG_EPOCH_LEN" "local" "$train_len" "$n_classes" "$in_channels")"
        run_status="${out%%|*}"
        rest="${out#*|}"
        log_file="${rest%%|*}"
        rest="${rest#*|}"
        _run_name="${rest%%|*}"
        rest="${rest#*|}"
        bs_eff="$rest"
        last_log="$log_file"

        if [[ "$run_status" == "ok" ]]; then
          best_bs="$bs_eff"
          # Try to increase up to cap.
          next_bs=$(( local_bs * 2 ))
          if (( next_bs > MAX_BS_CAP )); then
            status="ok"
            break
          fi
          local_bs="$next_bs"
          status="ok"
          continue
        fi

        if is_oom_log "$log_file"; then
          local_bs=$(( local_bs / 2 ))
          if (( local_bs < 1 )); then
            break
          fi
          continue
        fi

        # Non-OOM failure => stop tuning this pair.
        status="fail"
        break
      done

      if (( best_bs <= 0 )); then
        best_bs=1
      fi

      echo -e "${ds}\t${family}\t${size}\t${target}\t${n_classes}\t${in_channels}\t${start_bs}\t${best_bs}\t${status}\t${last_log}" >> "$TUNED_TSV"
      echo "[debug] ${ds} ${family}/${size}: tuned_bs=${best_bs} status=${status}"
    done
  done
}

run_main_matrix() {
  if [[ ! -f "$TUNED_TSV" ]]; then
    echo "Tuned batch file not found: $TUNED_TSV"
    exit 1
  fi

  echo -e "dataset\tfamily\tsize\ttarget\tbatch_size\tstatus\trun_name\tlog_file" > "$RESULTS_TSV"

  tail -n +2 "$TUNED_TSV" | while IFS=$'\t' read -r ds family size target n_classes in_channels _start_bs tuned_bs tuned_status _log; do
    if [[ "$tuned_status" != "ok" ]]; then
      echo -e "${ds}\t${family}\t${size}\t${target}\t${tuned_bs}\tskip\t-\t-" >> "$RESULTS_TSV"
      continue
    fi

    read -r train_len _nc _ic <<< "$(dataset_meta "$ds")"
    out="$(run_once "$ds" "$family" "$target" "$size" "$tuned_bs" "main" "$MAIN_EPOCHS" "null" "comet" "$train_len" "$n_classes" "$in_channels")"
    run_status="${out%%|*}"
    rest="${out#*|}"
    log_file="${rest%%|*}"
    rest="${rest#*|}"
    run_name="${rest%%|*}"
    rest="${rest#*|}"
    bs_eff="$rest"

    echo -e "${ds}\t${family}\t${size}\t${target}\t${bs_eff}\t${run_status}\t${run_name}\t${log_file}" >> "$RESULTS_TSV"
    echo "[main] ${ds} ${family}/${size}: status=${run_status} bs=${bs_eff} run=${run_name}"
  done
}

case "$MODE" in
  debug)
    run_debug_tune
    ;;
  main)
    run_main_matrix
    ;;
  all)
    run_debug_tune
    run_main_matrix
    ;;
  *)
    echo "Unknown MODE=$MODE (use debug|main|all)"
    exit 2
    ;;
esac

echo "Done. Tuned: $TUNED_TSV"
echo "Done. Results: $RESULTS_TSV"
