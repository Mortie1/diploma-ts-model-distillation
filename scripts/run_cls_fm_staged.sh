#!/usr/bin/env bash
set -euo pipefail

# Sequential classification runs via fm_classification_train config.

source .venv/bin/activate

OUT_DIR="${OUT_DIR:-/tmp/cls_fm_staged}"
mkdir -p "$OUT_DIR"

REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
EPOCHS="${EPOCHS:-10}"
EPOCH_LEN="${EPOCH_LEN:-0}"  # 0 => full epoch
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
COMPILE_ENABLED="${COMPILE_ENABLED:-false}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
UCR_ROOT="${UCR_ROOT:-data/raw/ucr}"

RESULTS_TSV="$OUT_DIR/results.tsv"
echo -e "dataset\trun\tmodel_target\tmodel_size\tn_classes\ttrain_len\tbatch_size\tstatus\tbest_test_Accuracy\tcomet_url\tlog_file" > "$RESULTS_TSV"

# run_name|model_target|model_size
RUNS=(
  "moment_small_cls|src.model.MomentClassificationAdapter|small"
  "moment_base_cls|src.model.MomentClassificationAdapter|base"
  "moment_large_cls|src.model.MomentClassificationAdapter|large"
  "chronos2_cls|src.model.Chronos2ClassificationAdapter|base"
  "tirex_cls|src.model.TiRexClassificationAdapter|base"
  "tspulse_cls|src.model.TSPulseClassificationAdapter|r1"
)

DATASETS=(
  "ElectricDevices"
  "FordA"
  "StarLightCurves"
)

extract_metrics() {
  local log_file="$1"
  .venv/bin/python - "$log_file" <<'PY'
import re
import sys

acc = []
url = ""
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"test_Accuracy\s*:\s*([0-9.eE+-]+)", line)
        if m:
            acc.append(float(m.group(1)))
        m = re.search(r"(https://www\.comet\.com/\S+)", line)
        if m:
            url = m.group(1)

best = max(acc) if acc else float("nan")
print(best, url)
PY
}

infer_dataset_stats() {
  local dataset_name="$1"
  .venv/bin/python - "$dataset_name" "$UCR_ROOT" <<'PY'
import sys
from src.datasets.classification.ucr_dataset import UCRDataset

dataset_name = sys.argv[1]
root = sys.argv[2]
ds = UCRDataset(root=root, dataset_name=dataset_name, split="train", normalize=True)
n_classes = int(ds.labels.max()) + 1
print(len(ds), n_classes)
PY
}

run_one() {
  local dataset_name="$1"
  local name="$2"
  local model_target="$3"
  local model_size="$4"
  local n_classes="$5"
  local train_len="$6"

  local log_file="$OUT_DIR/${dataset_name}_${name}.log"
  local run_name="clsfm-${dataset_name}-${name}-e${EPOCHS}-bs${BATCH_SIZE}"
  local run_bs="$BATCH_SIZE"
  if (( run_bs > train_len )); then
    run_bs="$train_len"
  fi

  local cmd=(
    .venv/bin/python train.py
    -cn=fm_classification_train
    "model._target_=${model_target}"
    "model.model_size=${model_size}"
    "model.n_classes=${n_classes}"
    "model.fail_on_provider_fallback=true"
    "datasets.train.root=${UCR_ROOT}"
    "datasets.train.dataset_name=${dataset_name}"
    "datasets.val.root=${UCR_ROOT}"
    "datasets.val.dataset_name=${dataset_name}"
    "datasets.test.root=${UCR_ROOT}"
    "datasets.test.dataset_name=${dataset_name}"
    "trainer.require_cuda=${REQUIRE_CUDA}"
    "trainer.n_epochs=${EPOCHS}"
    "trainer.log_step=1"
    "trainer.compile_enabled=${COMPILE_ENABLED}"
    "trainer.compile_mode=${COMPILE_MODE}"
    "trainer.compile_backend=${COMPILE_BACKEND}"
    "dataloader.batch_size=${run_bs}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "writer.run_name=${run_name}"
    "writer.description=classification_ucr_${dataset_name}"
    "writer.tags=[classification,fm,${dataset_name},${name}]"
  )

  if [[ "$EPOCH_LEN" != "0" ]]; then
    cmd+=("trainer.epoch_len=${EPOCH_LEN}")
  else
    cmd+=("trainer.epoch_len=null")
  fi

  local rc=0
  set +e
  "${cmd[@]}" >"$log_file" 2>&1
  rc=$?
  set -e

  local status="ok"
  if [[ $rc -ne 0 ]]; then
    status="fail"
  fi

  local parsed best_acc comet_url
  parsed="$(extract_metrics "$log_file")"
  best_acc="$(echo "$parsed" | awk '{print $1}')"
  comet_url="$(echo "$parsed" | awk '{print $2}')"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$dataset_name" "$name" "$model_target" "$model_size" "$n_classes" "$train_len" "$run_bs" "$status" "$best_acc" "$comet_url" "$log_file" >> "$RESULTS_TSV"
}

for dataset_name in "${DATASETS[@]}"; do
  stats="$(infer_dataset_stats "$dataset_name")"
  train_len="$(echo "$stats" | awk '{print $1}')"
  n_classes="$(echo "$stats" | awk '{print $2}')"
  for row in "${RUNS[@]}"; do
    IFS='|' read -r name model_target model_size <<< "$row"
    run_one "$dataset_name" "$name" "$model_target" "$model_size" "$n_classes" "$train_len"
  done
done

echo "Done: $RESULTS_TSV"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS_TSV"
else
  cat "$RESULTS_TSV"
fi
