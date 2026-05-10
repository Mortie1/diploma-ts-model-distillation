#!/usr/bin/env bash
set -euo pipefail

# Sequential FM classification runs for datasets/protocols from arXiv:2509.00221.
#
# Example:
#   MODEL_FAMILY=moment MODEL_TARGET=src.model.MomentClassificationAdapter \
#   MODEL_SIZE=base BS=64 EPOCHS=5 EPOCH_LEN=100 scripts/run_article2509_fm_seq.sh

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

BS="${BS:-64}"
EPOCHS="${EPOCHS:-5}"
EPOCH_LEN="${EPOCH_LEN:-100}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MODEL_FAMILY="${MODEL_FAMILY:-moment}"
MODEL_TARGET="${MODEL_TARGET:-src.model.MomentClassificationAdapter}"
MODEL_SIZE="${MODEL_SIZE:-small}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
ONLINE="${ONLINE:-online}"   # online|offline
NUM_WORKERS="${NUM_WORKERS:-4}"
OUT_DIR="${OUT_DIR:-/tmp/${MODEL_FAMILY}_article2509_seq}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
mkdir -p "$OUT_DIR"

RESULTS="$OUT_DIR/results.tsv"
echo -e "dataset\tstatus\trun_name\tlog_file" > "$RESULTS"

run_dataset() {
  local ds="$1"
  local n_classes="$2"
  local in_channels="$3"
  local run_name="$4"
  local tags="$5"
  local log_file="$OUT_DIR/${ds}.log"

  echo "=== RUN ${ds} (bs=${BS}, epochs=${EPOCHS}, epoch_len=${EPOCH_LEN}, accum=${GRAD_ACCUM_STEPS}) ==="

  set +e
  python train.py -cn=fm_train \
    datasets="$ds" \
    model._target_="$MODEL_TARGET" \
    model.model_size="$MODEL_SIZE" \
    model.n_classes="$n_classes" \
    model.in_channels="$in_channels" \
    model.require_provider_model=true \
    model.fail_on_provider_fallback=true \
    trainer.require_cuda="$REQUIRE_CUDA" \
    trainer.n_epochs="$EPOCHS" \
    trainer.epoch_len="$EPOCH_LEN" \
    trainer.grad_accum_steps="$GRAD_ACCUM_STEPS" \
    trainer.log_step=5 \
    dataloader.batch_size="$BS" \
    dataloader.num_workers="$NUM_WORKERS" \
    writer=cometml \
    writer.mode="$ONLINE" \
    writer.run_name="$run_name" \
    writer.tags="$tags" \
    ${EXTRA_OVERRIDES} > "$log_file" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo -e "${ds}\tok\t${run_name}\t${log_file}" >> "$RESULTS"
  else
    echo -e "${ds}\tfail\t${run_name}\t${log_file}" >> "$RESULTS"
  fi
}

# Activity (wearables)
run_dataset "pamap2_wrist10s" "8" "3" \
  "${MODEL_FAMILY}-${MODEL_SIZE}-pamap2w10s-e${EPOCHS}-bs${BS}" \
  "[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},pamap2_wrist10s]"
run_dataset "opportunity" "4" "3" \
  "${MODEL_FAMILY}-${MODEL_SIZE}-oppo-e${EPOCHS}-bs${BS}" \
  "[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},opportunity]"
run_dataset "hhar" "6" "3" \
  "${MODEL_FAMILY}-${MODEL_SIZE}-hhar-e${EPOCHS}-bs${BS}" \
  "[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},hhar]"
run_dataset "motionsense" "6" "12" \
  "${MODEL_FAMILY}-${MODEL_SIZE}-motionsense-e${EPOCHS}-bs${BS}" \
  "[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},motionsense]"
run_dataset "pamap2_leg2s" "12" "3" \
  "${MODEL_FAMILY}-${MODEL_SIZE}-pamap2leg-e${EPOCHS}-bs${BS}" \
  "[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},pamap2_leg2s]"

# ECG / PPG
run_dataset "mitbih_arrhythmia" "2" "1" \
  "${MODEL_FAMILY}-${MODEL_SIZE}-mitbih-e${EPOCHS}-bs${BS}" \
  "[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},mitbih_arrhythmia]"
run_dataset "wesad" "4" "1" \
  "${MODEL_FAMILY}-${MODEL_SIZE}-wesad-e${EPOCHS}-bs${BS}" \
  "[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},wesad]"

echo "Done. Summary: $RESULTS"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS"
else
  cat "$RESULTS"
fi
