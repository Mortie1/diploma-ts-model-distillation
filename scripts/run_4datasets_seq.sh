#!/usr/bin/env bash
set -euo pipefail

# Sequential FM classification training on 4 datasets.
# Defaults: batch_size=128, n_epochs=5, epoch_len=100.

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

BS="${BS:-128}"
EPOCHS="${EPOCHS:-5}"
EPOCH_LEN="${EPOCH_LEN:-100}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MODEL_FAMILY="${MODEL_FAMILY:-moment}"
MODEL_TARGET="${MODEL_TARGET:-src.model.MomentClassificationAdapter}"
MODEL_SIZE="${MODEL_SIZE:-small}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
ONLINE="${ONLINE:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUT_DIR="${OUT_DIR:-/tmp/${MODEL_FAMILY}_4ds_seq}"
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
    writer.tags="$tags" > "$log_file" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo -e "${ds}\tok\t${run_name}\t${log_file}" >> "$RESULTS"
  else
    echo -e "${ds}\tfail\t${run_name}\t${log_file}" >> "$RESULTS"
  fi
}

run_dataset "pamap2" "12" "52" "${MODEL_FAMILY}-${MODEL_SIZE}-pamap2-e${EPOCHS}-bs${BS}" "[classification,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},pamap2]"
run_dataset "ptbxl" "5" "12" "${MODEL_FAMILY}-${MODEL_SIZE}-ptbxl-e${EPOCHS}-bs${BS}" "[classification,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},ptbxl]"
run_dataset "insect_wingbeat" "11" "1" "${MODEL_FAMILY}-${MODEL_SIZE}-insect-e${EPOCHS}-bs${BS}" "[classification,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},insect_wingbeat]"
run_dataset "cwru_bearing" "6" "1" "${MODEL_FAMILY}-${MODEL_SIZE}-cwru-e${EPOCHS}-bs${BS}" "[classification,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},cwru_bearing]"

echo "Done. Summary: $RESULTS"
column -t -s $'\t' "$RESULTS"
