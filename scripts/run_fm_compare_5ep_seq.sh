#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

OUT_DIR="${OUT_DIR:-/tmp/fm_compare_5ep_seq}"
mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-5}"
DATASET="${DATASET:-ltsf_electricity_h96}"
HORIZON="${HORIZON:-96}"
IN_CHANNELS="${IN_CHANNELS:-321}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
RUN_TIMEOUT_SECS="${RUN_TIMEOUT_SECS:-0}"
COMPILE_ENABLED="${COMPILE_ENABLED:-false}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"

RESULTS_TSV="$OUT_DIR/results.tsv"
echo -e "run\tmodel_target\tmodel_id\tfinetune_mode\tbatch_size\tstatus\tbest_test_MAE\tbest_test_RMSE\tcomet_url\tlog_file" > "$RESULTS_TSV"

# name|model_target|model_id|finetune_mode|batch_size|extra_overrides
runs=(
  "chronos2_head|src.model.ChronosForecastAdapter|amazon/chronos-2|none|256|model.require_provider_model=true"
  "chronos_t5_base_head|src.model.ChronosForecastAdapter|amazon/chronos-t5-base|none|256|model.require_provider_model=true"
  "chronos_t5_small_head|src.model.ChronosForecastAdapter|amazon/chronos-t5-small|none|256|model.require_provider_model=true"
  "chronos_t5_mini_head|src.model.ChronosForecastAdapter|amazon/chronos-t5-mini|none|256|model.require_provider_model=true"
  "chronos_t5_tiny_head|src.model.ChronosForecastAdapter|amazon/chronos-t5-tiny|none|256|model.require_provider_model=true"
  "chronos_bolt_base_head|src.model.ChronosForecastAdapter|amazon/chronos-bolt-base|none|256|model.require_provider_model=true"
  "chronos_bolt_small_head|src.model.ChronosForecastAdapter|amazon/chronos-bolt-small|none|256|model.require_provider_model=true"
  "chronos_bolt_mini_head|src.model.ChronosForecastAdapter|amazon/chronos-bolt-mini|none|256|model.require_provider_model=true"
  "chronos_bolt_tiny_head|src.model.ChronosForecastAdapter|amazon/chronos-bolt-tiny|none|256|model.require_provider_model=true"
  "timesfm20_500m_hf_head|src.model.TimesFMHFForecastAdapter|google/timesfm-2.0-500m-pytorch|none|256|model.require_provider_model=true"
  "timesfm25_200m_hf_head|src.model.TimesFMHFForecastAdapter|google/timesfm-2.5-200m-pytorch|none|256|model.require_provider_model=true"
)

extract_metrics() {
  local log_file="$1"
  .venv/bin/python - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
mae = []
rmse = []
url = ""

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"test_MAE\s*:\s*([0-9.eE+-]+)", line)
        if m:
            mae.append(float(m.group(1)))
        m = re.search(r"test_RMSE\s*:\s*([0-9.eE+-]+)", line)
        if m:
            rmse.append(float(m.group(1)))
        m = re.search(r"(https://www\.comet\.com/\S+)", line)
        if m:
            url = m.group(1)

best_mae = min(mae) if mae else float("nan")
best_rmse = min(rmse) if rmse else float("nan")
print(best_mae, best_rmse, url)
PY
}

run_one() {
  local name="$1"
  local model_target="$2"
  local model_id="$3"
  local finetune_mode="$4"
  local bs="$5"
  local extra="$6"

  local log_file="$OUT_DIR/${name}.log"
  local model_slug
  model_slug="$(echo "$model_id" | tr '/:.' '___')"
  local short_name="fmcmp-${DATASET}-h${HORIZON}-${name}-${model_slug}"
  local main_change="${name}-ft${finetune_mode}-bs${bs}"

  local cmd=(
    .venv/bin/python train.py
    -cn=fm_train
    "model._target_=${model_target}"
    "datasets=${DATASET}"
    "model.model_id=${model_id}"
    "model.finetune_mode=${finetune_mode}"
    "model.horizon=${HORIZON}"
    "model.in_channels=${IN_CHANNELS}"
    "trainer.require_cuda=${REQUIRE_CUDA}"
    "trainer.n_epochs=${EPOCHS}"
    "trainer.log_step=1"
    "trainer.amp_enabled=true"
    "trainer.amp_dtype=bf16"
    "trainer.compile_enabled=${COMPILE_ENABLED}"
    "trainer.compile_mode=${COMPILE_MODE}"
    "trainer.compile_backend=${COMPILE_BACKEND}"
    "+trainer.skip_oom=false"
    "dataloader.batch_size=${bs}"
    "dataloader.pin_memory=true"
    "dataloader.num_workers=6"
    "writer.auto_name=true"
    "writer.short_name=${short_name}"
    "writer.main_change=${main_change}"
  )

  # shellcheck disable=SC2206
  local extra_arr=( $extra )
  cmd+=("${extra_arr[@]}")

  local rc=0
  if [[ "$RUN_TIMEOUT_SECS" -gt 0 ]]; then
    set +e
    timeout --signal=SIGINT "${RUN_TIMEOUT_SECS}" "${cmd[@]}" >"$log_file" 2>&1
    rc=$?
    set -e
  else
    set +e
    "${cmd[@]}" >"$log_file" 2>&1
    rc=$?
    set -e
  fi

  local status="ok"
  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 124 ]]; then
      status="timeout"
    else
      status="fail"
    fi
  fi

  local parsed
  parsed="$(extract_metrics "$log_file")"
  local best_mae best_rmse comet_url
  best_mae="$(echo "$parsed" | awk '{print $1}')"
  best_rmse="$(echo "$parsed" | awk '{print $2}')"
  comet_url="$(echo "$parsed" | awk '{print $3}')"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$model_target" "$model_id" "$finetune_mode" "$bs" "$status" \
    "$best_mae" "$best_rmse" "$comet_url" "$log_file" >> "$RESULTS_TSV"
}

for row in "${runs[@]}"; do
  IFS='|' read -r name model_target model_id finetune_mode bs extra <<< "$row"
  run_one "$name" "$model_target" "$model_id" "$finetune_mode" "$bs" "$extra"
done

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS_TSV"
else
  cat "$RESULTS_TSV"
fi
