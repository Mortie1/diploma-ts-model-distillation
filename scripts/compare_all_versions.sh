#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

mkdir -p /tmp/fm_all_versions
printf "provider\tmodel_id\tbatch_size\tstatus\twall_sec\tbest_test_mae\tbest_test_rmse\tcomet_url\tnote\n" > /tmp/fm_all_versions/results.tsv

chronos_models=(
  amazon/chronos-bolt-base
  amazon/chronos-bolt-mini
  amazon/chronos-bolt-small
  amazon/chronos-bolt-tiny
  amazon/chronos-t5-base
  amazon/chronos-t5-large
  amazon/chronos-t5-mini
  amazon/chronos-t5-small
  amazon/chronos-t5-tiny
  amazon/chronos-2
)

timesfm_models=(
  google/timesfm-1.0-200m
  google/timesfm-1.0-200m-pytorch
  google/timesfm-2.0-500m-jax
  google/timesfm-2.0-500m-pytorch
  google/timesfm-2.5-200m-flax
  google/timesfm-2.5-200m-pytorch
  google/timesfm-2.5-200m-transformers
)

batch_candidates=(512 256 128)

slugify() {
  echo "$1" | tr '/:.' '___'
}

parse_metrics() {
  local log_file=$1
  .venv/bin/python - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
mae = []
rmse = []
url = ""
note = ""
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"Experiment is live on comet.com (https://www\.comet\.com/\S+)", line)
        if m:
            url = m.group(1)
        m = re.search(r"test_MAE\s*:\s*([0-9.eE+-]+)", line)
        if m:
            mae.append(float(m.group(1)))
        m = re.search(r"test_RMSE\s*:\s*([0-9.eE+-]+)", line)
        if m:
            rmse.append(float(m.group(1)))
if not mae:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "RuntimeError:" in line or "ValueError:" in line or "Error" in line:
            note = line.strip().replace("\t", " ")[:240]
            break
best_mae = min(mae) if mae else float("nan")
best_rmse = min(rmse) if rmse else float("nan")
print(best_mae, best_rmse, url, note)
PY
}

run_one_model() {
  local provider=$1
  local model_id=$2
  local model_slug
  model_slug=$(slugify "$model_id")
  local success=0
  local last_note=""
  for bs in "${batch_candidates[@]}"; do
    local change="${provider}_${model_slug}_cmp_bs${bs}_e2"
    local log_file="/tmp/fm_all_versions/${change}.log"
    echo "--- ${provider} ${model_id} bs=${bs} ---"

    local start_ts end_ts wall_sec
    start_ts=$(date +%s)
    local status="ok"
    local timeout_sec=600
    if [[ "$model_id" == "amazon/chronos-2" ]]; then
      timeout_sec=180
    fi

    set +e
    timeout "${timeout_sec}s" .venv/bin/python train.py \
        -cn=fm_train \
        model.provider="$provider" \
        model.model_id="$model_id" \
        model.require_provider_model=true \
        writer.main_change="$change" \
        trainer.require_cuda=true \
        trainer.n_epochs=1 \
        trainer.epoch_len=2 \
        trainer.amp_enabled=true \
        trainer.amp_dtype=bf16 \
        dataloader.batch_size="$bs" \
        optimizer.lr=1e-3 \
        > "$log_file" 2>&1
    exit_code=$?
    set -e
    if [[ "$exit_code" -ne 0 ]]; then
      if [[ "$exit_code" -eq 124 ]]; then
        status="timeout"
      else
        status="fail"
      fi
    fi
    end_ts=$(date +%s)
    wall_sec=$((end_ts - start_ts))

    parsed=$(parse_metrics "$log_file")
    best_mae=$(echo "$parsed" | awk '{print $1}')
    best_rmse=$(echo "$parsed" | awk '{print $2}')
    url=$(echo "$parsed" | awk '{print $3}')
    note=$(echo "$parsed" | cut -d' ' -f4-)
    if [[ -z "$note" ]]; then
      note="-"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$provider" "$model_id" "$bs" "$status" "$wall_sec" "$best_mae" "$best_rmse" "$url" "$note" \
      >> /tmp/fm_all_versions/results.tsv

    tail -n 10 "$log_file" | sed "s/^/  /"

    if [[ "$status" == "ok" ]]; then
      success=1
      break
    fi

    if [[ "$status" == "timeout" ]]; then
      # Long init/download stalls usually do not depend on batch size.
      break
    fi

    last_note="$note"
    if rg -q "Failed to initialize provider|require_provider_model" "$log_file"; then
      # Provider-model mismatch: changing batch size won't help.
      break
    fi
  done

  if [[ "$success" -eq 0 ]]; then
    echo "  -> no successful run for ${provider} ${model_id} (${last_note})"
  fi
}

echo "=== Compare Chronos Versions ==="
for mid in "${chronos_models[@]}"; do
  run_one_model chronos "$mid"
done

echo "=== Compare TimesFM Versions ==="
for mid in "${timesfm_models[@]}"; do
  run_one_model timesfm "$mid"
done

echo
echo "=== Results ==="
column -t -s $'\t' /tmp/fm_all_versions/results.tsv
