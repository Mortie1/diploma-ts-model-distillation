#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

LOG_ROOT="/tmp/fm_5ep_suite_logs"
mkdir -p "$LOG_ROOT"

run_exp() {
  local key="$1"; shift
  local log_file="$LOG_ROOT/${key}.log"
  local timeout_secs="${RUN_TIMEOUT_SECS:-3600}"
  echo "[$(date '+%F %T')] START $key"
  set +e
  timeout --signal=SIGINT "${timeout_secs}" python train.py "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAIL  $key rc=$rc"
  else
    echo "[$(date '+%F %T')] DONE  $key"
  fi
  local url
  url=$(rg -o "https://www\.comet\.com/[^ ]+" "$log_file" | head -n1 || true)
  if [[ -n "$url" ]]; then
    echo "COMET_URL $key $url"
  fi
  echo "$key\t$rc\t$url" >> "$LOG_ROOT/summary.tsv"
}

echo -e "run\trc\tcomet_url" > "$LOG_ROOT/summary.tsv"

COMMON=(trainer.n_epochs=5 trainer.epoch_len=5 trainer.log_step=1)

run_exp "electricity_timesfm25_200m_full_e5" \
  --config-name fm_train \
  "${COMMON[@]}" \
  datasets=ltsf_electricity_h96 \
  model.provider=timesfm_hf \
  model.model_id=google/timesfm-2.5-200m-pytorch \
  model.in_channels=321 \
  model.horizon=96 \
  model.finetune_mode=full \
  dataloader.batch_size=32 \
  writer.short_name=fm5ep-electricity-h96 \
  writer.main_change=timesfm25-200m-full-e5

run_exp "electricity_timesfm25_500m_lora_e5" \
  --config-name fm_train \
  "${COMMON[@]}" \
  datasets=ltsf_electricity_h96 \
  model.provider=timesfm_hf \
  model.model_id=google/timesfm-2.5-500m-pytorch \
  model.in_channels=321 \
  model.horizon=96 \
  model.finetune_mode=lora \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.lora_dropout=0.05 \
  model.lora_target_patterns='[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj]' \
  dataloader.batch_size=16 \
  writer.short_name=fm5ep-electricity-h96 \
  writer.main_change=timesfm25-500m-lora-e5

# Previous models
run_exp "electricity_chronos_t5_small_e5" \
  --config-name fm_train \
  "${COMMON[@]}" \
  datasets=ltsf_electricity_h96 \
  model.provider=chronos \
  model.model_id=amazon/chronos-t5-small \
  model.in_channels=321 \
  model.horizon=96 \
  dataloader.batch_size=32 \
  writer.short_name=fm5ep-electricity-h96 \
  writer.main_change=chronos-t5-small-e5

run_exp "electricity_timesfm20_500m_e5" \
  --config-name fm_train \
  "${COMMON[@]}" \
  datasets=ltsf_electricity_h96 \
  model.provider=timesfm \
  model.model_id=google/timesfm-2.0-500m-pytorch \
  model.in_channels=321 \
  model.horizon=96 \
  model.finetune_mode=none \
  dataloader.batch_size=64 \
  writer.short_name=fm5ep-electricity-h96 \
  writer.main_change=timesfm20-500m-e5

# Chronos-2 can hang on first download in some environments; keep it last with timeout.
run_exp "electricity_chronos2_e5" \
  --config-name fm_train \
  "${COMMON[@]}" \
  datasets=ltsf_electricity_h96 \
  model.provider=chronos \
  model.model_id=amazon/chronos-2 \
  model.in_channels=321 \
  model.horizon=96 \
  dataloader.batch_size=8 \
  writer.short_name=fm5ep-electricity-h96 \
  writer.main_change=chronos2-e5

echo "Suite finished. Summary: $LOG_ROOT/summary.tsv"
