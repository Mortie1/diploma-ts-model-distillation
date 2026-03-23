#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

mkdir -p /tmp/fm_compare_5ep
printf "run\tprovider\tmodel_id\tcomet_url\tbest_test_MAE\tbest_test_RMSE\n" > /tmp/fm_compare_5ep/results.tsv

runs=(
  "chronos_small chronos amazon/chronos-t5-small"
  "chronos_base chronos amazon/chronos-t5-base"
  "timesfm_500m timesfm google/timesfm-2.0-500m-pytorch"
)

for row in "${runs[@]}"; do
  set -- $row
  name=$1
  provider=$2
  model_id=$3
  log="/tmp/fm_compare_5ep/${name}.log"

  echo "=== RUN $name ($provider, $model_id) ==="
  .venv/bin/python train.py \
    -cn=fm_train \
    model.provider="$provider" \
    model.model_id="$model_id" \
    writer.main_change="${name}_5ep" \
    trainer.require_cuda=true \
    trainer.n_epochs=5 \
    trainer.epoch_len=20 \
    trainer.amp_enabled=true \
    trainer.amp_dtype=bf16 \
    dataloader.batch_size=64 \
    optimizer.lr=1e-3 \
    > "$log" 2>&1

  url=$(rg -n "Experiment is live on comet.com" "$log" | tail -1 | sed -E "s/.*(https:\/\/www\.comet\.com\/[^ ]+).*/\\1/")

  parsed=$(
    .venv/bin/python - "$log" <<'PY'
import re
import sys

path = sys.argv[1]
mae = []
rmse = []
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"test_MAE\s*:\s*([0-9.eE+-]+)", line)
        if m:
            mae.append(float(m.group(1)))
        m = re.search(r"test_RMSE\s*:\s*([0-9.eE+-]+)", line)
        if m:
            rmse.append(float(m.group(1)))
print((min(mae) if mae else float("nan")), (min(rmse) if rmse else float("nan")))
PY
  )

  best_test_mae=$(echo "$parsed" | awk "{print \$1}")
  best_test_rmse=$(echo "$parsed" | awk "{print \$2}")

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$provider" "$model_id" "$url" "$best_test_mae" "$best_test_rmse" \
    >> /tmp/fm_compare_5ep/results.tsv

  tail -n 20 "$log" | sed "s/^/  /"
done

column -t -s $'\t' /tmp/fm_compare_5ep/results.tsv
