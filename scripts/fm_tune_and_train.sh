#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

mkdir -p /tmp/fm_tuning
printf "model_tag\tprovider\tmodel_id\tbatch_size\tepochs\tepoch_len\twall_sec\tbest_test_mae\tbest_test_rmse\tcomet_url\tstatus\n" > /tmp/fm_tuning/tuning_results.tsv

# Foundation models available in current setup.
models=(
  "chronos chronos amazon/chronos-t5-small"
  "timesfm timesfm google/timesfm-2.0-500m-pytorch"
)

# Prioritize throughput search over batch size.
batch_sizes=(128 256 512 1024 1536)

parse_metrics() {
  local log_file=$1
  .venv/bin/python - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
mae = []
rmse = []
url = ""
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
best_mae = min(mae) if mae else float("nan")
best_rmse = min(rmse) if rmse else float("nan")
print(best_mae, best_rmse, url)
PY
}

echo "=== TUNING BATCH SIZE (2 epochs) ==="
for model_row in "${models[@]}"; do
  set -- $model_row
  model_tag=$1
  provider=$2
  model_id=$3
  for bs in "${batch_sizes[@]}"; do
    change="${model_tag}_tune_bs${bs}_e2"
    log_file="/tmp/fm_tuning/${change}.log"
    echo "--- ${change} ---"

    start_ts=$(date +%s)
    status="ok"
    if ! .venv/bin/python train.py \
        -cn=fm_train \
        model.provider="$provider" \
        model.model_id="$model_id" \
        writer.main_change="$change" \
        trainer.require_cuda=true \
        trainer.n_epochs=2 \
        trainer.epoch_len=20 \
        trainer.amp_enabled=true \
        trainer.amp_dtype=bf16 \
        dataloader.batch_size="$bs" \
        optimizer.lr=1e-3 \
        > "$log_file" 2>&1; then
      status="fail"
    fi
    end_ts=$(date +%s)
    wall_sec=$((end_ts - start_ts))

    parsed=$(parse_metrics "$log_file")
    best_mae=$(echo "$parsed" | awk '{print $1}')
    best_rmse=$(echo "$parsed" | awk '{print $2}')
    url=$(echo "$parsed" | awk '{print $3}')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$model_tag" "$provider" "$model_id" "$bs" "2" "20" \
      "$wall_sec" "$best_mae" "$best_rmse" "$url" "$status" \
      >> /tmp/fm_tuning/tuning_results.tsv

    tail -n 12 "$log_file" | sed "s/^/  /"
  done
done

echo
echo "=== SELECT BEST BATCH PER MODEL ==="
.venv/bin/python - <<'PY'
import csv
from pathlib import Path

in_path = Path("/tmp/fm_tuning/tuning_results.tsv")
out_path = Path("/tmp/fm_tuning/best_configs.tsv")
rows = []
with in_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        if r["status"] != "ok":
            continue
        if r["best_test_mae"] in {"nan", "", "None"}:
            continue
        r["best_test_mae"] = float(r["best_test_mae"])
        r["best_test_rmse"] = float(r["best_test_rmse"]) if r["best_test_rmse"] not in {"nan", "", "None"} else float("inf")
        r["wall_sec"] = int(r["wall_sec"])
        r["batch_size"] = int(r["batch_size"])
        rows.append(r)

groups = {}
for r in rows:
    groups.setdefault(r["model_tag"], []).append(r)

selected = []
for model_tag, gr in groups.items():
    min_mae = min(x["best_test_mae"] for x in gr)
    # keep quality close to best, then maximize speed
    threshold = min_mae * 1.05
    candidates = [x for x in gr if x["best_test_mae"] <= threshold]
    best = min(candidates, key=lambda x: x["wall_sec"])
    selected.append(best)

with out_path.open("w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["model_tag","provider","model_id","batch_size","tune_best_test_mae","tune_wall_sec","comet_url"])
    for r in selected:
        writer.writerow([r["model_tag"], r["provider"], r["model_id"], r["batch_size"], r["best_test_mae"], r["wall_sec"], r["comet_url"]])

print(out_path.read_text(), end="")
PY

echo
echo "=== FINAL TRAIN (10 epochs) ==="
printf "model_tag\tprovider\tmodel_id\tbatch_size\tepochs\tepoch_len\twall_sec\tbest_test_mae\tbest_test_rmse\tcomet_url\tstatus\n" > /tmp/fm_tuning/final_results.tsv

tail -n +2 /tmp/fm_tuning/best_configs.tsv | while IFS=$'\t' read -r model_tag provider model_id batch_size tune_best_test_mae tune_wall_sec tune_url; do
  change="${model_tag}_final10_bs${batch_size}"
  log_file="/tmp/fm_tuning/${change}.log"
  echo "--- ${change} ---"

  start_ts=$(date +%s)
  status="ok"
  if ! .venv/bin/python train.py \
      -cn=fm_train \
      model.provider="$provider" \
      model.model_id="$model_id" \
      writer.main_change="$change" \
      trainer.require_cuda=true \
      trainer.n_epochs=10 \
      trainer.epoch_len=20 \
      trainer.amp_enabled=true \
      trainer.amp_dtype=bf16 \
      dataloader.batch_size="$batch_size" \
      optimizer.lr=1e-3 \
      > "$log_file" 2>&1; then
    status="fail"
  fi
  end_ts=$(date +%s)
  wall_sec=$((end_ts - start_ts))

  parsed=$(parse_metrics "$log_file")
  best_mae=$(echo "$parsed" | awk '{print $1}')
  best_rmse=$(echo "$parsed" | awk '{print $2}')
  url=$(echo "$parsed" | awk '{print $3}')

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$model_tag" "$provider" "$model_id" "$batch_size" "10" "20" \
    "$wall_sec" "$best_mae" "$best_rmse" "$url" "$status" \
    >> /tmp/fm_tuning/final_results.tsv

  tail -n 12 "$log_file" | sed "s/^/  /"
done

echo
echo "=== TUNING RESULTS ==="
column -t -s $'\t' /tmp/fm_tuning/tuning_results.tsv
echo
echo "=== FINAL RESULTS ==="
column -t -s $'\t' /tmp/fm_tuning/final_results.tsv
