#!/usr/bin/env bash
set -euo pipefail

# Recompute fold metrics from best checkpoints.
# Usage:
#   scripts/recompute_folds_from_best.sh /tmp/<run_dir1> /tmp/<run_dir2> ...
#   scripts/recompute_folds_from_best.sh /tmp/<run_dir>/results.tsv
# If a run dir is provided, script reads <run_dir>/results.tsv and writes:
#   <parent>/<run_dir_name>-fixed-scores/results_recomputed.tsv
#
# You can also edit RUN_DIRS below and run without CLI args.
# Optional env overrides:
#   DEVICE=auto REQUIRE_CUDA=false BATCH_SIZE= NUM_WORKERS= OUTPUT_TSV=

# Optional hardcoded list (used only when no CLI args are passed).
RUN_DIRS=(
    /tmp/distill-hubertl-fl0-mantis-student-nodistill/results.tsv
  # "/tmp/pamap_cv_no_distill_hubert_l8_wristacc6"
  # "/tmp/pamap_cv_distill_lf10_hubert_l8_wristacc6"
)

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

DEVICE="${DEVICE:-auto}"
REQUIRE_CUDA="${REQUIRE_CUDA:-false}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
OUTPUT_TSV="${OUTPUT_TSV:-}"

TARGETS=()
if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
elif [[ ${#RUN_DIRS[@]} -gt 0 ]]; then
  TARGETS=("${RUN_DIRS[@]}")
else
  echo "No inputs provided."
  echo "Usage: $0 <run_dir_or_results.tsv> [more_dirs...]"
  echo "Or fill RUN_DIRS array inside this script."
  exit 1
fi

ok=0
fail=0

for target in "${TARGETS[@]}"; do
  target="${target%/}"
  results_tsv=""
  out_tsv=""

  if [[ -f "$target" && "$target" == *.tsv ]]; then
    # Backward-compatible mode: explicit results.tsv passed.
    results_tsv="$target"
    run_dir="$(dirname "$results_tsv")"
    run_name="$(basename "$run_dir")"
    parent_dir="$(dirname "$run_dir")"
    out_dir="${parent_dir}/${run_name}-fixed-scores"
    mkdir -p "$out_dir"
    out_tsv="${out_dir}/results_recomputed.tsv"
  elif [[ -d "$target" ]]; then
    run_dir="$target"
    run_name="$(basename "$run_dir")"
    parent_dir="$(dirname "$run_dir")"
    results_tsv="${run_dir}/results.tsv"
    out_dir="${parent_dir}/${run_name}-fixed-scores"
    mkdir -p "$out_dir"
    out_tsv="${out_dir}/results_recomputed.tsv"
  else
    echo "[FAIL] Not a directory or TSV file: $target"
    fail=$((fail + 1))
    continue
  fi

  if [[ ! -f "$results_tsv" ]]; then
    echo "[FAIL] Missing results.tsv: $results_tsv"
    fail=$((fail + 1))
    continue
  fi

  # For batch mode OUTPUT_TSV should not be set globally; each target has its own path.
  if [[ -n "$OUTPUT_TSV" && ${#TARGETS[@]} -gt 1 ]]; then
    echo "[FAIL] OUTPUT_TSV is set but multiple targets provided. Unset OUTPUT_TSV."
    fail=$((fail + 1))
    continue
  fi
  if [[ -n "$OUTPUT_TSV" ]]; then
    out_tsv="$OUTPUT_TSV"
  fi

  cmd=(
    python scripts/recompute_fold_metrics_from_checkpoints.py
    --results-tsv "$results_tsv"
    --output-tsv "$out_tsv"
    --device "$DEVICE"
  )
  if [[ "$REQUIRE_CUDA" == "true" ]]; then
    cmd+=(--require-cuda)
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch-size "$BATCH_SIZE")
  fi
  if [[ -n "$NUM_WORKERS" ]]; then
    cmd+=(--num-workers "$NUM_WORKERS")
  fi

  echo
  echo "=== Recompute: $run_name ==="
  echo "Input : $results_tsv"
  echo "Output: $out_tsv"
  echo "Running: ${cmd[*]}"
  if "${cmd[@]}"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi
done

echo
echo "Done. success=$ok failed=$fail total=${#TARGETS[@]}"
if [[ $fail -gt 0 ]]; then
  exit 1
fi
