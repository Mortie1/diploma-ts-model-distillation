#!/usr/bin/env bash
set -euo pipefail

# Sequential distillation runs for article-style classification datasets:
# - hhar
# - motionsense
# - pamap2_wrist10s
# - pamap2_leg2s
#
# Example:
#   TEACHER_MODEL_NAME=facebook/hubert-base-ls960 \
#   BS=64 EPOCHS=5 EPOCH_LEN=100 \
#   scripts/run_distill_article_seq.sh

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MODEL_CONFIG="${MODEL_CONFIG:-student_classification}"
MODEL_TARGET="${MODEL_TARGET:-src.model.StudentClassifier}"
MODEL_TAG="${MODEL_TAG:-student}"

REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
COMET_MODE="${COMET_MODE:-online}"   # online|offline
BS="${BS:-64}"
EPOCHS="${EPOCHS:-10}"
EPOCH_LEN="${EPOCH_LEN:-null}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_STEP="${LOG_STEP:-10}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

TEACHER_BACKEND="${TEACHER_BACKEND:-hf}" # mock|hf
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-facebook/hubert-large-ll60k}"
TEACHER_HIDDEN_DIM="${TEACHER_HIDDEN_DIM:-3072}"
TEACHER_LAYER_IDX="${TEACHER_LAYER_IDX:-1}"
TEACHER_PER_CHANNEL="${TEACHER_PER_CHANNEL:-true}"
TEACHER_CHANNEL_REDUCE="${TEACHER_CHANNEL_REDUCE:-concat}"
TEACHER_CHANNEL_CHUNK_SIZE="${TEACHER_CHANNEL_CHUNK_SIZE:-16}"
TEACHER_PROJECTION="${TEACHER_PROJECTION:-none}" # none|pca
TEACHER_PCA_DIM="${TEACHER_PCA_DIM:-512}"
TEACHER_PCA_FIT_BATCHES="${TEACHER_PCA_FIT_BATCHES:-40}"
TEACHER_PCA_CENTER="${TEACHER_PCA_CENTER:-true}"

LAMBDA_FEAT="${LAMBDA_FEAT:-1.0}"
LAMBDA_LOGIT="${LAMBDA_LOGIT:-0.0}"
FEAT_LOSS_TYPE="${FEAT_LOSS_TYPE:-cosine}" # mse|cosine

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

DATASETS="${DATASETS:-pamap2_leg2s opportunity}"

RUN_PREFIX="${RUN_PREFIX:-distill-article}"
OUT_DIR="${OUT_DIR:-/tmp/${RUN_PREFIX}}"
mkdir -p "$OUT_DIR"

RESULTS="$OUT_DIR/results.tsv"
echo -e "dataset\tstatus\ttest_Accuracy\ttest_MacroF1\ttest_MicroF1\ttest_AUROC\ttest_AUC\trun_name\tcomet_url\tlog_file" > "$RESULTS"

extract_comet_url() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys
url = ""
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"(https://www\.comet\.com/\S+)", line)
        if m:
            url = m.group(1)
print(url)
PY
}

extract_test_metrics() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import math
import re
import sys

keys = {
    "test_Accuracy": float("nan"),
    "test_MacroF1": float("nan"),
    "test_MicroF1": float("nan"),
    "test_AUROC": float("nan"),
    "test_AUC": float("nan"),
}

patterns = {
    "test_Accuracy": r"test_Accuracy\s*:\s*([0-9.eE+-]+)",
    "test_MacroF1": r"test_MacroF1\s*:\s*([0-9.eE+-]+)",
    "test_MicroF1": r"test_MicroF1\s*:\s*([0-9.eE+-]+)",
    "test_AUROC": r"test_AUROC\s*:\s*([0-9.eE+-]+)",
    "test_AUC": r"test_AUC\s*:\s*([0-9.eE+-]+)",
}

with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        for k, p in patterns.items():
            m = re.search(p, line)
            if m:
                try:
                    keys[k] = float(m.group(1))
                except Exception:
                    pass

def fmt(v):
    return "nan" if not math.isfinite(v) else str(v)

print(
    "\t".join(
        [
            fmt(keys["test_Accuracy"]),
            fmt(keys["test_MacroF1"]),
            fmt(keys["test_MicroF1"]),
            fmt(keys["test_AUROC"]),
            fmt(keys["test_AUC"]),
        ]
    )
)
PY
}

run_dataset() {
  local ds="$1"
  local n_classes="$2"
  local in_channels="$3"
  local extra="$4"

  local run_name="${RUN_PREFIX}-${MODEL_TAG}-${ds}-e${EPOCHS}-bs${BS}"
  local tags="[classification,distill,${ds},${MODEL_TAG}]"
  local log_file="$OUT_DIR/${ds}.train.log"

  echo "=== RUN ${ds} (bs=${BS}, epochs=${EPOCHS}, epoch_len=${EPOCH_LEN}) ==="

  set +e
  python train.py -cn=distill_train \
    datasets="$ds" \
    model="${MODEL_CONFIG}" \
    model._target_="${MODEL_TARGET}" \
    model.n_classes="${n_classes}" \
    model.in_channels="${in_channels}" \
    trainer.require_cuda="${REQUIRE_CUDA}" \
    trainer.n_epochs="${EPOCHS}" \
    trainer.epoch_len="${EPOCH_LEN}" \
    trainer.grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    trainer.log_step="${LOG_STEP}" \
    dataloader.batch_size="${BS}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    writer=cometml \
    writer.mode="${COMET_MODE}" \
    writer.run_name="${run_name}" \
    writer.tags="${tags}" \
    distillation.enabled=true \
    distillation.teacher_backend="${TEACHER_BACKEND}" \
    distillation.teacher_model_name="${TEACHER_MODEL_NAME}" \
    distillation.teacher_hidden_dim="${TEACHER_HIDDEN_DIM}" \
    distillation.teacher_layer_idx="${TEACHER_LAYER_IDX}" \
    distillation.teacher_per_channel="${TEACHER_PER_CHANNEL}" \
    distillation.teacher_channel_reduce="${TEACHER_CHANNEL_REDUCE}" \
    distillation.teacher_channel_chunk_size="${TEACHER_CHANNEL_CHUNK_SIZE}" \
    distillation.teacher_projection="${TEACHER_PROJECTION}" \
    distillation.teacher_pca_dim="${TEACHER_PCA_DIM}" \
    distillation.teacher_pca_fit_batches="${TEACHER_PCA_FIT_BATCHES}" \
    distillation.teacher_pca_center="${TEACHER_PCA_CENTER}" \
    loss_function.lambda_feat="${LAMBDA_FEAT}" \
    loss_function.lambda_logit="${LAMBDA_LOGIT}" \
    loss_function.feat_loss_type="${FEAT_LOSS_TYPE}" \
    ${extra} \
    ${EXTRA_OVERRIDES} > "$log_file" 2>&1
  rc=$?
  set -e

  local test_metrics="nan\tnan\tnan\tnan\tnan"
  if [[ $rc -eq 0 ]]; then
    status="ok"
    test_metrics="$(extract_test_metrics "$log_file")"
  else
    status="fail"
  fi
  comet_url="$(extract_comet_url "$log_file")"
  echo -e "${ds}\t${status}\t${test_metrics}\t${run_name}\t${comet_url}\t${log_file}" >> "$RESULTS"
}

dataset_meta() {
  case "$1" in
    hhar)             echo "6 3";;
    motionsense)      echo "6 12";;
    pamap2_wrist10s)  echo "8 3";;
    pamap2_leg2s)     echo "12 3";;
    *) echo "Unknown dataset: $1" >&2; exit 1;;
  esac
}

for ds in ${DATASETS}; do
  read -r n_classes in_channels <<< "$(dataset_meta "$ds")"
  run_dataset "$ds" "$n_classes" "$in_channels" ""
done

echo "Done. Results: $RESULTS"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS"
else
  cat "$RESULTS"
fi
