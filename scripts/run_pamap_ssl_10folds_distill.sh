#!/usr/bin/env bash
set -euo pipefail

# Run N random subject-wise PAMAP2 distillation folds (ssl_wearables protocol):
# each fold sets one subject as val and one as test (ordered pair), train = rest.
#
# Example:
#   N_FOLDS=10 FOLD_SEED=42 \
#   TEACHER_BACKEND=hf TEACHER_MODEL_NAME=facebook/hubert-base-ls960 \
#   scripts/run_pamap_ssl_10folds_distill.sh

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MODEL_TAG="${MODEL_TAG:-student}"
N_FOLDS="${N_FOLDS:-10}"
FOLD_SEED="${FOLD_SEED:-42}"

REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
COMET_MODE="${COMET_MODE:-online}"   # online|offline
EPOCHS="${EPOCHS:-5}"
EPOCH_LEN="${EPOCH_LEN:-null}"
BS="${BS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_STEP="${LOG_STEP:-10}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

N_CLASSES="${N_CLASSES:-8}"
IN_CHANNELS="${IN_CHANNELS:-3}"

TEACHER_BACKEND="${TEACHER_BACKEND:-hf}" # mock|hf
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-facebook/hubert-large-ll60k}"
TEACHER_HIDDEN_DIM="${TEACHER_HIDDEN_DIM:-768}"
TEACHER_LAYER_IDX="${TEACHER_LAYER_IDX:-8}"
TEACHER_PER_CHANNEL="${TEACHER_PER_CHANNEL:-true}"
TEACHER_CHANNEL_REDUCE="${TEACHER_CHANNEL_REDUCE:-mean}"
TEACHER_CHANNEL_CHUNK_SIZE="${TEACHER_CHANNEL_CHUNK_SIZE:-16}"
TEACHER_PROJECTION="${TEACHER_PROJECTION:-none}"   # none|pca
TEACHER_PCA_DIM="${TEACHER_PCA_DIM:-512}"
TEACHER_PCA_FIT_BATCHES="${TEACHER_PCA_FIT_BATCHES:-50}"
TEACHER_PCA_CENTER="${TEACHER_PCA_CENTER:-true}"
LAMBDA_FEAT="${LAMBDA_FEAT:-0.05}"
LAMBDA_LOGIT="${LAMBDA_LOGIT:-0.0}"
FEAT_LOSS_TYPE="${FEAT_LOSS_TYPE:-cosine}"   # mse|cosine

LAMBDA_TAG="$(echo "${LAMBDA_FEAT}" | tr '.' 'p' | tr -cd '[:alnum:]')"
RUN_BASENAME="${RUN_BASENAME:-dist-${MODEL_TAG}-lf${LAMBDA_TAG}}"
OUT_DIR="${OUT_DIR:-/tmp/${RUN_BASENAME}}"
mkdir -p "$OUT_DIR"

RESULTS="$OUT_DIR/results.tsv"
echo -e "fold\tval_subject\ttest_subject\tstatus\ttest_Accuracy\ttest_MacroF1\ttest_MicroF1\trun_name\tcomet_url\ttrain_log\tinfer_log" > "$RESULTS"

mapfile -t FOLDS < <(
python - <<PY
import itertools
import random

subjects = [101, 102, 103, 104, 105, 106, 107, 108]
n_folds = int("${N_FOLDS}")
seed = int("${FOLD_SEED}")

pairs = [(v, t) for v, t in itertools.permutations(subjects, 2)]
rng = random.Random(seed)
rng.shuffle(pairs)
picked = pairs[: min(n_folds, len(pairs))]
for i, (v, t) in enumerate(picked, 1):
    print(f"{i}|{v}|{t}")
PY
)

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
import re
import sys

acc = float("nan")
macro = float("nan")
micro = float("nan")
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"test_Accuracy\s*:\s*([0-9.eE+-]+)", line)
        if m:
            acc = float(m.group(1))
        m = re.search(r"test_MacroF1\s*:\s*([0-9.eE+-]+)", line)
        if m:
            macro = float(m.group(1))
        m = re.search(r"test_MicroF1\s*:\s*([0-9.eE+-]+)", line)
        if m:
            micro = float(m.group(1))
print(acc, macro, micro)
PY
}

for row in "${FOLDS[@]}"; do
  IFS='|' read -r fold_idx val_subj test_subj <<< "$row"

  run_name="${RUN_BASENAME}-f${fold_idx}-v${val_subj}-t${test_subj}"
  tags="[classification,distill,pamap2,ssl_wearables,${MODEL_TAG},fold_${fold_idx},val_${val_subj},test_${test_subj}]"
  log_file="$OUT_DIR/fold_${fold_idx}_v${val_subj}_t${test_subj}.train.log"
  infer_log="$OUT_DIR/fold_${fold_idx}_v${val_subj}_t${test_subj}.infer.log"

  echo "=== RUN distill fold=${fold_idx} val=${val_subj} test=${test_subj} ==="

  set +e
  python train.py -cn=distill_train \
    datasets=pamap2 \
    model.n_classes="${N_CLASSES}" \
    model.in_channels="${IN_CHANNELS}" \
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
    datasets.train.protocol_mode=ssl_wearables \
    datasets.val.protocol_mode=ssl_wearables \
    datasets.test.protocol_mode=ssl_wearables \
    datasets.train.val_subjects="[${val_subj}]" \
    datasets.val.val_subjects="[${val_subj}]" \
    datasets.test.val_subjects="[${val_subj}]" \
    datasets.train.test_subjects="[${test_subj}]" \
    datasets.val.test_subjects="[${test_subj}]" \
    datasets.test.test_subjects="[${test_subj}]" \
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
    loss_function.feat_loss_type="${FEAT_LOSS_TYPE}" > "$log_file" 2>&1
  rc=$?
  set -e

  status="ok"
  test_acc="nan"
  test_macro="nan"
  test_micro="nan"
  if [[ $rc -ne 0 ]]; then
    status="fail"
  else
    ckpt="saved/${run_name}/model_best.pth"
    if [[ ! -f "$ckpt" ]]; then
      status="fail_no_best_ckpt"
    else
      set +e
      python inference.py -cn=distill_inference \
        datasets=pamap2 \
        model.n_classes="${N_CLASSES}" \
        model.in_channels="${IN_CHANNELS}" \
        inferencer.require_cuda="${REQUIRE_CUDA}" \
        inferencer.from_pretrained="${ckpt}" \
        dataloader.batch_size="${BS}" \
        dataloader.num_workers="${NUM_WORKERS}" \
        datasets.train.protocol_mode=ssl_wearables \
        datasets.val.protocol_mode=ssl_wearables \
        datasets.test.protocol_mode=ssl_wearables \
        datasets.train.val_subjects="[${val_subj}]" \
        datasets.val.val_subjects="[${val_subj}]" \
        datasets.test.val_subjects="[${val_subj}]" \
        datasets.train.test_subjects="[${test_subj}]" \
        datasets.val.test_subjects="[${test_subj}]" \
        datasets.test.test_subjects="[${test_subj}]" > "$infer_log" 2>&1
      infer_rc=$?
      set -e
      if [[ $infer_rc -ne 0 ]]; then
        status="fail_infer"
      else
        parsed="$(extract_test_metrics "$infer_log")"
        test_acc="$(echo "$parsed" | awk '{print $1}')"
        test_macro="$(echo "$parsed" | awk '{print $2}')"
        test_micro="$(echo "$parsed" | awk '{print $3}')"
      fi
    fi
  fi
  comet_url="$(extract_comet_url "$log_file")"
  echo -e "${fold_idx}\t${val_subj}\t${test_subj}\t${status}\t${test_acc}\t${test_macro}\t${test_micro}\t${run_name}\t${comet_url}\t${log_file}\t${infer_log}" >> "$RESULTS"
done

echo "Done. Results: $RESULTS"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS"
else
  cat "$RESULTS"
fi

echo
echo "Aggregated (ok folds):"
python - "$RESULTS" <<'PY'
import csv
import math
import sys
from statistics import mean, stdev

path = sys.argv[1]
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
ok = [r for r in rows if str(r.get("status", "")).strip() == "ok"]

print(f"total_folds={len(rows)} ok_folds={len(ok)}")
for key in ("test_Accuracy", "test_MacroF1", "test_MicroF1"):
    vals = []
    for r in ok:
        try:
            v = float(str(r.get(key, "")).strip())
            if math.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if len(vals) == 0:
        print(f"{key}: n=0 mean=nan std=nan")
    elif len(vals) == 1:
        print(f"{key}: n=1 mean={vals[0]:.6f} std=0.000000")
    else:
        print(f"{key}: n={len(vals)} mean={mean(vals):.6f} std={stdev(vals):.6f}")
PY
