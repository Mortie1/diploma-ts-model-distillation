#!/usr/bin/env bash
set -euo pipefail

# Universal CV runner for article-2509 datasets.
# Supports:
# - k-fold datasets: hhar, motionsense
# - subject-pair CV datasets: pamap2_wrist10s, pamap2_leg2s, opportunity, wesad
# - fixed split dataset: mitbih_arrhythmia (single run)

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

DISTILL_ENABLED="${DISTILL_ENABLED:-false}"

if [[ "${DISTILL_ENABLED}" == "true" ]]; then
  MODEL_FAMILY="${MODEL_FAMILY:-student}"
  MODEL_TARGET="${MODEL_TARGET:-src.model.StudentClassifier}"
  MODEL_SIZE="${MODEL_SIZE:-}"
  TRAIN_CONFIG="${TRAIN_CONFIG:-distill_train}"
  INFER_CONFIG="${INFER_CONFIG:-distill_inference}"
else
  MODEL_FAMILY="${MODEL_FAMILY:-moment}"
  MODEL_TARGET="${MODEL_TARGET:-src.model.MomentClassificationAdapter}"
  MODEL_SIZE="${MODEL_SIZE:-small}"
  TRAIN_CONFIG="${TRAIN_CONFIG:-fm_train}"
  INFER_CONFIG="${INFER_CONFIG:-fm_inference}"
fi

DATASETS="${DATASETS:-pamap2_wrist10s opportunity hhar motionsense pamap2_leg2s mitbih_arrhythmia wesad}"
N_FOLDS="${N_FOLDS:-5}"
FOLD_SEED="${FOLD_SEED:-42}"

BS="${BS:-64}"
EPOCHS="${EPOCHS:-5}"
EPOCH_LEN="${EPOCH_LEN:-null}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_STEP="${LOG_STEP:-10}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
COMET_MODE="${COMET_MODE:-online}" # online|offline
EXTRA_OVERRIDES_TRAIN="${EXTRA_OVERRIDES_TRAIN:-}"
EXTRA_OVERRIDES_INFER="${EXTRA_OVERRIDES_INFER:-}"

# Distillation settings (used when DISTILL_ENABLED=true)
TEACHER_BACKEND="${TEACHER_BACKEND:-hf}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-facebook/hubert-large-ll60k}"
TEACHER_HIDDEN_DIM="${TEACHER_HIDDEN_DIM:-3072}"
TEACHER_LAYER_IDX="${TEACHER_LAYER_IDX:-1}"
TEACHER_PER_CHANNEL="${TEACHER_PER_CHANNEL:-true}"
TEACHER_CHANNEL_REDUCE="${TEACHER_CHANNEL_REDUCE:-concat}"
TEACHER_CHANNEL_CHUNK_SIZE="${TEACHER_CHANNEL_CHUNK_SIZE:-3}"
TEACHER_PROJECTION="${TEACHER_PROJECTION:-none}"
TEACHER_PCA_DIM="${TEACHER_PCA_DIM:-512}"
TEACHER_PCA_FIT_BATCHES="${TEACHER_PCA_FIT_BATCHES:-50}"
TEACHER_PCA_CENTER="${TEACHER_PCA_CENTER:-true}"
LAMBDA_FEAT="${LAMBDA_FEAT:-1.0}"
LAMBDA_LOGIT="${LAMBDA_LOGIT:-0.0}"
FEAT_LOSS_TYPE="${FEAT_LOSS_TYPE:-cosine}"

# Per-dataset epoch overrides (fall back to EPOCHS if not set).
# Override via env, e.g.: EPOCHS_hhar=3 EPOCHS_motionsense=3 bash scripts/run_article2509_cv.sh
EPOCHS_pamap2_wrist10s="${EPOCHS_pamap2_wrist10s:-${EPOCHS}}"
EPOCHS_opportunity="${EPOCHS_opportunity:-${EPOCHS}}"
EPOCHS_hhar="${EPOCHS_hhar:-${EPOCHS}}"
EPOCHS_motionsense="${EPOCHS_motionsense:-${EPOCHS}}"
EPOCHS_pamap2_leg2s="${EPOCHS_pamap2_leg2s:-${EPOCHS}}"
EPOCHS_mitbih_arrhythmia="${EPOCHS_mitbih_arrhythmia:-${EPOCHS}}"
EPOCHS_wesad="${EPOCHS_wesad:-${EPOCHS}}"

OUT_DIR="${OUT_DIR:-/tmp/${MODEL_FAMILY}_article2509_cv}"
mkdir -p "$OUT_DIR"
RESULTS="${OUT_DIR}/results.tsv"
RESULTS_AGG="${OUT_DIR}/results_agg_by_dataset.tsv"
echo -e "dataset\tfold\tstatus\ttest_Accuracy\ttest_MacroF1\ttest_MicroF1\trun_name\tcomet_url\ttrain_log\tinfer_log" > "$RESULTS"

extract_comet_url() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re, sys
url=""
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
import re, sys
acc=float("nan"); macro=float("nan"); micro=float("nan")
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"test_Accuracy\s*:\s*([0-9.eE+-]+)", line)
        if m: acc = float(m.group(1))
        m = re.search(r"test_MacroF1\s*:\s*([0-9.eE+-]+)", line)
        if m: macro = float(m.group(1))
        m = re.search(r"test_MicroF1\s*:\s*([0-9.eE+-]+)", line)
        if m: micro = float(m.group(1))
print(acc, macro, micro)
PY
}

dataset_meta() {
  local ds="$1"
  case "$ds" in
    pamap2_wrist10s) echo "8 3";;
    opportunity) echo "4 3";;
    hhar) echo "6 3";;
    motionsense) echo "6 12";;
    pamap2_leg2s) echo "12 3";;
    mitbih_arrhythmia) echo "2 1";;
    wesad) echo "4 1";;
    *) echo "Unknown dataset: $ds" >&2; exit 1;;
  esac
}

run_one_fold() {
  local ds="$1"
  local fold_tag="$2"
  local n_classes="$3"
  local in_channels="$4"
  local train_overrides="$5"
  local infer_overrides="$6"
  local n_epochs="$7"

  local run_name="a2509-${MODEL_FAMILY}-${MODEL_SIZE}-${ds}-${fold_tag}-e${n_epochs}-bs${BS}"
  run_name="${run_name:0:120}"
  local tags="[classification,article2509,${MODEL_FAMILY},${MODEL_FAMILY}-${MODEL_SIZE},${ds},${fold_tag}]"
  local log_train="${OUT_DIR}/${ds}_${fold_tag}.train.log"
  local log_infer="${OUT_DIR}/${ds}_${fold_tag}.infer.log"

  echo "=== RUN ${ds} / ${fold_tag} (epochs=${n_epochs}, distill=${DISTILL_ENABLED}) ==="

  local distill_train_args=""
  local distill_infer_args=""
  local fm_train_args=""
  local fm_infer_args=""

  if [[ "${DISTILL_ENABLED}" == "true" ]]; then
    distill_train_args="distillation.enabled=true \
      distillation.teacher_backend=${TEACHER_BACKEND} \
      distillation.teacher_model_name=${TEACHER_MODEL_NAME} \
      distillation.teacher_hidden_dim=${TEACHER_HIDDEN_DIM} \
      distillation.teacher_layer_idx=${TEACHER_LAYER_IDX} \
      distillation.teacher_per_channel=${TEACHER_PER_CHANNEL} \
      distillation.teacher_channel_reduce=${TEACHER_CHANNEL_REDUCE} \
      distillation.teacher_channel_chunk_size=${TEACHER_CHANNEL_CHUNK_SIZE} \
      distillation.teacher_projection=${TEACHER_PROJECTION} \
      distillation.teacher_pca_dim=${TEACHER_PCA_DIM} \
      distillation.teacher_pca_fit_batches=${TEACHER_PCA_FIT_BATCHES} \
      distillation.teacher_pca_center=${TEACHER_PCA_CENTER} \
      loss_function.lambda_feat=${LAMBDA_FEAT} \
      loss_function.lambda_logit=${LAMBDA_LOGIT} \
      loss_function.feat_loss_type=${FEAT_LOSS_TYPE}"
  else
    fm_train_args="model.model_size=${MODEL_SIZE} \
      model.require_provider_model=true \
      model.fail_on_provider_fallback=true"
    fm_infer_args="model.model_size=${MODEL_SIZE} \
      model.require_provider_model=true \
      model.fail_on_provider_fallback=true"
  fi

  set +e
  python train.py -cn="${TRAIN_CONFIG}" \
    datasets="${ds}" \
    model._target_="${MODEL_TARGET}" \
    model.n_classes="${n_classes}" \
    model.in_channels="${in_channels}" \
    trainer.require_cuda="${REQUIRE_CUDA}" \
    trainer.n_epochs="${n_epochs}" \
    trainer.epoch_len="${EPOCH_LEN}" \
    trainer.grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    trainer.log_step="${LOG_STEP}" \
    dataloader.batch_size="${BS}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    writer=cometml \
    writer.mode="${COMET_MODE}" \
    writer.run_name="${run_name}" \
    writer.tags="${tags}" \
    ${fm_train_args} \
    ${distill_train_args} \
    ${train_overrides} \
    ${EXTRA_OVERRIDES_TRAIN} > "${log_train}" 2>&1
  rc=$?
  set -e

  local status="ok"
  local acc="nan"
  local macro="nan"
  local micro="nan"
  local ckpt="saved/${run_name}/model_best.pth"

  if [[ $rc -ne 0 ]] && [[ ! -f "$ckpt" ]]; then
    status="fail"
  else
    [[ $rc -ne 0 ]] && { status="ok_interrupted"; echo "  Training interrupted — model_best.pth found, running inference."; }
    if [[ ! -f "$ckpt" ]]; then
      status="fail_no_best_ckpt"
    else
      set +e
      python inference.py -cn="${INFER_CONFIG}" \
        datasets="${ds}" \
        model._target_="${MODEL_TARGET}" \
        model.n_classes="${n_classes}" \
        model.in_channels="${in_channels}" \
        inferencer.require_cuda="${REQUIRE_CUDA}" \
        inferencer.from_pretrained="${ckpt}" \
        dataloader.batch_size="${BS}" \
        dataloader.num_workers="${NUM_WORKERS}" \
        ${fm_infer_args} \
        ${infer_overrides} \
        ${EXTRA_OVERRIDES_INFER} > "${log_infer}" 2>&1
      rc_infer=$?
      set -e
      if [[ $rc_infer -ne 0 ]]; then
        status="fail_infer"
      else
        parsed="$(extract_test_metrics "${log_infer}")"
        acc="$(echo "$parsed" | awk '{print $1}')"
        macro="$(echo "$parsed" | awk '{print $2}')"
        micro="$(echo "$parsed" | awk '{print $3}')"
      fi
    fi
  fi

  comet_url="$(extract_comet_url "${log_train}")"
  echo -e "${ds}\t${fold_tag}\t${status}\t${acc}\t${macro}\t${micro}\t${run_name}\t${comet_url}\t${log_train}\t${log_infer}" >> "$RESULTS"
}

for ds in ${DATASETS}; do
  read -r n_classes in_channels <<< "$(dataset_meta "$ds")"
  # Resolve per-dataset epoch count: EPOCHS_<dataset> env var overrides global EPOCHS.
  ds_var="EPOCHS_${ds}"
  ds_epochs="${!ds_var:-${EPOCHS}}"
  case "$ds" in
    hhar|motionsense)
      for ((i=0; i<${N_FOLDS}; i++)); do
        fold_tag="f$((i+1))"
        ov="datasets.train.fold_index=${i} datasets.val.fold_index=${i} datasets.test.fold_index=${i} datasets.train.n_folds=${N_FOLDS} datasets.val.n_folds=${N_FOLDS} datasets.test.n_folds=${N_FOLDS}"
        run_one_fold "$ds" "$fold_tag" "$n_classes" "$in_channels" "$ov" "$ov" "$ds_epochs"
      done
      ;;
    pamap2_wrist10s)
      # Leave-one-subject-out style: random subject pairs (val+test), N_FOLDS iterations.
      mapfile -t pairs < <(
      python - <<PY
import itertools, random
subs = [101,102,103,104,105,106,107,108]
pairs = [(v,t) for v,t in itertools.permutations(subs,2)]
random.Random(int("${FOLD_SEED}")).shuffle(pairs)
for i,(v,t) in enumerate(pairs[:int("${N_FOLDS}")],1):
    print(f"{i}|{v}|{t}")
PY
      )
      for row in "${pairs[@]}"; do
        IFS='|' read -r i v t <<< "$row"
        fold_tag="f${i}-v${v}-t${t}"
        ov="datasets.train.val_subjects=[${v}] datasets.val.val_subjects=[${v}] datasets.test.val_subjects=[${v}] datasets.train.test_subjects=[${t}] datasets.val.test_subjects=[${t}] datasets.test.test_subjects=[${t}]"
        run_one_fold "$ds" "$fold_tag" "$n_classes" "$in_channels" "$ov" "$ov" "$ds_epochs"
      done
      ;;
    pamap2_leg2s)
      # 5-fold subject-group CV following Haresamudram et al. [2022].
      # Subjects [101-108] are split into 5 groups; fold i uses group[i] as test, group[(i+1)%5] as val.
      mapfile -t leg_folds < <(
      python - <<PY
import numpy as np
subs = [101,102,103,104,105,106,107,108]
groups = [g.tolist() for g in np.array_split(np.array(subs), 5)]
for i, test_grp in enumerate(groups):
    val_grp = groups[(i + 1) % 5]
    test_str = ",".join(str(s) for s in test_grp)
    val_str  = ",".join(str(s) for s in val_grp)
    print(f"{i+1}|{test_str}|{val_str}")
PY
      )
      for row in "${leg_folds[@]}"; do
        IFS='|' read -r i test_subs val_subs <<< "$row"
        fold_tag="f${i}"
        ov="datasets.train.val_subjects=[${val_subs}] datasets.val.val_subjects=[${val_subs}] datasets.test.val_subjects=[${val_subs}] datasets.train.test_subjects=[${test_subs}] datasets.val.test_subjects=[${test_subs}] datasets.test.test_subjects=[${test_subs}]"
        run_one_fold "$ds" "$fold_tag" "$n_classes" "$in_channels" "$ov" "$ov" "$ds_epochs"
      done
      ;;
    opportunity)
      mapfile -t pairs < <(
      python - <<PY
import itertools, random
subs = [1,2,3,4]
pairs = [(v,t) for v,t in itertools.permutations(subs,2)]
random.Random(int("${FOLD_SEED}")).shuffle(pairs)
for i,(v,t) in enumerate(pairs[:int("${N_FOLDS}")],1):
    print(f"{i}|{v}|{t}")
PY
      )
      for row in "${pairs[@]}"; do
        IFS='|' read -r i v t <<< "$row"
        fold_tag="f${i}-v${v}-t${t}"
        ov="datasets.train.val_subject=${v} datasets.val.val_subject=${v} datasets.test.val_subject=${v} datasets.train.test_subject=${t} datasets.val.test_subject=${t} datasets.test.test_subject=${t}"
        run_one_fold "$ds" "$fold_tag" "$n_classes" "$in_channels" "$ov" "$ov" "$ds_epochs"
      done
      ;;
    wesad)
      mapfile -t pairs < <(
      python - <<PY
import itertools, random
subs = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
pairs = [(v,t) for v,t in itertools.permutations(subs,2)]
random.Random(int("${FOLD_SEED}")).shuffle(pairs)
for i,(v,t) in enumerate(pairs[:int("${N_FOLDS}")],1):
    print(f"{i}|{v}|{t}")
PY
      )
      for row in "${pairs[@]}"; do
        IFS='|' read -r i v t <<< "$row"
        fold_tag="f${i}-v${v}-t${t}"
        ov="datasets.train.val_subject=${v} datasets.val.val_subject=${v} datasets.test.val_subject=${v} datasets.train.test_subject=${t} datasets.val.test_subject=${t} datasets.test.test_subject=${t}"
        run_one_fold "$ds" "$fold_tag" "$n_classes" "$in_channels" "$ov" "$ov" "$ds_epochs"
      done
      ;;
    mitbih_arrhythmia)
      run_one_fold "$ds" "fixed" "$n_classes" "$in_channels" "" "" "$ds_epochs"
      ;;
    *)
      echo "Unknown dataset in loop: $ds" >&2
      exit 1
      ;;
  esac
done

echo "Done. Results: $RESULTS"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS"
else
  cat "$RESULTS"
fi

python - "$RESULTS" "$RESULTS_AGG" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from statistics import mean, stdev

src = sys.argv[1]
dst = sys.argv[2]

with open(src, "r", encoding="utf-8", errors="ignore") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

ok = [r for r in rows if str(r.get("status", "")).strip() in ("ok", "ok_interrupted")]
by_ds = defaultdict(list)
for r in ok:
    by_ds[str(r.get("dataset", "")).strip()].append(r)

metrics = ("test_Accuracy", "test_MacroF1", "test_MicroF1")

def stat(vals):
    if len(vals) == 0:
        return ("0", "nan", "nan")
    if len(vals) == 1:
        return ("1", f"{vals[0]:.6f}", "0.000000")
    return (str(len(vals)), f"{mean(vals):.6f}", f"{stdev(vals):.6f}")

with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(
        [
            "dataset",
            "ok_folds",
            "test_Accuracy_mean",
            "test_Accuracy_std",
            "test_MacroF1_mean",
            "test_MacroF1_std",
            "test_MicroF1_mean",
            "test_MicroF1_std",
        ]
    )
    for ds in sorted(by_ds):
        rows_ds = by_ds[ds]
        out = [ds, str(len(rows_ds))]
        for m in metrics:
            vals = []
            for r in rows_ds:
                try:
                    v = float(str(r.get(m, "")).strip())
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            _, mu, sd = stat(vals)
            out.extend([mu, sd])
        w.writerow(out)
PY

echo "Per-dataset aggregation: $RESULTS_AGG"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS_AGG"
else
  cat "$RESULTS_AGG"
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
ok = [r for r in rows if str(r.get("status", "")).strip() in ("ok", "ok_interrupted")]

print(f"total_runs={len(rows)} ok_runs={len(ok)}")
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
