#!/bin/bash
#SBATCH --job-name=votie_std_cr
#SBATCH --account=f202600030aivlabdeucaliong
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --array=0-7%4
#SBATCH --output=logs/std_cr_%A_%a.out
#SBATCH --error=logs/std_cr_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# VotIE — standard benchmark, partial re-run for the camera-ready version.
#
# WHY THIS EXISTS. run_lomo_camera_ready.slurm retrains every LOMO cell at a
# uniform batch size 8 / learning rate 5e-5. Four of the eight Table 1 rows were
# published under different settings, so without this job Table 1 and Table 2
# would describe models trained differently, and §5's "35% relative drop" would
# be measuring a hyperparameter change as well as a domain shift:
#
#   config        published        camera-ready    Table 1 row
#   deberta_crf   b16 / 5e-5   ->  b8 / 5e-5       DeBERTa-CRF
#   xlmr_crf      b16 / 5e-5   ->  b8 / 5e-5       XLM-R-CRF   (abstract: 93.2)
#   bert_linear   b8  / 2e-5   ->  b8 / 5e-5       BERTimbau-Lin.
#   xlmr_linear   b8  / 2e-5   ->  b8 / 5e-5       XLM-R-Lin.
#
# It also makes §4 true as written: "AdamW optimization (learning rate 5e-5) ...
# without any hyperparameter tuning" does not describe the two linear heads as
# published (they used 2e-5), and configs/deberta_crf.yaml carried the comment
# "Best grid search configuration", which is tuning.
#
# bert_crf and deberta_linear were ALREADY b8 / 5e-5 and are deliberately NOT
# re-run. They are the control: if BERTimbau-CRF's aggregate moves away from the
# published 90.5±1.5, something other than these four configs changed.
#
# Target matrix — 8 tasks
#   deberta_crf  x seeds 42, 13, 123
#   xlmr_crf     x seeds 42, 13, 123
#   bert_linear  x seed 42
#   xlmr_linear  x seed 42
#
# ISOLATION — nothing here touches the previously reported results:
#   new results  ->  results/standard_cr/<model>/seed_s<SEED>_evaluation.json
#   old results  ->  results/discriminative_models/<model>/seeds/   (never read)
#   model dirs   ->  models/<model>/<model>_s<SEED>_<TAG>  (TAG default "cr_b8")
#
# Overrides (sbatch --export=ALL,VAR=...) behave as in run_lomo_camera_ready.slurm.
#   MODEL_FILTER / SEED_FILTER / EXPERIMENT_TAG / RESULTS_SUBDIR
#   SKIP_IF_DONE / FORCE_OVERWRITE / DRY_RUN
# Filters change the indexing here too — narrow --array to match, and read the
# manifest size each task prints.
#
# Examples
#   sbatch run_standard_camera_ready.slurm                    # all 8
#   sbatch --export=ALL,DRY_RUN=1 run_standard_camera_ready.slurm
#   sbatch --array=0-2 --export=ALL,MODEL_FILTER="deberta_crf" run_standard_camera_ready.slurm
# ─────────────────────────────────────────────────────────────────────────────

set -o pipefail

MODEL_FILTER="${MODEL_FILTER:-}"
SEED_FILTER="${SEED_FILTER:-42 13 123}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-cr_b8}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-standard_cr}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE_DIR="$(dirname "${REPO_DIR}")"
VENV_DIR="${BASE_DIR}/votie"
HF_CACHE_DIR="${BASE_DIR}/hf_cache"

cd "${REPO_DIR}" || exit 1

ml OpenMPI/5.0.3-GCC-13.3.0 CUDA/12.4.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0
source "${VENV_DIR}/bin/activate"

# Models must be pre-cached on the login node; block downloads from the node.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/hub"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

RESULTS_ROOT="results/${RESULTS_SUBDIR}"
mkdir -p logs "${RESULTS_ROOT}"

# ── Build the manifest ───────────────────────────────────────────────────────
# Only the configs whose hyperparameters changed. Adding a model here without a
# matching config change would silently re-run something already correct.
MULTI_SEED_MODELS="deberta_crf xlmr_crf"
SINGLE_SEED_MODELS="bert_linear xlmr_linear"

in_filter() {
    local needle="$1" haystack="$2"
    [[ -z "${haystack}" ]] && return 0
    for item in ${haystack}; do
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

MANIFEST=()
for model in ${MULTI_SEED_MODELS}; do
    in_filter "${model}" "${MODEL_FILTER}" || continue
    for seed in ${SEED_FILTER}; do
        MANIFEST+=("${model}|${seed}")
    done
done
for model in ${SINGLE_SEED_MODELS}; do
    in_filter "${model}" "${MODEL_FILTER}" || continue
    in_filter "42" "${SEED_FILTER}" || continue
    MANIFEST+=("${model}|42")
done

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if [[ "${IDX}" -ge "${#MANIFEST[@]}" ]]; then
    echo "[NOOP] array index ${IDX} is beyond the manifest (${#MANIFEST[@]} entries). Nothing to do."
    exit 0
fi

IFS='|' read -r MODEL SEED <<< "${MANIFEST[$IDX]}"

CONFIG="configs/${MODEL}.yaml"
RUN_NAME="${MODEL}_s${SEED}_${EXPERIMENT_TAG}"
RUN_DIR="results/discriminative_models/${MODEL}/${RUN_NAME}"
MODEL_DIR="models/${MODEL}/${RUN_NAME}"

OUT_DIR="${RESULTS_ROOT}/${MODEL}"
OUT_EVAL="${OUT_DIR}/seed_s${SEED}_evaluation.json"
OUT_PRED="${OUT_DIR}/seed_s${SEED}_predictions.jsonl"

echo "================================================================================"
echo "VotIE standard benchmark — camera-ready partial re-run (batch size 8, lr 5e-5)"
echo "================================================================================"
echo "Array job:      ${SLURM_ARRAY_JOB_ID:-n/a} task ${IDX} of ${#MANIFEST[@]}"
echo "Manifest size:  ${#MANIFEST[@]} (expected 8 with default filters)"
echo "Node:           ${SLURM_NODELIST:-n/a}"
echo "Model:          ${MODEL}"
echo "Seed:           ${SEED}"
echo "Config:         ${CONFIG}"
echo "Run name:       ${RUN_NAME}"
echo "Destination:    ${OUT_EVAL}"
echo "Started:        $(date)"
echo "================================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] would run the above; exiting."
    exit 0
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[FAIL] config not found: ${CONFIG}"
    exit 1
fi

# Guard the premise of this job: if the config is not at the camera-ready
# settings, re-running it produces neither the published numbers nor the new
# ones. Cheaper to catch here than after a GPU-hour.
python3 - "${CONFIG}" <<'PY' || exit 1
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
training = cfg.get("training", {})
batch, lr = training.get("batch_size"), str(training.get("learning_rate"))
if batch != 8 or lr != "5e-5":
    print(f"[FAIL] {sys.argv[1]} is batch_size={batch}, learning_rate={lr}; "
          f"expected 8 / 5e-5. This job exists to standardise those.")
    sys.exit(1)
print(f"  config check: batch_size=8, learning_rate=5e-5")
PY

mkdir -p "${OUT_DIR}"

PROVENANCE="${RESULTS_ROOT}/PROVENANCE.txt"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${GIT_DIRTY}" != "0" ]]; then
    echo "[WARN] ${GIT_DIRTY} uncommitted file(s): the recorded commit does NOT"
    echo "       identify the code running here. Commit before a run you intend to publish."
fi
if [[ ! -f "${PROVENANCE}" ]] && (set -C; : > "${PROVENANCE}") 2>/dev/null; then
    {
        echo "VotIE standard benchmark — camera-ready partial re-run"
        echo "array job id : ${SLURM_ARRAY_JOB_ID:-n/a}"
        echo "started      : $(date)"
        echo "experiment   : ${EXPERIMENT_TAG}"
        echo "git commit   : $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
        echo "git dirty    : ${GIT_DIRTY} file(s) modified"
        echo ""
        echo "Re-runs only the four Table 1 rows whose hyperparameters changed:"
        echo "  deberta_crf, xlmr_crf   batch 16 -> 8"
        echo "  bert_linear, xlmr_linear   lr 2e-5 -> 5e-5"
        echo "bert_crf and deberta_linear were already batch 8 / lr 5e-5 and are"
        echo "NOT re-run; BERTimbau-CRF's published 90.5+-1.5 is the control."
        echo ""
        echo "Pairs with results/lomo_cr/ (tag '${EXPERIMENT_TAG}'). Table 1 and"
        echo "Table 2 are only comparable when both come from that tag."
    } > "${PROVENANCE}"
    echo "[PROVENANCE] wrote ${PROVENANCE}"
fi

# Already collected in THIS results root — nothing to do.
if [[ -f "${OUT_EVAL}" && "${SKIP_IF_DONE}" == "1" && "${FORCE_OVERWRITE}" != "1" ]]; then
    echo "[SKIP] result already present: ${OUT_EVAL}"
    exit 0
fi

# This batch's own run finished but was never collected — copy rather than spend
# another GPU-hour. Keyed on RUN_NAME, so it can only match this batch.
if [[ -f "${RUN_DIR}/evaluation.json" && "${FORCE_OVERWRITE}" != "1" ]]; then
    echo "[COLLECT] evaluation exists in ${RUN_DIR}, copying without re-running"
    cp "${RUN_DIR}/evaluation.json" "${OUT_EVAL}"
    [[ -f "${RUN_DIR}/predictions.jsonl" ]] && cp "${RUN_DIR}/predictions.jsonl" "${OUT_PRED}"
    exit 0
fi

# Resume part-way: run_pipeline.py refuses to overwrite, so the stage flag has to
# match exactly which artifacts are already on disk.
STAGE_FLAG=()
if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
    :
elif [[ -f "${RUN_DIR}/predictions.jsonl" ]]; then
    echo "[RESUME] predictions exist, running --evaluate-only"
    STAGE_FLAG=(--evaluate-only)
elif [[ -d "${MODEL_DIR}" ]]; then
    echo "[RESUME] model exists, running --predict-only"
    STAGE_FLAG=(--predict-only)
fi

FORCE_FLAG=()
[[ "${FORCE_OVERWRITE}" == "1" ]] && FORCE_FLAG=(--force)

if srun python3 scripts/run_pipeline.py \
        --config "${CONFIG}" \
        --name   "${RUN_NAME}" \
        --seed   "${SEED}" \
        "${STAGE_FLAG[@]}" \
        "${FORCE_FLAG[@]}"; then
    echo "[OK] pipeline finished for ${MODEL} seed ${SEED}"
else
    echo "[FAIL] pipeline failed for ${MODEL} seed ${SEED}"
    exit 1
fi

if [[ -f "${RUN_DIR}/evaluation.json" ]]; then
    cp "${RUN_DIR}/evaluation.json" "${OUT_EVAL}"
    echo "[COLLECT] ${OUT_EVAL}"
else
    echo "[FAIL] pipeline reported success but ${RUN_DIR}/evaluation.json is missing"
    exit 1
fi

if [[ -f "${RUN_DIR}/predictions.jsonl" ]]; then
    cp "${RUN_DIR}/predictions.jsonl" "${OUT_PRED}"
    echo "[COLLECT] ${OUT_PRED}"
fi

echo "Finished: $(date)"
