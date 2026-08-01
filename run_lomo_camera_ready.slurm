#!/bin/bash
#SBATCH --job-name=votie_lomo_cr
#SBATCH --account=f202600030aivlabdeucaliong
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --array=0-83%8
#SBATCH --output=logs/lomo_cr_%A_%a.out
#SBATCH --error=logs/lomo_cr_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# VotIE — Leave-One-Municipality-Out, full re-run for the camera-ready version.
#
# This is a COMPLETE re-run of every LOMO cell, not a top-up. The previously
# reported deberta_crf and xlmr_crf LOMO runs used batch size 16; all configs
# are now uniformly batch size 8 (see scripts/make_lomo_configs.py), so old and
# new results are not comparable and must not be mixed.
#
# ISOLATION — nothing here touches the previously reported results:
#   new results  ->  results/lomo_cr/<model>/<M>_seed_s<SEED>_evaluation.json
#   old results  ->  results/lomo/...            (left untouched, never read)
#   model dirs   ->  models/<model>/<model>_<M>_s<SEED>_<TAG>   (TAG defaults
#                    to "cr_b8", which no previous run used)
#
# Target matrix — 84 tasks
#   encoder-CRF   bert_crf, deberta_crf, xlmr_crf          × 6 muni × 3 seeds = 54
#   linear heads  bert_linear, deberta_linear, xlmr_linear × 6 muni × seed 42 = 18
#   BiLSTM-CRF    bilstm_fasttext                          × 6 muni × seed 42 =  6
#   feature CRF   crf                                      × 6 muni × seed 42 =  6
#
# Overrides (sbatch --export=ALL,VAR=...)
#   MODEL_FILTER    space-separated model names to include (default: all)
#   MUNI_FILTER     space-separated municipalities        (default: M01..M06)
#   SEED_FILTER     space-separated seeds                 (default: 42 13 123)
#   EXPERIMENT_TAG  run-name suffix (default "cr_b8"); keep stable to resume
#   RESULTS_SUBDIR  results root leaf   (default "lomo_cr")
#   SKIP_IF_DONE    "1" (default) skip cells already present under the new root
#   FORCE_OVERWRITE "1" pass --force to run_pipeline.py (retrain from scratch)
#   DRY_RUN         "1" print this task's assignment and exit without running
#
# Examples
#   sbatch run_lomo_camera_ready.slurm                       # all 84
#   sbatch --array=0-83 --export=ALL,DRY_RUN=1 run_lomo_camera_ready.slurm
#   sbatch --array=0-17 --export=ALL,MODEL_FILTER="xlmr_crf" run_lomo_camera_ready.slurm
#
# FILTERS CHANGE THE INDEXING. Every task rebuilds the same manifest and indexes
# into it, so task N means the same run on every resubmission — but only while
# the three filters are unchanged. Narrow the filters and the manifest shortens
# and every index shifts. Two consequences worth knowing before you resubmit:
#   * narrow --array to match, or tasks past the end no-op (harmless, just idle);
#   * SEED_FILTER also gates the single-seed block, which is pinned to seed 42.
#     Resubmitting with SEED_FILTER="13 123" to top up encoder seeds silently
#     drops all 30 linear/BiLSTM/CRF tasks. Use MODEL_FILTER for that instead.
# Each task prints its own manifest size; if that number moves between
# submissions, the indices moved with it.
#
# The 6 feature-CRF tasks are sklearn-crfsuite and hold an A100 they never touch
# (~19 min each). Left in for one-command completeness; run them with
# MODEL_FILTER="crf" on a CPU partition if the GPU queue is contended.
# ─────────────────────────────────────────────────────────────────────────────

set -o pipefail

MODEL_FILTER="${MODEL_FILTER:-}"
MUNI_FILTER="${MUNI_FILTER:-M01 M02 M03 M04 M05 M06}"
SEED_FILTER="${SEED_FILTER:-42 13 123}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-cr_b8}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-lomo_cr}"
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

# Weights must be pre-cached on the login node; block downloads from the node.
#   python download_all_models.py --cache-dir "${HF_CACHE_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/hub"
# FastText vectors for the 6 bilstm_fasttext tasks, cached alongside the
# encoders by the same script. src/embeddings_cache.py would also find them via
# HF_HOME; setting it explicitly keeps that working if HF_HOME ever moves.
export VOTIE_FASTTEXT_PATH="${HF_CACHE_DIR}/fasttext/cc.pt.300.bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

RESULTS_ROOT="results/${RESULTS_SUBDIR}"
mkdir -p logs "${RESULTS_ROOT}"

# ── Build the manifest ───────────────────────────────────────────────────────
# Every array task builds the identical list and indexes into it, so the mapping
# from SLURM_ARRAY_TASK_ID to a run is stable across resubmissions with the same
# filters (see the note above). Highest-value cells run first: the three
# encoder-CRF models rebuild Table 2 itself, then the baselines the meta-review
# asked to add to it. That front half is also the expensive half — 54 of the 84
# tasks and roughly 80 of the ~98 GPU-hours — so a partial run still yields the
# table, but budget accordingly rather than expecting the tail to dominate.
MULTI_SEED_MODELS="xlmr_crf deberta_crf bert_crf"
SINGLE_SEED_MODELS="xlmr_linear deberta_linear bert_linear bilstm_fasttext crf"

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
    for muni in ${MUNI_FILTER}; do
        for seed in ${SEED_FILTER}; do
            MANIFEST+=("${model}|${muni}|${seed}")
        done
    done
done
for model in ${SINGLE_SEED_MODELS}; do
    in_filter "${model}" "${MODEL_FILTER}" || continue
    in_filter "42" "${SEED_FILTER}" || continue
    for muni in ${MUNI_FILTER}; do
        MANIFEST+=("${model}|${muni}|42")
    done
done

IDX="${SLURM_ARRAY_TASK_ID:-0}"

if [[ "${IDX}" -ge "${#MANIFEST[@]}" ]]; then
    echo "[NOOP] array index ${IDX} is beyond the manifest (${#MANIFEST[@]} entries). Nothing to do."
    exit 0
fi

IFS='|' read -r MODEL MUNI SEED <<< "${MANIFEST[$IDX]}"

CONFIG="configs/municipality_experiments/${MODEL}_${MUNI}.yaml"
RUN_NAME="${MODEL}_${MUNI}_s${SEED}_${EXPERIMENT_TAG}"
RUN_DIR="results/discriminative_models/${MODEL}/${RUN_NAME}"
MODEL_DIR="models/${MODEL}/${RUN_NAME}"

LOMO_DIR="${RESULTS_ROOT}/${MODEL}"
LOMO_EVAL="${LOMO_DIR}/${MUNI}_seed_s${SEED}_evaluation.json"
LOMO_PRED="${LOMO_DIR}/${MUNI}_seed_s${SEED}_predictions.jsonl"

echo "================================================================================"
echo "VotIE LOMO — camera-ready full re-run (batch size 8 throughout)"
echo "================================================================================"
echo "Array job:      ${SLURM_ARRAY_JOB_ID:-n/a} task ${IDX} of ${#MANIFEST[@]}"
echo "Manifest size:  ${#MANIFEST[@]} (expected 84 with default filters; a"
echo "                different number means the filters shifted every index)"
echo "Node:           ${SLURM_NODELIST:-n/a}"
echo "Model:          ${MODEL}"
echo "Held-out muni:  ${MUNI}"
echo "Seed:           ${SEED}"
echo "Config:         ${CONFIG}"
echo "Run name:       ${RUN_NAME}"
echo "Destination:    ${LOMO_EVAL}"
echo "Started:        $(date)"
echo "================================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] would run the above; exiting."
    exit 0
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[FAIL] config not found: ${CONFIG}"
    echo "       Regenerate the LOMO configs first:"
    echo "         python scripts/make_lomo_configs.py --overwrite"
    exit 1
fi

# The BiLSTM is the only model with a weight file outside the HF cache. Check it
# up front: the loader raises rather than silently using random vectors, but
# failing here costs a second instead of a queue slot plus model construction.
if [[ "${MODEL}" == "bilstm_fasttext" && ! -f "${VOTIE_FASTTEXT_PATH}" ]]; then
    echo "[FAIL] FastText vectors not found: ${VOTIE_FASTTEXT_PATH}"
    echo "       Fetch them on the login node (no internet from compute nodes):"
    echo "         python download_all_models.py --only-fasttext --cache-dir \"${HF_CACHE_DIR}\""
    exit 1
fi

mkdir -p "${LOMO_DIR}"

# Provenance stamp recording what this batch of results is, so it can never be
# confused with results/lomo/. Written by whichever task gets here first rather
# than by task 0, which a narrowed --array may never schedule. `set -C` makes the
# create atomic, so concurrent tasks cannot interleave into a half-written file.
PROVENANCE="${RESULTS_ROOT}/PROVENANCE.txt"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${GIT_DIRTY}" != "0" ]]; then
    echo "[WARN] ${GIT_DIRTY} uncommitted file(s): the recorded commit does NOT"
    echo "       identify the code running here. Commit before a run you intend to publish."
fi
if [[ ! -f "${PROVENANCE}" ]] && (set -C; : > "${PROVENANCE}") 2>/dev/null; then
    {
        echo "VotIE LOMO — camera-ready re-run"
        echo "array job id : ${SLURM_ARRAY_JOB_ID:-n/a}"
        echo "started      : $(date)"
        echo "experiment   : ${EXPERIMENT_TAG}"
        echo "git commit   : $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
        echo "git dirty    : ${GIT_DIRTY} file(s) modified"
        echo ""
        echo "All configs regenerated by scripts/make_lomo_configs.py --overwrite:"
        echo "  batch size 8, learning rate 5e-5, CUDA ordinal 0 for every"
        echo "  Transformer configuration."
        echo "Supersedes results/lomo/, where deberta_crf and xlmr_crf used batch size 16."
        echo "Do not aggregate the two directories together."
        echo ""
        echo "The standard-benchmark rows trained under the same settings live in"
        echo "results/discriminative_models/<model>/<model>_s<SEED>_${EXPERIMENT_TAG}/"
        echo "(see run_standard_camera_ready.slurm). Table 1 and Table 2 are only"
        echo "comparable when both come from tag '${EXPERIMENT_TAG}'."
    } > "${PROVENANCE}"
    echo "[PROVENANCE] wrote ${PROVENANCE}"
fi

# Already collected in THIS results root — nothing to do.
if [[ -f "${LOMO_EVAL}" && "${SKIP_IF_DONE}" == "1" && "${FORCE_OVERWRITE}" != "1" ]]; then
    echo "[SKIP] result already present: ${LOMO_EVAL}"
    exit 0
fi

# This batch's own run finished but was never collected — copy rather than
# spend another GPU-hour. Keyed on RUN_NAME, so it can only match this batch.
if [[ -f "${RUN_DIR}/evaluation.json" && "${FORCE_OVERWRITE}" != "1" ]]; then
    echo "[COLLECT] evaluation exists in ${RUN_DIR}, copying without re-running"
    cp "${RUN_DIR}/evaluation.json" "${LOMO_EVAL}"
    [[ -f "${RUN_DIR}/predictions.jsonl" ]] && cp "${RUN_DIR}/predictions.jsonl" "${LOMO_PRED}"
    exit 0
fi

# Resume part-way: run_pipeline.py refuses to overwrite, so the stage flag has
# to match exactly which artifacts are already on disk.
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
    echo "[OK] pipeline finished for ${MODEL} ${MUNI} seed ${SEED}"
else
    echo "[FAIL] pipeline failed for ${MODEL} ${MUNI} seed ${SEED}"
    exit 1
fi

if [[ -f "${RUN_DIR}/evaluation.json" ]]; then
    cp "${RUN_DIR}/evaluation.json" "${LOMO_EVAL}"
    echo "[COLLECT] ${LOMO_EVAL}"
else
    echo "[FAIL] pipeline reported success but ${RUN_DIR}/evaluation.json is missing"
    exit 1
fi

if [[ -f "${RUN_DIR}/predictions.jsonl" ]]; then
    cp "${RUN_DIR}/predictions.jsonl" "${LOMO_PRED}"
    echo "[COLLECT] ${LOMO_PRED}"
fi

echo "Finished: $(date)"
