#!/bin/bash
#SBATCH --job-name=votie_all_configs
#SBATCH --account=f202600030aivlabdeucaliong
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --output=logs/votie_all_configs_%j.out
#SBATCH --error=logs/votie_all_configs_%j.err

# ── Optional overrides ───────────────────────────────────────────────────────
# CONFIG_FILTER   : space-separated config basenames (without .yaml) to run
#                   default: every *.yaml directly under configs/
# EXPERIMENT_TAG  : suffix appended to each run name. Stable by default
#                   ("deucalion") so resubmissions can resume; set to something
#                   unique (e.g. "deucalion_${SLURM_JOB_ID}") for an independent run.
# SKIP_IF_DONE    : "1" (default) skip configs whose evaluation/ result file
#                   already exists for this tag. Set to "0" to disable.
# FORCE_OVERWRITE : "1" to add --force to run_pipeline.py (re-trains, overwrites
#                   predictions and results even if they exist). default: 0
# Example:
#   sbatch --export=ALL,CONFIG_FILTER="xlmr_crf deberta_crf" run_all_configs.slurm
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_FILTER="${CONFIG_FILTER:-}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-deucalion}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

# SLURM_SUBMIT_DIR = the repo root (where sbatch was called). The venv and the
# shared HF model cache live one level up, as siblings of the repo — the same
# layout the other jobs here use, so models resolve from ${BASE_DIR}/hf_cache.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE_DIR="$(dirname "${REPO_DIR}")"

VENV_DIR="${BASE_DIR}/votie"
HF_CACHE_DIR="${BASE_DIR}/hf_cache"



cd "${REPO_DIR}"

ml OpenMPI/5.0.3-GCC-13.3.0 CUDA/12.4.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

source "${VENV_DIR}/bin/activate"

# Block any accidental HuggingFace downloads during the job.
# Models / tokenizers must be pre-cached on the login node into ${HF_CACHE_DIR}
# by download_all_models.py, which writes them under ${HF_CACHE_DIR}/hub/.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/hub"

# Resolve imports from the venv, not a stray ~/.local copy: on Deucalion the
# EasyBuild module python can land pip installs in ~/.local and silently shadow
# the venv, so the job would look in the wrong place for its packages.
export PYTHONNOUSERSITE=1

# Make src/ scripts/ configs/ importable
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

mkdir -p logs

# Build the list of configs to run.
# We deliberately do NOT recurse into configs/municipality_experiments — those
# are handled by run_lomo_deberta.slurm.
CONFIGS=()
if [[ -n "${CONFIG_FILTER}" ]]; then
    for name in ${CONFIG_FILTER}; do
        cfg="configs/${name}.yaml"
        if [[ -f "${cfg}" ]]; then
            CONFIGS+=("${cfg}")
        else
            echo "WARNING: requested config not found, skipping: ${cfg}"
        fi
    done
else
    while IFS= read -r -d '' cfg; do
        CONFIGS+=("${cfg}")
    done < <(find configs -maxdepth 1 -type f -name '*.yaml' -print0 | sort -z)
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: no configs to run."
    exit 1
fi

# Extract the baseline name (model.name in YAML) from a config file. This is
# the same key run_pipeline.py uses to derive model_dir / predictions / results
# paths, so we need it to match exactly when checking for existing artifacts.
get_baseline_name() {
    local cfg="$1"
    python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['name'])" "${cfg}"
}

FORCE_FLAG=()
[[ "${FORCE_OVERWRITE}" == "1" ]] && FORCE_FLAG=(--force)

echo "================================================================================"
echo "VotIE — full configs pipeline"
echo "================================================================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            ${SLURM_NODELIST}"
echo "Repo dir:        ${REPO_DIR}"
echo "Venv:            ${VENV_DIR}"
echo "HF cache:        ${HF_CACHE_DIR}"
echo "Experiment tag:  ${EXPERIMENT_TAG}"
echo "Skip if done:    ${SKIP_IF_DONE}"
echo "Force overwrite: ${FORCE_OVERWRITE}"
echo "Configs (${#CONFIGS[@]}):"
for cfg in "${CONFIGS[@]}"; do
    echo "  - ${cfg}"
done
echo "Started:         $(date)"
echo "================================================================================"

FAILED=()
SUCCEEDED=()
SKIPPED=()

for cfg in "${CONFIGS[@]}"; do
    base="$(basename "${cfg}" .yaml)"
    baseline="$(get_baseline_name "${cfg}")"
    run_name="${base}_${EXPERIMENT_TAG}"

    model_dir="models/${baseline}/${run_name}"
    predictions_file="predictions/${baseline}_${run_name}.jsonl"
    results_file="evaluation/${baseline}_${run_name}.json"

    # Resume logic: inspect existing artifacts and decide whether to skip,
    # resume from prediction (--predict-only), or resume from evaluation
    # (--evaluate-only). FORCE_OVERWRITE bypasses all of this.
    stage_flag=()
    if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
        :
    elif [[ -f "${results_file}" ]]; then
        if [[ "${SKIP_IF_DONE}" == "1" ]]; then
            echo "[SKIP] ${base} — results already exist (${results_file})"
            SKIPPED+=("${base}")
            continue
        fi
    elif [[ -f "${predictions_file}" ]]; then
        echo "[RESUME] ${base} — predictions exist, running --evaluate-only"
        stage_flag=(--evaluate-only)
    elif [[ -d "${model_dir}" ]]; then
        echo "[RESUME] ${base} — model exists, running --predict-only"
        stage_flag=(--predict-only)
    fi

    echo ""
    echo "################################################################################"
    echo "# Running config: ${cfg}"
    echo "# Run name:       ${run_name}"
    echo "# Started:        $(date)"
    echo "################################################################################"

    if srun python3 scripts/run_pipeline.py \
            --config "${cfg}" \
            --name   "${run_name}" \
            "${stage_flag[@]}" \
            "${FORCE_FLAG[@]}"; then
        SUCCEEDED+=("${base}")
        echo "[OK]   ${base}"
    else
        FAILED+=("${base}")
        echo "[FAIL] ${base} — continuing with remaining configs"
    fi
done

echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
echo "Skipped   (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo "Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "Finished: $(date)"

# Exit non-zero if anything failed so SLURM marks the job appropriately.
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
