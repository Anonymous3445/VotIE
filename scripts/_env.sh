# Shared bootstrap for the per-model benchmark scripts. Sourced, not executed.
#
# Activates the project conda environment. Getting this right in a non-interactive
# shell is the fiddly part: `conda activate` is a shell function that only exists
# after conda's profile script has been sourced, so calling it directly in a
# script fails with "CommandNotFoundError".

CONDA_ENV="${CONDA_ENV:-Citilink-VotIE}"

# Repo root, regardless of where the script was invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$_conda_base" && -f "$_conda_base/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$_conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "WARNING: could not activate conda env '$CONDA_ENV'; using $(command -v python)" >&2
  fi
fi

echo "env        : ${CONDA_DEFAULT_ENV:-<none>}"
echo "python     : $(command -v python)"

# tiktoken lives only in the project env. Missing it does not stop a run — it
# silently produces one with no cost data, discovered only after paying for it.
if ! python -c "import tiktoken" 2>/dev/null; then
  echo "ERROR: tiktoken not importable in this interpreter." >&2
  echo "       GPT cost cannot be reported. Fix with:  pip install tiktoken" >&2
  exit 1
fi

mkdir -p logs/llm_cr results/llm_extraction_cr

# Strategies come from the command line, or default to both.
STRATEGIES=("$@")
if [[ ${#STRATEGIES[@]} -eq 0 ]]; then
  STRATEGIES=(zero_shot few_shot)
fi
echo "strategies : ${STRATEGIES[*]}"
echo
