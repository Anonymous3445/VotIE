#!/usr/bin/env bash
# Gemini 2.5 Pro — standard benchmark (529-document test set).
#
#   bash scripts/run_gemini_benchmark.sh              # zero-shot then few-shot
#   bash scripts/run_gemini_benchmark.sh few_shot     # one strategy only
#
# ~8 hours, ~$46. Measured at $0.0436/document, of which 94% is thinking tokens
# billed at the output rate — the checklist's old ~$2.60 estimate ignored those.
#
# Resumes: re-run the same command after an interruption and it continues from
# the last checkpoint, retrying only the documents that errored.

set -euo pipefail
source "$(dirname "$0")/_env.sh" "$@"

MODEL="gemini-2.5-pro"
OUT="results/llm_extraction_cr/gemini"

for S in "${STRATEGIES[@]}"; do
  echo "=== Gemini $MODEL — $S ==="
  python scripts/llm_extraction/extract_gemini_spans.py \
    --strategy "$S" \
    --model "$MODEL" \
    --temperature 0.0 \
    --test-file data/citilink-votie/test.jsonl \
    --output-file "$OUT/$S.jsonl" \
    2>&1 | tee "logs/llm_cr/gemini_$S.log"
done

echo
echo "=== Cost and discard rate ==="
python scripts/llm_extraction/alignment_report.py "$OUT"/*.jsonl

echo
echo "Done. Predictions in $OUT/"
echo "Score them with:  bash scripts/run_generative_camera_ready.sh report"
