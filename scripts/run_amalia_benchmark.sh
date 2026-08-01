#!/usr/bin/env bash
# AMALIA (self-hosted vLLM) — standard benchmark (529-document test set).
#
#   bash scripts/run_amalia_benchmark.sh              # zero-shot then few-shot
#   bash scripts/run_amalia_benchmark.sh few_shot     # one strategy only
#
# ~21 GPU-hours, no API cost. Longer than the published run, because documents
# that previously died at the 120s client timeout now run to completion.
#
# --timeout 600: in the published run 49 of the 58 zero-shot failures were
# "Request timed out" at exactly 120.1s. Timeouts are deliberately not retried,
# so each became a total recall miss. A failure rate measured at 120s describes
# our client, not the model.
#
# --model is omitted on purpose: the served id is auto-detected from /v1/models
# and recorded per document. The server exposes CitiLink LoRA adapters alongside
# the base model, so recording which one answered matters.
#
# max_output_tokens stays at the published 8192. It is a circuit breaker, not a
# budget: a faithful extraction needs 59 output tokens at the median and 252 at
# worst, so 8192 is ~33x the hardest legitimate case. Raising it would only let
# a runaway generation run longer.
#
# Resumes: re-run the same command after an interruption and it continues from
# the last checkpoint, retrying only the documents that errored.

set -euo pipefail
source "$(dirname "$0")/_env.sh" "$@"

OUT="results/llm_extraction_cr/amalia"
BASE_URL="${AMALIA_BASE_URL:-http://amalia.inesctec.pt:8000}"

echo "=== Checking the AMALIA server before committing hours to it ==="
if ! curl -sf --max-time 15 "$BASE_URL/v1/models" -o /tmp/amalia_models.json; then
  echo "ERROR: cannot reach $BASE_URL — is the vLLM server up and are you on the network?" >&2
  exit 1
fi
python - <<'PY'
import json
data = json.load(open("/tmp/amalia_models.json")).get("data", [])
print(f"  server is up, {len(data)} model(s) available:")
for m in data:
    parent = f"  (LoRA on {m['parent']})" if m.get("parent") else ""
    ctx = f"  max_model_len={m['max_model_len']:,}" if m.get("max_model_len") else ""
    print(f"    {m['id']}{ctx}{parent}")
print(f"  -> the extractor will use {data[0]['id']!r} (first entry, the base model)")
PY
echo

for S in "${STRATEGIES[@]}"; do
  echo "=== AMALIA — $S ==="
  python scripts/llm_extraction/extract_amalia_spans.py \
    --strategy "$S" \
    --temperature 0.0 \
    --timeout 600 \
    --test-file data/citilink-votie/test.jsonl \
    --output-file "$OUT/$S.jsonl" \
    2>&1 | tee "logs/llm_cr/amalia_$S.log"
done

echo
echo "=== Discard rate, document failures, truncation ==="
python scripts/llm_extraction/alignment_report.py "$OUT"/*.jsonl

echo
echo "Done. Predictions in $OUT/"
echo "Score them with:  bash scripts/run_generative_camera_ready.sh report"
