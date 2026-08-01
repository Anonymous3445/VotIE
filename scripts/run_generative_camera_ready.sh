#!/usr/bin/env bash
# Camera-ready re-run of the generative baselines, instrumented.
#
# Covers checklist items N1-N7, R7, R8, A3, A4, M6 for the standard benchmark and
# L9-L14 for LOMO: one pass per configuration yields the span-discard rate, the
# token cost, the document-level failure rate and the latency together.
#
# Results land in results/llm_extraction_cr/ and results/lomo_cr/llm/, so the
# published results/ tree is untouched until the numbers have been compared (N7).
#
# Prerequisites
#   .env contains GEMINI_KEY and IAEDU_API_KEY
#   AMALIA's vLLM endpoint is up (AMALIA_BASE_URL, default amalia.inesctec.pt:8000)
#
# Usage
#   bash scripts/run_generative_camera_ready.sh verify   # offline checks, free
#   bash scripts/run_generative_camera_ready.sh smoke    # 5 docs per config, ~$0.15
#   bash scripts/run_generative_camera_ready.sh api      # Gemini + GPT, full test set
#   bash scripts/run_generative_camera_ready.sh amalia   # AMALIA only (needs the GPU server)
#   bash scripts/run_generative_camera_ready.sh lomo     # cross-municipality, all providers
#   bash scripts/run_generative_camera_ready.sh report   # aggregate whatever exists

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-verify}"
OUT="results/llm_extraction_cr"
LOMO_OUT="results/lomo_cr/llm"
LOGS="logs/llm_cr"
mkdir -p "$LOGS"

GEMINI_MODEL="gemini-2.5-pro"
LIMIT_ARG=""
if [[ "$MODE" == "smoke" ]]; then
  LIMIT_ARG="--limit 5"
  OUT="results/llm_extraction_smoke"
fi

# Every paid mode runs the offline checks first. They are the gate that would
# have caught the LOMO prompt leakage and the dropped diagnostics.
preflight () {
  echo ">>> Pre-flight checks"
  python scripts/llm_extraction/verify_extraction_setup.py
}

run_gemini () {  # $1 = strategy
  echo ">>> Gemini $1"
  python scripts/llm_extraction/extract_gemini_spans.py \
    --strategy "$1" --model "$GEMINI_MODEL" --temperature 0.0 \
    --test-file data/citilink-votie/test.jsonl \
    --output-file "$OUT/gemini/$1.jsonl" \
    $LIMIT_ARG 2>&1 | tee "$LOGS/gemini_$1.log"
}

run_gpt () {
  echo ">>> GPT-5.5 $1"
  # No --temperature: the gateway's handling of extra fields is untested.
  # Probe it separately before claiming greedy decoding in the paper.
  python scripts/llm_extraction/extract_gpt_spans.py \
    --strategy "$1" \
    --test-file data/citilink-votie/test.jsonl \
    --output-file "$OUT/gpt5.5/$1.jsonl" \
    $LIMIT_ARG 2>&1 | tee "$LOGS/gpt_$1.log"
}

run_amalia () {
  echo ">>> AMALIA $1"
  # --model omitted so the served id is auto-detected and recorded; the paper's
  # "8B" claim needs to be checked against whatever the server reports.
  #
  # --timeout 600 (default 120): in the published run, 49 of the 58 zero-shot
  # failures were "Request timed out" at exactly 120.1s — our client limit, not
  # the model. Timeouts are deliberately not retried, so each one became a total
  # recall miss. A failure rate measured at 120s describes our configuration;
  # this measures AMALIA.
  #
  # --max-output-tokens stays at the published 8192 on purpose. The few-shot
  # failures are JSON truncation, which looks like degenerate repetition rather
  # than a cap set too tight; keeping it fixed leaves that finding measurable and
  # comparable, and the new diagnostics (truncation_recovered, output_tokens)
  # will show which it is.
  python scripts/llm_extraction/extract_amalia_spans.py \
    --strategy "$1" --temperature 0.0 --timeout 600 \
    --test-file data/citilink-votie/test.jsonl \
    --output-file "$OUT/amalia/$1.jsonl" \
    $LIMIT_ARG 2>&1 | tee "$LOGS/amalia_$1.log"
}

run_lomo () {  # $1 = provider, $2 = strategy
  echo ">>> LOMO $1 $2"
  python scripts/llm_extraction/run_cross_municipality.py \
    --provider "$1" --strategy "$2" \
    --output-dir "$LOMO_OUT/$1" \
    --skip-existing 2>&1 | tee "$LOGS/lomo_$1_$2.log"
}

case "$MODE" in
  verify)
    preflight
    echo "Checks passed. Next: bash scripts/run_generative_camera_ready.sh smoke"
    exit 0
    ;;
  smoke)
    preflight
    run_gemini few_shot; run_gpt few_shot
    echo ">>> Verifying the smoke output is actually instrumented"
    python - <<'PY'
import json, sys
paths = ["results/llm_extraction_smoke/gemini/few_shot.jsonl",
         "results/llm_extraction_smoke/gpt5.5/few_shot.jsonl"]
for p in paths:
    rows = [json.loads(l) for l in open(p)]
    inst = [r for r in rows if r.get("diagnostics")]
    tok = sum((r["diagnostics"].get("usage") or {}).get("input_tokens", 0) for r in inst)
    src = {(r["diagnostics"].get("usage") or {}).get("source") for r in inst}
    print(f"{p}: {len(inst)}/{len(rows)} instrumented, {tok} input tokens, source={src}")
    assert len(inst) == len(rows), "diagnostics missing — do not start the full run"
    if tok == 0:
        raise SystemExit(
            f"\n{p}: no token usage captured, so cost cannot be reported.\n"
            "  source='unavailable' means the provider hid usage AND the local\n"
            "  fallback was unavailable. Check that tiktoken is installed in THIS\n"
            "  interpreter (run scripts/llm_extraction/verify_extraction_setup.py),\n"
            "  then re-run this smoke test.")
print("OK: instrumentation verified, safe to run the full set")
PY
    ;;
  api)
    preflight
    for s in zero_shot few_shot; do run_gemini "$s"; run_gpt "$s"; done
    ;;
  amalia)
    preflight
    for s in zero_shot few_shot; do run_amalia "$s"; done
    ;;
  lomo)
    preflight
    for p in gemini gpt amalia; do
      for s in few_shot zero_shot; do run_lomo "$p" "$s"; done
    done
    ;;
  report) ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

# --- Aggregate: scores, then cost/discard/failure report -------------------
shopt -s nullglob

if compgen -G "$OUT/*/*.jsonl" > /dev/null; then
  echo ">>> Evaluating standard benchmark"
  for f in "$OUT"/*/*.jsonl; do
    python scripts/llm_extraction/evaluate_spans.py "$f" \
      --ground-truth data/citilink-votie/test.jsonl --relaxed \
      --output-file "${f%.jsonl}_evaluation.json" 2>&1 | tail -12
  done
fi

echo ">>> Cost, discard rate, document failures"
ALL=("$OUT"/*/*.jsonl "$LOMO_OUT"/*/*.jsonl)
if [ ${#ALL[@]} -gt 0 ]; then
  python scripts/llm_extraction/alignment_report.py "${ALL[@]}" \
    --out "$OUT/alignment_report.json"
fi

echo
echo "Done. Before substituting into Tables 1-2 (N7), compare against the published run:"
echo "  new: $OUT/*/*_evaluation.json  and  $LOMO_OUT/*/*_evaluation.json"
echo "  old: results/llm_extraction/*/*_evaluation.json"
echo "Expect small drift: multi-line spans that were previously discarded now align."
