#!/usr/bin/env bash
# GPT-5.5 via the IAEDU gateway — standard benchmark (529-document test set).
#
#   bash scripts/run_gpt_benchmark.sh              # zero-shot then few-shot
#   bash scripts/run_gpt_benchmark.sh few_shot     # one strategy only
#
# ~5.6 hours. Cost is a LOWER BOUND unless the gateway reports usage: tiktoken
# counts only the prompt and the visible response, and GPT-5.5 bills reasoning
# tokens at $30/M that cannot be counted client-side. For Gemini those were 94%
# of the bill. Check with:  grep 'SSE chunk type=' logs/llm_cr/gpt_*.log
#
# No --temperature: this gateway documents only four form fields, and whether an
# extra one is ignored or rejected is untested. A rejection would kill the run.
# Probe it separately with --limit 2 before claiming greedy decoding in the paper.
#
# RATE LIMITS. The gateway signals them as an HTTP-200 event carrying
# "Rate limit reached (429)", which raise_for_status cannot see. Failed requests
# now back off exponentially (30s, 60s, ... capped at 600s, 6 attempts).
# --request-interval paces requests to avoid tripping the limit at all; raise it
# if you still see 429s. The whole benchmark is 1,058 requests, so if the quota
# is small this run may need it raised on the IAEDU side.
#
# Resumes: re-run the same command after an interruption and it continues from
# the last checkpoint, retrying only the documents that errored.

set -euo pipefail
source "$(dirname "$0")/_env.sh" "$@"

OUT="results/llm_extraction_cr/gpt5.5"
INTERVAL="${REQUEST_INTERVAL:-3}"   # seconds between requests; override with env

echo "=== Gateway reachability and quota check ==="
python - <<'PY'
import json, os, sys, uuid, requests
from dotenv import load_dotenv
load_dotenv(".env")
from src.llm_extraction.gpt.config import GptConfig
cfg = GptConfig()
r = requests.post(cfg.endpoint, headers={"x-api-key": cfg.api_key}, stream=True, timeout=60,
                  files={"channel_id": (None, cfg.channel_id),
                         "thread_id": (None, str(uuid.uuid4())),
                         "user_info": (None, "{}"),
                         "message": (None, "Responde apenas: OK")})
r.raise_for_status()
status = "ok"
for line in r.iter_lines(decode_unicode=True):
    if not line:
        continue
    try:
        c = json.loads(line)
    except json.JSONDecodeError:
        continue
    if c.get("type") == "error":
        status = str(c.get("content"))
        break
if status == "ok":
    print("  gateway responding normally")
else:
    print(f"  GATEWAY REFUSING: {status}")
    print("  The run would spend its time backing off. Wait for the quota window")
    print("  to reset, or ask IAEDU to raise it, then re-run this script.")
    sys.exit(1)
PY
echo

for S in "${STRATEGIES[@]}"; do
  echo "=== GPT-5.5 — $S ==="
  python scripts/llm_extraction/extract_gpt_spans.py \
    --strategy "$S" \
    --request-interval "$INTERVAL" \
    --test-file data/citilink-votie/test.jsonl \
    --output-file "$OUT/$S.jsonl" \
    2>&1 | tee "logs/llm_cr/gpt_$S.log"
done

echo
echo "=== Cost and discard rate ==="
python scripts/llm_extraction/alignment_report.py "$OUT"/*.jsonl

echo
echo "Model string the gateway disclosed (the only evidence of what actually ran):"
grep -h "Gateway reported model" logs/llm_cr/gpt_*.log | sort -u || echo "  (none disclosed)"

echo
echo "Done. Predictions in $OUT/"
echo "Score them with:  bash scripts/run_generative_camera_ready.sh report"
