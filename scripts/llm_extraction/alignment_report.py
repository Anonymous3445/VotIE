#!/usr/bin/env python3
"""Report the LLM span-discard rate and API cost from instrumented extraction runs.

Answers reviewer 3fsk's question: "How many generated spans were discarded
because they could not be matched exactly to the input?"

This requires predictions produced *after* the extractors were instrumented to
persist a `diagnostics` field (see src/llm_extraction/{gemini,gpt}/extractor.py).
Runs predating that instrumentation carry no pre-alignment record: for those the
`alignment_fidelity` block in the evaluation JSONs is computed over spans that
already survived alignment and is therefore trivially 1.0. This script refuses to
report on such files rather than restating that artifact as a finding.

Usage:
    python scripts/llm_extraction/alignment_report.py \
        results/llm_extraction/gemini/few_shot.jsonl \
        results/llm_extraction/gpt5.5/few_shot.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# USD per 1M tokens. Update to the rates in force for the billed run.
PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gpt-5.5": {"input": 1.25, "output": 10.00},
}


def load(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def report(path):
    records = load(path)
    instrumented = [r for r in records if r.get("diagnostics")]

    print(f"\n{'=' * 68}\n{path}\n{'=' * 68}")
    print(f"documents            : {len(records)}")

    if not instrumented:
        print(
            "\n  NO DIAGNOSTICS FOUND.\n"
            "  This run predates the extractor instrumentation, so the number of\n"
            "  discarded spans is unrecoverable from it. Re-run the extraction to\n"
            "  measure it. Do NOT report 0% — the saved spans are post-alignment,\n"
            "  so any fidelity computed over them is 1.0 by construction."
        )
        return None

    if len(instrumented) < len(records):
        print(f"  warning: only {len(instrumented)}/{len(records)} documents carry diagnostics")

    raw = sum(d["diagnostics"].get("n_raw_generated", 0) for d in instrumented)
    typed = sum(d["diagnostics"].get("n_after_type_filter", 0) for d in instrumented)
    aligned = sum(d["diagnostics"].get("n_aligned", 0) for d in instrumented)
    dropped_type = raw - typed
    dropped_align = sum(d["diagnostics"].get("n_dropped_alignment", 0) for d in instrumented)

    print(f"raw spans generated  : {raw}")
    print(f"  dropped, bad type  : {dropped_type} ({dropped_type / raw:.2%})" if raw else "")
    print(f"  dropped, no align  : {dropped_align} ({dropped_align / raw:.2%})" if raw else "")
    print(f"retained for scoring : {aligned} ({aligned / raw:.2%})" if raw else "")

    by_type = Counter()
    for d in instrumented:
        for f in d["diagnostics"].get("failed_alignments", []):
            by_type[f.get("type", "?")] += 1
    if by_type:
        print("\nunalignable spans by type:")
        for t, c in by_type.most_common():
            print(f"    {t:<22} {c}")

    # Token usage / cost
    tok_in = tok_out = 0
    for d in instrumented:
        u = d["diagnostics"].get("usage") or {}
        tok_in += u.get("prompt_token_count") or u.get("prompt_tokens") or u.get("input_tokens") or 0
        tok_out += u.get("candidates_token_count") or u.get("completion_tokens") or u.get("output_tokens") or 0

    model = records[0].get("model", "")
    if tok_in or tok_out:
        print(f"\ntokens  in={tok_in:,}  out={tok_out:,}")
        price = PRICING.get(model)
        if price:
            cost = tok_in / 1e6 * price["input"] + tok_out / 1e6 * price["output"]
            print(f"est. cost ({model}): ${cost:.2f}  (${cost / len(instrumented) * 1000:.2f} / 1k docs)")
        else:
            print(f"  no pricing entry for {model!r}; add one to PRICING")
    else:
        print("\ntokens: not reported by langextract for this provider/version")

    times = [r.get("processing_time") for r in records if r.get("processing_time")]
    if times:
        print(f"latency: mean {sum(times) / len(times):.1f}s/doc, total {sum(times) / 3600:.2f}h")

    return {
        "path": str(path),
        "model": model,
        "documents": len(records),
        "raw_spans": raw,
        "dropped_unknown_type": dropped_type,
        "dropped_alignment": dropped_align,
        "retained": aligned,
        "discard_rate": (raw - aligned) / raw if raw else None,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+", help="*.jsonl prediction files")
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    summaries = [s for p in args.predictions if (s := report(p))]

    if args.out and summaries:
        Path(args.out).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nWrote summary -> {args.out}")


if __name__ == "__main__":
    main()
