#!/usr/bin/env python3
"""Report span-discard rate, document-failure rate, cost and latency for the
generative extractors.

Answers three reviewer questions from a single instrumented run:
  * "How many generated spans were discarded because they could not be matched
    exactly to the input?"                                    (Review 1, C2)
  * API cost, latency and throughput next to the encoder GPU hours (Review 1, W4)
  * how often a document fails outright rather than extracting badly (audit A3)

This requires predictions produced *after* the extractors were instrumented to
persist a `diagnostics` field (see src/llm_extraction/*/extractor.py and
src/llm_extraction/shared/data_utils.py:result_to_record). Runs predating that
instrumentation carry no pre-alignment record: for those the `alignment_fidelity`
block in the evaluation JSONs is computed over spans that already survived
alignment and is therefore trivially 1.0. This script refuses to report on such
files rather than restating that artifact as a finding.

Usage:
    python scripts/llm_extraction/alignment_report.py \
        results/llm_extraction_cr/gemini/few_shot.jsonl \
        results/llm_extraction_cr/gpt5.5/few_shot.jsonl \
        --out results/alignment_report.json
"""

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

# USD per 1M tokens, list price as of August 2026. Output rates include thinking/
# reasoning tokens for both providers, which is why the report bills
# `billable_output_tokens` rather than `output_tokens`.
#
#   gemini-2.5-pro  $1.25 in / $10 out  for prompts <=200k tokens
#                   ($2.50 / $15 above 200k; our chunks are ~20k chars, so the
#                    lower tier applies)
#   gpt-5.5         $5.00 in / $30 out
#
# GPT-5.5 is reached through the IAEDU institutional gateway, not OpenAI directly,
# so the amount actually billed may differ from list price. Reconcile against the
# provider's console before quoting a figure in the paper.
PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
}
SELF_HOSTED = ("amalia",)


def load(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def _norm(s: str) -> str:
    return " ".join((s or "").split()).casefold()


def classify_failure(span_text: str, source_text: str, threshold: float = 0.8) -> str:
    """Say why a generated span could not be aligned.

    "Discarded because it could not be matched" covers two opposite situations,
    and reporting them as one number misrepresents what alignment does:

      near_miss     — most of the span really is in the source, so rejecting it
                      loses a span the model essentially got right. Lossy.
      hallucinated  — the text is simply not in the document. Rejecting it is
                      correct and prevents a false positive.

    Measured as the longest contiguous run of the span found verbatim in the
    source, as a fraction of the span's length.
    """
    span, source = _norm(span_text), _norm(source_text)
    if not span:
        return "empty"
    match = SequenceMatcher(None, span, source, autojunk=False).find_longest_match(
        0, len(span), 0, len(source)
    )
    return "near_miss" if match.size / len(span) >= threshold else "hallucinated"


def _is_self_hosted(model: str) -> bool:
    return any(tag in (model or "").lower() for tag in SELF_HOSTED)


def report(path):
    records = load(path)
    instrumented = [r for r in records if r.get("diagnostics")]
    model = records[0].get("model", "") if records else ""

    print(f"\n{'=' * 68}\n{path}\n{'=' * 68}")
    print(f"model                : {model}")
    print(f"documents            : {len(records)}")

    # Document-level generation failures — a different phenomenon from poor
    # extraction quality, and one that depresses recall on the affected documents.
    failed = [r for r in records if r.get("error")]
    rate = len(failed) / len(records) if records else 0
    print(f"document failures    : {len(failed)} ({rate:.2%})")
    if failed:
        causes = Counter((r.get("error") or "")[:60] for r in failed)
        for cause, n in causes.most_common(5):
            print(f"    {n:>4}  {cause}")

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

    # Span accounting is only meaningful over documents that produced spans;
    # failed documents contribute usage and latency but no generation.
    produced = [r for r in instrumented if not r["diagnostics"].get("document_failed")]

    raw = sum(d["diagnostics"].get("n_raw_generated", 0) for d in produced)
    typed = sum(d["diagnostics"].get("n_after_type_filter", 0) for d in produced)
    aligned = sum(d["diagnostics"].get("n_aligned", 0) for d in produced)
    dropped_type = raw - typed
    dropped_align = sum(d["diagnostics"].get("n_dropped_alignment", 0) for d in produced)

    print(f"\nraw spans generated  : {raw}")
    if raw:
        print(f"  dropped, bad type  : {dropped_type} ({dropped_type / raw:.2%})")
        print(f"  dropped, no align  : {dropped_align} ({dropped_align / raw:.2%})")
        print(f"retained for scoring : {aligned} ({aligned / raw:.2%})")

    truncated = sum(1 for r in produced if r["diagnostics"].get("truncation_recovered"))
    if truncated:
        print(f"truncated + salvaged : {truncated} documents (hit max_output_tokens)")

    by_type = Counter()
    by_cause = Counter()
    examples = {}
    for d in produced:
        for f in d["diagnostics"].get("failed_alignments", []):
            by_type[f.get("type", "?")] += 1
            cause = classify_failure(f.get("text", ""), d.get("text", ""))
            by_cause[cause] += 1
            examples.setdefault(cause, (f.get("text", ""), f.get("type", "?"), d.get("id")))

    if by_cause:
        total_failed = sum(by_cause.values())
        print("\nwhy spans failed to align:")
        for cause, c in by_cause.most_common():
            note = {
                "hallucinated": "not in the document — rejecting it prevents a false positive",
                "near_miss": "mostly present — rejection loses a span the model nearly had",
                "empty": "empty span text",
            }.get(cause, "")
            print(f"    {cause:<14} {c:>5} ({c / total_failed:.1%})  {note}")
            text, etype, doc = examples[cause]
            print(f"                   e.g. {etype} {text[:60]!r} in {doc}")

    if by_type:
        print("\nunalignable spans by type:")
        for t, c in by_type.most_common():
            print(f"    {t:<22} {c}")

    # Token usage and cost
    tok_in = tok_out = calls = 0
    sources = Counter()
    for d in instrumented:
        u = d["diagnostics"].get("usage") or {}
        sources[u.get("source", "unavailable")] += 1
        tok_in += u.get("input_tokens", 0) or 0
        tok_out += u.get("billable_output_tokens", u.get("output_tokens", 0)) or 0
        calls += u.get("api_calls", 0) or 0

    cost = None
    if tok_in or tok_out:
        source = ", ".join(f"{s} ({n})" for s, n in sources.most_common())
        print(f"\ntokens  in={tok_in:,}  out={tok_out:,}  (source: {source})")
        print(f"api calls: {calls} over {len(instrumented)} documents "
              f"({calls / len(instrumented):.2f} chunks/doc)")
        price = PRICING.get(model)
        if price:
            cost = tok_in / 1e6 * price["input"] + tok_out / 1e6 * price["output"]
            print(f"est. cost ({model}): ${cost:.2f}  "
                  f"(${cost / len(instrumented) * 1000:.2f} / 1k documents)")
        elif _is_self_hosted(model):
            print(f"  {model} is self-hosted: report GPU hours, not API cost")
        else:
            print(f"  no pricing entry for {model!r}; add one to PRICING")
    elif _is_self_hosted(model):
        print("\ntokens: not reported by the vLLM endpoint for this run")
    else:
        print("\ntokens: NOT CAPTURED — cost cannot be reported for this run")

    times = [r.get("processing_time") for r in records if r.get("processing_time")]
    if times:
        print(f"latency: mean {sum(times) / len(times):.1f}s/doc, "
              f"total {sum(times) / 3600:.2f}h wall-clock")

    return {
        "path": str(path),
        "model": model,
        "strategy": records[0].get("strategy") if records else None,
        "documents": len(records),
        "document_failures": len(failed),
        "document_failure_rate": rate,
        "instrumented_documents": len(instrumented),
        "raw_spans": raw,
        "dropped_unknown_type": dropped_type,
        "dropped_alignment": dropped_align,
        "dropped_alignment_by_cause": dict(by_cause),
        "retained": aligned,
        "discard_rate": (raw - aligned) / raw if raw else None,
        "truncated_documents": truncated,
        "api_calls": calls,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "token_source": dict(sources),
        "est_cost_usd": cost,
        "mean_seconds_per_doc": sum(times) / len(times) if times else None,
        "total_hours": sum(times) / 3600 if times else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+", help="*.jsonl prediction files")
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    summaries = [s for p in args.predictions if (s := report(p))]

    if args.out and summaries:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nWrote summary -> {args.out}")


if __name__ == "__main__":
    main()
