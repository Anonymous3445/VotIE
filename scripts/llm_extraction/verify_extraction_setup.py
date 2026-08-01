#!/usr/bin/env python3
"""Pre-flight checks for the generative extraction stack. Run before spending anything.

Every check here corresponds to a defect that reached published results:

  leakage       four of five curated demonstrations are gold documents from Porto
                and Covilha, which a LOMO fold evaluates in full
  alignment     the whitespace-recovery path returned offsets into a normalised
                string, so it could only ever fail — silently discarding
                multi-line SUBJECT spans
  denominator   exact and relaxed macro used different, data-dependent denominators
  diagnostics   extraction scripts rebuilt records by hand and dropped the
                diagnostics block, making cost and discard rate unrecoverable
  dedup         LOMO folds pooled the oversampled training split without
                deduplicating, re-scoring and re-paying for duplicated segments

Runs entirely offline. No API keys, no network, no cost.

Usage:
    python scripts/llm_extraction/verify_extraction_setup.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
logging.disable(logging.WARNING)

from src.llm_extraction.lomo_examples import select_lomo_examples, scaffold_example
from src.llm_extraction.span_alignment import align_span, align_extraction_result
from src.llm_extraction.schemas import SpanEntity, SpanExtractionResult, ENTITY_TYPES
from src.llm_extraction.shared.data_utils import result_to_record
from src.llm_extraction.shared.evaluation import (
    evaluate_span_extraction,
    macro_over_entity_types,
    overlap_length,
)
from src.llm_extraction.usage import UsageTracker

MUNICIPALITIES = ["M01", "M02", "M03", "M04", "M05", "M06"]
EXPECTED_FOLD_SIZES = {"M01": 504, "M02": 398, "M03": 720, "M04": 235, "M05": 556, "M06": 472}

_failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        _failures.append(name)
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))


def section(title):
    print(f"\n{title}\n" + "-" * len(title))


def check_environment():
    section("0. Runtime environment")
    print(f"  interpreter: {sys.executable}")

    # tiktoken is how GPT cost gets estimated when the gateway hides usage.
    # Missing it does not fail the run — it silently produces a run with no
    # cost data, which is only discovered after paying for it.
    from src.llm_extraction.usage import count_tokens
    n = count_tokens("Ponderado e analisado o assunto")
    check("tiktoken available (GPT cost estimation)", n is not None,
          "pip install tiktoken — without it GPT runs report zero tokens"
          if n is None else f"{n} tokens")

    for module, why in [("langextract", "all extractors"),
                        ("dotenv", ".env loading"),
                        ("google.genai", "Gemini provider"),
                        ("seqeval", "exact-match evaluation")]:
        try:
            __import__(module)
            ok = True
        except ImportError:
            ok = False
        check(f"{module} importable ({why})", ok)

    import os
    for key, needed_by in [("GEMINI_KEY", "Gemini"), ("IAEDU_API_KEY", "GPT-5.5")]:
        # Not a failure: a LOMO-only or AMALIA-only run needs neither.
        state = "set" if os.environ.get(key) else "not set in this shell"
        print(f"  [INFO] {key} ({needed_by}): {state}")


def check_leakage():
    section("1. Few-shot leakage (the defect that reached Table 2)")
    for held_out in MUNICIPALITIES:
        examples, provenance = select_lomo_examples(held_out)
        sources = {p.municipality for p in provenance}
        check(
            f"{held_out}: 5 demonstrations, none from {held_out}",
            len(examples) == 5 and held_out not in sources,
            f"from {sorted(sources)}"
            + (f", relaxed: {[p.phenomenon for p in provenance if p.tier == 'relaxed']}"
               if any(p.tier == "relaxed" for p in provenance) else ""),
        )
    _, prov = scaffold_example("M06")
    check("zero-shot scaffold also respects the fold boundary",
          prov[0].municipality != "M06", f"scaffold from {prov[0].municipality}")

    # Determinism: the same fold must reproduce byte-identically.
    a = [p.id for p in select_lomo_examples("M03")[1]]
    b = [p.id for p in select_lomo_examples("M03")[1]]
    check("selection is deterministic across calls", a == b)


def check_alignment():
    section("2. Span alignment")
    src = "Ponderado o assunto, deliberou aprovar a proposta\nde alteracao ao regulamento."
    cases = [
        ("verbatim span aligns", "deliberou", True),
        ("case difference aligns", "DELIBEROU", True),
        ("multi-line span aligns (was discarded)",
         "a proposta de alteracao ao regulamento", True),
        ("paraphrase is rejected", "uma proposta inventada", False),
        ("wrong-tail paraphrase is rejected", "a proposta de alteracao ao codigo", False),
    ]
    for name, text, should in cases:
        out = align_span(src, SpanEntity(text=text, type="SUBJECT"), strict=True)
        check(name, (out is not None) == should)

    out = align_span(src, SpanEntity(text="a proposta de alteracao ao regulamento", type="SUBJECT"))
    check("offsets index the ORIGINAL text, not a normalised copy",
          out is not None and src[out.start:out.end] == out.text)

    # The rebuild inside align_extraction_result used to drop these.
    res = SpanExtractionResult(
        id="d", text=src, entities=[SpanEntity(text="deliberou", type="VOTING")],
        model="m", strategy="few_shot", processing_time=1.0, api_time=0.5,
    )
    res.diagnostics = {"n_raw_generated": 1}
    aligned, _ = align_extraction_result(res, strict=True)
    check("alignment preserves api_time and diagnostics",
          aligned.api_time == 0.5 and aligned.diagnostics == {"n_raw_generated": 1})


def check_denominator():
    section("3. Macro-averaging denominator")
    check("ENTITY_TYPES has the 11 scored types", len(ENTITY_TYPES) == 11)

    # A type absent from the data must still count in the denominator.
    partial = {ENTITY_TYPES[0]: {"precision": 1.0, "recall": 1.0, "f1": 1.0}}
    p, r, f = macro_over_entity_types(partial)
    check("one perfect type out of 11 gives 1/11, not 1.0",
          abs(f - 1 / 11) < 1e-9, f"f1={f:.4f}")

    check("overlap_length handles unaligned spans",
          overlap_length(SpanEntity(text="a", type="VOTING"),
                         SpanEntity(text="b", type="VOTING", start=0, end=2)) == 0)

    # Exact and relaxed must report the same denominator on the same data.
    text = "o Executivo Municipal deliberou por unanimidade"
    gold = [{"id": "d1", "text": text, "spans": [
        {"start": 22, "end": 31, "label": "VOTING", "text": "deliberou"}]}]
    pred = [SpanExtractionResult(
        id="d1", text=text, model="m", strategy="few_shot", processing_time=0.1,
        entities=[SpanEntity(text="deliberou", type="VOTING", start=22, end=31)])]
    out = evaluate_span_extraction(pred, gold, compute_fidelity=False, relaxed_matching=True)
    check("exact and relaxed share one denominator",
          out["macro_denominator"] == out["relaxed_boundary_metrics"]["macro_denominator"] == 11)
    check("a single correct type scores 1/11 macro",
          abs(out["entity_level"]["f1"] - 1 / 11) < 1e-6,
          f"f1={out['entity_level']['f1']:.4f}")


def check_diagnostics_persist():
    section("4. Diagnostics survive serialization (else the run is wasted)")
    tracker = UsageTracker(source="api")
    with tracker.timed_call() as fields:
        fields.update(input_tokens=1500, output_tokens=300, reasoning_tokens=120)
    usage = tracker.snapshot()
    check("usage tracker accumulates tokens", usage["input_tokens"] == 1500)
    check("thinking tokens are billed as output",
          usage["billable_output_tokens"] == 420, f"{usage['billable_output_tokens']}")
    check("API time is measured", usage["api_seconds"] >= 0)

    res = SpanExtractionResult(
        id="d1", text="t", model="gemini-2.5-pro", strategy="few_shot",
        processing_time=1.0, api_time=usage["api_seconds"],
        entities=[SpanEntity(text="a", type="VOTING", start=0, end=1)])
    res.diagnostics = {"n_raw_generated": 3, "n_aligned": 1,
                       "n_dropped_alignment": 2, "usage": usage}
    record = result_to_record(res)
    check("result_to_record carries diagnostics", bool(record.get("diagnostics")))
    check("record is JSON-serializable", json.dumps(record) is not None)
    roundtrip = json.loads(json.dumps(record))
    check("token counts survive the round trip",
          roundtrip["diagnostics"]["usage"]["input_tokens"] == 1500)
    check("api_time is populated, not None", roundtrip["api_time"] is not None)


def check_fold_sizes():
    section("5. LOMO fold sizes (deduplication of the oversampled train split)")
    sys.path.insert(0, str(Path(__file__).parent))
    from run_cross_municipality import filter_by_municipality, deduplicate
    from src.llm_extraction.shared.data_utils import load_jsonl

    splits = [load_jsonl(f"data/citilink-votie/{s}.jsonl") for s in ("train", "dev", "test")]
    for muni, expected in EXPECTED_FOLD_SIZES.items():
        pooled = [r for split in splits for r in filter_by_municipality(split, muni)]
        unique = deduplicate(pooled)
        check(f"{muni}: {expected} unique segments",
              len(unique) == expected,
              f"got {len(unique)} from {len(pooled)} rows "
              f"({len(pooled) - len(unique)} duplicates dropped)")
    total = sum(EXPECTED_FOLD_SIZES.values())
    check(f"folds sum to the documented {total}", total == 2885)


def main():
    print("=" * 70)
    print("VotIE generative extraction — pre-flight checks")
    print("=" * 70)

    check_environment()
    check_leakage()
    check_alignment()
    check_denominator()
    check_diagnostics_persist()
    check_fold_sizes()

    print("\n" + "=" * 70)
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED — do not start a paid run:")
        for name in _failures:
            print(f"  - {name}")
        print("=" * 70)
        return 1
    print("All checks passed. Safe to run the smoke test.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
