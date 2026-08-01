# LLM Span Extraction

Generative baseline for VotIE using three LLM providers via the `langextract` library.

## Architecture

All three extractors follow the same two-stage pipeline:

1. **Generation** — langextract constructs the prompt, segments the input if needed, calls the LLM, and parses the JSON output.
2. **Alignment** — `span_alignment.py` finds exact character positions in the source text for each extracted span.

## Module Structure

```
src/llm_extraction/
├── schemas.py               # Pydantic schemas (SpanEntity, SpanExtractionResult)
├── span_alignment.py        # Post-processing: map LLM spans to character offsets
├── fixed_examples.py        # 5 curated few-shot examples (standard benchmark only)
├── lomo_examples.py         # per-fold, leakage-free demonstrations (LOMO only)
├── usage.py                 # provider token/latency capture for cost reporting
│
├── gemini/                  # Gemini 2.5 Pro via langextract built-in Google provider
│   ├── extractor.py         # GeminiSpanExtractor
│   └── config.py            # GeminiConfig
│
├── gpt/                     # GPT 5.5 via langextract + custom streaming provider
│   ├── extractor.py         # GptSpanExtractor
│   ├── langextract_provider.py  # GptLanguageModel (BaseLanguageModel subclass)
│   └── config.py            # GptConfig
│
├── amalia/                  # AMALIA via langextract + vLLM OpenAI-compatible endpoint
│   ├── extractor.py         # AmaliaSpanExtractor
│   ├── prompts.py           # Shared prompt description (used by all three extractors)
│   └── config.py            # AmaliaConfig
│
└── shared/
    ├── base_extractor.py    # BaseSpanExtractor abstract class
    ├── data_utils.py        # load_jsonl / save_jsonl helpers
    └── evaluation.py        # Span-level evaluation utilities
```

## Key Design Decisions

- **Few-shot examples**: `fixed_examples.py` provides the same 5 curated examples to all extractors, covering 11 of the 12 schema types plus an empty-extraction case.
- **LOMO uses a different set.** Four of the five curated examples are gold documents from Porto and Covilha, and a LOMO fold evaluates *every* split of the held-out municipality — so reusing them would put test documents and their answers in the prompt. `lomo_examples.py` selects five demonstrations per fold from the five training municipalities instead, covering the same five phenomena. `run_cross_municipality.py` aborts a fold if any demonstration originates in the held-out municipality.
- **Shared prompt**: `amalia/prompts.py:get_prompt_description()` is used by all three extractors to ensure identical system instructions.
- **No schema-constrained decoding.** `use_schema_constraints=False` for all three providers. What is actually applied is JSON-mode decoding, and it differs per provider: Gemini sets `response_mime_type="application/json"`, AMALIA uses OpenAI JSON mode (routed through vLLM's xgrammar), and GPT relies on `fence_output=True` with no constraint at all. Describe it this way in the paper — "schema-constrained decoding" would overstate it.
- **Span alignment** is applied to all extractor outputs to recover exact character offsets, since LLMs generate text spans without positions.


## Measurement

`usage.py` captures token counts and API time at the provider client, because
langextract's `AnnotatedDocument` exposes only `extractions` and `text` — it
discards the response object and with it the usage. Counts are accumulated per
document and written into `diagnostics` by `shared/data_utils.result_to_record()`,
which every extraction script must go through: hand-built record dicts previously
dropped the block, making the span-discard rate and cost unrecoverable from a
completed run.

Thinking/reasoning tokens are folded into `billable_output_tokens` — both Gemini
2.5 Pro and GPT-5.5 bill them at the output rate but report them separately.

Report with:

    python scripts/llm_extraction/alignment_report.py results/llm_extraction_cr/*/*.jsonl

It refuses to report on runs that predate the instrumentation rather than
restating their `alignment_fidelity = 1.0`, which is 1.0 by construction: the
saved spans are post-alignment, so any fidelity recomputed over them is trivially
perfect.

## Before a paid run

    python scripts/llm_extraction/verify_extraction_setup.py

Offline, free, and covers the defects that previously reached published results:
few-shot leakage, the alignment recovery path, the macro denominator, diagnostics
persistence, and LOMO fold deduplication.
