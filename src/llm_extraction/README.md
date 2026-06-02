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
├── fixed_examples.py        # 4-5 fixed few-shot examples shared by all extractors
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
├── amalia/                  # AMALIA 8B via langextract + vLLM OpenAI-compatible endpoint
│   ├── extractor.py         # AmaliaSpanExtractor
│   ├── vllm_openai.py       # Async vLLM client
│   ├── prompts.py           # Shared prompt description (used by all three extractors)
│   └── config.py            # AmaliaConfig
│
└── shared/
    ├── base_extractor.py    # BaseSpanExtractor abstract class
    ├── data_utils.py        # load_jsonl / save_jsonl helpers
    └── evaluation.py        # Span-level evaluation utilities
```

## Key Design Decisions

- **Fixed few-shot examples**: `fixed_examples.py` provides the same 4–5 examples to all extractors, covering 11 of 12 entity types plus an empty-extraction case. Not dynamically selected.
- **Shared prompt**: `amalia/prompts.py:get_prompt_description()` is used by all three extractors to ensure identical system instructions.
- **Schema constraints disabled** for GPT and AMALIA (use `fence_output=True` or `use_schema_constraints=False`) because these providers don't support structured output natively via langextract.
- **Span alignment** is applied to all extractor outputs to recover exact character offsets, since LLMs generate text spans without positions.
