#!/usr/bin/env python3
"""
Extract spans using AMALIA API via langextract.

Uses langextract for prompt construction, text segmentation, and response
parsing. AMALIA's 32k-token context allows a generous max_char_buffer (default
20000), so most segments are processed in a single chunk without splitting.
Four fixed few-shot examples covering 11/12 entity types are included by default.

Usage:
    # Quick test on 5 examples
    python scripts/llm_extraction/extract_amalia_spans.py --strategy few_shot --limit 5

    # Full zero-shot run
    python scripts/llm_extraction/extract_amalia_spans.py --strategy zero_shot

    # Full few-shot run (4 fixed examples, default)
    python scripts/llm_extraction/extract_amalia_spans.py --strategy few_shot

    # Custom chunk size (reduce if hitting OOM or rate limits)
    python scripts/llm_extraction/extract_amalia_spans.py --strategy few_shot --max-char-buffer 10000

    # Evaluate results
    python scripts/llm_extraction/evaluate_spans.py results/llm_extraction/amalia/zero_shot.jsonl --relaxed
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm_extraction.amalia.extractor import AmaliaSpanExtractor
from src.llm_extraction.amalia.config import AmaliaConfig
from src.llm_extraction.shared.data_utils import load_jsonl, save_jsonl


# Configure logging before absl (used by langextract) can hijack it
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_handler)

# Also ensure the extractor and alignment loggers are visible
for _name in ['src.llm_extraction.amalia.extractor', 'src.llm_extraction.amalia.vllm_openai',
              'src.llm_extraction.span_alignment', 'src.llm_extraction.shared.data_utils']:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    _log.addHandler(_handler)


def main():
    parser = argparse.ArgumentParser(description="Extract spans using AMALIA via langextract")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Extraction strategy",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://amalia.inesctec.pt:8000",
        help="AMALIA API base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="amalia-base",
        help="Model ID (auto-detected if not set)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/citilink-votie/test.jsonl",
        help="Path to test file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output file (auto-generated if not set)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of fixed few-shot examples (max 5, covers 11 entity types + empty-extraction case)",
    )
    parser.add_argument(
        "--max-char-buffer",
        type=int,
        default=20000,
        help="Max characters per chunk for langextract segmentation (default: 20000 for AMALIA 32k context)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Max completion tokens per chunk (default: 8192; input uses ~3200 tokens leaving ~29500 headroom in AMALIA's 32k context)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120; increase for long documents)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts on transient errors (default: 3; use 1 to disable retries)",
    )

    args = parser.parse_args()

    # Auto-generate output file name
    if args.output_file is None:
        args.output_file = f"results/llm_extraction/amalia/{args.strategy}.jsonl"

    logger.info(f"Starting AMALIA extraction: strategy={args.strategy}")

    # Create config
    config = AmaliaConfig(
        base_url=args.base_url,
        model_id=args.model,
        temperature=args.temperature,
        max_char_buffer=args.max_char_buffer,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    # Create extractor
    extractor = AmaliaSpanExtractor(
        config=config,
        strategy=args.strategy,
        num_examples=args.num_examples,
    )

    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    test_data = load_jsonl(args.test_file)

    if args.limit:
        test_data = test_data[:args.limit]
        logger.info(f"Limited to {args.limit} examples")

    # Resume: load existing results and skip successfully processed examples
    prior_results: dict = {}
    ordered_ids = [ex["id"] for ex in test_data]
    if Path(args.output_file).exists():
        existing = load_jsonl(args.output_file)
        prior_results = {r["id"]: r for r in existing if r.get("error") is None}
        skip_ids = set(prior_results.keys())
        test_data = [ex for ex in test_data if ex["id"] not in skip_ids]
        n_errored = len(existing) - len(skip_ids)
        logger.info(
            f"Resuming: {len(skip_ids)} already successful, "
            f"{n_errored} errored (will retry), {len(test_data)} pending"
        )

    # Extract spans
    results = []
    for i, example in enumerate(test_data, 1):
        logger.info(f"Processing {i}/{len(test_data)}: {example['id']}")

        result = extractor.extract(
            text=example["text"],
            document_id=example["id"],
        )

        # Convert to dict for JSON serialization
        result_dict = {
            "id": result.id,
            "text": result.text,
            "entities": [
                {
                    "text": e.text,
                    "type": e.type,
                    "start": e.start,
                    "end": e.end,
                }
                for e in result.entities
            ],
            "model": result.model,
            "strategy": result.strategy,
            "processing_time": result.processing_time,
            "api_time": result.api_time,
            "error": result.error,
        }

        results.append(result_dict)

        # Log progress
        n_entities = len(result.entities)
        status = "OK" if result.error is None else f"ERROR: {result.error}"
        logger.info(f"  -> {n_entities} entities extracted [{status}]")

        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(test_data)} examples")
            # Checkpoint: merge with prior results and flush to disk so --resume
            # can pick up from here if the run is interrupted.
            checkpoint = dict(prior_results)
            for r in results:
                checkpoint[r["id"]] = r
            checkpoint_list = [checkpoint[id_] for id_ in ordered_ids if id_ in checkpoint]
            save_jsonl(checkpoint_list, args.output_file)
            logger.info(f"  Checkpoint saved ({len(checkpoint_list)} examples so far)")

    # Merge with prior successful results and restore original order
    if prior_results:
        for r in results:
            prior_results[r["id"]] = r
        results = [prior_results[id_] for id_ in ordered_ids if id_ in prior_results]

    # Save results
    logger.info(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)

    # Summary
    successful = sum(1 for r in results if r["error"] is None)
    total_entities = sum(len(r["entities"]) for r in results)
    avg_time = (
        sum(r["processing_time"] for r in results) / len(results) if results else 0
    )

    logger.info(f"\nExtraction complete!")
    logger.info(f"  Total examples: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {len(results) - successful}")
    logger.info(f"  Total entities extracted: {total_entities}")
    logger.info(f"  Avg processing time: {avg_time:.2f}s")
    logger.info(f"  Output: {args.output_file}")


if __name__ == "__main__":
    main()
