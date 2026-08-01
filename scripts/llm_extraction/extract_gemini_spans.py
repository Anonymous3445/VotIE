#!/usr/bin/env python3
"""
Extract spans using Gemini API via langextract.

Uses the same langextract pipeline, prompts, and few-shot examples as the
AMALIA extractor, but wired to Gemini via langextract's GeminiLanguageModel.
This enables a fair comparison of langextract outputs across providers.

Usage:
    # Quick test on 5 examples
    python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot --limit 5

    # Full zero-shot run
    python scripts/llm_extraction/extract_gemini_spans.py --strategy zero_shot

    # Full few-shot run (4 fixed examples, default)
    python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot

    # Evaluate results (same script as other extractors)
    python scripts/llm_extraction/evaluate_spans.py results/llm_extraction/gemini/few_shot.jsonl --relaxed
"""

import argparse
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.llm_extraction.gemini.extractor import GeminiSpanExtractor
from src.llm_extraction.gemini.config import GeminiConfig
from src.llm_extraction.shared.data_utils import load_jsonl, result_to_record, save_jsonl


# Configure logging before absl (used by langextract) can hijack it
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_handler)

for _name in ['src.llm_extraction.gemini.extractor', 'src.llm_extraction.span_alignment',
              'src.llm_extraction.shared.data_utils']:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    _log.addHandler(_handler)


def main():
    parser = argparse.ArgumentParser(description="Extract spans using Gemini via langextract")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Extraction strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model ID",
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
        help="Number of fixed few-shot examples (max 4, covers 11/12 entity types)",
    )
    parser.add_argument(
        "--max-char-buffer",
        type=int,
        default=20000,
        help="Max characters per chunk for langextract segmentation (default: 20000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    # Auto-generate output file name
    if args.output_file is None:
        args.output_file = f"results/llm_extraction/gemini/{args.strategy}.jsonl"

    logger.info(f"Starting langextract Gemini extraction: strategy={args.strategy}, model={args.model}")

    # Create config
    config = GeminiConfig(
        model_id=args.model,
        temperature=args.temperature,
        max_char_buffer=args.max_char_buffer,
    )

    # Create extractor
    extractor = GeminiSpanExtractor(
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

    # Resume from existing output if available
    output_path = Path(args.output_file)
    results = []
    processed_ids = set()
    if output_path.exists():
        results = load_jsonl(args.output_file)
        successful = [r for r in results if r.get("error") is None]
        errored = [r for r in results if r.get("error") is not None]
        # Only skip examples that completed successfully; re-run errored ones
        processed_ids = {r["id"] for r in successful}
        # Drop errored entries so they are rewritten cleanly on success
        results = successful
        logger.info(
            f"Resuming from checkpoint: {len(successful)} successful, "
            f"{len(errored)} errored (will retry)"
        )
        stale = sum(1 for r in successful if not r.get("diagnostics"))
        if stale:
            logger.warning(
                f"{stale}/{len(successful)} resumed records carry no diagnostics "
                f"(pre-instrumentation run). They will NOT be re-run, so span-discard "
                f"and cost figures would cover only part of the set. Write to a fresh "
                f"--output-file if you need a fully instrumented run."
            )

    pending = [ex for ex in test_data if ex["id"] not in processed_ids]
    logger.info(f"Remaining examples to process: {len(pending)}")

    # Extract spans
    for i, example in enumerate(pending, 1):
        logger.info(f"Processing {i}/{len(pending)}: {example['id']}")

        result = extractor.extract(
            text=example["text"],
            document_id=example["id"],
        )

        # Convert to dict for JSON serialization (carries the diagnostics block)
        result_dict = result_to_record(result)

        results.append(result_dict)

        n_entities = len(result.entities)
        status = "OK" if result.error is None else f"ERROR: {result.error}"
        logger.info(f"  -> {n_entities} entities extracted [{status}]")

        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(pending)} examples")

        if i % 50 == 0:
            save_jsonl(results, args.output_file)
            logger.info(f"Checkpoint saved ({len(results)} total) -> {args.output_file}")

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
