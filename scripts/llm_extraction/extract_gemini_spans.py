#!/usr/bin/env python3
"""
Extract spans using Gemini API.

Usage:
    python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot --limit 10
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Suppress gRPC warnings from Google API
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm_extraction.gemini.extractor import GeminiSpanExtractor
from src.llm_extraction.gemini.config import GeminiConfig
from src.llm_extraction.few_shot_selector import load_and_format_few_shot_examples
from src.llm_extraction.shared.data_utils import load_jsonl, save_jsonl


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract spans using Gemini")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Extraction strategy"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model ID"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/citilink_spans/test.jsonl",
        help="Path to test file"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/citilink_spans/train.jsonl",
        help="Path to train file (for few-shot examples)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/llm_extraction/gemini_predictions.jsonl",
        help="Path to output file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of few-shot examples"
    )

    args = parser.parse_args()

    logger.info(f"Starting Gemini extraction with strategy: {args.strategy}")

    # Create config
    config = GeminiConfig(model_id=args.model)

    # Create extractor
    extractor = GeminiSpanExtractor(config=config, strategy=args.strategy)

    # Load few-shot examples if needed
    if args.strategy == "few_shot":
        logger.info(f"Loading {args.num_examples} few-shot examples from {args.train_file}")
        examples = load_and_format_few_shot_examples(args.train_file, num_examples=args.num_examples)
        extractor.load_few_shot_examples(examples)

    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    test_data = load_jsonl(args.test_file)

    if args.limit:
        test_data = test_data[:args.limit]
        logger.info(f"Limited to {args.limit} examples")

    # Extract spans
    results = []
    for i, example in enumerate(test_data, 1):
        logger.info(f"Processing {i}/{len(test_data)}: {example['id']}")

        result = extractor.extract(
            text=example['text'],
            document_id=example['id']
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
                    "end": e.end
                }
                for e in result.entities
            ],
            "model": result.model,
            "strategy": result.strategy,
            "processing_time": result.processing_time,
            "api_time": result.api_time,
            "error": result.error
        }

        results.append(result_dict)

        # Log progress
        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(test_data)} examples")

    # Save results
    logger.info(f"Saving results to {args.output_file}")
    save_jsonl(results, args.output_file)

    # Summary
    successful = sum(1 for r in results if r['error'] is None)
    logger.info(f"\nExtraction complete!")
    logger.info(f"  Total examples: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {len(results) - successful}")
    logger.info(f"  Output: {args.output_file}")


if __name__ == "__main__":
    main()
