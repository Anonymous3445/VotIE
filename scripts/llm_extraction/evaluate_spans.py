#!/usr/bin/env python3
"""
Evaluate span extraction predictions.

Usage:
    python scripts/llm_extraction/evaluate_spans.py results/llm_extraction/gemini_predictions.jsonl

Output:
    Saves to results/llm_extraction/{model_name}_evaluation.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm_extraction.shared.evaluation import evaluate_span_extraction, print_evaluation_results
from src.llm_extraction.shared.data_utils import load_jsonl
from src.llm_extraction.schemas import SpanEntity, SpanExtractionResult, ENTITY_TYPES


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_predictions(file_path: str) -> list[SpanExtractionResult]:
    """Load predictions from JSONL file."""
    data = load_jsonl(file_path)

    results = []
    for item in data:
        entities = []
        for e in item.get('entities', []):
            start = e.get('start')
            end = e.get('end')
            etype = e.get('type', '')

            # Skip entities that failed alignment (no position) or have unknown types
            if start is None or end is None:
                logger.debug(f"Skipping unaligned entity '{e.get('text', '')[:40]}' in {item['id']}")
                continue
            if etype not in ENTITY_TYPES:
                logger.warning(f"Unknown entity type '{etype}' in {item['id']} — skipping")
                continue

            entities.append(SpanEntity(
                text=e['text'],
                type=etype,
                start=start,
                end=end,
            ))

        result = SpanExtractionResult(
            id=item['id'],
            text=item['text'],
            entities=entities,
            model=item.get('model', 'unknown'),
            strategy=item.get('strategy', 'unknown'),
            processing_time=item.get('processing_time', 0.0),
            api_time=item.get('api_time'),
            error=item.get('error'),
        )

        results.append(result)

    return results


def transform_to_deberta_format(results: dict, predictions_file: str, num_examples: int) -> dict:
    """
    Transform evaluation results to match DeBERTa evaluation format.

    Args:
        results: Results from evaluate_span_extraction()
        predictions_file: Path to predictions file
        num_examples: Number of examples evaluated

    Returns:
        Results in DeBERTa-compatible format
    """
    entity_level = results["entity_level"]
    per_type = results["per_type_metrics"]

    # Transform per-type metrics to match DeBERTa format
    per_type_formatted = {}
    for entity_type, metrics in per_type.items():
        per_type_formatted[entity_type] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "support": metrics["support"]
        }

    formatted_results = {
        "metadata": {
            "predictions_file": predictions_file,
            "timestamp": datetime.now().isoformat(),
            "num_examples": num_examples,
            "include_events": False  # LLM extraction doesn't use event construction
        },
        "entity_level_metrics": {
            "entity_precision": entity_level["precision"],
            "entity_recall": entity_level["recall"],
            "entity_f1": entity_level["f1"],
            "per_type_metrics": per_type_formatted,
            "evaluation_method": "seqeval_entity_level"
        }
    }

    # Add alignment fidelity if present
    if "alignment_fidelity" in results:
        formatted_results["alignment_fidelity"] = results["alignment_fidelity"]

    # Add relaxed boundary metrics if present
    if "relaxed_boundary_metrics" in results:
        relaxed = results["relaxed_boundary_metrics"]
        formatted_results["relaxed_boundary_metrics"] = {
            "matching_criterion": "type_match_and_overlap",
            "entity_precision": relaxed["precision"],
            "entity_recall": relaxed["recall"],
            "entity_f1": relaxed["f1"],
            "delta_f1": relaxed["f1"] - entity_level["f1"],
            "per_type_metrics": {
                entity_type: {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1"],
                    "support": metrics["support"]
                }
                for entity_type, metrics in relaxed["per_type_metrics"].items()
            }
        }

    return formatted_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate span extraction predictions")
    parser.add_argument(
        "predictions_file",
        type=str,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/citilink-votie/test.jsonl",
        help="Path to ground truth file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save evaluation results (default: auto-generated in results/llm_extraction/)"
    )
    parser.add_argument(
        "--no-fidelity",
        action="store_true",
        help="Skip alignment fidelity computation"
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Compute relaxed boundary matching metrics (type match + any overlap)"
    )

    args = parser.parse_args()

    logger.info(f"Loading predictions from {args.predictions_file}")
    predictions = load_predictions(args.predictions_file)

    logger.info(f"Loading ground truth from {args.ground_truth}")
    ground_truth = load_jsonl(args.ground_truth)

    logger.info(f"Evaluating {len(predictions)} predictions")

    results = evaluate_span_extraction(
        predictions=predictions,
        ground_truth=ground_truth,
        compute_fidelity=not args.no_fidelity,
        relaxed_matching=args.relaxed
    )

    # Print results
    print_evaluation_results(results)

    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Auto-generate output filename based on predictions file
        predictions_path = Path(args.predictions_file)
        # Extract model name from filename (e.g., "gemini_zero_shot_predictions.jsonl" -> "gemini_zero_shot")
        predictions_stem = predictions_path.stem.replace("_predictions", "")
        output_filename = f"{predictions_stem}_evaluation.json"
        output_path = Path("results/llm_extraction") / output_filename

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Transform results to DeBERTa-compatible format
    formatted_results = transform_to_deberta_format(
        results=results,
        predictions_file=str(args.predictions_file),
        num_examples=len(predictions)
    )

    # Convert numpy types to Python native types for JSON serialization
    serializable_results = convert_to_json_serializable(formatted_results)

    # Save results
    logger.info(f"Saving evaluation results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
