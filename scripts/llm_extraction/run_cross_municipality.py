#!/usr/bin/env python3
"""
Run complete LOMO (Leave-One-Municipality-Out) cross-validation for Gemini.

This script runs experiments for all 6 municipalities (M01-M06):
- For each municipality: train on other 5, test on held-out municipality
- Evaluates generalization performance
- Generates comparative results table

Usage:
    python scripts/llm_extraction/run_all_cross_municipality.py [--municipalities M01 M02 ...]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm_extraction.shared.data_utils import load_jsonl
from src.llm_extraction.few_shot_selector import load_and_format_few_shot_examples
from src.llm_extraction.gemini.extractor import GeminiSpanExtractor
from src.llm_extraction.gemini.config import GeminiConfig
from src.llm_extraction.span_alignment import align_extraction_result
from src.llm_extraction.shared.evaluation import evaluate_span_extraction
from src.llm_extraction.schemas import SpanEntity, SpanExtractionResult


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_by_municipality(data: List[dict], municipality: str) -> List[dict]:
    """Filter data for specific municipality."""
    filtered = []
    for example in data:
        example_id = example.get('id', '')
        if example_id.startswith(municipality + '_'):
            filtered.append(example)
    return filtered


def run_single_experiment(
    test_municipality: str,
    train_municipalities: List[str],
    train_file: str,
    dev_file: str,
    test_file: str,
    output_dir: Path
) -> Dict:
    """
    Run single LOMO experiment: train on N-1 municipalities, test on 1.

    Strategy:
    - Few-shot: 1 example per training municipality (5 total)
    - Test: ALL examples from held-out municipality (train + dev + test splits)

    Args:
        test_municipality: Municipality to test on (e.g., 'M01')
        train_municipalities: Municipalities to select few-shot examples from
        train_file: Path to train.jsonl (for few-shot examples)
        dev_file: Path to dev.jsonl (for evaluation)
        test_file: Path to test.jsonl (for evaluation)
        output_dir: Directory to save results

    Returns:
        Dictionary with evaluation results
    """
    logger.info("="*80)
    logger.info(f"LOMO Experiment: Test on {test_municipality}")
    logger.info("="*80)
    logger.info(f"Train municipalities: {', '.join(train_municipalities)}")
    logger.info(f"Few-shot strategy: 1 example per training municipality ({len(train_municipalities)} total)")

    # Load ALL data from held-out municipality (train + dev + test)
    logger.info(f"Loading ALL examples from {test_municipality} (train + dev + test splits)...")

    all_train_data = load_jsonl(train_file)
    all_dev_data = load_jsonl(dev_file)
    all_test_data = load_jsonl(test_file)

    # Filter each split for held-out municipality
    train_examples = filter_by_municipality(all_train_data, test_municipality)
    dev_examples = filter_by_municipality(all_dev_data, test_municipality)
    test_examples = filter_by_municipality(all_test_data, test_municipality)

    # Combine all splits for full LOMO evaluation
    test_data = train_examples + dev_examples + test_examples

    logger.info(f"  Train split: {len(train_examples)} examples")
    logger.info(f"  Dev split:   {len(dev_examples)} examples")
    logger.info(f"  Test split:  {len(test_examples)} examples")
    logger.info(f"  TOTAL:       {len(test_data)} examples")

    if len(test_data) == 0:
        logger.error(f"No examples found for {test_municipality}")
        return None

    # Select few-shot examples from training municipalities (using train set only)
    # LOMO mode: exactly 1 example per training municipality
    logger.info(f"Selecting few-shot examples from training municipalities (1 per municipality)...")
    few_shot_examples = load_and_format_few_shot_examples(
        train_file,
        save_to_file=False,
        lomo_mode=True,
        train_municipalities=train_municipalities
    )

    logger.info(f"Selected {len(few_shot_examples)} examples:")
    for ex in few_shot_examples:
        municipality = ex['id'].split('_')[0]
        logger.info(f"  - {ex['id'][:35]}... (from {municipality})")

    # Initialize Gemini extractor
    logger.info("\nInitializing Gemini extractor...")
    config = GeminiConfig(
        model_id="gemini-2.5-flash",  # Gemini 2.5 Flash
        retry_delay=120  # Wait 2 minutes between retries (increased from 60s)
    )
    extractor = GeminiSpanExtractor(config=config, strategy="few_shot")
    extractor.load_few_shot_examples(few_shot_examples)

    # Check for existing predictions (resume functionality)
    predictions_file = output_dir / f'gemini_few_{test_municipality}.jsonl'
    existing_predictions = {}

    if predictions_file.exists():
        logger.info(f"\nFound existing predictions at {predictions_file}")
        logger.info("Loading existing predictions to resume from checkpoint...")
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    pred = json.loads(line.strip())
                    existing_predictions[pred['id']] = pred
            logger.info(f"Loaded {len(existing_predictions)} existing predictions")
        except Exception as e:
            logger.warning(f"Error loading existing predictions: {e}")
            logger.warning("Starting fresh...")
            existing_predictions = {}

    # Run predictions
    logger.info(f"\nRunning predictions on {len(test_data)} examples...")
    predictions = list(existing_predictions.values())  # Start with existing predictions
    errors = sum(1 for p in predictions if p.get('error'))
    skipped = 0

    for i, example in enumerate(test_data, 1):
        # Skip if already processed
        if example['id'] in existing_predictions:
            skipped += 1
            continue

        if (i - skipped) % 10 == 0:
            logger.info(f"  Processing {i}/{len(test_data)}... (errors: {errors}, skipped: {skipped})")

        try:
            # Extract entities
            result = extractor.extract(example['text'], example['id'])

            # Align spans to character positions
            aligned_result, stats = align_extraction_result(
                result,
                strict=True,
                use_voting_context=True
            )

            # Convert to dict for saving
            pred_dict = {
                'id': aligned_result.id,
                'text': aligned_result.text,
                'entities': [
                    {
                        'text': e.text,
                        'type': e.type,
                        'start': e.start,
                        'end': e.end
                    }
                    for e in aligned_result.entities
                ],
                'model': aligned_result.model,
                'strategy': 'few_shot_lomo',
                'processing_time': aligned_result.processing_time,
                'test_municipality': test_municipality,
                'train_municipalities': train_municipalities
            }

            if aligned_result.error:
                pred_dict['error'] = aligned_result.error
                errors += 1

            predictions.append(pred_dict)

            # Save incrementally (append to file after each prediction)
            with open(predictions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')

            # Rate limiting (to avoid API quota issues)
            # 2 seconds = 30 RPM (safe for most quotas)
            time.sleep(2.0)

        except Exception as e:
            logger.error(f"Error processing {example['id']}: {e}")
            errors += 1
            error_dict = {
                'id': example['id'],
                'text': example['text'],
                'entities': [],
                'model': config.model_id,
                'strategy': 'few_shot_lomo',
                'error': str(e),
                'test_municipality': test_municipality,
                'train_municipalities': train_municipalities
            }
            predictions.append(error_dict)

            # Save error incrementally too
            with open(predictions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_dict, ensure_ascii=False) + '\n')

    logger.info(f"Completed predictions. Total: {len(test_data)}, New: {len(test_data) - skipped}, Skipped: {skipped}, Errors: {errors}")
    logger.info(f"All predictions saved to {predictions_file}")

    # Evaluate
    logger.info("\nEvaluating predictions...")
    pred_results = []
    for pred in predictions:
        entities = [
            SpanEntity(
                text=e['text'],
                type=e['type'],
                start=e['start'],
                end=e['end']
            )
            for e in pred.get('entities', [])
        ]
        pred_results.append(SpanExtractionResult(
            id=pred['id'],
            text=pred['text'],
            entities=entities,
            model=pred['model'],
            strategy=pred['strategy'],
            processing_time=pred.get('processing_time', 0.0),
            error=pred.get('error')
        ))

    evaluation_results = evaluate_span_extraction(
        predictions=pred_results,
        ground_truth=test_data,
        compute_fidelity=True,
        relaxed_threshold=0.0
    )

    # Add metadata
    evaluation_results['metadata'] = {
        'test_municipality': test_municipality,
        'train_municipalities': train_municipalities,
        'num_few_shot_examples': len(train_municipalities),  # 1 per municipality
        'num_test_examples': len(test_data),
        'num_errors': errors,
        'timestamp': datetime.now().isoformat()
    }

    # Save evaluation
    eval_file = output_dir / f'gemini_few_{test_municipality}_evaluation.json'
    logger.info(f"Saving evaluation to {eval_file}")

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    evaluation_results_serializable = convert_numpy_types(evaluation_results)

    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results_serializable, f, indent=2, ensure_ascii=False)

    # Print summary
    entity_metrics = evaluation_results['entity_level']
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS: {test_municipality}")
    logger.info(f"{'='*80}")
    logger.info(f"Precision: {entity_metrics['precision']:.4f}")
    logger.info(f"Recall:    {entity_metrics['recall']:.4f}")
    logger.info(f"F1:        {entity_metrics['f1']:.4f}")
    logger.info(f"{'='*80}\n")

    return evaluation_results


def generate_summary_table(results: Dict[str, Dict], output_dir: Path):
    """
    Generate comparative table of LOMO results across all municipalities.

    Args:
        results: Dictionary mapping municipality -> evaluation results
        output_dir: Directory to save summary
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING SUMMARY TABLE")
    logger.info("="*80)

    # Sort municipalities
    municipalities = sorted(results.keys())

    # Create markdown table
    lines = []
    lines.append("# Gemini Few-Shot LOMO Cross-Validation Results\n\n")
    lines.append("## Leave-One-Municipality-Out Generalization\n\n")
    lines.append("Each row shows results when testing on the held-out municipality ")
    lines.append("and training on the other 5 municipalities.\n\n")

    # Main results table
    lines.append("### Entity-Level Span Extraction Results\n\n")
    lines.append("| Test Municipality | Examples | Precision | Recall | F1 | TP | FN | FP |\n")
    lines.append("|-------------------|----------|-----------|--------|-----|----|----|----|\n")

    for muni in municipalities:
        if results[muni] is None:
            continue

        entity_metrics = results[muni]['entity_level']
        metadata = results[muni]['metadata']

        lines.append(
            f"| {muni} | {metadata['num_test_examples']} | "
            f"{entity_metrics['precision']:.3f} | "
            f"{entity_metrics['recall']:.3f} | "
            f"{entity_metrics['f1']:.3f} | "
            f"{entity_metrics['true_positives']} | "
            f"{entity_metrics['false_negatives']} | "
            f"{entity_metrics['false_positives']} |\n"
        )

    # Calculate average (only if there are successful results)
    successful_results = [r for r in results.values() if r]
    if successful_results:
        avg_p = sum(r['entity_level']['precision'] for r in successful_results) / len(successful_results)
        avg_r = sum(r['entity_level']['recall'] for r in successful_results) / len(successful_results)
        avg_f1 = sum(r['entity_level']['f1'] for r in successful_results) / len(successful_results)
        lines.append(f"| **Average** | - | **{avg_p:.3f}** | **{avg_r:.3f}** | **{avg_f1:.3f}** | - | - | - |\n")
    else:
        logger.warning("No successful results to calculate averages")
        lines.append(f"| **Average** | - | N/A | N/A | N/A | - | - | - |\n")

    # Per-entity type breakdown
    lines.append("\n### Per-Entity-Type F1 Scores\n\n")
    lines.append("| Entity Type | " + " | ".join(municipalities) + " | Avg |\n")
    lines.append("|-------------|" + "|".join(["-------"] * len(municipalities)) + "|-----|\n")

    # Get all entity types from first result
    entity_types = []
    for muni in municipalities:
        if results[muni] and 'per_type' in results[muni]['entity_level']:
            entity_types = list(results[muni]['entity_level']['per_type'].keys())
            break

    for etype in entity_types:
        scores = []
        for muni in municipalities:
            if results[muni] and etype in results[muni]['entity_level']['per_type']:
                f1 = results[muni]['entity_level']['per_type'][etype]['f1']
                scores.append(f1)
            else:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        score_strs = [f"{s:.3f}" for s in scores]
        lines.append(f"| {etype} | " + " | ".join(score_strs) + f" | **{avg_score:.3f}** |\n")

    # Fidelity metrics
    lines.append("\n### Alignment Fidelity\n\n")
    lines.append("| Municipality | Total Extracted | Aligned | Alignment Rate |\n")
    lines.append("|--------------|-----------------|---------|----------------|\n")

    for muni in municipalities:
        if results[muni] and 'fidelity' in results[muni]:
            fidelity = results[muni]['fidelity']
            total = fidelity['total_extracted']
            aligned = fidelity['successfully_aligned']
            rate = fidelity['alignment_rate']
            lines.append(f"| {muni} | {total} | {aligned} | {rate:.3f} |\n")

    # Experimental details
    lines.append("\n### Experimental Setup\n\n")
    lines.append("- **Model**: Gemini 2.0 Flash Experimental\n")
    lines.append("- **Strategy**: Few-shot (5 examples)\n")
    lines.append("- **Example Selection**: Diverse selection from training municipalities\n")
    lines.append("- **Evaluation**: Character-level span matching (exact match)\n")
    lines.append("- **Data Source**: `data/citilink_spans/train.jsonl`\n\n")

    # Save summary
    summary_file = output_dir / 'lomo_summary.md'
    logger.info(f"Saving summary to {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))

    # Also save as JSON for programmatic access
    summary_json = {
        'average_metrics': {
            'precision': avg_p,
            'recall': avg_r,
            'f1': avg_f1
        },
        'per_municipality': {
            muni: {
                'precision': results[muni]['entity_level']['precision'],
                'recall': results[muni]['entity_level']['recall'],
                'f1': results[muni]['entity_level']['f1'],
                'examples': results[muni]['metadata']['num_test_examples']
            }
            for muni in municipalities if results[muni]
        }
    }

    summary_json_file = output_dir / 'lomo_summary.json'
    with open(summary_json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved JSON summary to {summary_json_file}")

    # Print table to console
    logger.info("\n" + "="*80)
    logger.info("LOMO CROSS-VALIDATION SUMMARY")
    logger.info("="*80)
    for line in lines[4:]:  # Skip header
        if line.strip():
            logger.info(line.strip())
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete LOMO cross-validation for Gemini'
    )
    parser.add_argument(
        '--municipalities',
        nargs='+',
        default=['M01', 'M02', 'M03', 'M04', 'M05', 'M06'],
        help='Municipalities to test (default: all M01-M06)'
    )
    parser.add_argument(
        '--train-file',
        default='data/citilink_spans/train.jsonl',
        help='Training data file (for few-shot examples)'
    )
    parser.add_argument(
        '--dev-file',
        default='data/citilink_spans/dev.jsonl',
        help='Dev data file (for evaluation)'
    )
    parser.add_argument(
        '--test-file',
        default='data/citilink_spans/test.jsonl',
        help='Test data file (for evaluation)'
    )
    parser.add_argument(
        '--output-dir',
        default='results/llm_extraction/cross_municipality',
        help='Output directory'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip municipalities with existing evaluation files'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("GEMINI LOMO CROSS-VALIDATION")
    logger.info("="*80)
    logger.info(f"Municipalities: {', '.join(args.municipalities)}")
    logger.info(f"Train file (few-shot): {args.train_file}")
    logger.info(f"Dev file (evaluation): {args.dev_file}")
    logger.info(f"Test file (evaluation): {args.test_file}")
    logger.info(f"Evaluation strategy: ALL splits (train+dev+test) from held-out municipality")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Few-shot strategy: 1 example per training municipality")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("="*80 + "\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All possible municipalities
    all_municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']

    # Run experiments
    results = {}

    for test_muni in args.municipalities:
        # Check if already exists
        eval_file = output_dir / f'gemini_few_{test_muni}_evaluation.json'
        if args.skip_existing and eval_file.exists():
            logger.info(f"Skipping {test_muni} (evaluation file already exists)")
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    results[test_muni] = json.load(f)
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted JSON file for {test_muni}: {e}")
                logger.warning(f"Deleting corrupted file and re-running: {eval_file}")
                eval_file.unlink()
                # Continue to run the experiment

        # Get training municipalities (all except test)
        train_munis = [m for m in all_municipalities if m != test_muni]

        # Run experiment
        try:
            result = run_single_experiment(
                test_municipality=test_muni,
                train_municipalities=train_munis,
                train_file=args.train_file,
                dev_file=args.dev_file,
                test_file=args.test_file,
                output_dir=output_dir
            )
            results[test_muni] = result

        except Exception as e:
            logger.error(f"Failed to run experiment for {test_muni}: {e}")
            results[test_muni] = None

    # Generate summary table
    generate_summary_table(results, output_dir)

    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {output_dir / 'lomo_summary.md'}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
