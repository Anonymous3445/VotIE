#!/usr/bin/env python3
"""
Run municipality leave-one-out cross-validation experiments.

This script:
1. Trains 6 models (one for each municipality as test set)
2. Generates predictions on each test set
3. Evaluates predictions
4. Aggregates results across all municipalities

Usage:
    python scripts/run_municipality_experiments.py
    python scripts/run_municipality_experiments.py --municipalities M01 M02  # Run specific municipalities
    python scripts/run_municipality_experiments.py --skip-training  # Skip training, only predict/evaluate
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Municipalities
ALL_MUNICIPALITIES = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command to run as list of strings
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}")
        return False


def train_model(municipality: str, experiment_name: str) -> bool:
    """Train model for a specific municipality."""
    config_file = f"configs/municipality_experiments/deberta_crf_{municipality}.yaml"

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", config_file,
        "--experiment-name", experiment_name
    ]

    return run_command(cmd, f"Training model for {municipality}")


def generate_predictions(municipality: str, experiment_name: str, device: str = "cuda") -> bool:
    """Generate predictions for a specific municipality."""
    model_path = f"models/deberta_crf/{experiment_name}"
    test_data_path = f"data/citilink_spans_municipality_splits/{municipality}/test.jsonl"
    output_path = f"municipality_experiments/predictions/{municipality}_predictions.jsonl"

    cmd = [
        sys.executable,
        "scripts/predict.py",
        model_path,
        test_data_path,
        output_path,
        device
    ]

    return run_command(cmd, f"Generating predictions for {municipality}")


def evaluate_predictions(municipality: str) -> bool:
    """Evaluate predictions for a specific municipality."""
    predictions_path = f"municipality_experiments/predictions/{municipality}_predictions.jsonl"
    output_path = f"municipality_experiments/evaluation/{municipality}_results.json"

    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        predictions_path,
        output_path
    ]

    return run_command(cmd, f"Evaluating predictions for {municipality}")


def load_evaluation_results(municipality: str) -> Dict[str, Any]:
    """Load evaluation results for a municipality."""
    results_file = Path(f"municipality_experiments/evaluation/{municipality}_results.json")

    if not results_file.exists():
        print(f"Warning: Results file not found for {municipality}: {results_file}")
        return None

    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_results(municipalities: List[str]) -> Dict[str, Any]:
    """
    Aggregate results across all municipalities.

    Computes:
    - Mean and std of entity-level metrics
    - Per-municipality breakdown
    """
    all_results = {}

    for mun in municipalities:
        result = load_evaluation_results(mun)
        if result is not None:
            all_results[mun] = result

    if not all_results:
        print("Error: No results found to aggregate")
        return None

    # Extract metrics for aggregation
    entity_f1_scores = []
    entity_precision_scores = []
    entity_recall_scores = []

    for mun, result in all_results.items():
        entity_metrics = result.get('entity_level_metrics', {})
        entity_f1_scores.append(entity_metrics.get('entity_f1', 0.0))
        entity_precision_scores.append(entity_metrics.get('entity_precision', 0.0))
        entity_recall_scores.append(entity_metrics.get('entity_recall', 0.0))

    # Compute mean and std
    import statistics

    def mean_std(values):
        if not values:
            return 0.0, 0.0
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    entity_f1_mean, entity_f1_std = mean_std(entity_f1_scores)
    entity_precision_mean, entity_precision_std = mean_std(entity_precision_scores)
    entity_recall_mean, entity_recall_std = mean_std(entity_recall_scores)

    aggregated = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_municipalities': len(all_results),
            'municipalities': list(all_results.keys())
        },
        'entity_level_aggregated': {
            'entity_f1_mean': entity_f1_mean,
            'entity_f1_std': entity_f1_std,
            'entity_precision_mean': entity_precision_mean,
            'entity_precision_std': entity_precision_std,
            'entity_recall_mean': entity_recall_mean,
            'entity_recall_std': entity_recall_std,
            'per_municipality': {
                mun: {
                    'entity_f1': all_results[mun]['entity_level_metrics'].get('entity_f1', 0.0),
                    'entity_precision': all_results[mun]['entity_level_metrics'].get('entity_precision', 0.0),
                    'entity_recall': all_results[mun]['entity_level_metrics'].get('entity_recall', 0.0),
                }
                for mun in all_results.keys()
            }
        }
    }

    return aggregated


def print_summary(aggregated: Dict[str, Any]):
    """Print aggregated results summary."""
    print("\n" + "=" * 80)
    print("MUNICIPALITY LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
    print("=" * 80)

    metadata = aggregated['metadata']
    print(f"\nMunicipalities: {', '.join(metadata['municipalities'])}")
    print(f"Total: {metadata['num_municipalities']} experiments")

    print("\n" + "-" * 80)
    print("ENTITY-LEVEL METRICS (Mean ± Std)")
    print("-" * 80)

    entity_agg = aggregated['entity_level_aggregated']
    print(f"Entity F1:        {entity_agg['entity_f1_mean']:.4f} ± {entity_agg['entity_f1_std']:.4f}")
    print(f"Entity Precision: {entity_agg['entity_precision_mean']:.4f} ± {entity_agg['entity_precision_std']:.4f}")
    print(f"Entity Recall:    {entity_agg['entity_recall_mean']:.4f} ± {entity_agg['entity_recall_std']:.4f}")

    print("\nPer-Municipality Entity F1:")
    for mun, metrics in entity_agg['per_municipality'].items():
        print(f"  {mun}: {metrics['entity_f1']:.4f}")

    print("\n" + "=" * 80)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Run municipality leave-one-out cross-validation experiments"
    )
    parser.add_argument(
        '--municipalities',
        nargs='+',
        default=ALL_MUNICIPALITIES,
        choices=ALL_MUNICIPALITIES,
        help='Municipalities to process (default: all)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only run prediction/evaluation'
    )
    parser.add_argument(
        '--skip-prediction',
        action='store_true',
        help='Skip prediction and only run evaluation'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation and only run training/prediction'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['cuda', 'mps', 'cpu', 'auto'],
        help='Device for training/inference (default: auto - detects best available)'
    )

    args = parser.parse_args()

    municipalities = args.municipalities

    print("=" * 80)
    print("Municipality Leave-One-Out Cross-Validation Experiments")
    print("=" * 80)
    print(f"\nMunicipalities: {', '.join(municipalities)}")
    print(f"Skip Training: {args.skip_training}")
    print(f"Skip Prediction: {args.skip_prediction}")
    print(f"Skip Evaluation: {args.skip_evaluation}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Create output directories
    Path("municipality_experiments/predictions").mkdir(parents=True, exist_ok=True)
    Path("municipality_experiments/evaluation").mkdir(parents=True, exist_ok=True)

    # Process each municipality
    success_count = 0
    failed_municipalities = []

    for i, mun in enumerate(municipalities, 1):
        print(f"\n{'#' * 80}")
        print(f"# Processing Municipality {mun} ({i}/{len(municipalities)})")
        print(f"{'#' * 80}")

        experiment_name = f"municipality_{mun}"
        success = True

        # Step 1: Training
        if not args.skip_training:
            if not train_model(mun, experiment_name):
                print(f"✗ Training failed for {mun}")
                success = False
                failed_municipalities.append(mun)
                continue

        # Step 2: Prediction
        if not args.skip_prediction:
            if not generate_predictions(mun, experiment_name, args.device):
                print(f"✗ Prediction failed for {mun}")
                success = False
                failed_municipalities.append(mun)
                continue

        # Step 3: Evaluation
        if not args.skip_evaluation:
            if not evaluate_predictions(mun):
                print(f"✗ Evaluation failed for {mun}")
                success = False
                failed_municipalities.append(mun)
                continue

        if success:
            success_count += 1
            print(f"\n✓ Completed all steps for {mun}")

    # Aggregate results
    if not args.skip_evaluation:
        print("\n" + "=" * 80)
        print("Aggregating Results")
        print("=" * 80)

        successful_municipalities = [m for m in municipalities if m not in failed_municipalities]

        if successful_municipalities:
            aggregated = aggregate_results(successful_municipalities)

            if aggregated:
                # Save aggregated results
                output_file = Path("municipality_experiments/aggregated_results.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(aggregated, f, indent=2, ensure_ascii=False)

                print(f"\n✓ Aggregated results saved to {output_file}")

                # Print summary
                print_summary(aggregated)
        else:
            print("\n✗ No successful experiments to aggregate")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total municipalities processed: {len(municipalities)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_municipalities)}")

    if failed_municipalities:
        print(f"Failed municipalities: {', '.join(failed_municipalities)}")

    print("=" * 80)


if __name__ == '__main__':
    main()
