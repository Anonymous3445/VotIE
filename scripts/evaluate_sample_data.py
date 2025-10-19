#!/usr/bin/env python3
"""
Evaluate HuggingFace Model on Sample Data

This script loads the pre-trained DeBERTa-CRF-VotIE model from HuggingFace
and evaluates it on the sample examples in data/data_examples.json.

Perfect for reviewers to quickly test the model without needing the full dataset!

Usage:
    # Evaluate all 30 examples
    python scripts/evaluate_sample_data.py

    # Evaluate only first 5 examples (faster for testing)
    python scripts/evaluate_sample_data.py --limit 5

    # Custom output directory
    python scripts/evaluate_sample_data.py --output results/sample_evaluation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(data_path: str = "data/data_examples.json") -> List[Dict[str, Any]]:
    """Load sample examples from JSON file."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Sample data file not found: {data_path}")

    logger.info(f"Loading sample data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = data.get('examples', [])
    logger.info(f"✓ Loaded {len(examples)} examples")

    return examples


def load_hf_model():
    """Load the pre-trained model from HuggingFace."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "transformers library not found. Install it with: pip install transformers"
        )

    model_name = "Anonymous3445/DeBERTa-CRF-VotIE"
    logger.info(f"Loading model from HuggingFace: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    logger.info("✓ Model loaded successfully")

    return tokenizer, model


def align_predictions_to_tokens(predictions: List[Dict], tokens: List[str], text: str) -> List[str]:
    """
    Align word-level predictions from HF model to the original tokenization.

    The HF model returns predictions at word boundaries, but we need them
    aligned to the pre-tokenized tokens in the dataset.
    """
    # Create a simple alignment based on the text
    # This is a heuristic approach - the HF model's tokenization may differ slightly

    # Build a list of BIO labels matching the original tokens
    aligned_labels = ['O'] * len(tokens)

    # Create a mapping of text positions to token indices
    token_positions = []
    current_pos = 0
    for i, token in enumerate(tokens):
        # Find token in text (case-insensitive search from current position)
        token_start = text.lower().find(token.lower(), current_pos)
        if token_start >= 0:
            token_positions.append({
                'idx': i,
                'start': token_start,
                'end': token_start + len(token),
                'token': token
            })
            current_pos = token_start + len(token)

    # Map predictions to tokens based on position overlap
    for pred in predictions:
        if pred['label'] == 'O':
            continue

        pred_start = pred.get('start', -1)
        pred_end = pred.get('end', -1)

        if pred_start == -1:
            # No position info, try to match by word
            for i, token_info in enumerate(token_positions):
                if pred['word'].lower() in token_info['token'].lower():
                    aligned_labels[token_info['idx']] = pred['label']
                    break
        else:
            # Use position overlap
            for token_info in token_positions:
                # Check if there's overlap
                overlap_start = max(pred_start, token_info['start'])
                overlap_end = min(pred_end, token_info['end'])

                if overlap_start < overlap_end:
                    aligned_labels[token_info['idx']] = pred['label']

    return aligned_labels


def run_predictions(examples: List[Dict], tokenizer, model) -> List[Dict[str, Any]]:
    """Run predictions on all examples."""
    logger.info("Running predictions on examples...")

    predictions = []

    for i, example in enumerate(examples, 1):
        try:
            # Get example data
            example_id = example['id']
            text = example['text']
            tokens = example['tokens']
            gold_labels = example['labels']

            # Run model prediction
            inputs = tokenizer(text, return_tensors="pt")
            pred_words = model.decode(**inputs, tokenizer=tokenizer, text=text, return_offsets=True)

            # Align predictions to original tokens
            pred_labels = align_predictions_to_tokens(pred_words, tokens, text)

            # Create prediction entry
            predictions.append({
                'id': example_id,
                'tokens': tokens,
                'pred_labels': pred_labels,
                'gold_labels': gold_labels,
                'text': text,
                'municipality': example.get('municipality', 'unknown')
            })

            if i % 5 == 0:
                logger.info(f"  Processed {i}/{len(examples)} examples")

        except Exception as e:
            logger.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
            continue

    logger.info(f"✓ Generated predictions for {len(predictions)}/{len(examples)} examples")

    return predictions


def save_predictions(predictions: List[Dict], output_path: str):
    """Save predictions in JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving predictions to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    logger.info(f"✓ Predictions saved to {output_path}")


def print_sample_predictions(predictions: List[Dict], num_samples: int = 2):
    """Print a few sample predictions for inspection."""
    logger.info("\n" + "="*80)
    logger.info("Sample Predictions (first 2 examples):")
    logger.info("="*80)

    for pred in predictions[:num_samples]:
        print(f"\nExample: {pred['id']}")
        print(f"Text: {pred['text'][:100]}...")
        print(f"\n{'Token':<25} {'Predicted':<20} {'Gold':<20}")
        print("-" * 65)

        for token, pred_label, gold_label in zip(
            pred['tokens'][:10],  # Show first 10 tokens
            pred['pred_labels'][:10],
            pred['gold_labels'][:10]
        ):
            match = "✓" if pred_label == gold_label else "✗"
            print(f"{token:<25} {pred_label:<20} {gold_label:<20} {match}")

        if len(pred['tokens']) > 10:
            print(f"... ({len(pred['tokens']) - 10} more tokens)")

        print()


def main(output_dir: str = "results", limit: int = None):
    """Main execution function."""
    logger.info("="*80)
    logger.info("VotIE Sample Data Evaluation")
    logger.info("="*80)
    logger.info("")
    logger.info("This script evaluates the HuggingFace model on sample data.")
    logger.info("Perfect for reviewers to test the model without the full dataset!")
    logger.info("")

    try:
        # Step 1: Load sample data
        examples = load_sample_data()

        # Limit number of examples if specified
        if limit is not None and limit > 0:
            original_count = len(examples)
            examples = examples[:limit]
            logger.info(f"Limiting to {len(examples)} examples (out of {original_count})")

        # Step 2: Load HuggingFace model
        tokenizer, model = load_hf_model()

        # Step 3: Run predictions
        predictions = run_predictions(examples, tokenizer, model)

        # Step 4: Save predictions
        predictions_path = Path(output_dir) / "sample_predictions.jsonl"
        save_predictions(predictions, predictions_path)

        # Step 5: Print some samples
        print_sample_predictions(predictions)

        # Step 6: Run evaluation
        logger.info("="*80)
        logger.info("Running Evaluation...")
        logger.info("="*80)

        from scripts.evaluate import evaluate

        eval_output_path = Path(output_dir) / "sample_evaluation.json"
        results = evaluate(
            predictions_path=str(predictions_path),
            output_path=str(eval_output_path),
            include_events=True
        )

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS - Model Performance on Sample Data")
        logger.info("="*80)

        entity_metrics = results['entity_level_metrics']
        logger.info("\nEntity-Level Metrics:")
        logger.info(f"  F1 Score:  {entity_metrics.get('entity_f1', 0.0):.2%}")
        logger.info(f"  Precision: {entity_metrics.get('entity_precision', 0.0):.2%}")
        logger.info(f"  Recall:    {entity_metrics.get('entity_recall', 0.0):.2%}")

        event_metrics = results['event_level_metrics']
        if event_metrics and 'error' not in event_metrics:
            logger.info("\nEvent-Level Metrics:")
            logger.info(f"  Event Detection F1: {event_metrics.get('event_detection_f1', 0.0):.2%}")
            logger.info(f"  Subject F1:         {event_metrics.get('subject_f1', 0.0):.2%}")
            logger.info(f"  Counting F1:        {event_metrics.get('counting_f1', 0.0):.2%}")
            logger.info(f"  Participant F1:     {event_metrics.get('participant_vote_f1', 0.0):.2%}")
            logger.info(f"  Complete Event F1:  {event_metrics.get('complete_event_f1', 0.0):.2%}")

        logger.info("\n" + "="*80)
        logger.info(f"✓ Evaluation complete! Results saved to: {eval_output_path}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error("Stack trace:", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate HuggingFace model on sample data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for predictions and evaluation results (default: results/)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit evaluation to first N examples (default: all 30 examples)'
    )

    args = parser.parse_args()

    main(output_dir=args.output, limit=args.limit)
