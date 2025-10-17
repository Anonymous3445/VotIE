#!/usr/bin/env python3
"""
VotIE Evaluation Script

Loads predictions from JSONL file and computes comprehensive evaluation metrics.
Separated from prediction for better separation of concerns.

Usage:
    from scripts.evaluate import evaluate
    evaluate(
        predictions_path='predictions/test_predictions.jsonl',
        output_path='evaluation/results.json'
    )
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.entity_metrics import EntityLevelEvaluator
from src.evaluation.event_metrics import EventLevelEvaluator
from src.utils.event_constructor import EventConstructor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_predictions(predictions_path: str) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions_path = Path(predictions_path)
    
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                pred = json.loads(line.strip())
                
                # Validate required fields
                required_fields = ['id', 'tokens', 'pred_labels']
                for field in required_fields:
                    if field not in pred:
                        raise ValueError(f"Missing required field '{field}' in prediction")
                
                predictions.append(pred)
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            except Exception as e:
                raise ValueError(f"Error processing line {line_num}: {e}")
    
    logger.info(f"✓ Loaded {len(predictions)} predictions from {predictions_path}")
    return predictions


def extract_labels(predictions: List[Dict[str, Any]]):
    """Extract predicted and gold labels from predictions."""
    pred_labels_list = []
    gold_labels_list = []
    
    for pred in predictions:
        pred_labels_list.append(pred['pred_labels'])
        
        # Use gold labels if available, otherwise create dummy labels
        if 'gold_labels' in pred:
            gold_labels_list.append(pred['gold_labels'])
        else:
            # Create dummy gold labels (all 'O') - useful for prediction-only evaluation
            dummy_labels = ['O'] * len(pred['pred_labels'])
            gold_labels_list.append(dummy_labels)
            
    return pred_labels_list, gold_labels_list


def compute_entity_level_metrics(pred_labels: List[List[str]], gold_labels: List[List[str]]) -> Dict[str, Any]:
    """Compute entity-level evaluation metrics."""
    logger.info("Computing entity-level metrics...")
    
    evaluator = EntityLevelEvaluator()
    metrics = evaluator.compute_metrics(pred_labels, gold_labels)
    
    logger.info(f"Entity F1: {metrics.get('entity_f1', 0.0):.4f}")
    logger.info(f"Entity Precision: {metrics.get('entity_precision', 0.0):.4f}")
    logger.info(f"Entity Recall: {metrics.get('entity_recall', 0.0):.4f}")
    
    return metrics


def compute_event_level_metrics(predictions: List[Dict[str, Any]], pred_labels: List[List[str]], gold_labels: List[List[str]]) -> Dict[str, Any]:
    """Compute event-level evaluation metrics."""
    logger.info("Computing event-level metrics...")
    
    # Construct events from predictions and gold labels
    event_constructor = EventConstructor()
    
    # Prepare examples for event construction
    pred_examples = [
        {'id': pred['id'], 'tokens': pred['tokens']}
        for pred in predictions
    ]
    gold_examples = [
        {'id': pred['id'], 'tokens': pred['tokens']}
        for pred in predictions
    ]
    
    # Construct events
    pred_events = event_constructor.construct_events(pred_examples, pred_labels)
    gold_events = event_constructor.construct_events(gold_examples, gold_labels)
    
    # Count events
    pred_event_count = sum(1 for e in pred_events if e.get('has_voting_event', False))
    gold_event_count = sum(1 for e in gold_events if e.get('has_voting_event', False))
    
    logger.info(f"Predicted events: {pred_event_count}/{len(pred_events)}")
    logger.info(f"Gold events: {gold_event_count}/{len(gold_events)}")
    
    # Compute event-level metrics
    evaluator = EventLevelEvaluator()
    metrics = evaluator.compute_metrics(pred_events, gold_events)
    
    logger.info(f"Event detection accuracy: {metrics.get('event_detection_accuracy', 0.0):.4f}")
    logger.info(f"Complete event accuracy: {metrics.get('complete_event_accuracy', 0.0):.4f}")
    
    return metrics


def evaluate(predictions_path: str, output_path: str = None, include_events: bool = True) -> Dict[str, Any]:
    """
    Evaluate predictions and save comprehensive results.
    
    Args:
        predictions_path: Path to predictions JSONL file
        output_path: Path to output evaluation results JSON (optional)
        include_events: Whether to compute event-level metrics (default: True)
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    logger.info("="*80)
    logger.info("VotIE Evaluation")
    logger.info("="*80)
    
    # Load predictions
    logger.info(f"Loading predictions from: {predictions_path}")
    predictions = load_predictions(predictions_path)
    
    # Extract labels
    pred_labels, gold_labels = extract_labels(predictions)
    
    # Compute entity-level metrics
    entity_metrics = compute_entity_level_metrics(pred_labels, gold_labels)
    
    # Compute event-level metrics if requested
    event_metrics = {}
    if include_events:
        try:
            event_metrics = compute_event_level_metrics(predictions, pred_labels, gold_labels)
        except Exception as e:
            logger.error(f"Event-level evaluation failed: {e}")
            logger.error("Stack trace:", exc_info=True)
            logger.warning("Continuing with entity-level metrics only...")
            # Provide default metrics structure (5 key F1 scores only)
            event_metrics = {
                "error": str(e),
                "event_detection_f1": 0.0,
                "subject_f1": 0.0,
                "counting_f1": 0.0,
                "participant_vote_f1": 0.0,
                "complete_event_f1": 0.0
            }
    
    # Compile results
    results = {
        'metadata': {
            'predictions_file': str(predictions_path),
            'timestamp': datetime.now().isoformat(),
            'num_examples': len(predictions),
            'include_events': include_events
        },
        'entity_level_metrics': entity_metrics,
        'event_level_metrics': event_metrics
    }
    
    # Save results if output path provided
    if output_path:
        logger.info(f"Saving evaluation results to: {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Results saved to {output_path}")
    
    logger.info("="*80)
    logger.info("Evaluation Summary:")
    logger.info(f"  Entity F1:        {entity_metrics.get('entity_f1', 0.0):.4f}")
    logger.info(f"  Entity Precision: {entity_metrics.get('entity_precision', 0.0):.4f}")
    logger.info(f"  Entity Recall:    {entity_metrics.get('entity_recall', 0.0):.4f}")
    
    if include_events and 'error' not in event_metrics:
        # Event-level F1 metrics (NEW)
        logger.info(f"  Event Detection F1: {event_metrics.get('event_detection_f1', 0.0):.4f}")
        logger.info(f"  Subject F1:         {event_metrics.get('subject_f1', 0.0):.4f}")
        logger.info(f"  Counting F1:        {event_metrics.get('counting_f1', 0.0):.4f}")
        logger.info(f"  Participant F1:     {event_metrics.get('participant_vote_f1', 0.0):.4f}")
        logger.info(f"  Complete Event F1:  {event_metrics.get('complete_event_f1', 0.0):.4f}")
    
    logger.info("="*80)
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate.py PREDICTIONS_PATH [OUTPUT_PATH] [--no-events]")
        print("\nExamples:")
        print("  python scripts/evaluate.py predictions/test_predictions.jsonl")
        print("  python scripts/evaluate.py predictions/test_predictions.jsonl evaluation/results.json")
        print("  python scripts/evaluate.py predictions/test_predictions.jsonl evaluation/results.json --no-events")
        sys.exit(1)
    
    predictions_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    include_events = '--no-events' not in sys.argv
    
    evaluate(predictions_path, output_path, include_events)
