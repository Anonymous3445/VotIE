#!/usr/bin/env python3
"""
VotIE Evaluation Script

Loads predictions from JSONL file and computes entity-level evaluation metrics.
Separated from prediction for better separation of concerns.

Usage:
    from scripts.evaluate import evaluate
    evaluate(
        predictions_path='predictions/test_predictions.jsonl',
        output_path='results/evaluation.json',
        relaxed_threshold=0.1  # Optional: for ±10% boundary matching
    )
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.entity_metrics import EntityLevelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _spans_to_bio_via_offsets(
    offsets: List[List[int]], gold_spans: List[Dict[str, Any]]
) -> List[str]:
    """Convert gold spans (char-level) to BIO labels aligned to subword offsets."""
    labels = ['O'] * len(offsets)
    for span in sorted(gold_spans, key=lambda s: s['start']):
        span_type = span.get('type') or span.get('label')
        if not span_type:
            continue
        span_start, span_end = span['start'], span['end']
        first_in_span = True
        for i, (sw_start, sw_end) in enumerate(offsets):
            if sw_end <= span_start or sw_start >= span_end:
                continue
            if labels[i] != 'O':
                continue
            labels[i] = f'B-{span_type}' if first_in_span else f'I-{span_type}'
            first_in_span = False
    return labels


def _normalize_to_token_format(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize offset-aligned predictions to the legacy tokens/pred_labels format.

    The offset-aligned pipeline (transformer models) writes subword_offsets +
    subword_pred_labels. Legacy format (CRF/BiLSTM) writes tokens + pred_labels.
    This function unifies both so the rest of the evaluator stays unchanged.
    """
    if 'tokens' in pred and 'pred_labels' in pred:
        return pred  # already legacy format

    offsets = pred.get('subword_offsets', [])
    pred_labels = pred.get('subword_pred_labels', [])
    text = pred.get('text', '')

    tokens = [text[s:e] for s, e in offsets] if text else [''] * len(offsets)

    normalized: Dict[str, Any] = {
        'id': pred['id'],
        'tokens': tokens,
        'pred_labels': pred_labels,
        'token_offsets': offsets,
        'text': text,
    }

    if 'gold_spans' in pred and pred['gold_spans']:
        normalized['gold_labels'] = _spans_to_bio_via_offsets(offsets, pred['gold_spans'])
    elif 'gold_labels' in pred:
        normalized['gold_labels'] = pred['gold_labels']

    return normalized


def load_predictions(predictions_path: str) -> List[Dict[str, Any]]:
    """Load predictions from JSON or JSONL file."""
    predictions_path = Path(predictions_path)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions = []

    # Try to load as JSON first (array format)
    try:
        with open(predictions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # If it's a list, treat as JSON array format
        if isinstance(data, list):
            predictions = data
        else:
            raise ValueError("JSON file must contain an array of predictions")

    except json.JSONDecodeError:
        # If JSON fails, try JSONL format (one JSON object per line)
        with open(predictions_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    pred = json.loads(line)
                    predictions.append(pred)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")

    # Validate: every prediction must have an id and at least one label format
    for i, pred in enumerate(predictions, 1):
        if 'id' not in pred:
            raise ValueError(f"Prediction {i} missing required field 'id'")
        has_legacy = 'tokens' in pred and 'pred_labels' in pred
        has_offset = 'subword_offsets' in pred and 'subword_pred_labels' in pred
        if not has_legacy and not has_offset:
            raise ValueError(
                f"Prediction {i} must have either ('tokens', 'pred_labels') "
                f"or ('subword_offsets', 'subword_pred_labels')"
            )

    # Normalize offset-aligned format to legacy token/label format
    predictions = [_normalize_to_token_format(p) for p in predictions]

    logger.info(f"✓ Loaded {len(predictions)} predictions from {predictions_path}")
    return predictions


def extract_labels(predictions: List[Dict[str, Any]]):
    """Extract predicted labels, gold labels, and tokens from predictions."""
    pred_labels_list = []
    gold_labels_list = []
    tokens_list = []

    for pred in predictions:
        pred_labels_list.append(pred['pred_labels'])
        tokens_list.append(pred['tokens'])

        # Use gold labels if available, otherwise create dummy labels
        if 'gold_labels' in pred:
            gold_labels_list.append(pred['gold_labels'])
        else:
            # Create dummy gold labels (all 'O') - useful for prediction-only evaluation
            dummy_labels = ['O'] * len(pred['pred_labels'])
            gold_labels_list.append(dummy_labels)

    return pred_labels_list, gold_labels_list, tokens_list


def bio_to_spans(
    tokens: List[str],
    labels: List[str],
    token_offsets: List[List[int]] = None,
    text: str = None,
) -> List[Dict[str, Any]]:
    """
    Convert BIO labels to span entities with character positions.

    When ``token_offsets`` (and optionally ``text``) are provided, span
    boundaries are taken directly from the original character offsets
    produced during tokenization.  This avoids the lossy assumption that
    every token is separated by a single space, and yields span text
    that faithfully matches the source document (preserving punctuation
    adjacency, newlines, etc.).

    When offsets are *not* available the function falls back to the
    legacy heuristic (``char_pos += len(token) + 1``) so that existing
    prediction files without offsets continue to work.

    Args:
        tokens: List of tokens.
        labels: List of BIO labels (same length as *tokens*).
        token_offsets: Optional list of ``[start, end)`` character
            positions in the original text, one per token.
        text: Optional original text.  Used together with
            *token_offsets* to extract span text via
            ``text[start:end]`` instead of joining tokens with spaces.

    Returns:
        List of span dicts with keys ``text``, ``type``, ``start``,
        ``end``, ``tokens``.
    """
    use_offsets = (
        token_offsets is not None
        and len(token_offsets) == len(tokens)
    )

    spans = []
    current_entity = None
    char_pos = 0  # only used in legacy fallback

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if use_offsets:
            tok_start, tok_end = token_offsets[i]
        else:
            tok_start = char_pos
            tok_end = char_pos + len(token)

        if label.startswith('B-'):
            if current_entity:
                _finalize_span(current_entity, text, use_offsets)
                spans.append(current_entity)

            entity_type = label[2:]
            current_entity = {
                'type': entity_type,
                'start': tok_start,
                'end': tok_end,
                'tokens': [token],
                '_first_idx': i,
            }

        elif label.startswith('I-') and current_entity:
            entity_type = label[2:]
            if entity_type == current_entity['type']:
                current_entity['end'] = tok_end
                current_entity['tokens'].append(token)

        elif label == 'O':
            if current_entity:
                _finalize_span(current_entity, text, use_offsets)
                spans.append(current_entity)
                current_entity = None

        if not use_offsets:
            char_pos += len(token) + 1

    if current_entity:
        _finalize_span(current_entity, text, use_offsets)
        spans.append(current_entity)

    return spans


def _finalize_span(
    span: Dict[str, Any],
    text: str = None,
    use_offsets: bool = False,
) -> None:
    """Set the ``text`` field of a span and clean up internal keys.

    For offset-based input (subword-level), the SentencePiece ``▁`` boundary
    encodes the leading space into the next subword's offset, so a recovered
    span's start/end may straddle adjacent whitespace. Trim those edges so
    the recovered text matches the human-meaningful word boundaries that
    annotators use.
    """
    if use_offsets and text:
        s, e = span['start'], span['end']
        # Trim leading/trailing whitespace from the recovered char range.
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        span['start'] = s
        span['end'] = e
        span['text'] = text[s:e]
    else:
        span['text'] = ' '.join(span['tokens'])
    span.pop('_first_idx', None)


def relaxed_span_match(pred_span: Dict[str, Any], gold_span: Dict[str, Any]) -> bool:
    """
    Check if predicted span matches gold span with relaxed criteria.

    A span is considered correct if:
    1. The entity type matches exactly
    2. There is any overlap between the predicted and gold spans

    This follows the overlap-based relaxed matching paradigm used in the LLM evaluation.

    Args:
        pred_span: Predicted span dict with 'type', 'start', 'end'
        gold_span: Gold span dict with 'type', 'start', 'end'

    Returns:
        True if type matches and spans overlap
    """
    # Type must match exactly
    if pred_span['type'] != gold_span['type']:
        return False

    # Check for overlap: pred_start < gold_end AND pred_end > gold_start
    return pred_span['start'] < gold_span['end'] and pred_span['end'] > gold_span['start']


def compute_relaxed_metrics(
    pred_labels_list: List[List[str]],
    gold_labels_list: List[List[str]],
    tokens_list: List[List[str]]
) -> Dict[str, Any]:
    """
    Compute relaxed boundary matching metrics with macro-averaging.

    Uses overlap-based matching: a span is correct if the type matches
    and there is any overlap between predicted and gold spans.
    
    Computes macro-averaged metrics by computing per-type metrics first,
    then averaging across all entity types (equal weight per class).

    Args:
        pred_labels_list: List of predicted label sequences
        gold_labels_list: List of gold label sequences
        tokens_list: List of token sequences

    Returns:
        Dictionary with relaxed metrics (macro-averaged)
    """
    logger.info("Computing relaxed boundary metrics (overlap-based, macro-averaged)...")

    # Track per-type metrics
    per_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred_labels, gold_labels, tokens in zip(pred_labels_list, gold_labels_list, tokens_list):
        # Convert to spans
        pred_spans = bio_to_spans(tokens, pred_labels)
        gold_spans = bio_to_spans(tokens, gold_labels)

        # Track which gold spans have been matched
        matched_gold = set()

        # For each predicted span, try to find a matching gold span
        for pred_span in pred_spans:
            matched = False
            for j, gold_span in enumerate(gold_spans):
                if j in matched_gold:
                    continue

                if relaxed_span_match(pred_span, gold_span):
                    # True positive
                    per_type_stats[pred_span['type']]['tp'] += 1
                    matched_gold.add(j)
                    matched = True
                    break

            if not matched:
                # False positive
                per_type_stats[pred_span['type']]['fp'] += 1

        # Count unmatched gold spans as false negatives
        for j, gold_span in enumerate(gold_spans):
            if j not in matched_gold:
                per_type_stats[gold_span['type']]['fn'] += 1

    # Compute per-type metrics first
    per_type_metrics = {}
    per_type_precisions = []
    per_type_recalls = []
    per_type_f1s = []
    
    for entity_type, stats in per_type_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        type_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        type_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0

        per_type_metrics[entity_type] = {
            'precision': type_precision,
            'recall': type_recall,
            'f1_score': type_f1,
            'support': tp + fn
        }
        
        per_type_precisions.append(type_precision)
        per_type_recalls.append(type_recall)
        per_type_f1s.append(type_f1)

    # Compute macro-averaged overall metrics (average of per-type metrics)
    macro_precision = sum(per_type_precisions) / len(per_type_precisions) if per_type_precisions else 0.0
    macro_recall = sum(per_type_recalls) / len(per_type_recalls) if per_type_recalls else 0.0
    macro_f1 = sum(per_type_f1s) / len(per_type_f1s) if per_type_f1s else 0.0

    logger.info(f"Relaxed F1 (macro):        {macro_f1:.4f}")
    logger.info(f"Relaxed Precision (macro): {macro_precision:.4f}")
    logger.info(f"Relaxed Recall (macro):    {macro_recall:.4f}")

    return {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'per_type_metrics': per_type_metrics
    }


def compute_entity_level_metrics(pred_labels: List[List[str]], gold_labels: List[List[str]]) -> Dict[str, Any]:
    """Compute entity-level evaluation metrics using macro-averaging."""
    logger.info("Computing entity-level metrics (macro-averaged)...")
    logger.info("  Macro-averaging: Equal weight to all entity types")

    evaluator = EntityLevelEvaluator()
    metrics = evaluator.compute_metrics(pred_labels, gold_labels)

    logger.info(f"\nExact Match (macro-averaged):")
    logger.info(f"  F1:        {metrics.get('entity_f1', 0.0):.4f}")
    logger.info(f"  Precision: {metrics.get('entity_precision', 0.0):.4f}")
    logger.info(f"  Recall:    {metrics.get('entity_recall', 0.0):.4f}")

    return metrics


def evaluate(predictions_path: str, output_path: str = None, relaxed_matching: bool = False) -> Dict[str, Any]:
    """
    Evaluate predictions and save results.

    Args:
        predictions_path: Path to predictions JSONL file
        output_path: Path to output evaluation results JSON (optional)
        relaxed_matching: If True, also compute relaxed boundary metrics (overlap-based)

    Returns:
        Dictionary with evaluation results
    """
    logger.info("="*80)
    logger.info("VotIE Evaluation - Entity-Level Metrics")
    logger.info("="*80)

    # Load predictions
    logger.info(f"Loading predictions from: {predictions_path}")
    predictions = load_predictions(predictions_path)

    # Extract labels and tokens
    pred_labels, gold_labels, tokens = extract_labels(predictions)

    # Compute exact entity-level metrics
    entity_metrics = compute_entity_level_metrics(pred_labels, gold_labels)

    # Compile results
    results = {
        'metadata': {
            'predictions_file': str(predictions_path),
            'timestamp': datetime.now().isoformat(),
            'num_examples': len(predictions)
        },
        'entity_level_metrics': entity_metrics
    }

    # Compute relaxed metrics if requested
    if relaxed_matching:
        logger.info("")
        relaxed_metrics = compute_relaxed_metrics(pred_labels, gold_labels, tokens)
        results['relaxed_boundary_metrics'] = relaxed_metrics

        # Compute delta
        delta_f1 = relaxed_metrics['f1'] - entity_metrics['entity_f1']
        results['relaxed_boundary_metrics']['delta_f1'] = delta_f1

    # Save results if output path provided
    if output_path:
        logger.info(f"\nSaving evaluation results to: {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Results saved to {output_path}")

    logger.info("="*80)
    logger.info("Evaluation Summary (Macro-Averaged):")
    logger.info("")
    logger.info("Exact Match:")
    logger.info(f"  F1:        {entity_metrics.get('entity_f1', 0.0):.4f}")
    logger.info(f"  Precision: {entity_metrics.get('entity_precision', 0.0):.4f}")
    logger.info(f"  Recall:    {entity_metrics.get('entity_recall', 0.0):.4f}")

    if relaxed_matching:
        relaxed_metrics = results['relaxed_boundary_metrics']
        logger.info("")
        logger.info("Relaxed Match (overlap-based):")
        logger.info(f"  F1:        {relaxed_metrics['f1']:.4f}")
        logger.info(f"  Precision: {relaxed_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {relaxed_metrics['recall']:.4f}")
        logger.info(f"  Improvement: +{delta_f1:.4f} F1")

    logger.info("="*80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate VotIE predictions')
    parser.add_argument('predictions_path', help='Path to predictions JSONL file')
    parser.add_argument('output_path', nargs='?', default=None, help='Path to output evaluation JSON (optional)')
    parser.add_argument('--relaxed', action='store_true',
                        help='Enable relaxed boundary matching (overlap-based)')

    args = parser.parse_args()

    evaluate(args.predictions_path, args.output_path, args.relaxed)
