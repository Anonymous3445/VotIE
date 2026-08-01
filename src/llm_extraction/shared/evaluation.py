"""
Evaluation metrics for span extraction.

Computes entity-level F1 scores and alignment fidelity metrics.
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

try:
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2
except ImportError:
    raise ImportError(
        "seqeval is required for evaluation. "
        "Install with: pip install seqeval"
    )

from ..schemas import SpanExtractionResult, SpanEntity, ENTITY_TYPES
from ..span_alignment import calculate_fidelity_metrics

# Import the same tokenization used for gold standard data
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.span_to_bio import tokenize_text_with_offsets, align_spans_to_tokens


logger = logging.getLogger(__name__)


def spans_to_bio_labels(text: str, entities: List[SpanEntity]) -> Tuple[List[str], List[str]]:
    """
    Convert span entities to BIO labels for seqeval evaluation.
    
    Uses the SAME tokenization as the gold standard data (punctuation-aware)
    to ensure fair comparison between encoder and generative models.

    Args:
        text: Source text
        entities: List of span entities

    Returns:
        Tuple of (tokens, labels)
    """
    # Use the same tokenization as gold standard data
    tokens_with_offsets = tokenize_text_with_offsets(text)
    
    # Convert entities to span format expected by align_spans_to_tokens
    spans = [
        {
            'start': entity.start,
            'end': entity.end,
            'label': entity.type,
            'text': entity.text
        }
        for entity in entities
    ]
    
    # Align spans to tokens (same logic as gold standard creation)
    labels = align_spans_to_tokens(text, spans, tokens_with_offsets)
    
    # Extract just the tokens
    tokens = [token for token, _, _ in tokens_with_offsets]

    return tokens, labels


def macro_over_entity_types(
    per_type: Dict[str, Dict[str, float]]
) -> Tuple[float, float, float]:
    """Macro-average over the fixed schema type list, not over observed types.

    The denominator is always ``len(ENTITY_TYPES)`` (11 — ``Count-Against`` is in
    the schema but has zero support anywhere and is excluded from the list), so
    exact and relaxed matching share one denominator, and LOMO folds stay
    comparable even when a municipality never uses some type. Types with no gold
    and no prediction in a given evaluation set contribute 0.

    Args:
        per_type: mapping of entity type to a dict with precision/recall/f1.

    Returns:
        (precision, recall, f1), each averaged over the 11 schema types.
    """
    n = len(ENTITY_TYPES)
    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for entity_type in ENTITY_TYPES:
        metrics = per_type.get(entity_type) or {}
        for key in totals:
            totals[key] += metrics.get(key, 0.0)
    return totals["precision"] / n, totals["recall"] / n, totals["f1"] / n


def overlap_length(a: SpanEntity, b: SpanEntity) -> int:
    """Number of characters shared by two spans (0 if disjoint or unaligned)."""
    if any(v is None for v in (a.start, a.end, b.start, b.end)):
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def relaxed_span_match(pred_entity: SpanEntity, gold_entity: SpanEntity) -> bool:
    """
    Check if predicted span matches gold span with relaxed criteria.

    A span is considered correct if:
    1. The entity type matches exactly
    2. There is any overlap between the predicted and gold spans

    Args:
        pred_entity: Predicted entity
        gold_entity: Gold standard entity

    Returns:
        True if type matches and spans overlap
    """
    if pred_entity.type != gold_entity.type:
        return False

    # Guard against unaligned spans (None positions)
    if any(v is None for v in (pred_entity.start, pred_entity.end,
                                gold_entity.start, gold_entity.end)):
        return False

    return pred_entity.start < gold_entity.end and pred_entity.end > gold_entity.start


def evaluate_span_extraction(
    predictions: List[SpanExtractionResult],
    ground_truth: List[Dict[str, Any]],
    compute_fidelity: bool = True,
    relaxed_matching: bool = False
) -> Dict[str, Any]:
    """
    Evaluate span extraction results.

    Args:
        predictions: List of extraction results
        ground_truth: List of ground truth examples with 'id', 'text', 'spans'
        compute_fidelity: Whether to compute alignment fidelity metrics
        relaxed_matching: If True, use relaxed matching (type match + overlap)

    Returns:
        Dictionary with evaluation metrics
    """
    # Create mapping from id to ground truth
    gt_map = {ex['id']: ex for ex in ground_truth}

    # Collect predictions and labels for seqeval
    all_pred_labels = []
    all_true_labels = []

    # Process each prediction
    matched_count = 0
    for pred in predictions:
        if pred.id not in gt_map:
            logger.warning(f"Prediction {pred.id} not found in ground truth")
            continue

        gt = gt_map[pred.id]
        matched_count += 1

        # Convert ground truth spans to entities
        gt_entities = []
        for span in gt.get('spans', []):
            gt_entities.append(SpanEntity(
                text=span['text'],
                type=span.get('label', span.get('type', 'UNKNOWN')),
                start=span['start'],
                end=span['end']
            ))

        # Convert to BIO labels
        _, pred_labels = spans_to_bio_labels(pred.text, pred.entities)
        _, true_labels = spans_to_bio_labels(gt['text'], gt_entities)

        # Ensure same length (pad or truncate)
        max_len = max(len(pred_labels), len(true_labels))
        pred_labels.extend(["O"] * (max_len - len(pred_labels)))
        true_labels.extend(["O"] * (max_len - len(true_labels)))

        all_pred_labels.append(pred_labels)
        all_true_labels.append(true_labels)

    logger.info(f"Matched {matched_count}/{len(predictions)} predictions to ground truth")

    # Exact boundary matching via seqeval. We take per-type scores from the
    # report and average them ourselves rather than calling average='macro',
    # because seqeval's macro divides by the types it observes — a denominator
    # that changes between models and between LOMO folds.
    try:
        report = classification_report(
            all_true_labels,
            all_pred_labels,
            mode='strict',
            scheme=IOB2,
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        logger.error(f"Error computing seqeval metrics: {e}")
        report = {}

    exact_per_type = extract_per_type_metrics(report)
    entity_precision, entity_recall, entity_f1 = macro_over_entity_types(exact_per_type)

    # Compute relaxed matching metrics if requested
    relaxed_metrics = None
    if relaxed_matching:
        relaxed_metrics = compute_relaxed_metrics(predictions, gt_map)

    # Compute alignment fidelity if requested
    fidelity_metrics = {}
    if compute_fidelity:
        fidelity_metrics = calculate_fidelity_metrics(predictions)

    # Compile results
    results = {
        "entity_level": {
            "precision": entity_precision,
            "recall": entity_recall,
            "f1": entity_f1
        },
        "per_type_metrics": exact_per_type,
        "macro_denominator": len(ENTITY_TYPES),
        "total_predictions": len(predictions),
        "matched_to_ground_truth": matched_count
    }

    if compute_fidelity:
        results["alignment_fidelity"] = fidelity_metrics

    if relaxed_metrics is not None:
        results["relaxed_boundary_metrics"] = relaxed_metrics

    return results


def compute_relaxed_metrics(
    predictions: List[SpanExtractionResult],
    gt_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 with relaxed span matching.

    Accepts spans as correct if the type matches and there is any overlap.

    Args:
        predictions: List of predictions
        gt_map: Mapping from document ID to ground truth

    Returns:
        Dictionary with relaxed metrics
    """
    # Track TP, FP, FN per entity type
    per_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred in predictions:
        if pred.id not in gt_map:
            continue

        gt = gt_map[pred.id]

        # Convert ground truth spans to entities
        gt_entities = []
        for span in gt.get('spans', []):
            gt_entities.append(SpanEntity(
                text=span['text'],
                type=span.get('label', span.get('type', 'UNKNOWN')),
                start=span['start'],
                end=span['end']
            ))

        # Track which gold entities have been matched
        matched_gold = set()

        # For each predicted entity, take the gold entity it overlaps most.
        # Selecting the first overlap instead would make the score depend on the
        # order gold spans happen to appear in.
        for pred_entity in pred.entities:
            best_match = None
            best_overlap = 0
            for i, gold_entity in enumerate(gt_entities):
                if i in matched_gold or not relaxed_span_match(pred_entity, gold_entity):
                    continue
                overlap = overlap_length(pred_entity, gold_entity)
                if overlap > best_overlap:
                    best_match, best_overlap = i, overlap

            if best_match is not None:
                # True positive
                per_type_stats[pred_entity.type]["tp"] += 1
                matched_gold.add(best_match)
            else:
                # False positive
                per_type_stats[pred_entity.type]["fp"] += 1

        # Unmatched gold entities are false negatives
        for i, gold_entity in enumerate(gt_entities):
            if i not in matched_gold:
                per_type_stats[gold_entity.type]["fn"] += 1

    # Compute per-type metrics first
    per_type_metrics = {}
    for entity_type, stats in per_type_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        type_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        type_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0

        per_type_metrics[entity_type] = {
            "precision": type_precision,
            "recall": type_recall,
            "f1": type_f1,
            "support": tp + fn  # Total gold entities of this type
        }

    # Same fixed denominator as exact match, so the two are comparable.
    precision, recall, f1 = macro_over_entity_types(per_type_metrics)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_denominator": len(ENTITY_TYPES),
        "per_type_metrics": per_type_metrics
    }


def extract_per_type_metrics(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract per-entity-type metrics from seqeval report.

    Args:
        report: Classification report from seqeval

    Returns:
        Dictionary mapping entity types to metrics
    """
    per_type = {}

    for entity_type in ENTITY_TYPES:
        if entity_type in report:
            per_type[entity_type] = {
                "precision": report[entity_type].get("precision", 0.0),
                "recall": report[entity_type].get("recall", 0.0),
                "f1": report[entity_type].get("f1-score", 0.0),
                "support": report[entity_type].get("support", 0)
            }

    return per_type


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Print evaluation results in a readable format.

    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "=" * 80)
    print("SPAN EXTRACTION EVALUATION RESULTS")
    print("=" * 80)

    # Entity-level metrics (exact boundaries)
    entity_metrics = results["entity_level"]
    print(f"\nEntity-Level Metrics (Exact Boundary Matching):")
    print(f"  Precision: {entity_metrics['precision']:.4f}")
    print(f"  Recall:    {entity_metrics['recall']:.4f}")
    print(f"  F1:        {entity_metrics['f1']:.4f}")

    # Relaxed boundary metrics if present
    if "relaxed_boundary_metrics" in results:
        relaxed = results["relaxed_boundary_metrics"]
        print(f"\nRelaxed Span Matching (Type Match + Overlap):")
        print(f"  Precision: {relaxed['precision']:.4f}")
        print(f"  Recall:    {relaxed['recall']:.4f}")
        print(f"  F1:        {relaxed['f1']:.4f}")
        delta_f1 = relaxed['f1'] - entity_metrics['f1']
        print(f"  Δ F1:      {delta_f1:+.4f} ({delta_f1*100:+.1f} points)")

    # Per-type metrics
    print(f"\nPer-Entity-Type Metrics:")
    print(f"  {'Entity Type':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-' * 60}")

    per_type = results["per_type_metrics"]
    for entity_type in sorted(per_type.keys()):
        metrics = per_type[entity_type]
        print(f"  {entity_type:<25} {metrics['precision']:>6.2f} {metrics['recall']:>6.2f} "
              f"{metrics['f1']:>6.2f} {metrics['support']:>8}")

    # Alignment fidelity
    if "alignment_fidelity" in results:
        fidelity = results["alignment_fidelity"]
        print(f"\nAlignment Fidelity:")
        print(f"  Total spans:          {fidelity['total_spans']}")
        print(f"  Aligned spans:        {fidelity['aligned_spans']}")
        print(f"  Failed alignments:    {fidelity['failed_spans']}")
        print(f"  Alignment fidelity:   {fidelity['alignment_fidelity']:.2%}")
        print(f"  Non-verbatim errors:  {fidelity['non_verbatim_error_count']}")

    print("\n" + "=" * 80)
