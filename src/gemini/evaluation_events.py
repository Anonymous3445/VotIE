"""
Event-Level Evaluation for Gemini Event Extraction

Implements event-level metrics for voting event extraction.
"""

import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EventMetrics:
    """Metrics for event-level evaluation"""
    precision: float
    recall: float
    f1: float
    correct: int
    incorrect: int
    partial: int
    missed: int
    spurious: int


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Lowercase
    text = text.lower()
    return text

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Lowercase
    text = text.lower()
    return text


def compute_span_overlap(pred_text: str, gold_text: str) -> float:
    """
    Compute overlap between predicted and gold text spans.

    Returns Jaccard similarity (intersection / union) of normalized tokens.

    Args:
        pred_text: Predicted text span
        gold_text: Gold standard text span

    Returns:
        Overlap score between 0 and 1
    """
    pred_tokens = set(normalize_text(pred_text).split())
    gold_tokens = set(normalize_text(gold_text).split())

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    intersection = len(pred_tokens & gold_tokens)
    union = len(pred_tokens | gold_tokens)

    return intersection / union if union > 0 else 0.0


def compare_events(pred_event: Dict[str, Any], gold_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare predicted event to gold event at component level.

    Returns scores for each component (subject, participants, counting, voting_expression).

    Args:
        pred_event: Predicted event dictionary
        gold_event: Gold event dictionary

    Returns:
        Dictionary with component-level scores
    """
    scores = {
        "subject": 0.0,
        "participants": 0.0,
        "counting": 0.0,
        "voting_expression": 0.0,
        "overall": 0.0
    }

    # Check has_voting_event
    pred_has_event = pred_event.get("has_voting_event", False)
    gold_has_event = gold_event.get("has_voting_event", False)

    if pred_has_event != gold_has_event:
        return scores  # Mismatch on basic classification

    if not gold_has_event:
        # Both correctly identified no event
        return {k: 1.0 for k in scores.keys()}

    pred = pred_event.get("event", {})
    gold = gold_event.get("event", {})

    if not pred or not gold:
        return scores

    # Subject matching (handle both singular and list formats)
    pred_subj = pred.get("subject") or (pred.get("subjects", []) or [None])[0]
    gold_subj = gold.get("subject") or (gold.get("subjects", []) or [None])[0]
    if pred_subj and gold_subj:
        scores["subject"] = compute_span_overlap(pred_subj, gold_subj)

    # Voting expression matching (handle both singular and list formats)
    pred_voting = pred.get("voting_expression") or (pred.get("voting_expressions", []) or [None])[0]
    gold_voting = gold.get("voting_expression") or (gold.get("voting_expressions", []) or [None])[0]
    if pred_voting and gold_voting:
        scores["voting_expression"] = compute_span_overlap(pred_voting, gold_voting)

    # Counting matching (handle both singular dict and list formats)
    pred_counting = pred.get("counting")
    gold_counting = gold.get("counting")
    
    # Convert lists to single items if needed
    if isinstance(pred_counting, list):
        pred_counting = pred_counting[0] if pred_counting else None
    if isinstance(gold_counting, list):
        gold_counting = gold_counting[0] if gold_counting else None
    
    if pred_counting and gold_counting:
        # Check type match
        if pred_counting.get("type") == gold_counting.get("type"):
            # Check text overlap
            scores["counting"] = compute_span_overlap(pred_counting.get("text", ""),
                                                     gold_counting.get("text", ""))
        else:
            scores["counting"] = 0.0

    # Participants matching (most complex)
    pred_participants = pred.get("participants", [])
    gold_participants = gold.get("participants", [])

    if gold_participants:
        participant_matches = 0

        for gold_p in gold_participants:
            best_match = 0.0
            for pred_p in pred_participants:
                # Must match position
                if pred_p.get("position") == gold_p.get("position"):
                    overlap = compute_span_overlap(pred_p.get("text", ""), gold_p.get("text", ""))
                    best_match = max(best_match, overlap)

            if best_match >= 0.5:  # Partial match threshold
                participant_matches += 1

        scores["participants"] = participant_matches / len(gold_participants) if gold_participants else 0.0

    # Overall score (average of components)
    component_scores = [scores[k] for k in ["subject", "participants", "counting", "voting_expression"] if scores[k] > 0 or gold.get(k)]
    scores["overall"] = sum(component_scores) / len(component_scores) if component_scores else 0.0

    return scores


def compute_event_metrics(predictions: List[Dict[str, Any]],
                         gold_events: List[Dict[str, Any]]) -> EventMetrics:
    """
    Compute event-level metrics.

    Args:
        predictions: List of predicted event dictionaries
        gold_events: List of gold event dictionaries

    Returns:
        EventMetrics object with precision, recall, f1
    """
    if len(predictions) != len(gold_events):
        raise ValueError(f"Predictions and gold must have same length: {len(predictions)} vs {len(gold_events)}")

    correct = 0
    partial = 0
    missed = 0
    spurious = 0

    for pred, gold in zip(predictions, gold_events):
        scores = compare_events(pred, gold)
        overall_score = scores["overall"]

        if overall_score >= 0.95:  # Exact match
            correct += 1
        elif overall_score >= 0.5:  # Partial match
            partial += 1
        elif gold.get("has_voting_event") and not pred.get("has_voting_event"):
            missed += 1
        elif not gold.get("has_voting_event") and pred.get("has_voting_event"):
            spurious += 1

    # Calculate metrics
    total_pred = sum(1 for p in predictions if p.get("has_voting_event"))
    total_gold = sum(1 for g in gold_events if g.get("has_voting_event"))

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EventMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        correct=correct,
        incorrect=len(predictions) - correct - partial,
        partial=partial,
        missed=missed,
        spurious=spurious
    )


def evaluate_gemini_events(test_data: List[Dict[str, Any]],
                          predictions: List[Dict[str, Any]],
                          gold_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate Gemini predictions at event level.

    Args:
        test_data: List of test segments (not used, kept for API compatibility)
        predictions: List of predicted events from Gemini
        gold_events: List of gold standard events

    Returns:
        Dictionary with event-level metrics
    """
    # Event-level metrics
    event_metrics = compute_event_metrics(predictions, gold_events)

    return {
        "event_level": {
            "precision": event_metrics.precision,
            "recall": event_metrics.recall,
            "f1": event_metrics.f1,
            "correct": event_metrics.correct,
            "incorrect": event_metrics.incorrect,
            "partial": event_metrics.partial,
            "missed": event_metrics.missed,
            "spurious": event_metrics.spurious
        }
    }
