#!/usr/bin/env python
"""
Entity-Level NER Evaluation using seqeval library.

This module provides entity-level evaluation metrics for Named Entity Recognition (NER)
using the seqeval library, which is the standard evaluation framework for sequence
labeling tasks in NLP.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import seqeval for standard NER evaluation
from seqeval.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from seqeval.scheme import IOB2


class EntityLevelEvaluator:
    """
    Entity-level NER evaluation using seqeval library.

    Provides strict entity-level evaluation based on exact boundary matching
    using the IOB2 tagging scheme.
    """

    def __init__(self, id_to_label: Optional[Dict[int, str]] = None):
        """
        Initialize evaluator.

        Args:
            id_to_label: Optional mapping from label IDs to strings
        """
        self.id_to_label = id_to_label or {}

    def convert_ids_to_labels(self, predictions: List[List[int]]) -> List[List[str]]:
        """Convert predicted label IDs to label strings."""

        if not self.id_to_label:
            return predictions

        return [
            [self.id_to_label.get(pred_id, 'O') for pred_id in pred_seq]
            for pred_seq in predictions
        ]

    def compute_metrics(
        self,
        pred_labels: List[List[str]],
        true_labels: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Compute entity-level evaluation metrics using seqeval.

        Uses macro-averaging: computes metrics for each entity type independently,
        then averages them, giving equal weight to all entity types regardless of frequency.
        This is appropriate for imbalanced datasets where all classes are equally important.

        Args:
            pred_labels: Predicted BIO label sequences
            true_labels: Ground truth BIO label sequences

        Returns:
            Dictionary with comprehensive evaluation metrics including:
            - Macro-averaged precision, recall, F1 (equal weight per class)
            - Per-entity-type metrics (precision, recall, F1, support)
        """
        try:
            # Macro-averaged metrics (averages per-class metrics - equal weight per class)
            macro_precision = precision_score(true_labels, pred_labels, scheme=IOB2, average='macro')
            macro_recall = recall_score(true_labels, pred_labels, scheme=IOB2, average='macro')
            macro_f1 = f1_score(true_labels, pred_labels, scheme=IOB2, average='macro')
            accuracy = accuracy_score(true_labels, pred_labels)

            results = {
                "entity_precision": float(macro_precision),
                "entity_recall": float(macro_recall),
                "entity_f1": float(macro_f1),
                "entity_accuracy": float(accuracy),
            }

            # Per-type metrics using classification_report
            report = classification_report(
                true_labels,
                pred_labels,
                scheme=IOB2,
                output_dict=True
            )
            
            per_type_metrics = {}
            for entity_type, metrics in report.items():
                # Skip overall metrics
                if entity_type not in ['micro avg', 'macro avg', 'weighted avg']:
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        per_type_metrics[entity_type] = {
                            'precision': float(metrics['precision']),
                            'recall': float(metrics['recall']),
                            'f1_score': float(metrics['f1-score']),
                            'support': int(metrics['support'])
                        }

            results["per_type_metrics"] = per_type_metrics
            return results

        except Exception as e:
            logger.error(f"Error computing entity-level metrics: {e}")
            return {
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_f1": 0.0,
                "entity_accuracy": 0.0,
                "per_type_metrics": {},
            }