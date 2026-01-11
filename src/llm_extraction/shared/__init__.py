"""Shared utilities for span extraction."""

from .base_extractor import BaseSpanExtractor
from .evaluation import evaluate_span_extraction, print_evaluation_results
from .data_utils import load_jsonl, save_jsonl, load_citilink_data, load_votie_bio_data

__all__ = [
    "BaseSpanExtractor",
    "evaluate_span_extraction",
    "print_evaluation_results",
    "load_jsonl",
    "save_jsonl",
    "load_citilink_data",
    "load_votie_bio_data"  # Backward compatibility
]
