"""
LLM Span Extraction Module

This module provides generative LLM-based span extraction for the VotIE task.
Supports both Gemini and LLaMA models with zero-shot and few-shot strategies.
"""

from .schemas import (
    SpanEntity,
    SpanExtractionOutput,
    SpanExtractionResult,
    ENTITY_TYPES,
    validate_entity_type
)

__all__ = [
    "SpanEntity",
    "SpanExtractionOutput",
    "SpanExtractionResult",
    "ENTITY_TYPES",
    "validate_entity_type"
]
