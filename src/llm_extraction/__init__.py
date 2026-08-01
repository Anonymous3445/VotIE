"""
LLM Span Extraction Module

This module provides generative LLM-based span extraction for the VotIE task.
Supports Gemini, GPT and AMALIA in zero-shot and few-shot settings.
"""

from .schemas import (
    SpanEntity,
    SpanExtractionResult,
    ENTITY_TYPES,
    validate_entity_type
)

__all__ = [
    "SpanEntity",
    "SpanExtractionResult",
    "ENTITY_TYPES",
    "validate_entity_type"
]
