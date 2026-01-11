"""Gemini-based span extraction."""

from .extractor import GeminiSpanExtractor
from .config import GeminiConfig, DEFAULT_CONFIG

__all__ = ["GeminiSpanExtractor", "GeminiConfig", "DEFAULT_CONFIG"]
