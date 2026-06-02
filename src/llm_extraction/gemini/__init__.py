"""Gemini-based span extraction via langextract."""

from .extractor import GeminiSpanExtractor
from .config import GeminiConfig

__all__ = ["GeminiSpanExtractor", "GeminiConfig"]
