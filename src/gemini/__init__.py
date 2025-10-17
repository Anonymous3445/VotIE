"""
Gemini Event Extraction Module

Direct event extraction from Portuguese municipal voting documents using Gemini API.
"""

from .shared.model import GeminiEventExtractor
from .evaluation_events import evaluate_gemini_events
from .config import get_config

__version__ = "1.0.0"

__all__ = [
    "GeminiEventExtractor",
    "evaluate_gemini_events",
    "get_config"
]
