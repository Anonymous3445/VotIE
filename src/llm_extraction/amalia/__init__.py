"""AMALIA-based span extraction via langextract + OpenAI-compatible vLLM API."""

from .extractor import AmaliaSpanExtractor
from .config import AmaliaConfig, DEFAULT_CONFIG

__all__ = ["AmaliaSpanExtractor", "AmaliaConfig", "DEFAULT_CONFIG"]
