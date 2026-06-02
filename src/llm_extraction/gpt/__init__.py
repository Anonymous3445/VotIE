"""GPT-5.5 span extractor via langextract custom provider."""

from .extractor import GptSpanExtractor
from .config import GptConfig

__all__ = ["GptSpanExtractor", "GptConfig"]
