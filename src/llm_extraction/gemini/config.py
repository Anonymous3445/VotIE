"""
Configuration for Gemini span extraction via langextract.

Uses langextract's GeminiLanguageModel with a GEMINI_API_KEY.
"""

import os


class GeminiConfig:
    """Configuration for Gemini API extractor via langextract."""

    def __init__(
        self,
        api_key: str = None,
        # Must match the model actually reported in the paper. The previous
        # default (gemini-2.5-flash-preview-04-17) was never used in any reported
        # run — the CLI default overrode it — so a reproducer silently ran Flash.
        model_id: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        max_char_buffer: int = 20000,
    ):
        """
        Initialize langextract Gemini configuration.

        Args:
            api_key: Gemini API key (falls back to GEMINI_API_KEY env var)
            model_id: Gemini model ID
            temperature: Sampling temperature (0.0 = deterministic)
            max_char_buffer: Max characters per chunk for langextract segmentation
        """
        self.api_key = api_key or os.environ.get("GEMINI_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_KEY environment variable, "
                "add it to .env, or pass api_key parameter."
            )
        self.model_id = model_id
        self.temperature = temperature
        self.max_char_buffer = max_char_buffer

    def __repr__(self) -> str:
        return (
            f"GeminiConfig(model={self.model_id}, "
            f"temperature={self.temperature}, max_char_buffer={self.max_char_buffer})"
        )
