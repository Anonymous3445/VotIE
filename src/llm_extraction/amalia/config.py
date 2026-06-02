"""
Configuration for AMALIA span extraction via langextract.

AMALIA is a Portuguese LLM served via vLLM with an OpenAI-compatible API.
The current deployment has a 32k-token context window.
"""

import os


class AmaliaConfig:
    """Configuration for AMALIA API extractor via langextract."""

    def __init__(
        self,
        base_url: str = "http://amalia.inesctec.pt:8000",
        model_id: str = None,
        temperature: float = 0.0,
        max_char_buffer: int = 20000,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: int = 10,
        max_output_tokens: int = 8192,
    ):
        """
        Initialize AMALIA configuration.

        Args:
            base_url: AMALIA API base URL (without /v1 suffix)
            model_id: Model ID (auto-detected from /v1/models if None)
            temperature: Sampling temperature (0.0 = deterministic)
            max_char_buffer: Max characters per chunk for langextract segmentation.
                             Default 20000 chars (~15k tokens) leaves headroom for
                             prompt + few-shot examples within AMALIA's 32k context.
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            max_output_tokens: Max output tokens. Input uses ~3200 tokens for the
                               largest documents, leaving ~29500 headroom in AMALIA's
                               32k context. 8192 lets the model finish naturally.
        """
        self.base_url = os.getenv("AMALIA_BASE_URL", base_url).rstrip("/")
        self.model_id = model_id
        self.temperature = temperature
        self.max_char_buffer = max_char_buffer
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_output_tokens = max_output_tokens

    def __repr__(self) -> str:
        return (
            f"AmaliaConfig(base_url={self.base_url}, model={self.model_id}, "
            f"temperature={self.temperature}, max_char_buffer={self.max_char_buffer})"
        )


DEFAULT_CONFIG = AmaliaConfig()
