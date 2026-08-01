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
                             Kept at 20000 across all three providers so the
                             chunking regime is identical and the paradigm
                             comparison stays controlled — NOT because AMALIA
                             needs it. Measured, not assumed: document text runs
                             ~3.56 chars/token and the prompt plus five examples
                             costs a fixed ~2,340 tokens, so a 20000-char chunk is
                             ~5,620 + 2,340 = ~7,960 input tokens. With 8192
                             reserved for output that is ~16,150 of the 32,768
                             window, i.e. 49% utilisation. (An earlier note here
                             claimed "~15k tokens", which assumed roughly 1.3
                             chars/token — about 3x off.)
            timeout: Request timeout in seconds. 120 was too low: it produced 49
                     of the 58 zero-shot failures in the published run, all at
                     exactly 120.1s, which measured the client rather than the model.
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            max_output_tokens: Max output tokens per chunk. Acts as a circuit
                               breaker on runaway generation, not as a budget.
                               Measured against gold, a faithful extraction needs a
                               median of 59 output tokens and 252 in the worst case
                               (11 spans, 9,455 chars), so 8192 is ~33x the hardest
                               legitimate need. When AMALIA hits it, the JSON has run
                               past line 1,000 for documents whose answer is ~11
                               entities — a generation loop. 33 of the 89 few-shot
                               failures are malformed JSON rather than truncation, so
                               more room could not fix them either. Raising this
                               would spend GPU time letting the loop run longer.
                               There is space if it were ever wanted: a 20000-char
                               chunk leaves ~24,800 tokens of output room.
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
