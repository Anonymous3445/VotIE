"""Configuration for GPT-5.5 span extraction via langextract."""

import os


class GptConfig:
    """Configuration for GPT-5.5 via the IAEDU streaming API."""

    ENDPOINT = "https://api.iaedu.pt/agent-chat//api/v1/agent/cmor5objoex9gfp01vm7p95jh/stream"
    CHANNEL_ID = "cmp2evpr4ewaglx01u8kbh5tn"
    MODEL_ID = "gpt-5.5"

    def __init__(
        self,
        api_key: str = None,
        endpoint: str = ENDPOINT,
        channel_id: str = CHANNEL_ID,
        model_id: str = MODEL_ID,
        max_char_buffer: int = 20000,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        self.api_key = api_key or os.environ.get("IAEDU_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GPT API key required. Set IAEDU_API_KEY env var or pass api_key."
            )
        self.endpoint = endpoint
        self.channel_id = channel_id
        self.model_id = model_id
        self.max_char_buffer = max_char_buffer
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def __repr__(self) -> str:
        return (
            f"GptConfig(model={self.model_id}, endpoint={self.endpoint}, "
            f"max_char_buffer={self.max_char_buffer})"
        )
