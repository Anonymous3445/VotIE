"""Configuration for GPT-5.5 span extraction via langextract."""

import os


class GptConfig:
    """Configuration for GPT-5.5 via the IAEDU streaming API."""

    # Verified against the live service: the /agent-chat prefix is required
    # (dropping it returns 404 "Resource not found") and the agent id is validated
    # server-side (a bogus id returns 404 "Agent '...' not found"). An earlier form
    # of this URL had a doubled slash after agent-chat; the server normalises it, so
    # both spellings behave identically — this is the tidy one.
    AGENT_ID = "cmor5objoex9gfp01vm7p95jh"
    ENDPOINT = f"https://api.iaedu.pt/agent-chat/api/v1/agent/{AGENT_ID}/stream"
    CHANNEL_ID = "cmp2evpr4ewaglx01u8kbh5tn"
    # A label only. The agent decides the model; see the note in __init__.
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
        temperature: float = None,
        rate_limit_delay: int = 30,
        rate_limit_retries: int = 6,
        request_interval: float = 0.0,
    ):
        """
        Note on `model_id` and `temperature`:

        This endpoint is an agent gateway, not the OpenAI API. It accepts only
        `channel_id`, `thread_id`, `user_info` and `message`; the model is fixed
        by the configured agent. So `model_id` is a label recorded in the output,
        NOT a routing instruction — it cannot be used to select a model, and it is
        not evidence of which model served the request.

        `temperature` is therefore off by default: whether the gateway honours or
        rejects an extra field is untested. Probe with a 2-document run before
        enabling it, and if the gateway ignores it, say so in the paper rather
        than claiming greedy decoding.
        """
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
        self.temperature = temperature
        # The gateway returns 'Rate limit reached (429)' as an HTTP-200 error event.
        # Back off exponentially from rate_limit_delay; request_interval paces
        # requests client-side to stay under the limit in the first place.
        self.rate_limit_delay = rate_limit_delay
        self.rate_limit_retries = rate_limit_retries
        self.request_interval = request_interval

    def __repr__(self) -> str:
        return (
            f"GptConfig(model={self.model_id}, endpoint={self.endpoint}, "
            f"max_char_buffer={self.max_char_buffer}, "
            f"temperature={self.temperature if self.temperature is not None else 'gateway default'})"
        )
