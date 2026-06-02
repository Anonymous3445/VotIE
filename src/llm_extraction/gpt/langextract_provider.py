"""
langextract-compatible language model backed by the GPT-5.5 streaming API (IAEDU endpoint).

SSE response format (one JSON per line, blank line separator):
  {"type": "start",   "content": "Processing"}
  {"type": "token",   "content": "<token>"}
  {"type": "message", "content": {"type": "ai", "content": "<full text>", ...}}
  {"type": "done",    ...}

We wait for the "message" chunk and return content.content as the full output.
Each call uses a fresh UUID thread_id to avoid conversation state bleeding.
"""

import json
import logging
import time
import uuid
from typing import Iterator, Sequence

import requests
from langextract.core import types as core_types
from langextract.core.base_model import BaseLanguageModel

logger = logging.getLogger(__name__)


class GptLanguageModel(BaseLanguageModel):
    """langextract LanguageModel that routes inference through the IAEDU API."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        channel_id: str,
        model_id: str = "gpt-5.5",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.endpoint = endpoint
        self.channel_id = channel_id
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Tell langextract's parse_output to use JSON
        self.format_type = core_types.FormatType.JSON

    def _call_once(self, prompt: str) -> str:
        """One API call; returns the full assistant response text."""
        thread_id = str(uuid.uuid4())

        files = {
            "channel_id": (None, self.channel_id),
            "thread_id":  (None, thread_id),
            "user_info":  (None, "{}"),
            "message":    (None, prompt),
        }

        resp = requests.post(
            self.endpoint,
            headers={"x-api-key": self.api_key},
            files=files,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if chunk.get("type") == "message":
                content = chunk.get("content", {})
                if isinstance(content, dict):
                    return content.get("content", "")
                break

        raise RuntimeError(
            f"GPT API returned no 'message' chunk (thread={thread_id})"
        )

    def _call_with_retry(self, prompt: str) -> str:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_once(prompt)
            except Exception as e:
                msg = str(e).lower()
                transient = any(k in msg for k in ("timeout", "timed out", "connection", "read"))
                if transient and attempt < self.max_retries:
                    logger.warning(
                        f"GPT attempt {attempt}/{self.max_retries} failed ({e}); "
                        f"retrying in {self.retry_delay}s"
                    )
                    last_exc = e
                    time.sleep(self.retry_delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[core_types.ScoredOutput]]:
        """Yield one [ScoredOutput] per prompt (sequential, no batching)."""
        for prompt in batch_prompts:
            text = self._call_with_retry(prompt)
            logger.debug(f"GPT response ({len(text)} chars): {text[:300]}")
            yield [core_types.ScoredOutput(score=None, output=text)]
