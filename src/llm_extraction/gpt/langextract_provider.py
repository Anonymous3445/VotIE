"""
langextract-compatible language model backed by the GPT-5.5 streaming API (IAEDU endpoint).

SSE response format (one JSON per line, blank line separator). Observed shapes,
logged from a live run rather than assumed:

  {"type": "start",   "run_id": ..., "content": "<str>"}
  {"type": "token",   "content": "<token>"}
  {"type": "message", "run_id": ..., "content": {"content": "<full text>",
       "response_metadata": {...}, "custom_data": ..., "tool_calls": ..., ...}}
  {"type": "done",    "run_id": ..., "messageId": ..., "content": "<str>"}
  {"type": "error",   "run_id": ..., "messageId": ..., "content": "<message>"}

The gateway signals failure with an "error" chunk over HTTP 200 — raise_for_status
cannot see it. Treating that as "no message chunk" hides the reason, so the error
content is surfaced verbatim.

We read the whole stream rather than returning at "message", because usage and
the terminal status arrive afterwards. Each call uses a fresh UUID thread_id to
avoid conversation state bleeding.
"""

import json
import logging
import re
import time
import uuid
from typing import Iterator, Optional, Sequence

import requests
from langextract.core import types as core_types
from langextract.core.base_model import BaseLanguageModel

from ..usage import UsageTracker, count_tokens

logger = logging.getLogger(__name__)


class GatewayError(RuntimeError):
    """The gateway returned an explicit error event over HTTP 200.

    `raise_for_status()` cannot see these — the HTTP status is 200 and the failure
    rides in the stream body. Rate limits are the common case and are retryable
    with backoff; anything else is assumed permanent, since replaying an identical
    request generally reproduces it.
    """

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _is_rate_limit(detail: str) -> bool:
    d = (detail or "").lower()
    return "rate limit" in d or "429" in d or "too many requests" in d


_LOGGED_SHAPES: set = set()


def _log_chunk_shape(chunk) -> None:
    """Log each distinct SSE chunk shape once, keys only, never content.

    The gateway is undocumented. It does disclose the model, so it may also
    disclose token usage under a name `_find_usage` does not look for — and
    billed tokens are worth far more than a local estimate, because GPT-5.5
    charges for reasoning tokens that cannot be counted client-side.
    Keys only: the payloads contain the prompts.
    """
    if not isinstance(chunk, dict):
        return
    kind = str(chunk.get("type", "?"))
    content = chunk.get("content")
    shape = (kind, tuple(sorted(chunk)),
             tuple(sorted(content)) if isinstance(content, dict) else type(content).__name__)
    if shape in _LOGGED_SHAPES or kind == "token":  # token chunks stream per-token
        return
    _LOGGED_SHAPES.add(shape)
    logger.info(f"SSE chunk type={kind!r} keys={shape[1]} content_keys={shape[2]}")


def _find_usage(chunk: dict) -> dict:
    """Locate a token-usage payload anywhere in an SSE chunk.

    The IAEDU gateway is a proxy, and where (or whether) it surfaces usage is not
    documented, so we search the chunk recursively for the conventional key names
    instead of assuming a position.
    """
    if isinstance(chunk, dict):
        for key in ("usage", "usage_metadata", "token_usage"):
            value = chunk.get(key)
            if isinstance(value, dict) and value:
                return value
        # LangChain-style gateways nest counts under response_metadata.
        meta = chunk.get("response_metadata")
        if isinstance(meta, dict) and meta:
            logger.debug(f"response_metadata keys: {sorted(meta)}")
        for value in chunk.values():
            found = _find_usage(value)
            if found:
                return found
    elif isinstance(chunk, list):
        for item in chunk:
            found = _find_usage(item)
            if found:
                return found
    return {}


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
        temperature: float = None,
        rate_limit_delay: int = 30,
        rate_limit_retries: int = 6,
        request_interval: float = 0.0,
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
        self.temperature = temperature
        # Rate limits arrive as HTTP-200 error events; back off rather than retry
        # immediately, and optionally pace requests to stay under the limit.
        self.rate_limit_delay = rate_limit_delay
        self.rate_limit_retries = rate_limit_retries
        self.request_interval = request_interval
        self._last_request_at = 0.0
        # Tell langextract's parse_output to use JSON
        self.format_type = core_types.FormatType.JSON
        # Populated per document by GptSpanExtractor. The gateway may not report
        # usage; we fall back to a local tiktoken count of prompt and response.
        self.usage = UsageTracker(source="api")
        # Whatever model string the gateway discloses, if any (see _capture_model).
        self.reported_model: Optional[str] = None

    def _call_once(self, prompt: str) -> str:
        """One API call; returns the full assistant response text."""
        # Client-side pacing, when a minimum gap between requests is configured.
        if self.request_interval > 0:
            wait = self._last_request_at + self.request_interval - time.time()
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.time()

        thread_id = str(uuid.uuid4())
        started = time.time()

        files = {
            "channel_id": (None, self.channel_id),
            "thread_id":  (None, thread_id),
            "user_info":  (None, "{}"),
            "message":    (None, prompt),
        }
        # Opt-in only: this gateway documents no sampling parameters, so sending
        # one is a probe. Without it the run uses the gateway default and is NOT
        # greedy — which is what the paper must then say.
        if self.temperature is not None:
            files["temperature"] = (None, str(self.temperature))

        resp = requests.post(
            self.endpoint,
            headers={"x-api-key": self.api_key},
            files=files,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        output = None
        reported_usage: dict = {}
        gateway_error = None

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            _log_chunk_shape(chunk)
            if not reported_usage:
                reported_usage = _find_usage(chunk)
            self._capture_model(chunk)

            if chunk.get("type") == "error":
                # HTTP 200 with an error event: raise_for_status cannot see this.
                gateway_error = chunk.get("content") or "<no detail>"
                continue

            if output is None and chunk.get("type") == "message":
                content = chunk.get("content", {})
                if isinstance(content, dict):
                    output = content.get("content", "")
                # Keep reading: usage and status arrive in later chunks.

        if output is None:
            if gateway_error is not None:
                raise GatewayError(
                    f"IAEDU gateway error: {gateway_error}",
                    retryable=_is_rate_limit(str(gateway_error)),
                )
            raise RuntimeError(
                f"GPT API returned no 'message' chunk and no error event "
                f"(thread={thread_id})"
            )

        self._record_usage(prompt, output, reported_usage, time.time() - started)
        return output

    def _capture_model(self, chunk) -> None:
        """Record any model identifier the gateway reveals.

        `model_id` is only a label we assign — it is never sent to this endpoint,
        which routes to whatever model the configured agent uses. If the stream
        names the model anywhere, that is the only evidence of what actually ran.
        """
        if self.reported_model or not isinstance(chunk, dict):
            return
        for key in ("model", "model_name", "model_id"):
            value = chunk.get(key)
            if isinstance(value, str) and value:
                self.reported_model = value
                logger.info(f"Gateway reported model: {value}")
                return
        for value in chunk.values():
            if isinstance(value, (dict, list)):
                self._capture_model(value if isinstance(value, dict) else {})

    def _record_usage(
        self, prompt: str, output: str, reported: dict, api_seconds: float
    ) -> None:
        """Record billed tokens, or a local estimate when the gateway hides them."""
        if reported:
            self.usage.record(
                input_tokens=reported.get("prompt_tokens") or reported.get("input_tokens") or 0,
                output_tokens=reported.get("completion_tokens") or reported.get("output_tokens") or 0,
                reasoning_tokens=(reported.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                ),
                api_seconds=api_seconds,
            )
            return

        n_in, n_out = count_tokens(prompt), count_tokens(output)
        if n_in is None or n_out is None:
            self.usage.record(has_usage=False, api_seconds=api_seconds)
            return
        # Marked estimated via the tracker's source field; the gateway bills
        # reasoning tokens we cannot see, so this is a lower bound.
        self.usage._source = "estimated:tiktoken/o200k_base"
        self.usage.record(input_tokens=n_in, output_tokens=n_out, api_seconds=api_seconds)

    def _call_with_retry(self, prompt: str) -> str:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_once(prompt)
            except GatewayError as e:
                if not e.retryable:
                    # The gateway answered and said no. Replaying an identical
                    # request reproduces it; fail fast instead of burning 3x the time.
                    logger.error(f"{e}")
                    raise
                if attempt >= self.rate_limit_retries:
                    logger.error(f"{e} — giving up after {attempt} attempts")
                    raise
                # Exponential backoff. A fixed short delay makes a rate limit
                # worse: it keeps hammering the endpoint that just refused us.
                delay = min(self.rate_limit_delay * 2 ** (attempt - 1), 600)
                logger.warning(
                    f"{e} — attempt {attempt}/{self.rate_limit_retries}, "
                    f"backing off {delay}s"
                )
                last_exc = e
                time.sleep(delay)
                continue
            except Exception as e:
                msg = str(e).lower()
                # Match whole words: "read" as a substring also matches "thread",
                # which appears in our own error text and made every permanent
                # failure look transient.
                transient = any(
                    re.search(rf"\b{k}\b", msg)
                    for k in ("timeout", "timed out", "connection", "read")
                )
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
