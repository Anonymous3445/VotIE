"""Token-usage accounting for the generative extractors.

langextract's ``AnnotatedDocument`` carries only ``extractions`` and ``text`` —
it discards the provider response object, and with it the token counts. The
usage therefore has to be captured one level down, at the language-model client,
and accumulated per document.

Each extractor owns a :class:`UsageTracker`, resets it before ``lx.extract`` and
snapshots it afterwards into ``result.diagnostics['usage']``. Because langextract
chunks long documents, one document may issue several API calls; ``api_calls``
records how many, which is what makes the per-document cost interpretable.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class UsageTracker:
    """Thread-safe per-document accumulator of provider token counts and API time."""

    def __init__(self, source: str = "unavailable"):
        self._lock = threading.Lock()
        self._source = source
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.api_calls = 0
            self.calls_with_usage = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.reasoning_tokens = 0
            self.cached_input_tokens = 0
            self.api_seconds = 0.0

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        cached_input_tokens: int = 0,
        has_usage: bool = True,
        api_seconds: float = 0.0,
    ) -> None:
        with self._lock:
            self.api_calls += 1
            if has_usage:
                self.calls_with_usage += 1
            self.input_tokens += input_tokens or 0
            self.output_tokens += output_tokens or 0
            self.reasoning_tokens += reasoning_tokens or 0
            self.cached_input_tokens += cached_input_tokens or 0
            self.api_seconds += api_seconds or 0.0

    @contextmanager
    def timed_call(self):
        """Time one provider call, yielding a dict the caller fills with counts.

        Separating API time from total processing time lets Appendix C report
        generation latency apart from the alignment post-processing we add on top.
        """
        started = time.time()
        fields: Dict[str, Any] = {}
        try:
            yield fields
        finally:
            self.record(api_seconds=time.time() - started, **fields)

    def snapshot(self) -> Dict[str, Any]:
        """Return the accumulated counts for the document just processed.

        ``billable_output_tokens`` folds in reasoning/"thoughts" tokens, which
        thinking models bill at the output rate but report separately.
        """
        with self._lock:
            return {
                "source": self._source if self.calls_with_usage else "unavailable",
                "api_calls": self.api_calls,
                "calls_with_usage": self.calls_with_usage,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "billable_output_tokens": self.output_tokens + self.reasoning_tokens,
                "api_seconds": round(self.api_seconds, 3),
            }


def _as_int(obj: Any, *names: str) -> int:
    """First non-None integer among ``names``, read from an object or a dict."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return int(value)
    return 0


def attach_gemini_usage(lm) -> UsageTracker:
    """Wrap a langextract ``GeminiLanguageModel`` so token usage is recorded.

    ``_process_single_prompt`` drops the ``GenerateContentResponse`` after reading
    ``.text``, so we wrap the client call instead of the langextract method — that
    keeps langextract's schema/kwargs handling untouched.
    """
    tracker = UsageTracker(source="api")

    try:
        models = lm._client.models
        original = models.generate_content
    except AttributeError:  # pragma: no cover - provider internals changed
        logger.warning("Could not attach usage tracking to Gemini client; costs will not be recorded")
        return tracker

    def generate_content(*args, **kwargs):
        with tracker.timed_call() as fields:
            response = original(*args, **kwargs)
            meta = getattr(response, "usage_metadata", None)
            if meta is None:
                fields["has_usage"] = False
            else:
                fields.update(
                    input_tokens=_as_int(meta, "prompt_token_count"),
                    output_tokens=_as_int(meta, "candidates_token_count"),
                    # Gemini 2.5 Pro is a thinking model: thoughts are billed as
                    # output but reported in their own field.
                    reasoning_tokens=_as_int(meta, "thoughts_token_count"),
                    cached_input_tokens=_as_int(meta, "cached_content_token_count"),
                )
        return response

    models.generate_content = generate_content
    return tracker


def attach_openai_usage(lm) -> UsageTracker:
    """Wrap a langextract ``OpenAILanguageModel`` (used for AMALIA via vLLM)."""
    tracker = UsageTracker(source="api")

    try:
        completions = lm._client.chat.completions
        original = completions.create
    except AttributeError:  # pragma: no cover - provider internals changed
        logger.warning("Could not attach usage tracking to OpenAI client; costs will not be recorded")
        return tracker

    def create(*args, **kwargs):
        with tracker.timed_call() as fields:
            response = original(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is None:
                fields["has_usage"] = False
            else:
                details = getattr(usage, "completion_tokens_details", None)
                fields.update(
                    input_tokens=_as_int(usage, "prompt_tokens", "input_tokens"),
                    output_tokens=_as_int(usage, "completion_tokens", "output_tokens"),
                    reasoning_tokens=_as_int(details, "reasoning_tokens") if details else 0,
                )
        return response

    completions.create = create
    return tracker


_ENCODING = None
_ENCODING_FAILED = False


def count_tokens(text: str, encoding_name: str = "o200k_base") -> Optional[int]:
    """Local token count, for providers whose API hides usage.

    Returns ``None`` when tiktoken is unavailable or its vocabulary cannot be
    fetched, so callers can record "unavailable" rather than a fabricated zero.
    """
    global _ENCODING, _ENCODING_FAILED
    if _ENCODING_FAILED:
        return None
    if _ENCODING is None:
        try:
            import tiktoken

            _ENCODING = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"tiktoken unavailable ({e}); token counts will not be estimated")
            _ENCODING_FAILED = True
            return None
    return len(_ENCODING.encode(text or ""))
