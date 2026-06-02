"""
vLLM-friendly OpenAI provider subclass for langextract.

Three problems with the stock OpenAILanguageModel for AMALIA:

1. ``response_format={"type":"json_object"}`` triggers vLLM grammar-constrained
   token sampling (~300s stall per request). We skip that injection entirely.

2. AMALIA ignores langextract's Q&A completion convention ("A: ") and replies
   in natural-language Portuguese. Fix: assistant prefill — we add a partial
   assistant turn ``{"role":"assistant","content":"```json\\n"}`` so vLLM
   continues generation from that prefix, forcing JSON output.

3. Even with prefill, AMALIA writes a Portuguese explanation first (inside the
   ```json fence) and then appends the valid JSON in a second nested block.
   langextract's non-greedy fence regex only captures the first (bad) block.
   Fix: scan the raw response for the first valid JSON object with an
   "extractions" key using json.JSONDecoder.raw_decode, then re-wrap it in a
   clean single fence so the resolver can parse it unambiguously.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langextract.core import data, exceptions
from langextract.core import types as core_types
from langextract.providers.openai import OpenAILanguageModel

logger = logging.getLogger(__name__)

_JSON_PREFILL = "```json\n"


def _extract_extractions_json(text: str) -> str | None:
    """Return a clean fenced JSON string for the first valid extractions object.

    Scans ``text`` character-by-character for an opening ``{``, then attempts
    ``json.JSONDecoder.raw_decode`` at that position.  Returns the first dict
    that contains an ``"extractions"`` key, wrapped in a single ``\\`\\`\\`json``
    fence so langextract's resolver can parse it unambiguously.  Returns None if
    no valid object is found.
    """
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        try:
            idx = text.index("{", pos)
        except ValueError:
            break
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and "extractions" in obj:
                clean = json.dumps(obj, ensure_ascii=False)
                return f"```json\n{clean}\n```"
        except json.JSONDecodeError:
            pass
        pos = idx + 1
    return None


class VLLMOpenAILanguageModel(OpenAILanguageModel):
    """OpenAI-compatible provider for vLLM using assistant prefill for JSON."""

    @property
    def requires_fence_output(self) -> bool:
        return True

    def _process_single_prompt(
        self, prompt: str, config: dict
    ) -> core_types.ScoredOutput:
        try:
            normalized = self._normalize_reasoning_params(config)

            # Direct Portuguese instruction: AMALIA must output ONLY JSON.
            if self.format_type == data.FormatType.JSON:
                system_message = (
                    "Assistente de extração de entidades de votação. "
                    "RESPONDE APENAS com o JSON pedido. "
                    "PROIBIDO escrever texto antes, dentro ou depois do JSON. "
                    "PROIBIDO pedir mais informação. "
                    'Se não há entidades, devolve {"extractions": []}.'
                )
            elif self.format_type == data.FormatType.YAML:
                system_message = (
                    "You are a helpful assistant that responds in YAML format."
                )
            else:
                system_message = ""

            messages: list[dict[str, str]] = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # Assistant prefill: force vLLM to continue generation from the
            # start of a JSON fence instead of writing explanatory prose.
            use_prefill = self.format_type == data.FormatType.JSON
            if use_prefill:
                messages.append({"role": "assistant", "content": _JSON_PREFILL})

            logger.info(
                "AMALIA prompt (sys=%r, user_tail=%r)",
                system_message[:80] if system_message else "",
                prompt[-200:],
            )

            api_params: dict[str, Any] = {
                "model": self.model_id,
                "messages": messages,
                "n": 1,
            }

            temp = normalized.get("temperature", self.temperature)
            if temp is not None:
                api_params["temperature"] = temp
            if (v := normalized.get("max_output_tokens")) is not None:
                api_params["max_tokens"] = v
            if (v := normalized.get("top_p")) is not None:
                api_params["top_p"] = v
            for key in (
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "stop",
                "logprobs",
                "top_logprobs",
                "reasoning",
            ):
                if (v := normalized.get(key)) is not None:
                    api_params[key] = v
            # Deliberately NOT forwarding `response_format` — vLLM treats it
            # as guided decoding and stalls.

            response = self._client.chat.completions.create(**api_params)
            raw = response.choices[0].message.content or ""

            # Prepend the prefill so the scanner sees the complete response.
            if use_prefill and not raw.startswith(_JSON_PREFILL):
                full_raw = _JSON_PREFILL + raw
            else:
                full_raw = raw

            logger.info(
                "AMALIA raw output (%d chars): %r", len(full_raw), full_raw[:400]
            )

            # Robust post-processing: find the first valid {"extractions": [...]}
            # JSON object anywhere in the response (AMALIA may prepend Portuguese
            # explanation text even inside the ```json fence) and return it as a
            # single clean fenced block for the resolver.
            clean = _extract_extractions_json(full_raw)
            if clean is not None:
                logger.info("AMALIA extracted clean JSON: %r", clean[:200])
                return core_types.ScoredOutput(score=1.0, output=clean)

            # Fallback: return the raw prefill+response and let the resolver try.
            logger.warning(
                "AMALIA: no 'extractions' JSON found in response, passing raw output"
            )
            return core_types.ScoredOutput(score=1.0, output=full_raw)

        except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f"vLLM OpenAI error: {e}", original=e
            ) from e
