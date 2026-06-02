"""
AMALIA-based span extractor using langextract with OpenAI-compatible vLLM API.

Uses langextract for prompt construction, text segmentation, API calls,
and response parsing. Keeps span alignment post-processing.
"""

import json
import time
import logging
from typing import List, Optional

import requests
import langextract as lx
from langextract.core import data as lx_data
from langextract.providers.openai import OpenAILanguageModel
from langextract.prompt_validation import PromptValidationLevel

from ..shared.base_extractor import BaseSpanExtractor
from ..schemas import SpanEntity, SpanExtractionResult, ENTITY_TYPES
from ..span_alignment import align_extraction_result
from .prompts import get_prompt_description
from ..fixed_examples import get_fixed_examples
from .config import AmaliaConfig, DEFAULT_CONFIG


logger = logging.getLogger(__name__)


class AmaliaSpanExtractor(BaseSpanExtractor):
    """AMALIA-based span extractor via langextract + OpenAI provider."""

    def __init__(
        self,
        config: Optional[AmaliaConfig] = None,
        strategy: str = "zero_shot",
        num_examples: int = 5,
    ):
        """
        Args:
            config: AMALIA configuration
            strategy: "zero_shot" or "few_shot"
            num_examples: Number of fixed few-shot examples to use (max 5)
        """
        self.config = config or DEFAULT_CONFIG

        # Query server for model info and auto-detect model_id
        self._query_server()

        super().__init__(model_id=self.config.model_id, strategy=strategy)

        # Build langextract components
        self.prompt_description = get_prompt_description()
        # langextract 1.1.1 requires at least one example unconditionally.
        # For zero_shot we use 1 example as the minimum required scaffold.
        self.lx_examples = get_fixed_examples(num_examples if strategy == "few_shot" else 1)

        # Use stock OpenAI JSON mode: langextract injects
        # response_format={"type":"json_object"} which vLLM 0.8.2 routes
        # through xgrammar (fast constrained decoding, ~5s/example).
        # An earlier YAML workaround was introduced when AMALIA ran a pre-0.7
        # vLLM version where json_object triggered slow outlines (~300s);
        # that is no longer the case. Without a format constraint AMALIA
        # ignores the task and outputs plain Portuguese prose.
        # max_output_tokens caps completion to prevent runaway generation.
        self._lm = OpenAILanguageModel(
            model_id=self.config.model_id,
            api_key="not-needed",
            base_url=f"{self.config.base_url}/v1",
            temperature=self.config.temperature,
            format_type=lx_data.FormatType.JSON,
            max_output_tokens=self.config.max_output_tokens,
        )

        # langextract builds the openai.OpenAI client with read=600s and
        # max_retries=2, so a hung request blocks for up to 30 minutes
        # (3 attempts × 600s). Patch both so config.timeout is honoured.
        try:
            import openai as _openai
            # Positional arg sets the default for read/write/pool fields
            self._lm._client.timeout = _openai.Timeout(
                self.config.timeout, connect=10.0
            )
            self._lm._client.max_retries = 0  # no silent retries on timeout
            logger.info(
                f"OpenAI client timeout set to {self.config.timeout}s "
                f"(connect=10s, max_retries=0)"
            )
        except Exception as e:
            logger.warning(f"Could not set OpenAI client timeout: {e}")

        logger.info(
            f"Initialized AmaliaSpanExtractor with {self.config}, "
            f"strategy={strategy}, examples={len(self.lx_examples)}"
        )

    def _query_server(self):
        """Query the server for available models and context limit."""
        url = f"{self.config.base_url}/v1/models"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                raise RuntimeError("No models available on AMALIA server")
            model_info = data[0]
            if not self.config.model_id:
                self.config.model_id = model_info["id"]
            logger.info(f"Model: {self.config.model_id}")
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to AMALIA at {self.config.base_url}: {e}"
            )

    def _build_prompt(self, text: str, examples=None) -> str:
        """Not used directly — langextract handles prompt construction."""
        return self.prompt_description

    def _call_with_retry(self, text: str, document_id: str):
        """
        Call langextract with explicit retry on timeout / connection errors.

        AMALIA's vLLM endpoint occasionally hangs on queued requests; a fresh
        request usually succeeds. We keep per-attempt timeout at config.timeout
        and add up to config.max_retries follow-up attempts with a small delay.
        """
        last_exc = None
        for attempt in range(1, self.config.max_retries + 1):
            attempt_start = time.time()
            try:
                return lx.extract(
                    text_or_documents=text,
                    prompt_description=self.prompt_description,
                    examples=self.lx_examples,
                    model=self._lm,
                    max_char_buffer=self.config.max_char_buffer,
                    fence_output=False,  # JSON mode returns raw JSON, no fences
                    use_schema_constraints=False,
                    show_progress=False,
                    max_workers=1,
                    extraction_passes=1,
                    prompt_validation_level=PromptValidationLevel.OFF,
                )
            except Exception as e:
                elapsed = time.time() - attempt_start
                msg = str(e).lower()
                # Only retry on connection-level errors (network blip, reset).
                # Timeouts mean the server is hung — retrying the same request
                # will just time out again and waste config.timeout seconds.
                transient = (
                    "connection" in msg
                    or ("read" in msg and "504" in msg)
                )
                if transient and attempt < self.config.max_retries:
                    logger.warning(
                        f"  {document_id}: attempt {attempt}/{self.config.max_retries} "
                        f"failed after {elapsed:.0f}s ({e}); retrying in "
                        f"{self.config.retry_delay}s"
                    )
                    last_exc = e
                    time.sleep(self.config.retry_delay)
                    continue
                # Non-transient or final attempt: re-raise
                raise
        # Should be unreachable, but be explicit
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"AMALIA call failed for {document_id} after retries")

    def extract(
        self,
        text: str,
        document_id: str,
        align_spans: bool = True,
    ) -> SpanExtractionResult:
        """
        Extract spans from text using AMALIA via langextract.

        Args:
            text: Source text to extract from
            document_id: Document identifier
            align_spans: Whether to align spans to source text

        Returns:
            SpanExtractionResult with extracted entities
        """
        start_time = time.time()

        try:
            logger.info(f"Extracting from {document_id} ({len(text)} chars, strategy={self.strategy})")

            annotated_doc = self._call_with_retry(text, document_id)

            # Convert langextract result to SpanExtractionResult
            processing_time = time.time() - start_time
            result = self._convert_result(
                annotated_doc, document_id, text, processing_time
            )

            n_raw = len(annotated_doc.extractions) if annotated_doc.extractions else 0
            logger.info(
                f"  langextract returned {n_raw} extractions, "
                f"{len(result.entities)} valid entities ({processing_time:.1f}s)"
            )

            # Align spans if requested
            if align_spans:
                result, alignment_stats = align_extraction_result(
                    result, strict=True
                )
                logger.info(
                    f"  Aligned {alignment_stats['aligned_spans']}/{alignment_stats['total_spans']} spans "
                    f"(fidelity={alignment_stats['alignment_rate']:.0%})"
                )

            return result

        except Exception as e:
            err_msg = str(e)
            processing_time = time.time() - start_time

            # JSON truncation recovery: if the model hit max_output_tokens mid-JSON,
            # the partial output may contain completed entity objects we can salvage.
            if "unterminated" in err_msg.lower() or "truncat" in err_msg.lower():
                recovered = self._recover_truncated_entities(e)
                if recovered:
                    logger.warning(
                        f"  {document_id}: JSON truncated — recovered {len(recovered)} entities "
                        f"(increase --max-output-tokens if this is frequent)"
                    )
                    result = self._create_result(
                        document_id=document_id,
                        text=text,
                        entities=recovered,
                        processing_time=processing_time,
                        error=None,
                    )
                    if align_spans:
                        result, alignment_stats = align_extraction_result(result, strict=True)
                        logger.info(
                            f"  Aligned {alignment_stats['aligned_spans']}/{alignment_stats['total_spans']} spans "
                            f"(fidelity={alignment_stats['alignment_rate']:.0%})"
                        )
                    return result

            logger.error(f"Error extracting from {document_id}: {e}")
            return self._create_result(
                document_id=document_id,
                text=text,
                entities=[],
                processing_time=processing_time,
                error=err_msg,
            )

    def _recover_truncated_entities(self, exc: Exception) -> List[SpanEntity]:
        """Salvage entities from a truncated JSON string.

        langextract raises FormatParseError (message: "Failed to parse JSON
        content: ...") whose __cause__ is a json.JSONDecodeError whose .doc
        attribute IS the truncated JSON string that failed to parse.  We walk
        it with raw_decode to collect all complete entity dicts that appeared
        before the truncation point.
        """
        cause = getattr(exc, "__cause__", None)
        # json.JSONDecodeError.doc is the string that failed parsing
        raw_text: str = getattr(cause, "doc", None) or ""
        if not raw_text or "{" not in raw_text:
            return []

        entities: List[SpanEntity] = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(raw_text):
            try:
                idx = raw_text.index("{", pos)
            except ValueError:
                break
            try:
                obj, end = decoder.raw_decode(raw_text, idx)
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key in ENTITY_TYPES and isinstance(value, str) and value:
                            entities.append(
                                SpanEntity(text=value, type=key, start=None, end=None)
                            )
                pos = end
            except json.JSONDecodeError:
                pos = idx + 1
        return entities

    def _convert_result(
        self,
        annotated_doc: lx.data.AnnotatedDocument,
        document_id: str,
        text: str,
        processing_time: float,
    ) -> SpanExtractionResult:
        """Convert langextract AnnotatedDocument to SpanExtractionResult."""
        entities = []
        if annotated_doc.extractions:
            for extraction in annotated_doc.extractions:
                entity_type = extraction.extraction_class
                entity_text = extraction.extraction_text
                if entity_type in ENTITY_TYPES and entity_text:
                    entities.append(
                        SpanEntity(
                            text=entity_text,
                            type=entity_type,
                            start=None,
                            end=None,
                        )
                    )

        logger.info(f"  Converted {len(entities)} entities from langextract result")

        return self._create_result(
            document_id=document_id,
            text=text,
            entities=entities,
            processing_time=processing_time,
        )
