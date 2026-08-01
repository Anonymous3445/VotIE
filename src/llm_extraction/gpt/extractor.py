"""
GPT-5.5 span extractor using langextract with a custom GptLanguageModel.

Uses the same prompts and few-shot examples as the other extractors — only
the underlying LM provider changes.
"""

import time
import logging
from typing import Optional

import langextract as lx
from langextract.prompt_validation import PromptValidationLevel

from ..shared.base_extractor import BaseSpanExtractor
from ..schemas import SpanEntity, ENTITY_TYPES
from ..span_alignment import align_extraction_result
from ..amalia.prompts import get_prompt_description
from ..fixed_examples import get_fixed_examples
from .config import GptConfig
from .langextract_provider import GptLanguageModel

logger = logging.getLogger(__name__)


class GptSpanExtractor(BaseSpanExtractor):
    """GPT-5.5 span extractor via langextract."""

    def __init__(
        self,
        config: Optional[GptConfig] = None,
        strategy: str = "zero_shot",
        num_examples: int = 5,
    ):
        """
        Args:
            config: GPT API configuration
            strategy: "zero_shot" or "few_shot"
            num_examples: Number of fixed few-shot examples (max 5)
        """
        self.config = config or GptConfig()

        super().__init__(model_id=self.config.model_id, strategy=strategy)

        self.prompt_description = get_prompt_description()
        # langextract 1.1.1 requires at least one example unconditionally.
        # For zero_shot we use 1 example as the minimum required scaffold.
        self.lx_examples = get_fixed_examples(num_examples if strategy == "few_shot" else 1)

        self._lm = GptLanguageModel(
            api_key=self.config.api_key,
            endpoint=self.config.endpoint,
            channel_id=self.config.channel_id,
            model_id=self.config.model_id,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            temperature=self.config.temperature,
            rate_limit_delay=self.config.rate_limit_delay,
            rate_limit_retries=self.config.rate_limit_retries,
            request_interval=self.config.request_interval,
        )

        logger.info(
            f"Initialized GptSpanExtractor with {self.config}, "
            f"strategy={strategy}, examples={len(self.lx_examples)}"
        )

    def _build_prompt(self, text: str, examples=None) -> str:
        """Not used directly — langextract handles prompt construction."""
        return self.prompt_description

    def extract(
        self,
        text: str,
        document_id: str,
        align_spans: bool = True,
    ):
        start_time = time.time()
        self._lm.usage.reset()

        try:
            logger.info(
                f"Extracting from {document_id} ({len(text)} chars, strategy={self.strategy})"
            )

            # fence_output=True: GPT-5.5 may wrap JSON in markdown fences;
            # langextract will strip them before parsing.
            annotated_doc = lx.extract(
                text_or_documents=text,
                prompt_description=self.prompt_description,
                examples=self.lx_examples,
                model=self._lm,
                max_char_buffer=self.config.max_char_buffer,
                fence_output=True,
                use_schema_constraints=False,
                show_progress=False,
                max_workers=1,
                extraction_passes=1,
                prompt_validation_level=PromptValidationLevel.OFF,
            )

            processing_time = time.time() - start_time
            result = self._convert_result(
                annotated_doc, document_id, text, processing_time
            )

            n_raw = len(annotated_doc.extractions) if annotated_doc.extractions else 0
            logger.info(
                f"  langextract returned {n_raw} extractions, "
                f"{len(result.entities)} valid entities ({processing_time:.1f}s)"
            )

            # Snapshot pre-alignment output so the span-discard rate is recoverable
            # from the saved predictions rather than only from stdout logs.
            usage = self._lm.usage.snapshot()
            diagnostics = {
                "n_raw_generated": n_raw,
                "n_after_type_filter": len(result.entities),
                "n_dropped_unknown_type": n_raw - len(result.entities),
                "pre_alignment_entities": [
                    {"text": e.text, "type": e.type} for e in result.entities
                ],
                "raw_generated": [
                    {"text": x.extraction_text, "type": x.extraction_class}
                    for x in (annotated_doc.extractions or [])
                ],
                "usage": usage,
                "gateway_reported_model": self._lm.reported_model,
            }
            # Generation time alone; result.processing_time also covers alignment.
            result.api_time = usage["api_seconds"]

            if align_spans:
                result, alignment_stats = align_extraction_result(result, strict=True)
                logger.info(
                    f"  Aligned {alignment_stats['aligned_spans']}/{alignment_stats['total_spans']} "
                    f"spans (fidelity={alignment_stats['alignment_rate']:.0%})"
                )
                diagnostics.update(
                    {
                        "n_before_alignment": alignment_stats["total_spans"],
                        "n_aligned": alignment_stats["aligned_spans"],
                        "n_dropped_alignment": alignment_stats["failed_spans"],
                        "alignment_rate": alignment_stats["alignment_rate"],
                        "failed_alignments": alignment_stats["failed_alignments"],
                    }
                )

            result.diagnostics = diagnostics
            return result

        except Exception as e:
            logger.error(f"Error extracting from {document_id}: {e}")
            processing_time = time.time() - start_time
            result = self._create_result(
                document_id=document_id,
                text=text,
                entities=[],
                processing_time=processing_time,
                error=str(e),
            )
            # A failed document still burns tokens; record them so cost is not
            # under-reported and the failure rate is auditable per document.
            result.diagnostics = {
                "document_failed": True,
                "usage": self._lm.usage.snapshot(),
            }
            return result

    def _convert_result(self, annotated_doc, document_id, text, processing_time):
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
