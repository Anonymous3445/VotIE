"""
Gemini-based span extractor using langextract with GeminiLanguageModel.

Mirrors the AmaliaSpanExtractor flow but wired to Gemini via langextract,
enabling a fair comparison of langextract outputs across providers.
"""

import time
import logging
from typing import Optional

import langextract as lx
from langextract.providers.gemini import GeminiLanguageModel
from langextract.prompt_validation import PromptValidationLevel

from ..shared.base_extractor import BaseSpanExtractor
from ..schemas import SpanEntity, ENTITY_TYPES
from ..span_alignment import align_extraction_result
from ..amalia.prompts import get_prompt_description
from ..fixed_examples import get_fixed_examples
from .config import GeminiConfig


logger = logging.getLogger(__name__)


class GeminiSpanExtractor(BaseSpanExtractor):
    """Gemini-based span extractor via langextract."""

    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
        strategy: str = "zero_shot",
        num_examples: int = 4,
    ):
        """
        Args:
            config: Gemini configuration
            strategy: "zero_shot" or "few_shot"
            num_examples: Number of fixed few-shot examples to use (max 4)
        """
        self.config = config or GeminiConfig()

        super().__init__(model_id=self.config.model_id, strategy=strategy)

        self.prompt_description = get_prompt_description()
        # langextract 1.1.1 requires at least one example unconditionally.
        # For zero_shot we use 1 example as the minimum required scaffold.
        self.lx_examples = get_fixed_examples(num_examples if strategy == "few_shot" else 1)

        # response_mime_type forces JSON output at the API level; without it,
        # GeminiLanguageModel only sets this when a gemini_schema is attached,
        # which can silently produce empty responses when schema constraints fail.
        self._lm = GeminiLanguageModel(
            model_id=self.config.model_id,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            response_mime_type="application/json",
        )

        logger.info(
            f"Initialized GeminiSpanExtractor with {self.config}, "
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
        """
        Extract spans from text using Gemini via langextract.

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

            annotated_doc = lx.extract(
                text_or_documents=text,
                prompt_description=self.prompt_description,
                examples=self.lx_examples,
                model=self._lm,
                max_char_buffer=self.config.max_char_buffer,
                fence_output=False,
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
            logger.error(f"Error extracting from {document_id}: {e}")
            processing_time = time.time() - start_time
            return self._create_result(
                document_id=document_id,
                text=text,
                entities=[],
                processing_time=processing_time,
                error=str(e),
            )

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
