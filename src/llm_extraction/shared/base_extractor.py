"""
Abstract base class for span extractors.

Defines the interface that all span extractors (Gemini, LLaMA, etc.) must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

from ..schemas import SpanExtractionResult, SpanEntity


logger = logging.getLogger(__name__)


class BaseSpanExtractor(ABC):
    """Abstract base class for LLM span extractors."""

    def __init__(self, model_id: str, strategy: str = "zero_shot"):
        """
        Initialize the span extractor.

        Args:
            model_id: Identifier for the LLM model
            strategy: Extraction strategy ("zero_shot" or "few_shot")
        """
        if strategy not in ["zero_shot", "few_shot"]:
            raise ValueError(f"Invalid strategy: {strategy}. Must be 'zero_shot' or 'few_shot'")

        self.model_id = model_id
        self.strategy = strategy

        logger.info(f"Initialized {self.__class__.__name__} with model={model_id}, strategy={strategy}")

    @abstractmethod
    def extract(self, text: str, document_id: str) -> SpanExtractionResult:
        """
        Extract entity spans from text.

        Args:
            text: Source text to extract from
            document_id: Unique identifier for the document

        Returns:
            SpanExtractionResult with extracted entities

        Raises:
            Exception: If extraction fails
        """
        pass

    @abstractmethod
    def _build_prompt(self, text: str, examples: Optional[List[Dict]] = None) -> str:
        """
        Build the prompt for the LLM.

        Args:
            text: Source text to extract from
            examples: Optional few-shot examples

        Returns:
            Formatted prompt string
        """
        pass

    def _create_result(
        self,
        document_id: str,
        text: str,
        entities: List[SpanEntity],
        processing_time: float,
        api_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> SpanExtractionResult:
        """
        Create a SpanExtractionResult object.

        Args:
            document_id: Document identifier
            text: Source text
            entities: Extracted entities
            processing_time: Total time taken for extraction
            api_time: Time taken for API/LLM call only
            error: Optional error message

        Returns:
            SpanExtractionResult object
        """
        return SpanExtractionResult(
            id=document_id,
            text=text,
            entities=entities,
            model=self.model_id,
            strategy=self.strategy,
            processing_time=processing_time,
            api_time=api_time,
            error=error,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id}, strategy={self.strategy})"
