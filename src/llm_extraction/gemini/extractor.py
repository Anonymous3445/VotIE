"""
Gemini-based span extractor using response_json_schema.
"""

import time
import logging
import json
from typing import List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai is required for Gemini extractor. "
        "Install with: pip install google-genai"
    )

from ..shared.base_extractor import BaseSpanExtractor
from ..schemas import SpanEntity, SpanExtractionResult
from ..span_alignment import align_extraction_result
from .config import GeminiConfig, DEFAULT_CONFIG
from .prompts import build_zero_shot_prompt, build_few_shot_prompt


logger = logging.getLogger(__name__)


class GeminiSpanExtractor(BaseSpanExtractor):
    """Gemini-based span extractor with structured output."""

    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
        strategy: str = "zero_shot"
    ):
        """
        Initialize Gemini span extractor.

        Args:
            config: Gemini configuration (uses DEFAULT_CONFIG if None)
            strategy: Extraction strategy ("zero_shot" or "few_shot")
        """
        self.config = config or DEFAULT_CONFIG

        # Initialize parent
        super().__init__(model_id=self.config.model_id, strategy=strategy)

        # Create Gemini client (Vertex AI or AI Studio)
        if self.config.use_vertex_ai:
            logger.info(f"Using Vertex AI with project={self.config.project_id}, location={self.config.location}")
            self.client = genai.Client(
                vertexai=True,
                project=self.config.project_id,
                location=self.config.location
            )
        else:
            logger.info(f"Using AI Studio with API key")
            self.client = genai.Client(api_key=self.config.api_key)

        logger.info(f"Initialized GeminiSpanExtractor with {self.config}")

    def _get_json_schema(self) -> dict:
        """Get JSON schema for response validation (text-only, no positions)."""
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"}
                        },
                        "required": ["text", "type"]
                    }
                }
            },
            "required": ["entities"]
        }

    def _build_prompt(self, text: str, examples: Optional[List[dict]] = None) -> str:
        """Build prompt for Gemini."""
        if self.strategy == "few_shot" and examples:
            return build_few_shot_prompt(text, examples)
        else:
            return build_zero_shot_prompt(text)

    def extract(
        self,
        text: str,
        document_id: str,
        align_spans: bool = True
    ) -> SpanExtractionResult:
        """
        Extract spans from text using Gemini.

        Args:
            text: Source text to extract from
            document_id: Document identifier
            align_spans: Whether to align spans to source text

        Returns:
            SpanExtractionResult with extracted entities
        """
        start_time = time.time()

        try:
            # Build prompt
            if self.strategy == "few_shot" and self.few_shot_examples:
                prompt = self._build_prompt(text, self.few_shot_examples)
            else:
                prompt = self._build_prompt(text)

            logger.debug(f"Extracting from document {document_id} with {self.strategy} strategy")

            # Call Gemini API with retry logic (returns response and API time)
            response, api_time = self._call_with_retry(prompt)

            # Parse response
            entities = self._parse_response(response.text)

            # Create result
            processing_time = time.time() - start_time
            result = self._create_result(
                document_id=document_id,
                text=text,
                entities=entities,
                processing_time=processing_time,
                api_time=api_time,
                raw_response=response.text
            )

            # Align spans if requested
            if align_spans:
                result, alignment_stats = align_extraction_result(result, strict=True)
                logger.debug(
                    f"Aligned {alignment_stats['aligned_spans']}/{alignment_stats['total_spans']} spans "
                    f"(fidelity={alignment_stats['alignment_rate']:.2%})"
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
                error=str(e)
            )

    def _call_with_retry(self, prompt: str) -> tuple:
        """
        Call Gemini API with retry logic for quota errors.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (response object, api_time in seconds)

        Raises:
            Exception: If all retries fail
        """
        api_start = time.time()

        for attempt in range(self.config.max_retries):
            try:
                # Use new google-genai API
                call_start = time.time()
                response = self.client.models.generate_content(
                    model=self.config.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=self._get_json_schema(),
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens
                    )
                )
                call_time = time.time() - call_start
                total_api_time = time.time() - api_start

                logger.debug(f"API call completed in {call_time:.2f}s (total with retries: {total_api_time:.2f}s)")
                return response, total_api_time

            except Exception as e:
                error_msg = str(e).lower()

                # Check if it's a quota/rate limit error
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    if attempt < self.config.max_retries - 1:
                        logger.warning(
                            f"Quota/rate limit error (attempt {attempt + 1}/{self.config.max_retries}). "
                            f"Retrying in {self.config.retry_delay}s..."
                        )
                        time.sleep(self.config.retry_delay)
                        continue
                    else:
                        logger.error(f"Max retries reached for quota error")
                        raise

                # For other errors, raise immediately
                raise

        raise Exception("Failed after all retry attempts")

    def _parse_response(self, response_text: str) -> List[SpanEntity]:
        """
        Parse Gemini JSON response into span entities (without positions).

        Args:
            response_text: JSON response from Gemini

        Returns:
            List of SpanEntity objects (start/end will be None, filled by alignment)
        """
        try:
            # Parse JSON
            data = json.loads(response_text)

            # Extract entities (no positions from LLM)
            entities = []
            for entity_dict in data.get("entities", []):
                entity = SpanEntity(
                    text=entity_dict["text"],
                    type=entity_dict["type"],
                    start=None,  # Will be filled by alignment module
                    end=None     # Will be filled by alignment module
                )
                entities.append(entity)

            logger.debug(f"Parsed {len(entities)} entities from response (positions will be aligned)")
            return entities

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing response: {e}")
            logger.error(f"Response text (first 500 chars): {response_text[:500]}")
            logger.debug(f"Full response text: {response_text}")

            # Try to salvage partial JSON
            try:
                # Sometimes Gemini returns valid JSON with extra text - try to extract it
                # Look for the first { and last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned = response_text[start_idx:end_idx+1]
                    logger.warning("Attempting to parse cleaned JSON...")
                    data = json.loads(cleaned)
                    entities = []
                    for entity_dict in data.get("entities", []):
                        entity = SpanEntity(
                            text=entity_dict["text"],
                            type=entity_dict["type"],
                            start=None,
                            end=None
                        )
                        entities.append(entity)
                    logger.warning(f"Successfully recovered {len(entities)} entities from malformed response")
                    return entities
            except Exception as recovery_error:
                logger.debug(f"Recovery attempt failed: {recovery_error}")

            return []
