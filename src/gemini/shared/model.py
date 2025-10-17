"""
Gemini 2.0 Flash Model for Portuguese Voting Event Extraction

Uses direct Gemini API calls with JSON mode for structured output.
"""

import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("google-generativeai not installed. Install with: pip install google-generativeai")

logger = logging.getLogger(__name__)


@dataclass
class EventExtractionResult:
    """Result container for event extraction"""
    document_id: str
    text: str
    has_voting_event: bool
    event: Optional[Dict[str, Any]]
    processing_time: float = 0.0
    model_id: str = "gemini-2.0-flash-exp"
    error: Optional[str] = None
    raw_response: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class GeminiEventExtractor:
    """Gemini 2.0 Flash model for voting event extraction with JSON output"""

    def __init__(self, model_id: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        """
        Initialize Gemini model for event extraction

        Args:
            model_id: Gemini model ID
            api_key: Google API key (or set GOOGLE_API_KEY env var)
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")

        self.model_id = model_id
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('LANGEXTRACT_API_KEY')

        if not self.api_key:
            raise ValueError("No API key found. Set GOOGLE_API_KEY or LANGEXTRACT_API_KEY environment variable.")

        # Configure genai
        genai.configure(api_key=self.api_key)

        # Create model with JSON response format
        self.model = genai.GenerativeModel(
            model_name=model_id,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0  # Deterministic output
            }
        )

        logger.info(f"✓ Initialized Gemini {model_id} for event extraction with JSON output")

    def build_prompt(self, text: str, system_prompt: str, examples: List[Dict[str, Any]] = None) -> str:
        """
        Build few-shot prompt for event extraction

        Args:
            text: Input text to extract from
            system_prompt: System instructions
            examples: Few-shot examples (optional)

        Returns:
            Complete prompt string
        """
        prompt_parts = [system_prompt]

        if examples:
            prompt_parts.append("\n\n# Examples\n")

            for i, example in enumerate(examples[:5], 1):  # Use first 5 examples to save tokens
                example_text = example.get('text', '').strip()
                has_event = example.get('has_voting_event', False)

                if not has_event:
                    output = {"has_voting_event": False, "event": None}
                else:
                    event_data = example.get('event', {})

                    # Use hybrid format directly (subject singular, others lists)
                    subject = event_data.get('subject')
                    if subject is None:
                        # Handle old format where subject might be in 'subjects' list
                        subjects_list = event_data.get('subjects', [])
                        subject = subjects_list[0] if subjects_list else None

                    voting_expressions = event_data.get('voting_expressions', [])
                    if not voting_expressions:
                        # Handle old format where it might be singular 'voting_expression'
                        single_expr = event_data.get('voting_expression')
                        voting_expressions = [single_expr] if single_expr else []

                    counting = event_data.get('counting', [])
                    if counting and not isinstance(counting, list):
                        # Handle old format where counting might be a dict
                        counting = [counting]

                    participants = event_data.get('participants', [])

                    output = {
                        "has_voting_event": True,
                        "event": {
                            "subject": subject,
                            "participants": participants,
                            "counting": counting,
                            "voting_expressions": voting_expressions
                        }
                    }

                # Truncate long example texts
                truncated_text = example_text[:400] + "..." if len(example_text) > 400 else example_text

                prompt_parts.append(f"\n## Example {i}\n")
                prompt_parts.append(f"Input: {truncated_text}\n")
                prompt_parts.append(f"Output: {json.dumps(output, ensure_ascii=False)}\n")

        # Add the actual task
        prompt_parts.append("\n\n# Your Task\n")
        prompt_parts.append("Extract the voting event from the following text and return ONLY the JSON object:\n\n")
        prompt_parts.append(f"Input: {text}\n")
        prompt_parts.append("\nOutput:")

        return "\n".join(prompt_parts)

    def extract(self,
                text: str,
                system_prompt: str,
                examples: List[Dict[str, Any]] = None,
                document_id: str = "unknown",
                max_retries: int = 3,
                retry_delay: int = 60) -> EventExtractionResult:
        """
        Extract voting event from text

        Args:
            text: Document text
            system_prompt: System instructions
            examples: Few-shot examples
            document_id: Document identifier
            max_retries: Retry attempts for quota errors
            retry_delay: Delay between retries (seconds)

        Returns:
            EventExtractionResult with event or error
        """
        start_time = time.time()

        # Build prompt
        prompt = self.build_prompt(text, system_prompt, examples)

        for attempt in range(max_retries + 1):
            try:
                # Generate response
                response = self.model.generate_content(prompt)

                # Extract JSON from response
                try:
                    # Get the text response
                    response_text = response.text.strip()

                    # Parse JSON
                    result_json = json.loads(response_text)

                    # Validate structure
                    if not isinstance(result_json, dict) or 'has_voting_event' not in result_json:
                        raise ValueError(f"Invalid response structure: {response_text[:200]}")

                    has_event = result_json.get('has_voting_event', False)
                    event = result_json.get('event', None)

                    processing_time = time.time() - start_time
                    logger.info(f"✓ Extracted event from {document_id} (has_event={has_event})")

                    return EventExtractionResult(
                        document_id=document_id,
                        text=text,
                        has_voting_event=has_event,
                        event=event,
                        processing_time=processing_time,
                        model_id=self.model_id,
                        raw_response=response_text
                    )

                except (json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Failed to parse JSON response for {document_id}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(f"Response: {response.text[:500]}")

                    processing_time = time.time() - start_time
                    return EventExtractionResult(
                        document_id=document_id,
                        text=text,
                        has_voting_event=False,
                        event=None,
                        processing_time=processing_time,
                        model_id=self.model_id,
                        error=error_msg,
                        raw_response=response.text[:1000]
                    )

            except Exception as e:
                error_str = str(e).lower()

                # Check for quota/rate limit errors
                is_quota_error = any(pattern in error_str for pattern in [
                    'quota', 'rate limit', 'rate_limit_exceeded',
                    'resource_exhausted', 'resource has been exhausted', '429'
                ])

                # Retry on quota errors
                if is_quota_error and attempt < max_retries:
                    logger.warning(f"Quota limit for {document_id}. Waiting {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue

                # Failed - return error
                processing_time = time.time() - start_time
                error_msg = f"Extraction failed for {document_id}: {str(e)}"
                logger.error(error_msg)

                return EventExtractionResult(
                    document_id=document_id,
                    text=text,
                    has_voting_event=False,
                    event=None,
                    processing_time=processing_time,
                    model_id=self.model_id,
                    error=error_msg
                )

        # Max retries exceeded
        processing_time = time.time() - start_time
        error_msg = f"Max retries exceeded for {document_id}"
        logger.error(error_msg)

        return EventExtractionResult(
            document_id=document_id,
            text=text,
            has_voting_event=False,
            event=None,
            processing_time=processing_time,
            model_id=self.model_id,
            error=error_msg
        )
