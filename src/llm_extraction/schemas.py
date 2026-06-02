"""
Pydantic schemas for LLM span extraction.

Defines the output format for text-to-structure span extraction,
where LLMs generate JSON with exact spans from source text.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SpanEntity(BaseModel):
    """A single entity span extracted from text.

    The LLM generates only text and type fields.
    The start/end positions are added during post-processing alignment.
    """

    text: str = Field(description="Exact character span from the input text (do not paraphrase)")
    type: str = Field(
        description=(
            "Entity type: VOTER-FAVOR, VOTER-AGAINST, VOTER-ABSTENTION, VOTER-ABSENT, "
            "VOTING, SUBJECT, COUNTING-UNANIMITY, COUNTING-MAJORITY, "
            "COUNT-FAVOR, COUNT-BLANK, or VOTING-METHOD"
        )
    )
    start: Optional[int] = Field(default=None, description="Character position (0-indexed, inclusive) - filled by alignment post-processing")
    end: Optional[int] = Field(default=None, description="Character position (exclusive) - filled by alignment post-processing")

    class Config:
        # Customize JSON schema to expose only text/type to the LLM with enum constraint
        json_schema_extra = {
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Exact character span from the input text (do not paraphrase)"
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "VOTER-FAVOR",
                        "VOTER-AGAINST",
                        "VOTER-ABSTENTION",
                        "VOTER-ABSENT",
                        "VOTING",
                        "SUBJECT",
                        "COUNTING-UNANIMITY",
                        "COUNTING-MAJORITY",
                        "COUNT-FAVOR",
                        "COUNT-BLANK",
                        "VOTING-METHOD",
                    ],
                    "description": "Entity type (must be one of the allowed values)"
                }
            },
            "required": ["text", "type"],
            "example": {
                "text": "o Executivo Municipal",
                "type": "VOTER-FAVOR"
            }
        }


class SpanExtractionOutput(BaseModel):
    """Output format for span extraction (used for JSON schema in LLM APIs)."""

    entities: List[SpanEntity] = Field(
        description="List of extracted entity spans",
        default_factory=list
    )

    class Config:
        json_schema_extra = {
            "example": {
                "entities": [
                    {"text": "o Executivo Municipal", "type": "VOTER-FAVOR", "start": 29, "end": 50},
                    {"text": "deliberou", "type": "VOTING", "start": 51, "end": 60}
                ]
            }
        }


class SpanExtractionResult(BaseModel):
    """Complete result of a span extraction operation."""

    id: str = Field(description="Document identifier")
    text: str = Field(description="Original source text")
    entities: List[SpanEntity] = Field(description="Extracted entities", default_factory=list)
    model: str = Field(description="Model identifier (e.g., 'gemini-2.5-flash')")
    strategy: str = Field(description="Extraction strategy: 'zero_shot' or 'few_shot'")
    processing_time: float = Field(description="Total processing time in seconds (includes prompt building, API call, parsing, alignment)")
    api_time: Optional[float] = Field(default=None, description="API/generation time in seconds (just the LLM call)")
    error: Optional[str] = Field(default=None, description="Error message if extraction failed")
    raw_response: Optional[str] = Field(default=None, description="Raw LLM response for debugging")

    @property
    def success(self) -> bool:
        """Check if extraction was successful."""
        return self.error is None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "M01_cm_002_2023-01-18_seg008",
                "text": "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade...",
                "entities": [
                    {"text": "o Executivo Municipal", "type": "VOTER-FAVOR", "start": 29, "end": 50},
                    {"text": "deliberou", "type": "VOTING", "start": 51, "end": 60}
                ],
                "model": "gemini-2.5-flash",
                "strategy": "few_shot",
                "processing_time": 1.23,
                "api_time": 0.87,
                "error": None,
                "raw_response": None
            }
        }


# Valid entity types for VotIE task (aligned with citilink-votie annotation scheme)
ENTITY_TYPES = [
    "VOTER-FAVOR",
    "VOTER-AGAINST",
    "VOTER-ABSTENTION",
    "VOTER-ABSENT",
    "VOTING",
    "SUBJECT",
    "COUNTING-UNANIMITY",
    "COUNTING-MAJORITY",
    "COUNT-FAVOR",
    "COUNT-BLANK",
    "VOTING-METHOD",
]


def validate_entity_type(entity_type: str) -> bool:
    """Validate that an entity type is one of the allowed types."""
    return entity_type in ENTITY_TYPES
