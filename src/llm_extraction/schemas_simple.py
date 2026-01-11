"""
Simplified schemas without position fields for LLM extraction.
Positions will be added during post-processing alignment.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SimpleSpanEntity(BaseModel):
    """Entity with ONLY text and type (no positions)."""

    text: str = Field(description="Exact text span from input (character-for-character copy)")
    type: str = Field(description="Entity type: VOTER-FAVOR, VOTER-AGAINST, VOTER-ABSTENTION, VOTER-ABSENT, VOTING, SUBJECT, COUNTING-UNANIMITY, COUNTING-MAJORITY")


class SimpleSpanExtractionOutput(BaseModel):
    """Output format with simplified entities (positions added later)."""

    entities: List[SimpleSpanEntity] = Field(
        description="List of extracted entities (text and type only)",
        default_factory=list
    )


# For compatibility, create a function to convert to full SpanEntity format
def add_positions_to_entities(simple_output: SimpleSpanExtractionOutput, source_text: str):
    """Convert SimpleSpanEntity to full SpanEntity with position alignment."""
    from .schemas import SpanEntity
    from .span_alignment import find_span_in_text

    entities_with_positions = []

    for simple_entity in simple_output.entities:
        # Find positions in source text
        result = find_span_in_text(source_text, simple_entity.text)

        if result:
            start, end = result
            entity = SpanEntity(
                text=simple_entity.text,
                type=simple_entity.type,
                start=start,
                end=end
            )
        else:
            # No positions found - still include but with None
            entity = SpanEntity(
                text=simple_entity.text,
                type=simple_entity.type,
                start=None,
                end=None
            )

        entities_with_positions.append(entity)

    return entities_with_positions
