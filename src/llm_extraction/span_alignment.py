"""
Span alignment module for exact match validation.

This module handles aligning generated spans back to source text with character-level precision.
Non-verbatim matches are counted as errors.
"""

import logging
from typing import Tuple, Optional, List, Dict, Any
import unicodedata

from .schemas import SpanEntity, SpanExtractionResult


logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for matching (remove extra whitespace, normalize unicode).

    Offsets into the result do NOT correspond to offsets into the input — use
    :func:`collapse_whitespace_with_offsets` when the position is needed.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Normalize unicode (NFC form)
    text = unicodedata.normalize('NFC', text)
    # Collapse multiple spaces into one
    text = ' '.join(text.split())
    return text


def collapse_whitespace_with_offsets(text: str) -> Tuple[str, List[int]]:
    """Collapse whitespace runs to a single space, keeping a map back to `text`.

    Returns ``(collapsed, offsets)`` where ``offsets[i]`` is the index in the
    original ``text`` of ``collapsed[i]``. This is what makes whitespace-tolerant
    matching usable: a match found in the collapsed string can be translated back
    into real character positions, which a plain normalise-then-search cannot do.

    Unicode normalisation is deliberately excluded — NFC can change string length,
    which would break the mapping. Case and unicode differences are handled by
    :func:`spans_equivalent` at comparison time instead.
    """
    out: List[str] = []
    offsets: List[int] = []
    i, n = 0, len(text)

    while i < n:
        if text[i].isspace():
            run_start = i
            while i < n and text[i].isspace():
                i += 1
            if out:  # never emit a leading space
                out.append(' ')
                offsets.append(run_start)
        else:
            out.append(text[i])
            offsets.append(i)
            i += 1

    while out and out[-1] == ' ':  # never keep a trailing space
        out.pop()
        offsets.pop()

    return ''.join(out), offsets


def spans_equivalent(a: str, b: str) -> bool:
    """True if two strings differ only in case, unicode form, or whitespace runs.

    Used to decide whether a source slice is the same span the model emitted.
    A model that renders a line break inside a long SUBJECT as a single space is
    still pointing at the same span; it should be realigned, not discarded.
    """
    return normalize_text(a).casefold() == normalize_text(b).casefold()


def find_all_occurrences(source_text: str, span_text: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of span_text in source_text.

    Args:
        source_text: Original source text
        span_text: Text to find

    Returns:
        List of (start, end) tuples for all occurrences
    """
    occurrences = []
    start = 0

    while True:
        try:
            pos = source_text.index(span_text, start)
            occurrences.append((pos, pos + len(span_text)))
            start = pos + 1
        except ValueError:
            break

    return occurrences


def find_span_in_text(
    source_text: str,
    span_text: str,
    suggested_start: Optional[int] = None,
    window_size: int = 100,
    voting_positions: Optional[List[Tuple[int, int]]] = None,
    max_distance: int = 500,
) -> Optional[Tuple[int, int]]:
    """
    Find the exact position of a span in the source text.

    If multiple occurrences exist, uses proximity to VOTING entities to choose
    the most likely occurrence (matching annotation guidelines).

    Args:
        source_text: Original source text
        span_text: Text to find
        suggested_start: Optional suggested starting position (from LLM output)
        window_size: Window size around suggested_start to search first
        voting_positions: List of (start, end) positions of VOTING entities
        max_distance: Maximum distance from VOTING entity to consider (default: 500 chars)

    Returns:
        Tuple of (start, end) character positions if found, None otherwise
    """
    if not span_text or not source_text:
        return None

    # Try exact match first (unnormalized)
    occurrences = find_all_occurrences(source_text, span_text)

    if not occurrences:
        # Whitespace-tolerant match. Offsets are found in the collapsed string and
        # translated back through the offset map, so they remain valid positions in
        # `source_text`. Searching the collapsed string without translating back
        # yields positions that can never validate against the original.
        collapsed_source, offsets = collapse_whitespace_with_offsets(source_text)
        collapsed_span, _ = collapse_whitespace_with_offsets(span_text)
        for c_start, c_end in find_all_occurrences(collapsed_source, collapsed_span):
            if c_end > len(offsets):
                continue
            occurrences.append((offsets[c_start], offsets[c_end - 1] + 1))

    if not occurrences:
        # If suggested_start is provided, search in a window around it
        if suggested_start is not None and suggested_start >= 0:
            window_start = max(0, suggested_start - window_size)
            window_end = min(len(source_text), suggested_start + window_size)
            window = source_text[window_start:window_end]

            try:
                rel_start = window.index(span_text)
                abs_start = window_start + rel_start
                abs_end = abs_start + len(span_text)
                return (abs_start, abs_end)
            except ValueError:
                pass

        # Case-insensitive fallback (handles LLM case normalisation vs ALL-CAPS source)
        source_lower = source_text.lower()
        span_lower = span_text.lower()
        occurrences_ci = find_all_occurrences(source_lower, span_lower)
        if occurrences_ci:
            logger.debug(
                f"Case-insensitive match for '{span_text[:50]}' "
                f"({len(occurrences_ci)} occurrence(s))"
            )
            occurrences = occurrences_ci

    if not occurrences:
        logger.debug(f"Could not find span in text: '{span_text[:50]}'")
        return None

    # If only one occurrence, return it
    if len(occurrences) == 1:
        return occurrences[0]

    # Multiple occurrences - use proximity to VOTING entities to choose
    if voting_positions and len(voting_positions) > 0:
        best_occurrence = None
        min_distance = float('inf')

        for occ_start, occ_end in occurrences:
            occ_center = (occ_start + occ_end) // 2

            # Find minimum distance to any VOTING entity
            for vote_start, vote_end in voting_positions:
                vote_center = (vote_start + vote_end) // 2
                distance = abs(occ_center - vote_center)

                if distance < min_distance:
                    min_distance = distance
                    best_occurrence = (occ_start, occ_end)

        # Only use this occurrence if it's within max_distance
        if best_occurrence and min_distance <= max_distance:
            logger.debug(
                f"Multiple occurrences found for '{span_text[:30]}...'. "
                f"Selected occurrence at {best_occurrence[0]} "
                f"(distance {min_distance} from nearest VOTING entity)"
            )
            return best_occurrence

    # Fallback: return first occurrence
    logger.debug(
        f"Multiple occurrences found for '{span_text[:30]}...', "
        f"no VOTING entities for context. Returning first occurrence."
    )
    return occurrences[0]


def validate_span_boundaries(
    text: str,
    start: int,
    end: int,
    span_text: Optional[str] = None
) -> bool:
    """
    Validate that span boundaries are valid for the given text.

    Args:
        text: Source text
        start: Start position
        end: End position
        span_text: Optional span text to verify exact match

    Returns:
        True if boundaries are valid
    """
    # Check bounds
    if start < 0 or end > len(text) or start >= end:
        return False

    # If span_text provided, verify it matches exactly
    if span_text is not None:
        extracted = text[start:end]
        if extracted != span_text:
            logger.warning(
                f"Span text mismatch: expected '{span_text}', got '{extracted}' at ({start}, {end})"
            )
            return False

    return True


def align_span(
    source_text: str,
    span_entity: SpanEntity,
    strict: bool = True,
    voting_positions: Optional[List[Tuple[int, int]]] = None
) -> Optional[SpanEntity]:
    """
    Align a span entity to source text, validating exact match.

    Uses proximity to VOTING entities when multiple occurrences exist.

    Args:
        source_text: Original source text
        span_entity: Span entity to align
        strict: If True, require exact character match
        voting_positions: List of (start, end) positions of VOTING entities for context

    Returns:
        Aligned SpanEntity if successful, None if alignment fails
    """
    # If span already has valid positions, verify them
    if span_entity.start is not None and span_entity.end is not None:
        if validate_span_boundaries(
            source_text,
            span_entity.start,
            span_entity.end,
            span_entity.text if strict else None
        ):
            return span_entity

    # Try to find the span in the text
    result = find_span_in_text(
        source_text,
        span_entity.text,
        suggested_start=span_entity.start,
        voting_positions=voting_positions
    )

    if result is None:
        logger.warning(f"Failed to align span: '{span_entity.text}' (type: {span_entity.type})")
        return None

    start, end = result
    extracted = source_text[start:end]

    if extracted != span_entity.text:
        if strict and not spans_equivalent(extracted, span_entity.text):
            # Genuine mismatch — not a case, unicode or whitespace difference
            logger.warning(
                f"Non-verbatim match for '{span_entity.text}': found '{extracted}'"
            )
            return None
        # Same span, different rendering (case or internal whitespace):
        # normalise the entity text to the verbatim source slice.
        logger.debug(
            f"Normalised to source: '{span_entity.text}' → '{extracted}'"
        )

    return SpanEntity(
        text=extracted,   # always use the verbatim source slice
        type=span_entity.type,
        start=start,
        end=end,
    )


def align_extraction_result(
    result: SpanExtractionResult,
    strict: bool = True,
    use_voting_context: bool = True
) -> Tuple[SpanExtractionResult, Dict[str, Any]]:
    """
    Align all spans in an extraction result to the source text.

    Uses proximity to VOTING entities to resolve ambiguous matches.

    Args:
        result: Extraction result to align
        strict: If True, require exact character matches
        use_voting_context: If True, use VOTING entity positions for context-aware alignment

    Returns:
        Tuple of (aligned_result, alignment_stats)
    """
    # First, identify VOTING entity positions
    voting_positions = []
    if use_voting_context:
        for entity in result.entities:
            if entity.type == "VOTING" and entity.start is not None and entity.end is not None:
                # If VOTING entity already has positions, use them
                voting_positions.append((entity.start, entity.end))
            elif entity.type == "VOTING":
                # Try to find VOTING entity first (without context, to avoid circular dependency)
                vote_result = find_span_in_text(
                    result.text,
                    entity.text,
                    suggested_start=entity.start,
                    voting_positions=None  # No context for VOTING entities themselves
                )
                if vote_result:
                    voting_positions.append(vote_result)

    logger.debug(f"Found {len(voting_positions)} VOTING entities for context-aware alignment")

    aligned_entities = []
    failed_alignments = []

    for entity in result.entities:
        # Use voting context for non-VOTING entities
        context = voting_positions if use_voting_context and entity.type != "VOTING" else None

        aligned = align_span(result.text, entity, strict=strict, voting_positions=context)

        if aligned is not None:
            aligned_entities.append(aligned)
        else:
            failed_alignments.append({
                'text': entity.text,
                'type': entity.type,
                'suggested_start': entity.start,
                'suggested_end': entity.end
            })

    # Calculate alignment statistics
    total_spans = len(result.entities)
    aligned_spans = len(aligned_entities)
    failed_spans = len(failed_alignments)

    stats = {
        'total_spans': total_spans,
        'aligned_spans': aligned_spans,
        'failed_spans': failed_spans,
        'alignment_rate': aligned_spans / total_spans if total_spans > 0 else 0.0,
        'failed_alignments': failed_alignments,
        'voting_entities_used': len(voting_positions)
    }

    # Create aligned result
    aligned_result = SpanExtractionResult(
        id=result.id,
        text=result.text,
        entities=aligned_entities,
        model=result.model,
        strategy=result.strategy,
        processing_time=result.processing_time,
        # Carried over explicitly: this rebuild used to drop them, which would
        # silently discard the measured API time and the diagnostics block.
        api_time=result.api_time,
        error=result.error,
        diagnostics=result.diagnostics,
    )

    return aligned_result, stats


def calculate_fidelity_metrics(results: List[SpanExtractionResult]) -> Dict[str, Any]:
    """
    Calculate alignment fidelity metrics across multiple results.

    Args:
        results: List of extraction results

    Returns:
        Dictionary with fidelity metrics
    """
    total_documents = len(results)
    total_spans = 0
    aligned_spans = 0
    documents_with_errors = 0
    non_verbatim_errors = []

    for result in results:
        if result.error is not None:
            documents_with_errors += 1
            continue

        for entity in result.entities:
            total_spans += 1

            # Check if span is properly aligned
            if validate_span_boundaries(
                result.text,
                entity.start,
                entity.end,
                entity.text
            ):
                aligned_spans += 1
            else:
                non_verbatim_errors.append({
                    'document_id': result.id,
                    'entity_text': entity.text,
                    'entity_type': entity.type,
                    'suggested_start': entity.start,
                    'suggested_end': entity.end
                })

    fidelity = aligned_spans / total_spans if total_spans > 0 else 0.0

    return {
        'total_documents': total_documents,
        'documents_with_errors': documents_with_errors,
        'total_spans': total_spans,
        'aligned_spans': aligned_spans,
        'failed_spans': total_spans - aligned_spans,
        'alignment_fidelity': fidelity,
        'non_verbatim_error_count': len(non_verbatim_errors),
        'non_verbatim_errors': non_verbatim_errors[:10]  # Return first 10 for inspection
    }
