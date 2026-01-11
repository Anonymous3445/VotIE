#!/usr/bin/env python3
"""
Convert span-annotated data to BIO-tagged format.

Handles proper tokenization with punctuation and alignment of character-level
spans to token-level BIO tags, following NER best practices.
"""

import re
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)


def tokenize_text_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text into tokens with character offsets.

    Uses whitespace and punctuation-aware tokenization following standard NER practices.
    Preserves character offsets for accurate span alignment.

    Args:
        text: Input text to tokenize

    Returns:
        List of (token, start_offset, end_offset) tuples
    """
    if not text:
        return []

    # Pattern for tokenization:
    # - Words (including hyphenated words and apostrophes)
    # - Punctuation (each as separate token)
    # - Numbers (including decimals and dates)
    pattern = r'''
        (?:[A-Za-zÀ-ÿ]+(?:[-'][A-Za-zÀ-ÿ]+)*) |  # Words with hyphens/apostrophes
        (?:\d+(?:[.,]\d+)*(?:º|ª)?) |            # Numbers (with ordinals)
        (?:[.,;:!?(){}\[\]"\'`…—–\-/\\]) |       # Punctuation
        (?:\.{2,}) |                              # Ellipsis
        (?:\S)                                    # Any other non-whitespace
    '''

    tokens_with_offsets = []

    for match in re.finditer(pattern, text, re.VERBOSE):
        token = match.group(0)
        start = match.start()
        end = match.end()

        # Skip pure whitespace tokens
        if token.strip():
            tokens_with_offsets.append((token, start, end))

    return tokens_with_offsets


def align_spans_to_tokens(
    text: str,
    spans: List[Dict[str, Any]],
    tokens_with_offsets: List[Tuple[str, int, int]]
) -> List[str]:
    """
    Convert character-level spans to token-level BIO tags.

    Args:
        text: Original text
        spans: List of span dictionaries with 'start', 'end', 'label' keys
        tokens_with_offsets: List of (token, start_offset, end_offset)

    Returns:
        List of BIO tags (one per token)
    """
    num_tokens = len(tokens_with_offsets)
    bio_tags = ['O'] * num_tokens

    # Sort spans by start position to handle overlaps
    sorted_spans = sorted(spans, key=lambda s: (s['start'], s['end']))

    for span in sorted_spans:
        span_start = span['start']
        span_end = span['end']
        span_label = span['label']

        # Find tokens that overlap with this span
        matched_token_indices = []

        for token_idx, (token, token_start, token_end) in enumerate(tokens_with_offsets):
            # Check for overlap between token and span
            # Token overlaps if it starts before span ends AND ends after span starts
            if token_start < span_end and token_end > span_start:
                matched_token_indices.append(token_idx)

        # Assign BIO tags
        if matched_token_indices:
            # First token gets B- tag
            first_idx = matched_token_indices[0]
            bio_tags[first_idx] = f'B-{span_label}'

            # Subsequent tokens get I- tag
            for idx in matched_token_indices[1:]:
                bio_tags[idx] = f'I-{span_label}'

    return bio_tags


def validate_bio_tags(tokens: List[str], bio_tags: List[str], example_id: str = "unknown") -> bool:
    """
    Validate BIO tag sequence.

    Args:
        tokens: List of tokens
        bio_tags: List of BIO tags
        example_id: Example identifier for logging

    Returns:
        True if valid, False otherwise
    """
    if len(tokens) != len(bio_tags):
        logger.error(f"Token/tag mismatch in {example_id}: {len(tokens)} tokens, {len(bio_tags)} tags")
        return False

    if not tokens:
        logger.warning(f"Empty token sequence in {example_id}")
        return False

    # Check BIO format validity
    prev_tag = 'O'
    prev_entity_type = None

    for i, tag in enumerate(bio_tags):
        if tag == 'O':
            prev_tag = 'O'
            prev_entity_type = None
            continue

        if '-' not in tag:
            logger.error(f"Malformed BIO tag at position {i} in {example_id}: '{tag}'")
            return False

        tag_type, entity_type = tag.split('-', 1)

        if tag_type not in ['B', 'I']:
            logger.error(f"Invalid BIO tag type at position {i} in {example_id}: '{tag_type}'")
            return False

        # I- tag must follow B- or I- tag of same type
        if tag_type == 'I':
            if prev_tag == 'O':
                logger.warning(f"I- tag without B- tag at position {i} in {example_id}")
            elif prev_entity_type != entity_type:
                logger.warning(f"I- tag type mismatch at position {i} in {example_id}")

        prev_tag = tag_type
        prev_entity_type = entity_type

    return True


def convert_span_to_bio(
    example: Dict[str, Any],
    validate: bool = True
) -> Dict[str, Any]:
    """
    Convert a span-annotated example to BIO-tagged format.

    Args:
        example: Dictionary with 'id', 'text', 'spans' keys
        validate: Whether to validate the output

    Returns:
        Dictionary with 'id', 'tokens', 'labels' keys (plus original fields)
    """
    example_id = example.get('id', 'unknown')
    text = example.get('text', '')
    spans = example.get('spans', [])

    # Tokenize text with character offsets
    tokens_with_offsets = tokenize_text_with_offsets(text)

    # Extract just the tokens
    tokens = [token for token, _, _ in tokens_with_offsets]

    # Convert spans to BIO tags
    bio_tags = align_spans_to_tokens(text, spans, tokens_with_offsets)

    # Validate if requested
    if validate:
        if not validate_bio_tags(tokens, bio_tags, example_id):
            logger.warning(f"Validation failed for {example_id}")

    # Create output with both original and converted data
    output = example.copy()
    output['tokens'] = tokens
    output['labels'] = bio_tags

    return output


def convert_dataset_span_to_bio(
    examples: List[Dict[str, Any]],
    validate: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert a list of span-annotated examples to BIO format.

    Args:
        examples: List of span-annotated examples
        validate: Whether to validate outputs

    Returns:
        List of BIO-tagged examples
    """
    converted_examples = []

    for example in examples:
        try:
            converted = convert_span_to_bio(example, validate=validate)
            converted_examples.append(converted)
        except Exception as e:
            logger.error(f"Failed to convert example {example.get('id', 'unknown')}: {e}")
            continue

    logger.info(f"Converted {len(converted_examples)}/{len(examples)} examples to BIO format")

    return converted_examples


if __name__ == '__main__':
    # Test the conversion
    import json

    test_example = {
        "id": "test_001",
        "text": "O Executivo Municipal deliberou por unanimidade aprovar a proposta.",
        "spans": [
            {"start": 2, "end": 21, "label": "VOTER-FAVOR", "text": "Executivo Municipal"},
            {"start": 22, "end": 31, "label": "VOTING", "text": "deliberou"},
            {"start": 32, "end": 47, "label": "COUNTING-UNANIMITY", "text": "por unanimidade"},
            {"start": 56, "end": 65, "label": "SUBJECT", "text": "a proposta"}
        ]
    }

    print("Original example:")
    print(json.dumps(test_example, indent=2, ensure_ascii=False))

    converted = convert_span_to_bio(test_example)

    print("\nConverted to BIO:")
    print(f"Tokens: {converted['tokens']}")
    print(f"Labels: {converted['labels']}")

    # Show alignment
    print("\nToken-Label pairs:")
    for token, label in zip(converted['tokens'], converted['labels']):
        print(f"  {token:20s} -> {label}")
