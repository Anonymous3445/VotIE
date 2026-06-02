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
    # Preserve real character offsets for faithful text reconstruction.
    # Each entry is [start, end) in the original text, so
    # text[start:end] == token for every token.
    output['token_offsets'] = [[start, end] for _, start, end in tokens_with_offsets]

    return output


def convert_text_to_subword_features(
    text: str,
    spans: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 512,
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    """Tokenize raw text directly and assign BIO labels per subword by char-offset overlap.

    This is the offset-based alignment path that replaces the regex+per-word
    tokenization legacy. The model is fed the same subwords clients see at
    inference, so there is no train-inference tokenization gap.

    Conventions:
      * Special tokens ([CLS], [SEP], padding) get label='O' here but are
        masked to -100 in the label IDs (see ``subtoken_mask`` flag).
      * For each gold span, the FIRST overlapping non-special subword gets
        ``B-LABEL``; subsequent overlapping subwords get ``I-LABEL``.
      * Subwords overlapping no span get ``O``.
      * If a subword overlaps multiple spans, the EARLIEST-starting span
        wins (deterministic; gold spans should not cross-overlap).

    Args:
        text: Original text.
        spans: List of span dicts with ``start``, ``end``, ``label``.
        tokenizer: HuggingFace fast tokenizer (must support ``return_offsets_mapping``).
        max_length: Max sequence length including special tokens.
        add_special_tokens: Whether the tokenizer adds [CLS]/[SEP].

    Returns:
        ``{"subword_labels", "subword_offsets", "input_ids", "attention_mask",
           "subtoken_mask", "is_special"}`` — all aligned to the subword sequence.
        ``subtoken_mask[i] == True`` iff position i is a real (non-special,
        non-padding) subword whose label should contribute to loss/metrics.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        return_attention_mask=True,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]

    n = len(input_ids)
    sw_labels: List[str] = ["O"] * n
    is_special: List[bool] = [False] * n

    # Identify special-token positions. Fast tokenizers report (0, 0) for [CLS],
    # [SEP], padding, and any other special token. ``sequence_ids`` is the
    # robust signal but isn't available on all tokenizers, so use offsets +
    # ``special_tokens_mask`` if present.
    if "special_tokens_mask" in encoding:
        special_mask = encoding["special_tokens_mask"]
        for i, m in enumerate(special_mask):
            if m == 1:
                is_special[i] = True
    else:
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                is_special[i] = True

    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    for span in sorted_spans:
        sp_start, sp_end = span["start"], span["end"]
        sp_label = span["label"]
        first_match = True
        for i, (sub_start, sub_end) in enumerate(offsets):
            if is_special[i]:
                continue
            # Half-open overlap: subword and span overlap iff
            # sub_start < sp_end AND sub_end > sp_start.
            if sub_start < sp_end and sub_end > sp_start:
                # Don't overwrite an existing entity label from an earlier
                # span (deterministic: earliest-start wins).
                if sw_labels[i] != "O":
                    continue
                sw_labels[i] = f"B-{sp_label}" if first_match else f"I-{sp_label}"
                first_match = False

    subtoken_mask = [(not sp) and (am == 1) for sp, am in zip(is_special, attention_mask)]

    return {
        "subword_labels": sw_labels,
        "subword_offsets": [list(o) for o in offsets],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subtoken_mask": subtoken_mask,
        "is_special": is_special,
    }


def create_subword_windows(
    text: str,
    spans: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 512,
    overlap_subwords: int = 50,
) -> List[Dict[str, Any]]:
    """Produce one or more subword-level feature dicts covering ``text``.

    Tokenizes the full text *without* special tokens to get a complete subword
    sequence, then slices it into windows of ``max_length - 2`` (room for
    [CLS]/[SEP]) with ``overlap_subwords`` of overlap. Each window is a
    standalone feature dict with B/I/O labels assigned from the gold spans
    via char-offset overlap.

    Why this is safer than the legacy word-level windowing:
      * No regex pre-tokenizer dependency.
      * Window boundaries are at subword boundaries — the model never sees
        partial tokens.
      * Char-offset alignment is exact, so labels can never drift.

    Returns a list of feature dicts. For text fitting in one window, the list
    has length 1.
    """
    if not text:
        return []

    full = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    full_ids = full["input_ids"]
    full_offsets = [list(o) for o in full["offset_mapping"]]
    n = len(full_ids)

    if n == 0:
        return []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    inner_max = max_length - 2  # leave room for [CLS] and [SEP]
    if inner_max <= 0:
        raise ValueError(f"max_length={max_length} too small for special tokens")

    if n <= inner_max:
        starts = [0]
    else:
        step = max(inner_max - overlap_subwords, 1)
        starts = list(range(0, n, step))
        # Drop trailing windows fully covered by the previous one
        starts = [s for s in starts if s + inner_max < n + step]
        if starts[-1] + inner_max < n:
            starts.append(n - inner_max)
        starts = sorted(set(starts))

    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"]))

    windows: List[Dict[str, Any]] = []
    for w_start in starts:
        w_end = min(w_start + inner_max, n)
        win_ids = full_ids[w_start:w_end]
        win_offs = full_offsets[w_start:w_end]

        # Wrap with [CLS]/[SEP]
        input_ids = [cls_id] + win_ids + [sep_id]
        offsets = [[0, 0]] + win_offs + [[0, 0]]
        is_special = [True] + [False] * len(win_ids) + [True]

        # Pad to max_length
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_id] * pad_len
            offsets = offsets + [[0, 0]] * pad_len
            is_special = is_special + [True] * pad_len  # padding counts as special
        attention_mask = [0 if sp and i >= 1 + len(win_ids) + 1 else 1
                          for i, sp in enumerate(is_special)]
        # The above sets attention=1 for [CLS]/non-special/[SEP], 0 for padding.

        # Build labels per subword via offset overlap
        sw_labels = ["O"] * len(input_ids)
        for span in sorted_spans:
            sp_start, sp_end = span["start"], span["end"]
            sp_label = span["label"]
            first_match = True
            for i, (sub_s, sub_e) in enumerate(offsets):
                if is_special[i]:
                    continue
                if sub_s < sp_end and sub_e > sp_start:
                    if sw_labels[i] != "O":
                        continue
                    sw_labels[i] = f"B-{sp_label}" if first_match else f"I-{sp_label}"
                    first_match = False

        subtoken_mask = [(not sp) and (am == 1) for sp, am in zip(is_special, attention_mask)]

        windows.append({
            "subword_labels": sw_labels,
            "subword_offsets": offsets,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "subtoken_mask": subtoken_mask,
            "is_special": is_special,
            "window_subword_range": [w_start, w_end],
        })

    return windows


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
