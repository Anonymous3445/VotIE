"""
Data loading utilities for span extraction.

Provides functions to load and process data from the citilink dataset.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional


logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Path to the output file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(data)} examples to {file_path}")


def result_to_record(result, **extra: Any) -> Dict[str, Any]:
    """Serialize a ``SpanExtractionResult`` for JSONL storage.

    Every extraction script must go through this function. Hand-built record
    dicts previously omitted ``diagnostics``, which silently discarded the
    span-discard counts and token usage the extractors had already measured —
    the saved spans are all post-alignment, so any fidelity recomputed from
    them is 1.0 by construction and the run has to be repeated.

    Args:
        result: The extraction result to serialize.
        **extra: Additional fields to merge in (e.g. ``test_municipality``).
    """
    record = {
        "id": result.id,
        "text": result.text,
        "entities": [
            {"text": e.text, "type": e.type, "start": e.start, "end": e.end}
            for e in result.entities
        ],
        "model": result.model,
        "strategy": result.strategy,
        "processing_time": result.processing_time,
        "api_time": getattr(result, "api_time", None),
        "error": result.error,
        "diagnostics": getattr(result, "diagnostics", None),
    }
    record.update(extra)
    return record


def convert_bio_to_spans(
    tokens: List[str],
    labels: List[str],
    token_offsets: List[List[int]] = None,
    text: str = None,
) -> List[Dict[str, Any]]:
    """
    Convert BIO-tagged tokens to span format.

    When *token_offsets* are provided, character positions are taken
    from the original tokenization rather than approximated via
    cumulative ``len(token) + 1``.

    Args:
        tokens: List of tokens
        labels: List of BIO labels
        token_offsets: Optional ``[[start, end], ...]`` character
            positions in the original text, one per token.
        text: Optional original text for extracting span text via
            ``text[start:end]``.

    Returns:
        List of span dictionaries with text, type, start, end
    """
    use_offsets = (
        token_offsets is not None
        and len(token_offsets) == len(tokens)
    )

    spans = []
    current_span = None
    current_tokens = []
    char_offset = 0  # only used in legacy fallback

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if use_offsets:
            tok_start, tok_end = token_offsets[i]
        else:
            tok_start = char_offset
            tok_end = char_offset + len(token)

        if label.startswith('B-'):
            if current_span is not None:
                _finish_bio_span(current_span, current_tokens, text, use_offsets)
                spans.append(current_span)

            entity_type = label[2:]
            current_span = {
                'type': entity_type,
                'start': tok_start,
                'end': tok_end,
            }
            current_tokens = [token]

        elif label.startswith('I-') and current_span is not None:
            current_span['end'] = tok_end
            current_tokens.append(token)

        else:
            if current_span is not None:
                _finish_bio_span(current_span, current_tokens, text, use_offsets)
                spans.append(current_span)
                current_span = None
                current_tokens = []

        if not use_offsets:
            char_offset += len(token) + 1

    if current_span is not None:
        _finish_bio_span(current_span, current_tokens, text, use_offsets)
        spans.append(current_span)

    return spans


def _finish_bio_span(
    span: Dict[str, Any],
    tokens: List[str],
    text: str = None,
    use_offsets: bool = False,
) -> None:
    """Set the ``text`` field on a span being finalized."""
    if use_offsets and text:
        span['text'] = text[span['start']:span['end']]
    else:
        span['text'] = ' '.join(tokens)


def load_citilink_data(file_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load Citilink data and convert to span format.

    Args:
        file_path: Path to the citilink JSONL file
        limit: Optional limit on number of examples to load

    Returns:
        List of examples with id, text, tokens, labels, and spans
    """
    data = load_jsonl(file_path)

    if limit is not None:
        data = data[:limit]

    # Convert BIO labels to spans for each example
    for example in data:
        if 'tokens' in example and 'labels' in example:
            example['spans'] = convert_bio_to_spans(
                example['tokens'],
                example['labels'],
                token_offsets=example.get('token_offsets'),
                text=example.get('text'),
            )

    return data


# Backward compatibility alias
load_votie_bio_data = load_citilink_data


def format_example_for_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format an example for use in few-shot prompts.

    Args:
        example: Example dictionary with text and entities/spans

    Returns:
        Dictionary with 'text' and 'output' (JSON string)
    """
    # Get entities from either 'spans' or 'entities' field
    entities = example.get('spans', example.get('entities', []))

    # Convert to SpanEntity format
    formatted_entities = []
    for entity in entities:
        formatted_entities.append({
            'text': entity['text'],
            'type': entity.get('type', entity.get('label', 'UNKNOWN')),
            'start': entity['start'],
            'end': entity['end']
        })

    output = {
        'entities': formatted_entities
    }

    return {
        'text': example['text'],
        'output': json.dumps(output, ensure_ascii=False, indent=2)
    }
