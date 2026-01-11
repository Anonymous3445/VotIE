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


def convert_bio_to_spans(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    """
    Convert BIO-tagged tokens to span format.

    Args:
        tokens: List of tokens
        labels: List of BIO labels

    Returns:
        List of span dictionaries with text, type, start, end
    """
    spans = []
    current_span = None
    current_text = []
    char_offset = 0

    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            # Save previous span if exists
            if current_span is not None:
                current_span['text'] = ' '.join(current_text)
                current_span['end'] = char_offset - 1  # Remove trailing space
                spans.append(current_span)

            # Start new span
            entity_type = label[2:]  # Remove 'B-' prefix
            current_span = {
                'type': entity_type,
                'start': char_offset
            }
            current_text = [token]

        elif label.startswith('I-') and current_span is not None:
            # Continue current span
            current_text.append(token)

        else:  # O tag or I- without B-
            # Save previous span if exists
            if current_span is not None:
                current_span['text'] = ' '.join(current_text)
                current_span['end'] = char_offset - 1
                spans.append(current_span)
                current_span = None
                current_text = []

        # Update character offset (token + space)
        char_offset += len(token) + 1

    # Save final span if exists
    if current_span is not None:
        current_span['text'] = ' '.join(current_text)
        current_span['end'] = char_offset - 1
        spans.append(current_span)

    return spans


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
            example['spans'] = convert_bio_to_spans(example['tokens'], example['labels'])

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
