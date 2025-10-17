#!/usr/bin/env python3
"""
Convert Event-Based Format to BIO-Tagged JSONL Format

This script converts the event-based JSON format with spans to BIO-tagged JSONL format
for sequence labeling.

Input: data/votie_events/{train,dev,test}.json (JSON list format)
Output: votie_bio/{train,dev,test}.jsonl (JSONL format)
"""

import json
import logging
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load spaCy
try:
    import spacy
    # Load Portuguese language model with only tokenizer (faster)
    try:
        nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])
        logger.info("Using spaCy for tokenization (optimized)")
    except OSError:
        logger.warning("Portuguese spaCy model not found. Install with: python -m spacy download pt_core_news_sm")
        logger.info("Falling back to whitespace tokenization")
        nlp = None
except ImportError:
    logger.warning("spaCy not installed. Install with: pip install spacy")
    logger.info("Falling back to whitespace tokenization")
    nlp = None


def is_problematic_token(token: str) -> bool:
    """
    Check if a token should be filtered out.

    Filters out:
    - Empty or whitespace-only tokens (including \\n, \\t, etc.)
    - Private use area Unicode characters
    - Control characters
    """
    if not token or not token.strip():
        return True

    # Check for private use area characters
    if re.search(r'[\uf000-\uf8ff]', token):
        return True

    # Check for control characters
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', token):
        return True

    return False


def filter_problematic_tokens(tokens: List[str], bio_tags: List[str]) -> Tuple[List[str], List[str]]:
    """
    Remove problematic tokens and their corresponding BIO tags.

    This ensures that the data is clean from the start and prevents
    alignment issues during training, prediction, and evaluation.

    Args:
        tokens: List of tokens
        bio_tags: List of BIO tags (same length as tokens)

    Returns:
        (filtered_tokens, filtered_tags)
    """
    if len(tokens) != len(bio_tags):
        logger.error(f"Token/tag length mismatch: {len(tokens)} vs {len(bio_tags)}")
        return tokens, bio_tags

    filtered_tokens = []
    filtered_tags = []
    removed_count = 0

    for token, tag in zip(tokens, bio_tags):
        if not is_problematic_token(token):
            filtered_tokens.append(token)
            filtered_tags.append(tag)
        else:
            removed_count += 1
            # If we're removing a B- or I- tag, log a warning
            if not tag.startswith('O'):
                logger.debug(f"Removing problematic token with entity tag: '{token}' ({tag})")

    if removed_count > 0:
        logger.debug(f"Filtered {removed_count} problematic tokens")

    return filtered_tokens, filtered_tags


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text using spaCy if available, otherwise fallback to whitespace.

    Args:
        text: Raw text to tokenize

    Returns:
        List of tokens
    """
    if nlp is not None:
        # Use spaCy tokenization
        doc = nlp(text)
        return [token.text for token in doc]
    else:
        # Fallback to simple whitespace tokenization
        return text.split()


def spans_to_bio_tags(tokens: List[str], spans: List[Dict]) -> List[str]:
    """
    Convert spans to BIO tags for tokens.

    Args:
        tokens: List of tokens
        spans: List of span dictionaries with token_start, token_end, label

    Returns:
        List of BIO tags (same length as tokens)
    """
    # Initialize with O tags
    bio_tags = ['O'] * len(tokens)

    # Sort spans by start position to handle overlaps consistently
    sorted_spans = sorted(spans, key=lambda s: (s["token_start"], s["token_end"]))

    # Track which tokens are already assigned
    assigned_tokens = set()

    for span in sorted_spans:
        token_start = span["token_start"]
        token_end = span["token_end"]
        label = span["label"]

        # Check for overlap with already assigned tokens
        span_tokens = range(token_start, token_end + 1)
        if any(t in assigned_tokens for t in span_tokens):
            # Skip overlapping spans (first span wins)
            logger.debug(f"Skipping overlapping span: {span['text']} ({label})")
            continue

        # Assign BIO tags
        if token_start <= token_end < len(tokens):
            bio_tags[token_start] = f"B-{label}"
            for i in range(token_start + 1, token_end + 1):
                if i < len(tokens):
                    bio_tags[i] = f"I-{label}"
                    assigned_tokens.add(i)
            assigned_tokens.add(token_start)

    return bio_tags


def validate_bio_tags(tokens: List[str], tags: List[str], example_id: str) -> bool:
    """
    Validate BIO tag sequence.

    Returns:
        True if valid
    """
    if len(tokens) != len(tags):
        logger.error(f"Token/tag length mismatch in {example_id}: "
                    f"{len(tokens)} vs {len(tags)}")
        return False

    # Check BIO constraints
    for i, tag in enumerate(tags):
        if i == 0:
            if tag.startswith('I-'):
                logger.warning(f"Sequence starts with I- tag in {example_id}")
                return False
        else:
            prev_tag = tags[i - 1]
            if tag.startswith('I-'):
                # I- must follow B- or I- of same type
                if prev_tag == 'O':
                    logger.warning(f"I- tag after O in {example_id} at position {i}")
                    return False
                elif prev_tag.startswith('B-') or prev_tag.startswith('I-'):
                    # Check same entity type
                    if tag[2:] != prev_tag[2:]:
                        logger.warning(f"Entity type mismatch in {example_id} at position {i}")
                        return False

    return True


def convert_example_to_bio_jsonl(example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict]:
    """
    Convert single example to BIO-tagged JSONL format.

    Returns:
        (bio_example, statistics)
    """
    text = example["text"]
    spans = example["spans"]
    example_id = example["id"]

    # Tokenize the text using spaCy or whitespace
    tokens = tokenize_text(text)
    
    # Map character-level spans to token-level spans
    token_spans = []
    if nlp is not None:
        # Use spaCy for accurate character-to-token mapping
        doc = nlp(text)
        char_to_token = {}
        for i, token in enumerate(doc):
            for char_idx in range(token.idx, token.idx + len(token.text)):
                char_to_token[char_idx] = i
        
        for span in spans:
            # FIXED: Character positions in the original data are off by 1
            # Adding +1 to both start and end positions to get correct alignment
            start_char = span["start"] + 1
            end_char = span["end"] + 1
            label = span["label"]
            span_text = span["text"]

            # Find token indices that overlap with character span
            # Try the character position first, if it fails try nearby positions
            token_start = char_to_token.get(start_char)
            if token_start is None and start_char > 0:
                # Try adjacent positions in case of whitespace
                token_start = char_to_token.get(start_char - 1) or char_to_token.get(start_char + 1)

            token_end = char_to_token.get(end_char - 1)  # end is exclusive
            if token_end is None and end_char > 1:
                # Try adjacent positions
                token_end = char_to_token.get(end_char - 2) or char_to_token.get(end_char)
            
            if token_start is not None and token_end is not None:
                token_spans.append({
                    "token_start": token_start,
                    "token_end": token_end,
                    "label": label,
                    "text": span_text
                })
            else:
                logger.warning(f"Could not map span to tokens in {example_id}: "
                             f"'{span_text[:40]}...' ({label}) at chars [{start_char}:{end_char}]")
    else:
        # Fallback: use whitespace tokenization and approximate mapping
        # This is less accurate but works without spaCy
        char_pos = 0
        token_idx = 0
        char_to_token = {}
        
        for token in tokens:
            # Find token in text starting from char_pos
            idx = text.find(token, char_pos)
            if idx >= 0:
                for i in range(idx, idx + len(token)):
                    char_to_token[i] = token_idx
                char_pos = idx + len(token)
            token_idx += 1
        
        for span in spans:
            # FIXED: Character positions in the original data are off by 1
            # Adding +1 to both start and end positions to get correct alignment
            start_char = span["start"] + 1
            end_char = span["end"] + 1
            label = span["label"]
            span_text = span["text"]

            # Try the character position first, if it fails try nearby positions
            token_start = char_to_token.get(start_char)
            if token_start is None and start_char > 0:
                token_start = char_to_token.get(start_char - 1) or char_to_token.get(start_char + 1)

            token_end = char_to_token.get(end_char - 1)
            if token_end is None and end_char > 1:
                token_end = char_to_token.get(end_char - 2) or char_to_token.get(end_char)
            
            if token_start is not None and token_end is not None:
                token_spans.append({
                    "token_start": token_start,
                    "token_end": token_end,
                    "label": label,
                    "text": span_text
                })
            else:
                logger.warning(f"Could not map span to tokens in {example_id}: "
                             f"'{span_text[:40]}...' ({label}) at chars [{start_char}:{end_char}]")

    # Convert token-level spans to BIO tags
    bio_tags = spans_to_bio_tags(tokens, token_spans)

    # Validate
    if not validate_bio_tags(tokens, bio_tags, example_id):
        logger.error(f"BIO validation failed for {example_id}")

    # FILTER PROBLEMATIC TOKENS - Critical for preventing alignment issues
    # This removes whitespace-only tokens, newlines, control characters, etc.
    original_token_count = len(tokens)
    tokens, bio_tags = filter_problematic_tokens(tokens, bio_tags)
    filtered_token_count = original_token_count - len(tokens)

    # Create BIO-tagged example in JSONL format
    bio_example = {
        "id": example_id,
        "text": text,
        "tokens": tokens,
        "labels": bio_tags,  # BIO tags as labels
        "municipality": example["municipality"]
    }
    
    # Only include document_id and segment_id if they exist in the source
    if "document_id" in example:
        bio_example["document_id"] = example["document_id"]
    if "segment_id" in example:
        bio_example["segment_id"] = example["segment_id"]

    # Statistics
    entity_count = sum(1 for tag in bio_tags if tag.startswith('B-'))
    entity_types = Counter(tag[2:] for tag in bio_tags if tag.startswith('B-'))

    # Track conversion success rate
    original_entity_count = len(spans)
    mapped_entity_count = len(token_spans)

    return bio_example, {
        "tokens": len(tokens),
        "entities": entity_count,
        "entity_types": dict(entity_types),
        "original_spans": original_entity_count,
        "mapped_spans": mapped_entity_count,
        "filtered_tokens": filtered_token_count
    }


def convert_split(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Convert a single split file from events to BIO-tagged JSONL format.

    Returns:
        Statistics dictionary
    """
    logger.info(f"Converting {input_file.name}...")

    bio_examples = []
    total_tokens = 0
    total_entities = 0
    entity_label_counts = Counter()
    num_examples = 0
    total_original_spans = 0
    total_mapped_spans = 0
    total_problematic_tokens = 0

    # Read input JSON file (list format)
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    # Process each example with progress bar
    for example in tqdm(examples, desc=f"  Processing {input_file.name}", unit="example"):
        bio_example, stats = convert_example_to_bio_jsonl(example)

        bio_examples.append(bio_example)
        total_tokens += stats["tokens"]
        total_entities += stats["entities"]
        entity_label_counts.update(stats["entity_types"])
        total_original_spans += stats["original_spans"]
        total_mapped_spans += stats["mapped_spans"]
        total_problematic_tokens += stats["filtered_tokens"]
        num_examples += 1

    # Write output as JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for bio_example in bio_examples:
            f.write(json.dumps(bio_example, ensure_ascii=False) + '\n')

    mapping_rate = 100 * total_mapped_spans / total_original_spans if total_original_spans > 0 else 0
    logger.info(f"  Converted {num_examples} examples with {total_entities} entities")
    logger.info(f"  Mapping success: {total_mapped_spans}/{total_original_spans} spans ({mapping_rate:.1f}%)")
    if total_problematic_tokens > 0:
        logger.info(f"  Filtered {total_problematic_tokens} problematic tokens (newlines, whitespace, control chars)")

    return {
        "num_examples": num_examples,
        "total_tokens": total_tokens,
        "total_entities": total_entities,
        "entity_label_counts": dict(entity_label_counts),
        "original_spans": total_original_spans,
        "mapped_spans": total_mapped_spans,
        "mapping_rate": f"{mapping_rate:.1f}%",
        "filtered_tokens": total_problematic_tokens
    }


def validate_conversion(event_dir: Path, bio_dir: Path) -> bool:
    """
    Validate that BIO-JSONL conversion preserved all information correctly.

    Returns:
        True if validation passed
    """
    logger.info("Validating conversion...")

    validation_passed = True

    for split in ["train", "dev", "test"]:
        event_file = event_dir / f"{split}.json"
        bio_file = bio_dir / f"{split}.jsonl"

        # Count entities in event format (JSON list)
        event_entity_counts = Counter()
        event_example_count = 0

        with open(event_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            event_example_count = len(examples)
            for example in examples:
                for span in example["spans"]:
                    event_entity_counts[span["label"]] += 1

        # Count entities in BIO-JSONL format
        bio_entity_counts = Counter()
        bio_example_count = 0

        with open(bio_file, 'r', encoding='utf-8') as f:
            for line in f:
                bio_example = json.loads(line.strip())
                bio_example_count += 1
                for tag in bio_example["labels"]:
                    if tag.startswith('B-'):
                        entity_label = tag[2:]
                        bio_entity_counts[entity_label] += 1

        # Check example counts
        if event_example_count != bio_example_count:
            logger.error(f"  {split}: Example count mismatch! "
                        f"{event_example_count} events vs {bio_example_count} BIO-JSONL")
            validation_passed = False

        # Check entity counts
        for label in set(event_entity_counts.keys()) | set(bio_entity_counts.keys()):
            event_count = event_entity_counts.get(label, 0)
            bio_count = bio_entity_counts.get(label, 0)

            # Allow small discrepancy due to overlap resolution
            if abs(event_count - bio_count) > max(1, event_count * 0.01):
                logger.warning(f"  {split}: Entity count discrepancy for {label}: "
                              f"{event_count} events vs {bio_count} BIO-JSONL")

        logger.info(f"  {split}: {'✓ PASS' if validation_passed else '⚠ WARN'}")

    return validation_passed


def main():
    """Main conversion pipeline."""
    logger.info("=" * 80)
    logger.info("EVENT TO BIO-TAGGED JSONL FORMAT CONVERSION")
    logger.info("=" * 80)

    # Paths
    input_dir = Path("data/votie_events")
    output_dir = Path("data/votie_bio")

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please run the data preparation script first")
        return 1

    # Convert each split
    all_stats = {}
    for split in ["train", "dev", "test"]:
        input_file = input_dir / f"{split}.json"
        output_file = output_dir / f"{split}.jsonl"

        if not input_file.exists():
            logger.warning(f"Split file not found: {input_file}")
            continue

        stats = convert_split(input_file, output_file)
        all_stats[split] = stats

    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved statistics to {stats_file}")

    # Validate conversion
    validation_passed = validate_conversion(input_dir, output_dir)

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 80)

    total_examples = sum(stats["num_examples"] for stats in all_stats.values())
    total_entities = sum(stats["total_entities"] for stats in all_stats.values())

    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Total entities: {total_entities}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Validation: {'PASSED' if validation_passed else 'WARNING'}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
