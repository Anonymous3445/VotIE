#!/usr/bin/env python
"""
Dataset loading and processing for NER Tasks

Handles JSONL datasets from municipality documents with proper tokenization
and subword alignment following the Portuguese BERT paper methodology.
Works with any BIO-tagged NER task (voting, assunto, etc.)
"""

import json
import logging
import re
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def clean_problematic_characters(text: str) -> str:
    """
    Clean problematic characters that cause empty tokenization.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Replace private use area characters (like \uf0b7) with appropriate alternatives
    text = re.sub(r'[\uf000-\uf8ff]', '•', text)  # Replace private use chars with bullet
    
    # Remove control characters but preserve basic whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Remove control chars
    
    # Handle standalone newlines and whitespace-only tokens
    if text in ['\n', '\r\n', '\t', '\r'] or text.isspace():
        return ''  # Convert pure whitespace tokens to empty (will be skipped)
    
    return text

def is_problematic_token(token: str) -> bool:
    """Check if a token is problematic for BERT tokenization."""
    if not token or not token.strip():
        return True
    
    # Check for private use area characters
    if re.search(r'[\uf000-\uf8ff]', token):
        return True
    
    # Check for control characters
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', token):
        return True
    
    return False
VALID_TRANSITIONS = {  
    'BIO': {
        'B': ['B', 'I', 'O'],
        'I': ['B', 'I', 'O'],
        'O': ['B', 'O'],
    }
}

@dataclass
class VoteExample:
    """Single training/evaluation example."""
    doc_id: str
    tokens: List[str]
    labels: List[str]
    original_text: str
    
    def __len__(self):
        return len(self.tokens)

@dataclass 
class VoteFeatures:
    """Features for model input."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    subtoken_mask: torch.Tensor  # Mask for first subtokens only
    example_id: str
    orig_token_indices: List[int]  # Mapping from processed positions to original token indices

class VotIEDataset(Dataset):
    """PyTorch Dataset for stage 1 tasks with BIO tagging."""
    
    def __init__(self, features: List[VoteFeatures]):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            'input_ids': feature.input_ids,
            'attention_mask': feature.attention_mask,
            'labels': feature.labels,
            'subtoken_mask': feature.subtoken_mask,
            'example_id': feature.example_id,
            'orig_token_indices': feature.orig_token_indices
        }

def validate_bio_sequence(tag_sequence: List[str], scheme: str = 'BIO', example_id: str = "unknown") -> List[str]:
    """
    Validate and fix BIO tag sequence following Portuguese BERT paper methodology.
    
    Args:
        tag_sequence: List of BIO tags
        scheme: Tagging scheme ('BIO' or 'BILUO')
        example_id: Example identifier for logging
        
    Returns:
        Corrected tag sequence
    """
    if scheme not in VALID_TRANSITIONS:
        logger.warning(f"Unknown scheme {scheme}, skipping validation")
        return tag_sequence
    
    prev_tag = 'O'
    prev_type = 'O'
    corrected = []
    corrections_made = 0
    corrections_detail = []
    
    for i, tag_and_cls in enumerate(tag_sequence):
        if tag_and_cls == 'O':
            corrected.append('O')
            prev_tag = 'O'
            prev_type = 'O'
            continue
            
        # Parse tag and entity type
        if '-' in tag_and_cls:
            tag = tag_and_cls[0]
            type_ = tag_and_cls.split('-', 1)[1]
        else:
            # Malformed tag, convert to O
            corrected.append('O')
            prev_tag = 'O' 
            prev_type = 'O'
            corrections_made += 1
            corrections_detail.append(f"Position {i}: malformed tag '{tag_and_cls}' -> 'O'")
            continue
        
        # Check if transition is valid
        valid_transitions = VALID_TRANSITIONS[scheme][prev_tag]
        valid_tag = False
        
        if tag in valid_transitions:
            if tag == 'B' or tag == 'O':
                valid_tag = True
            elif tag == 'I' and type_ == prev_type:  # Same entity type
                valid_tag = True
        
        if valid_tag:
            corrected.append(tag_and_cls)
            prev_tag = tag
            prev_type = type_
        else:
            corrected.append('O')
            prev_tag = 'O'
            prev_type = 'O'
            corrections_made += 1
            corrections_detail.append(f"Position {i}: invalid transition '{tag_and_cls}' after '{prev_tag}-{prev_type}' -> 'O'")
    
    if corrections_made > 0:
        logger.info(f"Fixed {corrections_made} BIO tag violations")
        if corrections_made > 5:  # Only log if many corrections
            logger.info(f"Made {corrections_made} BIO tag corrections in {example_id}")
    
    return corrected

def validate_example_data(example_data: Dict[str, Any]) -> bool:
    """
    Comprehensive data validation following Portuguese BERT paper methodology.
    
    Args:
        example_data: Dictionary containing example data
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['id', 'tokens', 'labels']
    example_id = example_data.get('id', 'unknown')
    
    # Check required fields
    for field in required_fields:
        if field not in example_data:
            logger.warning(f"Missing required field '{field}' in example {example_id}")
            return False
    
    tokens = example_data['tokens']
    tags = example_data['labels']
    
    # Check basic alignment
    if len(tokens) != len(tags):
        logger.warning(f"Token/tag mismatch in {example_id}: {len(tokens)} tokens, {len(tags)} tags")
        return False
    
    # Check for empty sequences
    if not tokens:
        logger.warning(f"Empty token sequence in {example_id}")
        return False
    
    # Count problematic tokens before validation
    problematic_before = sum(1 for token in tokens if is_problematic_token(token))
    if problematic_before > 0:
        pass  # Track problematic tokens (reduced logging)
    
    # Validate BIO tags
    validated_tags = validate_bio_sequence(tags, example_id=example_id)
    if validated_tags != tags:
        logger.info(f"BIO tags corrected in {example_id}")
        example_data['labels'] = validated_tags  # Update with corrected labels
    
    # Validation completed successfully
    return True

def load_jsonl_file(
    file_path: Path
) -> List[Dict[str, Any]]:
    """
    Load JSONL file with vote identification data.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of example dictionaries
    """
    examples = []

    total_lines = 0
    invalid_json = 0
    invalid_examples = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                total_lines += 1
                try:
                    data = json.loads(line.strip())

                    # Validate data before adding
                    if validate_example_data(data):
                        examples.append(data)
                    else:
                        logger.warning(f"Skipping invalid example at line {line_no}")
                        invalid_examples += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_no} in {file_path}: {e}")
                    invalid_json += 1
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []

    # Log file statistics
    logger.info(f"Loaded {len(examples)} valid examples from {file_path}")

    return examples


def jsonl_to_examples_with_dynamic_windowing(
    jsonl_data: List[Dict[str, Any]], 
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    overlap_tokens: int = 50,
    enable_windowing: bool = True
) -> List[VoteExample]:
    """
    Convert JSONL data to VoteExample objects with dynamic windowing.
    
    This function handles full segments from the unified converter and applies
    windowing dynamically using the actual tokenizer.
    
    Args:
        jsonl_data: Raw JSONL data (full segments, no pre-windowing)
        tokenizer: Tokenizer to use for windowing decisions
        max_length: Maximum sequence length
        overlap_tokens: Overlap between windows
        enable_windowing: If False, treat as traditional pre-windowed data
        
    Returns:
        List of VoteExample objects (may include multiple windows per original segment)
    """
    examples = []
    windowing_stats = {
        'total_segments': 0,
        'single_window_segments': 0,
        'multi_window_segments': 0,
        'total_windows_created': 0,
        'max_windows_per_segment': 0
    }
    
    for item in jsonl_data:
        doc_id = item.get('id', 'unknown')
        original_text = item.get('text', '')
        tokens = item.get('tokens', [])
        tags = item.get('labels', [])
        
        # Validation
        if len(tokens) != len(tags):
            logger.warning(f"Token/tag mismatch in {doc_id}: {len(tokens)} tokens, {len(tags)} tags")
            continue
        
        if not tokens:
            logger.warning(f"Empty token sequence in {doc_id}")
            continue
        
        windowing_stats['total_segments'] += 1
        
        if enable_windowing:
            # Apply dynamic windowing
            windows = create_dynamic_windows(
                tokens=tokens,
                labels=tags,
                tokenizer=tokenizer,
                max_length=max_length,
                overlap_tokens=overlap_tokens,
                example_id=doc_id
            )
            
            windowing_stats['total_windows_created'] += len(windows)
            windowing_stats['max_windows_per_segment'] = max(
                windowing_stats['max_windows_per_segment'], 
                len(windows)
            )
            
            if len(windows) == 1:
                windowing_stats['single_window_segments'] += 1
            else:
                windowing_stats['multi_window_segments'] += 1
            
            # Create examples for each window
            for window_idx, (window_tokens, window_labels) in enumerate(windows):
                window_doc_id = f"{doc_id}_w{window_idx}" if len(windows) > 1 else doc_id
                window_text = " ".join(window_tokens)
                
                examples.append(VoteExample(
                    doc_id=window_doc_id,
                    tokens=window_tokens,
                    labels=window_labels,
                    original_text=window_text
                ))
        else:
            # Traditional mode - treat as pre-windowed data
            examples.append(VoteExample(
                doc_id=doc_id,
                tokens=tokens,
                labels=tags,
                original_text=original_text
            ))
            windowing_stats['single_window_segments'] += 1
            windowing_stats['total_windows_created'] += 1
    
    # Log windowing statistics
    if enable_windowing:
        logger.info(f"Dynamic windowing statistics:")
        logger.info(f"  Total segments: {windowing_stats['total_segments']}")
        logger.info(f"  Single-window segments: {windowing_stats['single_window_segments']}")
        logger.info(f"  Multi-window segments: {windowing_stats['multi_window_segments']}")
        logger.info(f"  Total windows created: {windowing_stats['total_windows_created']}")
        logger.info(f"  Max windows per segment: {windowing_stats['max_windows_per_segment']}")
        
        if windowing_stats['total_segments'] > 0:
            avg_windows = windowing_stats['total_windows_created'] / windowing_stats['total_segments']
            windowing_rate = (windowing_stats['multi_window_segments'] / windowing_stats['total_segments']) * 100
            logger.info(f"  Average windows per segment: {avg_windows:.2f}")
            logger.info(f"  Segments requiring windowing: {windowing_rate:.1f}%")
    
    return examples

def create_label_mapping(examples: List[VoteExample]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label to ID mapping."""
    all_labels = set()
    for example in examples:
        all_labels.update(example.labels)
    
    # Ensure O is at index 0 (required for paper methodology)
    sorted_labels = ['O'] + sorted([label for label in all_labels if label != 'O'])
    
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    logger.info(f"Created label mapping with {len(sorted_labels)} labels: {sorted_labels}")
    return label_to_id, id_to_label

def align_tokens_with_subwords(
    tokens: List[str], 
    labels: List[str], 
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    example_id: str = "unknown",
    allow_truncation: bool = True
) -> Tuple[List[str], List[int], List[int], List[bool], List[int]]:
    """
    Align original tokens with subwords.Solved tokenization issues by cleaning problematic characters.
    
    Args:
        tokens: Original word tokens
        labels: BIO labels for tokens
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (including [CLS] and [SEP])
        example_id: Identifier for logging
        allow_truncation: If False, process all tokens without truncation
    
    Returns:
        subwords: List of subword tokens
        input_ids: Token IDs
        label_ids: Label IDs (-100 for non-first subtokens)
        subtoken_mask: True for first subtoken of each word
        orig_token_indices: Mapping from processed positions to original token indices
    """
    subwords = []
    input_ids = []
    label_ids = []
    subtoken_mask = []
    
    # Track original token mapping (Portuguese BERT paper approach)
    orig_token_indices = []  # Maps processed token positions to original indices
    
    # Add [CLS] token
    subwords.append(tokenizer.cls_token)
    input_ids.append(tokenizer.cls_token_id)
    label_ids.append(-100)  # Ignore CLS for prediction
    subtoken_mask.append(False)
    orig_token_indices.append(-1)  # CLS has no original token
    
    # Track processing statistics
    skipped_tokens = 0
    truncated_tokens = 0
    cleaned_tokens = 0
    empty_tokenizations = 0
    
    # Starting token alignment
    
    for orig_token_idx, (token, label) in enumerate(zip(tokens, labels)):
        # Clean problematic characters first (Portuguese BERT paper approach)
        original_token = token
        cleaned_token = clean_problematic_characters(token)

        # Skip empty cleaned tokens (whitespace-only tokens like \n)
        if not cleaned_token or not cleaned_token.strip():
            skipped_tokens += 1
            continue

        # Track if token was cleaned
        if cleaned_token != original_token:
            cleaned_tokens += 1

        # Tokenize the cleaned word
        try:
            token_subwords = tokenizer.tokenize(cleaned_token)
        except Exception as e:
            logger.warning(f"Tokenization error for {cleaned_token!r}: {e}")
            token_subwords = []

        # Handle empty tokenization (should be rare now with cleaning)
        if not token_subwords:
            token_subwords = [tokenizer.unk_token]
            empty_tokenizations += 1
        
        # Check if we have space (reserve 1 for [SEP])
        if allow_truncation and len(subwords) + len(token_subwords) + 1 > max_length:
            logger.warning(f"Truncating sequence at token {orig_token_idx}: {original_token!r}")
            truncated_tokens = len(tokens) - orig_token_idx
            break
        
        # Add subwords with proper alignment tracking
        for i, subword in enumerate(token_subwords):
            
            subwords.append(subword)
            input_ids.append(tokenizer.convert_tokens_to_ids(subword))
            
            # All subtokens map to the same original token
            orig_token_indices.append(orig_token_idx)
            
            if i == 0:  # First subtoken gets the label
                label_ids.append(label)
                subtoken_mask.append(True)
            else:  # Subsequent subtokens get ignore index
                label_ids.append(-100)
                subtoken_mask.append(False)
    
    # Add [SEP] token
    subwords.append(tokenizer.sep_token)
    input_ids.append(tokenizer.sep_token_id)
    label_ids.append(-100)  # Ignore SEP for prediction
    subtoken_mask.append(False)
    orig_token_indices.append(-1)  # SEP has no original token
    
    # Log important statistics to console
    if cleaned_tokens > 5:  # Only log if significant cleaning happened
        logger.info(f"Cleaned {cleaned_tokens} problematic tokens in {example_id}")
    if skipped_tokens > 2:  # Only log if significant skipping happened
        logger.info(f"Skipped {skipped_tokens} empty/problematic tokens in {example_id}")
    if truncated_tokens > 0:
        logger.warning(f"Truncated {truncated_tokens} tokens due to length limit in {example_id}")
    
    return subwords, input_ids, label_ids, subtoken_mask, orig_token_indices

def convert_examples_to_features(
    examples: List[VoteExample],
    tokenizer: AutoTokenizer,
    label_to_id: Dict[str, int],
    max_length: int = 512
) -> List[VoteFeatures]:
    """Convert examples to model features."""
    features = []
        
    for example in examples:
        # Align tokens with subwords (using improved method with logging)
        subwords, input_ids, label_ids, subtoken_mask, orig_token_indices = align_tokens_with_subwords(
            example.tokens, example.labels, tokenizer, max_length, example.doc_id
        )
        
        # Convert string labels to IDs
        converted_label_ids = []
        for label_id in label_ids:
            if label_id == -100:
                converted_label_ids.append(-100)
            else:
                if label_id in label_to_id:
                    converted_label_ids.append(label_to_id[label_id])
                else:
                    logger.warning(f"Unknown label '{label_id}' in {example.doc_id}, using O")
                    converted_label_ids.append(label_to_id['O'])
        
        # Pad to max_length
        seq_length = len(input_ids)
        attention_mask = [1] * seq_length
        
        # Pad sequences
        input_ids += [tokenizer.pad_token_id] * (max_length - seq_length)
        attention_mask += [0] * (max_length - seq_length)
        converted_label_ids += [-100] * (max_length - seq_length)
        subtoken_mask += [False] * (max_length - seq_length)
        orig_token_indices += [-1] * (max_length - seq_length)  # Padding has no original token
        
        # Convert to tensors
        features.append(VoteFeatures(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            labels=torch.tensor(converted_label_ids, dtype=torch.long),
            subtoken_mask=torch.tensor(subtoken_mask, dtype=torch.bool),
            example_id=example.doc_id,
            orig_token_indices=orig_token_indices
        ))
    
    return features

def create_dynamic_windows(
    tokens: List[str],
    labels: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    overlap_tokens: int = 50,
    example_id: str = "unknown"
) -> List[Tuple[List[str], List[str]]]:
    """
    Create sliding windows dynamically using actual tokenizer sub-word expansion.

    Args:
        tokens: Original word tokens
        labels: BIO labels for tokens
        tokenizer: Actual tokenizer being used for training
        max_length: Maximum sequence length (including [CLS], [SEP])
        overlap_tokens: Number of original tokens to overlap between windows
        example_id: Example identifier for logging

    Returns:
        List of (window_tokens, window_labels) tuples
    """
    if len(tokens) != len(labels):
        logger.warning(f"Token/label mismatch in {example_id}: {len(tokens)} vs {len(labels)}")
        return []

    if not tokens:
        return []

    # Reserve space for [CLS] and [SEP] tokens, plus extra safety margin
    # Use 90% of available space to account for subword expansion
    effective_max_length = int((max_length - 2) * 0.9)

    # Always check if sequence fits after tokenization (not just by token count)
    subwords = tokenizer.tokenize(" ".join(tokens))

    if len(subwords) <= effective_max_length:
        # Sequence fits in one window
        return [(tokens, labels)]

    # Sequence needs windowing - create sliding windows
    windows = []
    start_idx = 0
    window_id = 0

    while start_idx < len(tokens):
        # Start with a conservative estimate: use fewer tokens than effective_max_length
        # because tokens can expand to multiple subwords
        initial_window_size = min(effective_max_length // 2, len(tokens) - start_idx)

        # Extract initial window
        end_idx = start_idx + initial_window_size
        window_tokens = tokens[start_idx:end_idx]
        window_labels = labels[start_idx:end_idx]

        # Check actual tokenized length and grow if possible (with cleaning)
        cleaned_window_tokens = [clean_problematic_characters(t) for t in window_tokens]
        cleaned_window_tokens = [t for t in cleaned_window_tokens if t and t.strip()]
        window_text = " ".join(cleaned_window_tokens)
        window_subwords = tokenizer.tokenize(window_text)

        # Try to grow the window to use more space (use full effective_max_length now)
        while len(window_subwords) < effective_max_length and end_idx < len(tokens):
            # Try adding more tokens
            candidate_end = min(end_idx + 10, len(tokens))
            candidate_tokens = tokens[start_idx:candidate_end]
            cleaned_candidate = [clean_problematic_characters(t) for t in candidate_tokens]
            cleaned_candidate = [t for t in cleaned_candidate if t and t.strip()]
            candidate_text = " ".join(cleaned_candidate)
            candidate_subwords = tokenizer.tokenize(candidate_text)

            if len(candidate_subwords) <= effective_max_length:
                # Still fits, use the larger window
                window_tokens = candidate_tokens
                window_labels = labels[start_idx:candidate_end]
                window_subwords = candidate_subwords
                end_idx = candidate_end
            else:
                # Would be too large, stop growing
                break

        # Ensure window definitely fits (safety check with cleaning)
        max_iterations = 50
        iteration = 0
        while len(window_subwords) > effective_max_length and len(window_tokens) > 1 and iteration < max_iterations:
            # Reduce by 10% and try again
            new_size = max(1, int(len(window_tokens) * 0.9))
            window_tokens = window_tokens[:new_size]
            window_labels = window_labels[:new_size]
            cleaned_window = [clean_problematic_characters(t) for t in window_tokens]
            cleaned_window = [t for t in cleaned_window if t and t.strip()]
            window_text = " ".join(cleaned_window)
            window_subwords = tokenizer.tokenize(window_text)
            iteration += 1

        if len(window_subwords) > effective_max_length:
            logger.error(f"Could not fit window for {example_id} - window has {len(window_subwords)} subwords, max is {effective_max_length}")
            logger.error(f"Window has {len(window_tokens)} tokens, tried {iteration} iterations")
            # Use a minimal window as fallback
            window_tokens = window_tokens[:1]
            window_labels = window_labels[:1]

        # Validate the final window size (with cleaning)
        final_cleaned = [clean_problematic_characters(t) for t in window_tokens]
        final_cleaned = [t for t in final_cleaned if t and t.strip()]
        final_check = " ".join(final_cleaned)
        final_subwords = tokenizer.tokenize(final_check)
        if len(final_subwords) > effective_max_length:
            logger.error(f"CRITICAL: Window still too large ({len(final_subwords)} > {effective_max_length}) for {example_id}")
            # Skip this problematic window rather than causing downstream errors
            if len(window_tokens) == 1:
                # Single token is too long - skip it
                logger.error(f"Skipping problematic single token: {window_tokens[0]!r}")
                start_idx += 1
                continue

        # Avoid cutting entities in half - look for good cut point
        if start_idx + len(window_tokens) < len(tokens) and len(window_tokens) > overlap_tokens:
            # Look backwards from end for a good cut point (preferably at "O" tag)
            cut_point = len(window_tokens)
            for i in range(len(window_tokens) - 1, max(0, len(window_tokens) - 20), -1):
                if window_labels[i] == "O":
                    cut_point = i + 1
                    break

            if cut_point < len(window_tokens) and cut_point > 0:
                window_tokens = window_tokens[:cut_point]
                window_labels = window_labels[:cut_point]

        windows.append((window_tokens, window_labels))

        # Check if we've covered all tokens with the ACTUAL window
        if start_idx + len(window_tokens) >= len(tokens):
            break

        # Move forward by window_size - overlap, ensuring progress
        step_size = max(len(window_tokens) - overlap_tokens, 1)
        start_idx = start_idx + step_size
        window_id += 1

        # Safety check to prevent infinite loops
        if window_id > 100:
            logger.error(f"Too many windows for {example_id}, stopping at {window_id}")
            break

    # Validate all windows (only log errors, use cleaning like actual processing)
    if len(windows) > 1:
        for i, (w_tokens, w_labels) in enumerate(windows):
            w_cleaned = [clean_problematic_characters(t) for t in w_tokens]
            w_cleaned = [t for t in w_cleaned if t and t.strip()]
            w_text = " ".join(w_cleaned)
            w_subwords = tokenizer.tokenize(w_text)
            if len(w_subwords) > effective_max_length:
                logger.error(f"Window {i} exceeds max length: {len(w_subwords)} > {effective_max_length}")

    return windows


def load_ner_dataset_with_dynamic_windowing(
    data_dir: Path,
    train_file: str,
    dev_file: str,
    test_file: str,
    tokenizer_name: str,
    max_length: int = 512,
    overlap_tokens: int = 50,
    enable_windowing: bool = True
) -> Tuple[VotIEDataset, VotIEDataset, VotIEDataset, Dict[str, int], Dict[int, str], AutoTokenizer]:
    """
    Load NER dataset with dynamic windowing at training time.

    This function works with full segments from the unified converter and applies
    windowing dynamically using the actual tokenizer, providing better control
    and logging than pre-computed windowing.

    Args:
        data_dir: Directory containing JSONL files
        train_file: Training file name
        dev_file: Development file name
        test_file: Test file name
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        overlap_tokens: Overlap between windows
        enable_windowing: If True, apply dynamic windowing. If False, use traditional loading

    Returns:
        train_dataset, dev_dataset, test_dataset, label_to_id, id_to_label, tokenizer
    """
    logger.info(f"Loading NER dataset with dynamic windowing: {enable_windowing}")
    logger.info(f"Max length: {max_length}, Overlap: {overlap_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load training data
    logger.info(f"Loading training file: {train_file}")
    train_data = load_jsonl_file(data_dir / train_file)
    train_examples = jsonl_to_examples_with_dynamic_windowing(
        train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        overlap_tokens=overlap_tokens,
        enable_windowing=enable_windowing
    )
    logger.info(f"  Loaded {len(train_examples)} examples from {train_file}")
    
    logger.info(f"Loading development file: {dev_file}")
    dev_data = load_jsonl_file(data_dir / dev_file)
    dev_examples = jsonl_to_examples_with_dynamic_windowing(
        dev_data,
        tokenizer=tokenizer, 
        max_length=max_length,
        overlap_tokens=overlap_tokens,
        enable_windowing=enable_windowing
    )
    logger.info(f"  Loaded {len(dev_examples)} examples from {dev_file}")
    
    logger.info(f"Loading test file: {test_file}")
    test_data = load_jsonl_file(data_dir / test_file)
    test_examples = jsonl_to_examples_with_dynamic_windowing(
        test_data,
        tokenizer=tokenizer,
        max_length=max_length, 
        overlap_tokens=overlap_tokens,
        enable_windowing=enable_windowing
    )
    logger.info(f"  Loaded {len(test_examples)} examples from {test_file}")
    
    # Create label mapping from all examples  
    all_examples = train_examples + dev_examples + test_examples
    label_to_id, id_to_label = create_label_mapping(all_examples)
    
    # Convert to features
    logger.info("Converting examples to features...")
    train_features = convert_examples_to_features(train_examples, tokenizer, label_to_id, max_length)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, label_to_id, max_length)
    test_features = convert_examples_to_features(test_examples, tokenizer, label_to_id, max_length)
    
    # Create datasets
    train_dataset = VotIEDataset(train_features)
    dev_dataset = VotIEDataset(dev_features)
    test_dataset = VotIEDataset(test_features)
    
    logger.info(f"Dataset loaded with dynamic windowing:")
    logger.info(f"  Train: {len(train_dataset)} examples")
    logger.info(f"  Dev: {len(dev_dataset)} examples") 
    logger.info(f"  Test: {len(test_dataset)} examples")
    logger.info(f"  Labels: {len(label_to_id)} ({list(label_to_id.keys())})")
    
    return train_dataset, dev_dataset, test_dataset, label_to_id, id_to_label, tokenizer

def calculate_paper_class_weights(examples: List[VoteExample], label_to_id: Dict[str, int], 
                                 weight_o: float = 0.01) -> torch.Tensor:
    """
    Calculate class weights.
    
    Args:
        examples: List of training examples
        label_to_id: Label to ID mapping
        weight_o: Weight for O tag (default 0.01)
        
    Returns:
        Class weights tensor
    """
    label_counts = Counter()
    total_tokens = 0
    
    for example in examples:
        for label in example.labels:
            label_counts[label] += 1
            total_tokens += 1
    
    # Calculate inverse frequency weights
    weights = torch.ones(len(label_to_id))
    for label, count in label_counts.items():
        if label in label_to_id and count > 0:
            # Inverse frequency weight
            weights[label_to_id[label]] = total_tokens / (len(label_to_id) * count)
    
    # Apply paper's O-tag weighting (configurable)
    if 'O' in label_to_id:
        weights[label_to_id['O']] = weight_o
    
    logger.info(f"Class weights calculated (O-weight={weight_o}): {weights}")
    
    # Log label distribution for analysis
    logger.info("Label distribution:")
    for label, count in label_counts.most_common():
        percentage = (count / total_tokens) * 100
        logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    return weights