"""
Postprocessing utilities for model output and token alignment.
"""

import torch
from typing import Optional


def postprocess_model_output(predictions, tokenized_inputs, tokenizer=None):
    """
    Postprocess model output using proper token-to-word alignment.
    
    Uses the tokenizer's word_ids() method for accurate token alignment,
    which works correctly with all tokenizer types (BERT, XLM-R, etc.).
    """
    batch_predictions = []
    
    # Handle tensor input
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy().tolist()
    
    batch_size = len(predictions)
    
    for batch_idx in range(batch_size):
        pred_seq = predictions[batch_idx]
        word_ids = tokenized_inputs.word_ids(batch_index=batch_idx)
        
        if word_ids is None:
            # Fallback for single sequence
            word_ids = tokenized_inputs.word_ids()
        
        out_predictions = []
        previous_word_idx = None
        
        for tok_idx, word_idx in enumerate(word_ids):
            # Skip special tokens (word_idx is None)
            if word_idx is None:
                continue
            
            # Only use the first subtoken of each word
            if word_idx != previous_word_idx:
                if tok_idx < len(pred_seq):
                    out_predictions.append(pred_seq[tok_idx])
                previous_word_idx = word_idx
        
        batch_predictions.append(out_predictions)
    
    return batch_predictions


def postprocess_labels_to_original(labels, tokenized_inputs, tokenizer=None):
    """
    Postprocess labels to align with original tokens using proper token-to-word alignment.
    
    Uses the tokenizer's word_ids() method for accurate token alignment,
    which works correctly with all tokenizer types (BERT, XLM-R, etc.).
    """
    batch_labels = []
    
    # Handle tensor input
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy().tolist()
    
    batch_size = len(labels)
    
    for batch_idx in range(batch_size):
        label_seq = labels[batch_idx]
        word_ids = tokenized_inputs.word_ids(batch_index=batch_idx)
        
        if word_ids is None:
            # Fallback for single sequence
            word_ids = tokenized_inputs.word_ids()
        
        out_labels = []
        previous_word_idx = None
        
        for tok_idx, word_idx in enumerate(word_ids):
            # Skip special tokens (word_idx is None)
            if word_idx is None:
                continue
            
            # Only use the first subtoken of each word
            if word_idx != previous_word_idx:
                if tok_idx < len(label_seq) and label_seq[tok_idx] != -100:
                    out_labels.append(label_seq[tok_idx])
                previous_word_idx = word_idx
        
        batch_labels.append(out_labels)
    
    return batch_labels
