#!/usr/bin/env python3
"""
VotIE Prediction Script

Prediction interface for Vote Information Extraction (VotIE) models.
Handles windowing automatically for long sequences.

Usage:
    from scripts.predict import predict
    predict(
        model_path='results/bert_crf/best_model',
        test_data_path='data/votie_bio/test.jsonl',
        output_path='predictions/test_predictions.jsonl'
    )
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bertimbau_models import BertimbauLinearVotIE, BertimbauCRFVotIE
from src.models.deberta_models import DebertaLinearVotIE, DebertaCRFVotIE
from src.models.xlmr_models import XLMRLinearVotIE, XLMRCRFVotIE
from src.models.bilstm_crf import BiLSTMFastTextVotIE
from src.data.dataset import load_jsonl_file, align_tokens_with_subwords, create_dynamic_windows

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = 'cuda'):
    """Load trained VotIE model from checkpoint."""
    model_path = Path(model_path)
    
    # Load config
    with open(model_path / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load label mapping - handle both formats
    if 'id2label' in config:
        # Transformer models store labels in config
        id_to_label = {int(k): v for k, v in config['id2label'].items()}
    else:
        # BiLSTM models store labels in separate file
        label_mapping_file = model_path / 'label_mapping.json'
        if label_mapping_file.exists():
            with open(label_mapping_file, 'r') as f:
                label_mapping = json.load(f)
            id_to_label = label_mapping['idx2label']
            # Convert string keys to int
            id_to_label = {int(k): v for k, v in id_to_label.items()}
        else:
            raise FileNotFoundError(f"Neither 'id2label' in config nor 'label_mapping.json' found in {model_path}")
    
    # Check if this is a BiLSTM model
    model_type = config.get('model_type', '')
    if model_type == 'bilstm_fasttext':
        # Load BiLSTM model using its from_pretrained method
        try:
            from src.models.bilstm_crf import FastTextEmbedding, CharacterVocab
            
            model = BiLSTMFastTextVotIE.from_pretrained(model_path)
            model.to(device)
            model.eval()
            
            # Load embedding handler and character vocabulary ONCE for all predictions
            embedding_handler = FastTextEmbedding.load_vocabulary(model_path)
            
            # Load char vocab if it exists
            char_vocab = None
            char_vocab_path = model_path / "char_vocab.json"
            if char_vocab_path.exists():
                with open(char_vocab_path, 'r') as f:
                    char_vocab_data = json.load(f)
                char_vocab = CharacterVocab(max_char_len=char_vocab_data.get('max_char_len', 20))
                char_vocab.char2idx = char_vocab_data['char2idx']
                char_vocab.idx2char = {int(k): v for k, v in char_vocab_data['idx2char'].items()}
            
            # Store these as attributes for prediction
            model._embedding_handler = embedding_handler
            model._char_vocab = char_vocab
            model._model_path = str(model_path)
            
            # BiLSTM models don't use transformers tokenizer
            tokenizer = None
            
            logger.info(f"✓ Loaded BiLSTM model from {model_path}")
            return model, tokenizer, id_to_label, config
        except Exception as e:
            logger.error(f"Failed to load BiLSTM model: {e}")
            raise
    
    # Check if this is a CRF model
    if model_type == 'crf':
        from src.models.crf import TraditionalCRFModel
        
        # Load CRF model using its load_model method
        model = TraditionalCRFModel.load_model(model_path)
        
        # CRF models don't use device or transformers tokenizer
        tokenizer = None
        
        logger.info(f"✓ Loaded CRF model from {model_path}")
        return model, tokenizer, id_to_label, config
    
    # Handle transformer models
    # Support both 'architecture' (old format) and 'architectures' (HuggingFace format)
    architecture = config.get('architecture', '')
    if not architecture and 'architectures' in config:
        # Extract from HuggingFace format (list)
        hf_arch = config['architectures'][0] if config['architectures'] else ''
        # Map HuggingFace architecture names to our internal names
        arch_mapping = {
            'BertimbauLinearForTokenClassification': 'BertimbauLinearVotIE',
            'BertimbauCRFForTokenClassification': 'BertimbauCRFVotIE',
            'DebertaLinearForTokenClassification': 'DebertaLinearVotIE',
            'DebertaCRFForTokenClassification': 'DebertaCRFVotIE',
            'XLMRLinearForTokenClassification': 'XLMRLinearVotIE',
            'XLMRCRFForTokenClassification': 'XLMRCRFVotIE',
        }
        architecture = arch_mapping.get(hf_arch, hf_arch)

    base_model_name = config.get('base_model', 'neuralmind/bert-base-portuguese-cased')

    model_classes = {
        'BertimbauLinearVotIE': BertimbauLinearVotIE,
        'BertimbauCRFVotIE': BertimbauCRFVotIE,
        'XLMRLinearVotIE': XLMRLinearVotIE,
        'XLMRCRFVotIE': XLMRCRFVotIE,
        'DebertaLinearVotIE': DebertaLinearVotIE,
        'DebertaCRFVotIE': DebertaCRFVotIE,
    }

    if architecture not in model_classes:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(model_classes.keys())}")
    
    model_class = model_classes[architecture]
    num_labels = len(id_to_label)
    
    # Load model
    model = model_class(base_model_name, num_labels)

    # Try to load from safetensors first (new format), fallback to pytorch_model.bin
    safetensors_path = model_path / 'model.safetensors'
    pytorch_path = model_path / 'pytorch_model.bin'

    if safetensors_path.exists():
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(str(safetensors_path))
        logger.info(f"Loaded model from safetensors: {safetensors_path}")
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location=device)
        logger.info(f"Loaded model from pytorch bin: {pytorch_path}")
    else:
        raise FileNotFoundError(f"No model weights found. Expected {safetensors_path} or {pytorch_path}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    logger.info(f"✓ Loaded {architecture} model from {model_path}")
    
    return model, tokenizer, id_to_label, config


def predict_single_window(model, tokenizer, tokens: List[str], device: str = 'cuda') -> List[int]:
    """Predict labels for a single window (guaranteed to fit in max_length)."""
    if not tokens:
        logger.warning("Empty token list provided to predict_single_window")
        return []

    original_token_count = len(tokens)
    dummy_labels = ['O'] * len(tokens)

    # Use allow_truncation=True as a safety measure - windowing should already ensure fit
    _, input_ids, _, subtoken_mask, _ = align_tokens_with_subwords(
        tokens=tokens,
        labels=dummy_labels,
        tokenizer=tokenizer,
        max_length=512,
        example_id="predict",
        allow_truncation=True  # Safety: truncate if windowing didn't work perfectly
    )

    if not input_ids:
        logger.error(f"align_tokens_with_subwords returned empty input_ids for tokens: {tokens[:5]}...")
        return [0] * len(tokens)  # Return all O labels as fallback

    # Count how many True values in subtoken_mask (should match original_token_count)
    num_original_tokens_in_mask = sum(subtoken_mask)
    if num_original_tokens_in_mask != original_token_count:
        logger.warning(f"Subtoken mask count mismatch: original_tokens={original_token_count}, "
                      f"mask_true_count={num_original_tokens_in_mask}")

    # Safety check: ensure we don't exceed max_length
    if len(input_ids) > 512:
        logger.warning(f"Window produced {len(input_ids)} input_ids (> 512), truncating to 512")
        input_ids = input_ids[:512]
        subtoken_mask = subtoken_mask[:512]

    # Pad to 512
    seq_length = len(input_ids)
    if seq_length > 512:
        # This should never happen after the check above, but just in case
        logger.error(f"CRITICAL: seq_length {seq_length} > 512 after safety check!")
        seq_length = 512
        input_ids = input_ids[:512]
        subtoken_mask = subtoken_mask[:512]

    input_ids += [tokenizer.pad_token_id] * (512 - seq_length)
    attention_mask = [1] * seq_length + [0] * (512 - seq_length)
    subtoken_mask += [False] * (512 - seq_length)

    # Final validation before creating tensors
    if len(input_ids) != 512 or len(attention_mask) != 512 or len(subtoken_mask) != 512:
        logger.error(f"Tensor size mismatch before creation: input_ids={len(input_ids)}, attention_mask={len(attention_mask)}, subtoken_mask={len(subtoken_mask)}")
        # Force correct size
        input_ids = (input_ids + [tokenizer.pad_token_id] * 512)[:512]
        attention_mask = (attention_mask + [0] * 512)[:512]
        subtoken_mask = (subtoken_mask + [False] * 512)[:512]

    # Convert to tensors
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    attention_mask_tensor = torch.tensor([attention_mask]).to(device)
    subtoken_mask_tensor = torch.tensor([subtoken_mask]).to(device)

    # Predict
    with torch.no_grad():
        predictions = model.decode(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            subtoken_mask=subtoken_mask_tensor,
            apply_bio_validation=False,
            id_to_label=None
        )

    # Validate predictions format
    if not predictions:
        logger.error(f"Model.decode returned empty predictions for {len(tokens)} tokens")
        return [0] * len(tokens)

    if not isinstance(predictions, list):
        logger.error(f"Model.decode returned unexpected type: {type(predictions)}")
        return [0] * len(tokens)

    if len(predictions) == 0:
        logger.error(f"Model.decode returned empty list")
        return [0] * len(tokens)

    # Validate prediction length
    pred_ids = predictions[0]
    if len(pred_ids) != original_token_count:
        logger.error(f"Prediction length mismatch: expected {original_token_count} tokens, "
                    f"got {len(pred_ids)} predictions")
        logger.error(f"  num_original_tokens_in_mask={num_original_tokens_in_mask}")
        logger.error(f"  First few tokens: {tokens[:5]}")
        # Pad or truncate to match expected length
        if len(pred_ids) < original_token_count:
            pred_ids = pred_ids + [0] * (original_token_count - len(pred_ids))
        else:
            pred_ids = pred_ids[:original_token_count]

    return pred_ids


def predict_with_windowing(model, tokenizer, tokens: List[str], device: str = 'cuda') -> List[int]:
    """Predict labels using windowing for long sequences."""
    if not tokens:
        logger.warning("Empty token list provided to predict_with_windowing")
        return []

    # Create windows using the same logic as training
    windows = create_dynamic_windows(
        tokens=tokens,
        labels=['O'] * len(tokens),  # Dummy labels
        tokenizer=tokenizer,
        max_length=512,
        overlap_tokens=50,
        example_id="predict"
    )

    if not windows:
        logger.error(f"create_dynamic_windows returned no windows for {len(tokens)} tokens")
        # Fallback: try direct prediction
        return predict_single_window(model, tokenizer, tokens, device)

    # Single window - no merging needed
    if len(windows) == 1:
        return predict_single_window(model, tokenizer, windows[0][0], device)

    # Multiple windows - predict and merge
    all_predictions = []
    for i, (window_tokens, _) in enumerate(windows):
        if not window_tokens:
            logger.warning(f"Empty window {i} in create_dynamic_windows")
            continue
        window_preds = predict_single_window(model, tokenizer, window_tokens, device)
        all_predictions.append(window_preds)
    
    if not all_predictions:
        logger.error(f"No predictions generated for {len(windows)} windows")
        return [0] * len(tokens)

    # Merge predictions from overlapping windows
    merged_predictions = [0] * len(tokens)
    coverage_map = {}

    start_idx = 0
    overlap_tokens = 50

    for window_idx, (window_tokens, _) in enumerate(windows):
        if window_idx >= len(all_predictions):
            logger.warning(f"Window {window_idx} has no predictions, skipping")
            continue
        window_preds = all_predictions[window_idx]
        window_len = len(window_tokens)

        # Validate prediction length matches window length
        if len(window_preds) != window_len:
            logger.error(f"Window {window_idx}: prediction length mismatch! "
                        f"window_tokens={window_len}, window_preds={len(window_preds)}")
            logger.error(f"  First few tokens: {window_tokens[:5]}")
            # Pad or truncate predictions to match window length
            if len(window_preds) < window_len:
                window_preds = window_preds + [0] * (window_len - len(window_preds))
            else:
                window_preds = window_preds[:window_len]

        for i in range(window_len):
            pos = start_idx + i
            if pos < len(tokens) and pos not in coverage_map:
                merged_predictions[pos] = window_preds[i]
                coverage_map[pos] = window_idx
        
        step_size = max(window_len - overlap_tokens, 1)
        start_idx = start_idx + step_size
    
    return merged_predictions


def predict_crf(model, tokens: List[str]) -> List[int]:
    """Predict labels for CRF model."""
    # CRF models work directly with tokens - no device or tensor conversion needed
    # Create a test example in the expected format
    test_example = {'tokens': tokens}
    
    # Use the CRF model's predict method
    predictions = model.predict([test_example])
    
    # predictions is a list of lists, get the first (and only) sequence
    pred_labels = predictions[0]
    
    # Convert label strings to IDs using the model's label list
    label_to_id = {label: idx for idx, label in enumerate(model.label_list)}
    pred_ids = [label_to_id.get(label, 0) for label in pred_labels]
    
    return pred_ids


def predict_bilstm(model, tokens: List[str], device: str = 'cuda') -> List[int]:
    """Predict labels for BiLSTM model."""
    try:
        # Use pre-loaded embedding handler and character vocabulary
        embedding_handler = model._embedding_handler
        char_vocab = model._char_vocab
        
        # Encode tokens
        token_ids = embedding_handler.encode_sentences([tokens])[0]
        
        # Encode characters if char_vocab available
        char_ids = None
        if char_vocab is not None:
            char_ids = char_vocab.encode_words(tokens)
            char_ids = torch.tensor([char_ids], dtype=torch.long).to(device)
        
        # Create tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Get predictions using model's decode method
        model.eval()
        with torch.no_grad():
            predictions = model.decode(input_ids, attention_mask, char_ids)
        
        # predictions is a list of lists of label IDs
        pred_ids = predictions[0]
        
        return pred_ids
        
    except Exception as e:
        logger.error(f"BiLSTM prediction failed: {e}")
        logger.exception("Full traceback:")
        # Fallback to all 'O' labels
        return [0] * len(tokens)


def predict_example(model, tokenizer, id_to_label: Dict[int, str], example: Dict[str, Any], device: str = 'cuda') -> Dict[str, Any]:
    """Generate predictions for a single example."""
    tokens = example['tokens']
    
    # Check model type for prediction
    if tokenizer is None:
        # Check if this is a traditional CRF model (has label_list and crf attributes)
        if hasattr(model, 'label_list') and hasattr(model, 'crf') and not hasattr(model, 'bilstm'):
            # Traditional CRF models (sklearn-crfsuite)
            pred_label_ids = predict_crf(model, tokens)
        elif hasattr(model, 'use_crf') or 'BiLSTM' in str(type(model).__name__):
            # BiLSTM models - use their decode method with proper encoding
            pred_label_ids = predict_bilstm(model, tokens, device)
        else:
            # Unknown model type
            logger.warning(f"Unknown model type: {type(model)}, using fallback prediction")
            pred_label_ids = [0] * len(tokens)  # All 'O' labels
    else:
        # Predict label IDs using transformer model
        pred_label_ids = predict_with_windowing(model, tokenizer, tokens, device)
    
    # Convert IDs to labels
    pred_labels = [id_to_label.get(lid, 'O') for lid in pred_label_ids]
    
    result = {
        'id': example['id'],
        'tokens': tokens,
        'pred_labels': pred_labels
    }
    
    # Include gold labels if available
    if 'labels' in example:
        result['gold_labels'] = example['labels']
    
    return result


def predict(model_path: str, test_data_path: str, output_path: str, device: str = 'cuda'):
    """
    Generate predictions for a test dataset and save to file.
    
    Args:
        model_path: Path to trained model directory
        test_data_path: Path to input JSONL file
        output_path: Path to output predictions file
        device: Device to use for inference ('cpu' or 'cuda')
    """
    logger.info("="*80)
    logger.info("VotIE Prediction")
    logger.info("="*80)
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model, tokenizer, id_to_label, config = load_model(model_path, device=device)
    
    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    examples = load_jsonl_file(test_data_path)
    logger.info(f"✓ Loaded {len(examples)} examples")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    
    for example in tqdm(examples, desc="Predicting"):
        pred = predict_example(model, tokenizer, id_to_label, example, device)
        predictions.append(pred)
    
    # Save predictions
    logger.info(f"Saving predictions to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    logger.info("="*80)
    logger.info(f"✓ Predictions saved to {output_path}")
    logger.info(f"✓ Total examples: {len(predictions)}")
    logger.info("="*80)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python scripts/predict.py MODEL_PATH TEST_DATA_PATH OUTPUT_PATH [DEVICE]")
        print("\nExample:")
        print("  python scripts/predict.py results/bert_crf/best_model data/votie_bio/test.jsonl predictions/test_predictions.jsonl")
        print("  python scripts/predict.py results/bert_crf/best_model data/votie_bio/test.jsonl predictions/test_predictions.jsonl cuda")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_path = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
    
    predict(model_path, test_data_path, output_path, device)
