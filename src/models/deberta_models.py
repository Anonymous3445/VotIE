"""
DeBERTa based NER models.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF
from typing import Optional
import logging

from .base import BaseVotIEModel
from src.data.postprocessing import postprocess_model_output

logger = logging.getLogger(__name__)


class DebertaLinearVotIE(BaseVotIEModel):
    """DeBERTa + Linear fine tuning code."""
    
    def __init__(self, model_name: str, num_labels: int, class_weights: Optional[torch.Tensor] = None,
                 weight_o: float = 0.01, bias_o: float = 6.0):
        super().__init__(model_name, num_labels, class_weights, weight_o, bias_o)
        
        self.deberta = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)
        
        # Initialize O-tag bias (Portuguese BERT paper methodology)
        self._initialize_o_bias(self.classifier)
        
        logger.info(f"Initialized DeBERTa-Linear with model: {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None, subtoken_mask=None):
        deberta_outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(deberta_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            outputs['loss'] = self._compute_weighted_loss(logits, labels, attention_mask)
        
        return outputs
    
    def decode(self, input_ids, attention_mask, subtoken_mask=None, orig_token_indices=None,
               apply_bio_validation=True, id_to_label=None):
        """Decode using argmax with proper reconstruction."""
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            # Use reference-style postprocessing if orig_token_indices provided
            if orig_token_indices is not None:
                batch_predictions = postprocess_model_output(
                    predictions, orig_token_indices
                )
            else:
                # Fallback to simple approach for backward compatibility
                batch_predictions = []
                for i, pred_seq in enumerate(predictions):
                    mask = attention_mask[i].bool()
                    if subtoken_mask is not None:
                        mask = mask & subtoken_mask[i].bool()
                    masked_preds = pred_seq[mask].cpu().numpy().tolist()
                    batch_predictions.append(masked_preds)
            
            # Apply BIO validation if requested and id_to_label mapping is provided
            if apply_bio_validation and id_to_label is not None:
                validated_predictions, validation_stats = self.apply_bio_validation(batch_predictions, id_to_label)
                return validated_predictions
            
            return batch_predictions


class DebertaCRFVotIE(BaseVotIEModel):
    """DeBERTa + CRF fine tuning code."""
    
    def __init__(self, model_name: str, num_labels: int, class_weights: Optional[torch.Tensor] = None,
                 weight_o: float = 0.01, bias_o: float = 6.0):
        super().__init__(model_name, num_labels, class_weights, weight_o, bias_o)
        
        self.deberta = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)
        
        # Initialize O-tag bias (Portuguese BERT paper methodology)
        self._initialize_o_bias(self.classifier)
        
        self.crf = CRF(num_labels, batch_first=True)
        
        logger.info(f"Initialized DeBERTa-CRF with model: {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None, subtoken_mask=None):
        deberta_outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(deberta_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            # Create mask for CRF - use attention_mask to satisfy CRF requirements
            mask = attention_mask.bool()
            
            # Prepare labels for CRF (handle -100 ignore labels)
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0  # CRF doesn't handle -100
            
            # Clamp labels to valid range
            if crf_labels.max() >= self.num_labels or crf_labels.min() < 0:
                crf_labels = torch.clamp(crf_labels, min=0, max=self.num_labels-1)
            
            # CRF loss (negative log-likelihood)
            log_likelihood = self.crf(logits, crf_labels, mask=mask, reduction='mean')
            outputs['loss'] = -log_likelihood
        
        return outputs
    
    def decode(self, input_ids, attention_mask, subtoken_mask=None, apply_bio_validation=True, id_to_label=None):
        """Decode using CRF Viterbi algorithm with subtoken_mask filtering."""
        with torch.no_grad():
            deberta_outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(deberta_outputs.last_hidden_state)
            logits = self.classifier(sequence_output)
            
            # Create mask for CRF decoding - use attention_mask only
            mask = attention_mask.bool()
            
            # Get CRF predictions
            predictions = self.crf.decode(logits, mask=mask)
            
            # Filter predictions using subtoken_mask to remove special tokens and sub-tokens
            if subtoken_mask is not None:
                batch_predictions = []
                for i, pred_seq in enumerate(predictions):
                    if i < subtoken_mask.size(0):
                        # Use subtoken_mask to filter valid positions
                        mask_seq = subtoken_mask[i].cpu().tolist()
                        filtered_pred = [pred for pred, is_valid in zip(pred_seq, mask_seq) if is_valid]
                        batch_predictions.append(filtered_pred)
                    else:
                        # Fallback if subtoken_mask is shorter
                        batch_predictions.append(pred_seq[1:-1] if len(pred_seq) > 2 else pred_seq)
            else:
                # Simple approach without word alignment - remove special tokens
                batch_predictions = []
                for pred_seq in predictions:
                    # Remove first and last tokens (special tokens)
                    if len(pred_seq) > 2:
                        pred_seq = pred_seq[1:-1]
                    batch_predictions.append(pred_seq)
            
            # Apply BIO validation if requested and id_to_label mapping is provided
            if apply_bio_validation and id_to_label is not None:
                validated_predictions, validation_stats = self.apply_bio_validation(batch_predictions, id_to_label)
                return validated_predictions
            
            return batch_predictions
