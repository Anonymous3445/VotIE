"""
XLM-R based NER models.
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


class XLMRLinearVotIE(BaseVotIEModel):
    """XLM-R + Linear (Fine-tuning approach similar to BERT paper)."""
    
    def __init__(self, model_name: str, num_labels: int, class_weights: Optional[torch.Tensor] = None,
                 weight_o: float = 0.01, bias_o: float = 6.0):
        super().__init__(model_name, num_labels, class_weights, weight_o, bias_o)
        
        self.xlmr = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.xlmr.config.hidden_size, num_labels)
        
        # Initialize classifier weights properly for stable training
        with torch.no_grad():
            # Xavier/Glorot initialization with reduced scale for XLM-R stability
            nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
            # Custom bias initialization to prevent collapse
            self._initialize_o_bias(self.classifier)
        
        logger.info(f"Initialized XLM-R-Linear with model: {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None, subtoken_mask=None, return_embeddings=False):
        xlmr_outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=return_embeddings)
        sequence_output = self.dropout(xlmr_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        if return_embeddings:
            outputs['embeddings'] = xlmr_outputs.last_hidden_state
            outputs['hidden_states'] = xlmr_outputs.hidden_states
        
        if labels is not None:
            outputs['loss'] = self._compute_weighted_loss(logits, labels, attention_mask)
        
        return outputs
    
    def decode(self, input_ids, attention_mask, subtoken_mask=None, tokenized_inputs=None, tokenizer=None,
               apply_bio_validation=True, id_to_label=None):
        """Decode using argmax with proper subtoken alignment (consistent with BERT models)."""
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            # Use proper token-to-word alignment if available
            if tokenized_inputs is not None:
                batch_predictions = postprocess_model_output(
                    predictions, tokenized_inputs, tokenizer
                )
            else:
                # Standard approach using subtoken_mask (consistent with other models)
                batch_predictions = []
                for i, pred_seq in enumerate(predictions):
                    mask = attention_mask[i].bool()
                    if subtoken_mask is not None:
                        # Only include predictions for first subtokens (where subtoken_mask is True)
                        mask = mask & subtoken_mask[i].bool()
                    masked_preds = pred_seq[mask].cpu().numpy().tolist()
                    batch_predictions.append(masked_preds)
            
            # Apply BIO validation if requested and id_to_label mapping is provided
            if apply_bio_validation and id_to_label is not None:
                validated_predictions, validation_stats = self.apply_bio_validation(batch_predictions, id_to_label)
                return validated_predictions
            
            return batch_predictions


class XLMRCRFVotIE(BaseVotIEModel):
    """XLM-R + CRF (Fine-tuning approach similar to BERT paper)."""
    
    def __init__(self, model_name: str, num_labels: int, class_weights: Optional[torch.Tensor] = None,
                 weight_o: float = 0.01, bias_o: float = 6.0):
        super().__init__(model_name, num_labels, class_weights, weight_o, bias_o)
        
        self.xlmr = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.xlmr.config.hidden_size, num_labels)
        
        # Initialize classifier weights properly for stable training
        with torch.no_grad():
            # Xavier/Glorot initialization with reduced scale for XLM-R stability
            nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
            # Custom bias initialization to prevent collapse
            self._initialize_o_bias(self.classifier)
        
        self.crf = CRF(num_labels, batch_first=True)
        
        logger.info(f"Initialized XLM-R-CRF with model: {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None, subtoken_mask=None, return_embeddings=False):
        xlmr_outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=return_embeddings)
        sequence_output = self.dropout(xlmr_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        if return_embeddings:
            outputs['embeddings'] = xlmr_outputs.last_hidden_state
            outputs['hidden_states'] = xlmr_outputs.hidden_states
        
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
    
    def decode(self, input_ids, attention_mask, subtoken_mask=None, tokenized_inputs=None, tokenizer=None, 
               apply_bio_validation=True, id_to_label=None):
        """Decode using CRF Viterbi algorithm with proper subtoken alignment (consistent with BERT models)."""
        with torch.no_grad():
            xlmr_outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(xlmr_outputs.last_hidden_state)
            logits = self.classifier(sequence_output)
            
            # Create mask for CRF decoding - use attention_mask only
            mask = attention_mask.bool()
            
            # Get CRF predictions
            predictions = self.crf.decode(logits, mask=mask)
            
            # Use proper token-to-word alignment if available
            if tokenized_inputs is not None:
                batch_predictions = postprocess_model_output(
                    predictions, tokenized_inputs, tokenizer
                )
            elif subtoken_mask is not None:
                # Standard approach using subtoken_mask (consistent with other models)
                batch_predictions = []
                for pred_seq, sub_mask in zip(predictions, subtoken_mask):
                    # Only keep predictions where subtoken_mask is True (first subtokens)
                    filtered_pred = [pred for pred, mask_val in zip(pred_seq, sub_mask) if mask_val]
                    batch_predictions.append(filtered_pred)
            else:
                # Fallback: remove special tokens
                batch_predictions = []
                for pred_seq in predictions:
                    if len(pred_seq) > 2:
                        pred_seq = pred_seq[1:-1]
                    batch_predictions.append(pred_seq)
            
            # Apply BIO validation if requested and id_to_label mapping is provided
            if apply_bio_validation and id_to_label is not None:
                validated_predictions, validation_stats = self.apply_bio_validation(batch_predictions, id_to_label)
                return validated_predictions
            
            return batch_predictions
