"""
BERT-based VotIE models (BERTimbau implementations).
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from typing import Optional
import logging

from .base import BaseVotIEModel
from src.data.postprocessing import postprocess_model_output

logger = logging.getLogger(__name__)


class BertimbauLinearVotIE(BaseVotIEModel):
    """BERTimbau + Linear (Fine-tuning approach from Portuguese BERT paper)."""
    
    def __init__(self, model_name: str, num_labels: int, class_weights: Optional[torch.Tensor] = None,
                 weight_o: float = 0.01, bias_o: float = 6.0):
        super().__init__(model_name, num_labels, class_weights, weight_o, bias_o)
        
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Initialize O-tag bias (Portuguese BERT paper methodology)
        self._initialize_o_bias(self.classifier)
        
        logger.info(f"Initialized BERT-Linear with model: {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None, subtoken_mask=None, return_embeddings=False):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=return_embeddings)
        sequence_output = self.dropout(bert_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        if return_embeddings:
            outputs['embeddings'] = bert_outputs.last_hidden_state
            outputs['hidden_states'] = bert_outputs.hidden_states
        
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


class BertimbauCRFVotIE(BaseVotIEModel):
    """BERT + CRF (Fine-tuning approach from Portuguese BERT paper)."""
    
    def __init__(self, model_name: str, num_labels: int, class_weights: Optional[torch.Tensor] = None,
                 weight_o: float = 0.01, bias_o: float = 6.0):
        super().__init__(model_name, num_labels, class_weights, weight_o, bias_o)
        
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Initialize O-tag bias (Portuguese BERT paper methodology)
        self._initialize_o_bias(self.classifier)
        
        self.crf = CRF(num_labels, batch_first=True)
        
        logger.info(f"Initialized BERT-CRF with model: {model_name}")
    
    def forward(self, input_ids, attention_mask, labels=None, subtoken_mask=None, return_embeddings=False):

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=return_embeddings)
        sequence_output = self.dropout(bert_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        if return_embeddings:
            outputs['embeddings'] = bert_outputs.last_hidden_state
            outputs['hidden_states'] = bert_outputs.hidden_states
        
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
        """Decode using CRF Viterbi algorithm (Portuguese BERT paper approach)."""
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(bert_outputs.last_hidden_state)
            logits = self.classifier(sequence_output)
            
            # Create mask for CRF decoding - use attention_mask only
            mask = attention_mask.bool()
            
            # Get CRF predictions
            predictions = self.crf.decode(logits, mask=mask)
            
            # Filter predictions using subtoken_mask to remove [CLS]/[SEP] and sub-tokens
            if subtoken_mask is not None:
                filtered_predictions = []
                for pred_seq, sub_mask in zip(predictions, subtoken_mask):
                    # Only keep predictions where subtoken_mask is True (first sub-tokens)
                    filtered_pred = [pred for pred, mask_val in zip(pred_seq, sub_mask) if mask_val]
                    filtered_predictions.append(filtered_pred)
                batch_predictions = filtered_predictions
            else:
                batch_predictions = predictions
            
            # Apply BIO validation if requested and id_to_label mapping is provided
            if apply_bio_validation and id_to_label is not None:
                validated_predictions, validation_stats = self.apply_bio_validation(batch_predictions, id_to_label)
                return validated_predictions
            
            return batch_predictions
