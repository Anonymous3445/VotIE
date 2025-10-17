"""
Base model class for all VotIE transformer models, used for the stage 1 - span extraction task framed as sequence labeling.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseVotIEModel(nn.Module):
    """Base class for all VotIE models."""
    
    def __init__(self, model_name: str, num_labels: int, class_weights: Optional[torch.Tensor] = None,
                 weight_o: float = 0.01, bias_o: float = 6.0):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        
        # class imbalance handling
        self.weight_o = weight_o
        self.bias_o = bias_o
        
        # Setup weighted loss function 
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            weights = torch.ones(num_labels)
            weights[0] = weight_o  # O-tag gets reduced weight
            self.class_weights = weights
        
        # Create weighted cross-entropy loss
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        
        logger.info(f"BaseVotIEModel initialized with O-weight={weight_o}, O-bias={bias_o}")
    
    def _initialize_o_bias(self, classifier_layer):
        """Initialize classifier bias to prevent model collapse."""
        if hasattr(classifier_layer, 'bias') and classifier_layer.bias is not None:
            with torch.no_grad():
                # Initialize all biases to prevent collapse
                classifier_layer.bias.data.zero_()  # Start with zero
                classifier_layer.bias.data[0] = self.bias_o  # O-tag gets high bias
                # Initialize other classes with small negative bias to prevent early dominance
                for i in range(1, classifier_layer.bias.size(0)):
                    classifier_layer.bias.data[i] = -0.5  # More conservative than -1.0
            logger.info(f"Classifier bias initialized: O-tag={self.bias_o}, others=-0.5")
    
    def _compute_weighted_loss(self, logits, labels, attention_mask):
        """
        Compute weighted cross-entropy loss with O-tag weighting.
        """
        # Move class weights to same device as logits
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
        
        # Create loss function if not exists or weights changed
        if not hasattr(self, 'loss_fct') or self.loss_fct.weight.device != logits.device:
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            self.loss_fct = self.loss_fct.to(logits.device)
        
        # Only compute loss on active tokens
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
        )
        
        # Debug: Check if we have valid labels to compute loss on
        valid_labels = (active_labels != -100).sum()
        if valid_labels == 0:
            logger.warning("No valid labels for loss computation - all labels are -100")
        
        loss = self.loss_fct(active_logits, active_labels)
        
        return loss

    def update_class_weights(self, new_weights: torch.Tensor):
        """Update class weights and loss function."""
        self.class_weights = new_weights
        if hasattr(self, 'loss_fct'):
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            if hasattr(self, 'device'):
                self.loss_fct = self.loss_fct.to(self.device)

    def apply_bio_validation(self, predictions: List[List[int]], id_to_label: Dict[int, str]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Apply BIO validation to model predictions.
        """
        from src.data.dataset import validate_bio_sequence
        
        validated_predictions = []
        
        # Create reverse mapping
        label_to_id = {v: k for k, v in id_to_label.items()}
        
        for seq_idx, pred_seq in enumerate(predictions):
            # Convert IDs to labels
            label_seq = [id_to_label.get(pred_id, 'O') for pred_id in pred_seq]

            # Apply validation
            validated_labels = validate_bio_sequence(
                label_seq,
                scheme='BIO',
                example_id=f"model_{seq_idx}"
            )

            # Convert back to IDs
            validated_ids = [label_to_id.get(label, label_to_id['O']) for label in validated_labels]
            validated_predictions.append(validated_ids)

        return validated_predictions, {}
