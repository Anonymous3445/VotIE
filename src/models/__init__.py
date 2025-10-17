"""
Models package for NER Vote Identification.

This package contains all the transformer-based NER models:
- BERT-based models (BERTimbau)
- XLM-R models 
- DeBERTa models
"""

from .base import BaseVotIEModel
from .bertimbau_models import BertimbauLinearVotIE, BertimbauCRFVotIE
from .xlmr_models import XLMRLinearVotIE, XLMRCRFVotIE
from .deberta_models import DebertaLinearVotIE, DebertaCRFVotIE

# Import postprocessing utilities from data directory
from src.data.postprocessing import postprocess_model_output, postprocess_labels_to_original

__all__ = [
    # Base class
    'BaseVotIEModel',
    # BERT models
    'BertimbauLinearVotIE', 'BertimbauCRFVotIE',
    # XLM-R models  
    'XLMRLinearVotIE', 'XLMRCRFVotIE',
    # DeBERTa models
    'DebertaLinearVotIE', 'DebertaCRFVotIE',
    # Utilities
    'postprocess_model_output', 'postprocess_labels_to_original'
]
