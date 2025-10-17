#!/usr/bin/env python3
"""
Traditional CRF-only Model for Portuguese Voting NER

Implements a pure CRF approach using sklearn-crfsuite with carefully designed
feature engineering for Portuguese municipal documents. This provides an
important baseline and comparison point for the transformer-based approaches.

Features include:
- Word shape and character-level features
- POS tagging and linguistic features
- Context window features
- Portuguese-specific features
- Voting domain-specific patterns
"""

import re
import string
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score
import sklearn_crfsuite
from sklearn_crfsuite import CRF, metrics

# Import BIO validation and evaluation from core
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from src.data.dataset import validate_bio_sequence
from src.evaluation.entity_metrics import EntityLevelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortugueseFeatureExtractor:
    """
    Feature extractor for Portuguese voting documents.

    Implements comprehensive feature engineering including:
    - Word-level features (shape, prefixes, suffixes)
    - Character-level features
    - Linguistic features (POS, dependency)
    - Context features
    - Portuguese-specific patterns
    - Voting domain patterns
    """

    def __init__(self):
        """
        Initialize the feature extractor. POS and dependency features that
        previously relied on spaCy have been removed to keep the baseline
        lightweight and dependency-free. The extractor still provides word
        shape, affix and domain-specific features.
        """
        self.use_spacy = False
        self.nlp = None

        # Portuguese voting patterns
        self.voting_patterns = {
            'voting_verbs': {
                'votar', 'aprovar', 'rejeitar', 'abster', 'abstém', 'vota', 'aprova',
                'rejeita', 'votou', 'aprovou', 'rejeitou', 'absteve', 'votaram',
                'aprovaram', 'rejeitaram', 'abstiveram'
            },
            'favor_terms': {
                'favor', 'favorável', 'favoráveis', 'sim', 'aprova', 'aprovou',
                'aprovação', 'apoio', 'concordo', 'concorda', 'concordam'
            },
            'against_terms': {
                'contra', 'contrário', 'contrários', 'não', 'rejeita', 'rejeitou',
                'rejeição', 'oposição', 'discordo', 'discorda', 'discordam'
            },
            'abstention_terms': {
                'abstenção', 'abstém', 'absteve', 'abstiveram', 'abstém-se',
                'absteve-se', 'abstiveram-se', 'neutra', 'neutro'
            },
            'counting_terms': {
                'unanimidade', 'maioria', 'unânime', 'unânimes', 'maioritário',
                'maioritária', 'total', 'todos', 'todas', 'nenhum', 'nenhuma'
            },
            'subject_indicators': {
                'proposta', 'projeto', 'deliberação', 'resolução', 'pedido',
                'requerimento', 'moção', 'ponto', 'item', 'assunto', 'questão',
                'matéria', 'tema', 'documento', 'deliberou','decidiu'
            }
        }

        # Portuguese title patterns
        self.titles = {
            'dr.', 'dra.', 'sr.', 'sra.', 'eng.', 'prof.', 'vereador', 'vereadora',
            'presidente', 'secretário', 'secretária', 'deputado', 'deputada'
        }

        # Portuguese prepositions and articles for context
        self.function_words = {
            'articles': {'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas'},
            'prepositions': {'de', 'em', 'para', 'por', 'com', 'sobre', 'entre', 'durante', 'através'},
            'conjunctions': {'e', 'ou', 'mas', 'porém', 'contudo', 'todavia', 'entretanto'}
        }

    def extract_word_features(self, word: str, position: int, sentence_length: int) -> Dict[str, Any]:
        """
        Extract word-level features.

        Args:
            word: The word to extract features from
            position: Position in the sentence
            sentence_length: Total sentence length

        Returns:
            Dictionary of word features
        """
        features = {}

        # Basic word properties
        features['word.lower'] = word.lower()
        features['word.isupper'] = word.isupper()
        features['word.istitle'] = word.istitle()
        features['word.isdigit'] = word.isdigit()
        features['word.isalpha'] = word.isalpha()
        features['word.length'] = len(word)

        # Position features
        features['word.position'] = position
        features['word.relative_position'] = position / sentence_length if sentence_length > 0 else 0
        features['word.is_first'] = position == 0
        features['word.is_last'] = position == sentence_length - 1

        # Word shape features
        word_shape = self._get_word_shape(word)
        features['word.shape'] = word_shape
        features['word.short_shape'] = self._get_short_shape(word_shape)

        # Character-level features
        features.update(self._extract_character_features(word))

        # Prefix and suffix features
        features.update(self._extract_affix_features(word))

        # Portuguese-specific features
        features.update(self._extract_portuguese_features(word))

        # Voting domain features
        features.update(self._extract_voting_features(word))

        return features

    def extract_context_features(self, tokens: List[str], position: int, window: int = 2) -> Dict[str, Any]:
        """
        Extract context features around the current token.

        Args:
            tokens: List of tokens in the sentence
            position: Current position
            window: Context window size

        Returns:
            Dictionary of context features
        """
        features = {}

        # Context window features
        for i in range(-window, window + 1):
            if i == 0:
                continue

            ctx_pos = position + i
            if 0 <= ctx_pos < len(tokens):
                ctx_token = tokens[ctx_pos]
                prefix = f'ctx[{i:+d}]'

                features[f'{prefix}.word'] = ctx_token.lower()
                features[f'{prefix}.shape'] = self._get_word_shape(ctx_token)
                features[f'{prefix}.is_voting'] = self._is_voting_term(ctx_token)
                features[f'{prefix}.is_title'] = ctx_token.lower() in self.titles
            else:
                prefix = f'ctx[{i:+d}]'
                features[f'{prefix}.word'] = '<PAD>'

        # Bigram and trigram features
        if position > 0:
            features['bigram_prev'] = f"{tokens[position-1].lower()}_{tokens[position].lower()}"
        if position < len(tokens) - 1:
            features['bigram_next'] = f"{tokens[position].lower()}_{tokens[position+1].lower()}"

        if position > 1:
            features['trigram_prev'] = f"{tokens[position-2].lower()}_{tokens[position-1].lower()}_{tokens[position].lower()}"
        if position < len(tokens) - 2:
            features['trigram_next'] = f"{tokens[position].lower()}_{tokens[position+1].lower()}_{tokens[position+2].lower()}"

        return features

    def extract_linguistic_features(self, sentence: List[str], position: int) -> Dict[str, Any]:
        """
        Extract linguistic features.

        Args:
            sentence: List of tokens
            position: Current position

        Returns:
            Dictionary of linguistic features
        """
        # Lightweight linguistic features that do not require spaCy. We keep
        # a small set of signals (lemma-like lowercase, is_alpha, is_punct)
        # which are useful for CRF models but avoid heavy dependencies.
        features: Dict[str, Any] = {}
        if not sentence or position >= len(sentence):
            return features

        word = sentence[position]
        features['lemma_like'] = word.lower()
        features['is_alpha'] = word.isalpha()
        features['is_digit'] = word.isdigit()
        features['is_punct'] = all(ch in string.punctuation for ch in word)
        return features

    def _get_word_shape(self, word: str) -> str:
        """Get word shape representation (Xx, XX, xx, 9, etc.)"""
        if not word:
            return ""

        shape = ""
        for char in word:
            if char.isupper():
                shape += "X"
            elif char.islower():
                shape += "x"
            elif char.isdigit():
                shape += "9"
            else:
                shape += char
        return shape

    def _get_short_shape(self, shape: str) -> str:
        """Get compressed word shape"""
        if not shape:
            return ""

        # Compress consecutive identical characters
        short_shape = ""
        prev_char = ""
        for char in shape:
            if char != prev_char:
                short_shape += char
                prev_char = char
        return short_shape

    def _extract_character_features(self, word: str) -> Dict[str, Any]:
        """Extract character-level features"""
        features = {}

        # Character type counts
        features['num_uppercase'] = sum(1 for c in word if c.isupper())
        features['num_lowercase'] = sum(1 for c in word if c.islower())
        features['num_digits'] = sum(1 for c in word if c.isdigit())
        features['num_punct'] = sum(1 for c in word if c in string.punctuation)

        # Character patterns
        features['has_hyphen'] = '-' in word
        features['has_apostrophe'] = "'" in word
        features['has_accent'] = any(c in 'áàâãéêíóôõúç' for c in word.lower())
        features['starts_with_upper'] = word[0].isupper() if word else False
        features['ends_with_punct'] = word[-1] in string.punctuation if word else False

        return features

    def _extract_affix_features(self, word: str) -> Dict[str, Any]:
        """Extract prefix and suffix features"""
        features = {}

        word_lower = word.lower()

        # Prefixes (1-4 characters)
        for i in range(1, min(5, len(word) + 1)):
            features[f'prefix_{i}'] = word_lower[:i]

        # Suffixes (1-4 characters)
        for i in range(1, min(5, len(word) + 1)):
            features[f'suffix_{i}'] = word_lower[-i:]

        return features

    def _extract_portuguese_features(self, word: str) -> Dict[str, Any]:
        """Extract Portuguese-specific features"""
        features = {}

        word_lower = word.lower()

        # Title detection
        features['is_title'] = word_lower in self.titles

        # Function words
        features['is_article'] = word_lower in self.function_words['articles']
        features['is_preposition'] = word_lower in self.function_words['prepositions']
        features['is_conjunction'] = word_lower in self.function_words['conjunctions']

        # Portuguese morphological patterns
        features['ends_with_ção'] = word_lower.endswith('ção')
        features['ends_with_mente'] = word_lower.endswith('mente')
        features['ends_with_dade'] = word_lower.endswith('dade')
        features['ends_with_ável'] = word_lower.endswith('ável')

        return features

    def _extract_voting_features(self, word: str) -> Dict[str, Any]:
        """Extract voting domain-specific features"""
        features = {}

        word_lower = word.lower()

        # Voting terminology
        features['is_voting_verb'] = word_lower in self.voting_patterns['voting_verbs']
        features['is_favor_term'] = word_lower in self.voting_patterns['favor_terms']
        features['is_against_term'] = word_lower in self.voting_patterns['against_terms']
        features['is_abstention_term'] = word_lower in self.voting_patterns['abstention_terms']
        features['is_counting_term'] = word_lower in self.voting_patterns['counting_terms']
        features['is_subject_indicator'] = word_lower in self.voting_patterns['subject_indicators']

        # Combined voting indicator
        features['is_voting_related'] = self._is_voting_term(word)

        return features

    def _is_voting_term(self, word: str) -> bool:
        """Check if word is related to voting"""
        word_lower = word.lower()
        for pattern_group in self.voting_patterns.values():
            if word_lower in pattern_group:
                return True
        return False

    def extract_sentence_features(self, tokens: List[str], position: int) -> Dict[str, Any]:
        """
        Extract all features for a token at a given position.

        Args:
            tokens: List of tokens in the sentence
            position: Position of current token

        Returns:
            Dictionary of all features
        """
        if position >= len(tokens):
            return {}

        word = tokens[position]
        sentence_length = len(tokens)

        # Combine all feature types
        features = {}
        features.update(self.extract_word_features(word, position, sentence_length))
        features.update(self.extract_context_features(tokens, position))
        features.update(self.extract_linguistic_features(tokens, position))

        return features


class TraditionalCRFModel:
    """
    Traditional CRF model for Portuguese voting NER.

    Uses sklearn-crfsuite with comprehensive feature engineering
    specifically designed for Portuguese municipal voting documents.
    """

    def __init__(
        self,
        algorithm: str = 'lbfgs',
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        all_possible_transitions: bool = True
    ):
        """
        Initialize the CRF model.

        Args:
            algorithm: CRF training algorithm ('lbfgs', 'l2sgd', 'ap', 'pa', 'arow')
            c1: L1 regularization coefficient
            c2: L2 regularization coefficient
            max_iterations: Maximum training iterations
            all_possible_transitions: Whether to allow all label transitions
        """
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.all_possible_transitions = all_possible_transitions

        # Initialize CRF model
        self.crf = CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions
        )

        # Initialize feature extractor
        self.feature_extractor = PortugueseFeatureExtractor()

        # Model metadata
        self.is_trained = False
        self.training_stats = {}
        self.label_list = None

        logger.info(f"Initialized CRF model with algorithm={algorithm}, c1={c1}, c2={c2}")

    def prepare_data(self, data: List[Dict[str, Any]]) -> Tuple[List[List[Dict]], List[List[str]]]:
        """
        Prepare training data by extracting features.

        Args:
            data: List of training examples with 'tokens' and 'labels'

        Returns:
            Tuple of (feature_sequences, label_sequences)
        """
        logger.info(f"Preparing data from {len(data)} examples...")

        X = []  # Feature sequences
        y = []  # Label sequences

        for example in data:
            tokens = example['tokens']
            labels = example['labels']

            if len(tokens) != len(labels):
                logger.warning(f"Token-label length mismatch: {len(tokens)} != {len(labels)}")
                continue

            # Extract features for each token
            features_seq = []
            for i, token in enumerate(tokens):
                features = self.feature_extractor.extract_sentence_features(tokens, i)
                features_seq.append(features)

            X.append(features_seq)
            y.append(labels)

        logger.info(f"Prepared {len(X)} sequences for training")
        return X, y

    def train(self, train_data: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
        """
        Train the CRF model.

        Args:
            train_data: Training data with 'tokens' and 'labels'
            verbose: Whether to print training progress

        Returns:
            Training statistics
        """
        logger.info("Starting CRF training...")
        start_time = time.time()

        # Prepare training data
        X_train, y_train = self.prepare_data(train_data)

        if not X_train:
            raise ValueError("No valid training examples found")

        # Get unique labels
        all_labels = set()
        for label_seq in y_train:
            all_labels.update(label_seq)
        self.label_list = sorted(list(all_labels))

        logger.info(f"Training on {len(X_train)} sequences with {len(self.label_list)} labels")
        logger.info(f"Labels: {self.label_list}")

        # Train the model
        if verbose:
            # Enable verbose output
            self.crf.set_params(verbose=True)

        self.crf.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Training statistics
        self.training_stats = {
            'training_time': training_time,
            'num_sequences': len(X_train),
            'num_labels': len(self.label_list),
            'labels': self.label_list,
            'algorithm': self.algorithm,
            'c1': self.c1,
            'c2': self.c2,
            'max_iterations': self.max_iterations
        }

        self.is_trained = True

        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Model trained with max_iterations={self.max_iterations}")

        return self.training_stats

    def predict(self, test_data: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Predict labels for test data.

        Args:
            test_data: Test data with 'tokens'

        Returns:
            List of predicted label sequences
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        logger.info(f"Predicting on {len(test_data)} examples...")

        # Prepare test data (only features, no labels)
        X_test = []
        for example in test_data:
            tokens = example['tokens']
            features_seq = []
            for i, token in enumerate(tokens):
                features = self.feature_extractor.extract_sentence_features(tokens, i)
                features_seq.append(features)
            X_test.append(features_seq)

        # Predict
        predictions = self.crf.predict(X_test)

        # Apply BIO validation to ensure valid tag sequences
        validated_predictions = []
        for i, pred_seq in enumerate(predictions):
            validated_seq = validate_bio_sequence(
                pred_seq,
                scheme='BIO',
                example_id=f"crf_{i}"
            )
            validated_predictions.append(validated_seq)

        logger.info(f"Generated predictions for {len(validated_predictions)} sequences")
        return validated_predictions

    def evaluate(self, test_data: List[Dict[str, Any]], detailed: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on test data using entity-level metrics (same as transformers).

        Args:
            test_data: Test data with 'tokens' and 'labels'
            detailed: Whether to compute detailed per-label metrics

        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        logger.info(f"Evaluating on {len(test_data)} examples...")

        # Predict labels
        predictions = self.predict(test_data)
        
        # Extract true labels
        true_labels = [example['labels'] for example in test_data]

        # Use entity-level evaluation
        evaluator = EntityLevelEvaluator()
        entity_metrics = evaluator.compute_metrics(predictions, true_labels)

        # Map to consistent field names for compatibility
        results = {
            'entity_f1_strict': entity_metrics.get('entity_f1', 0.0),
            'entity_precision': entity_metrics.get('entity_precision', 0.0),
            'entity_recall': entity_metrics.get('entity_recall', 0.0),
            'num_examples': len(test_data)
        }

        if detailed:
            # Per-type metrics from seqeval (entity-level) - same structure as transformers
            per_type_metrics = entity_metrics.get('per_type_metrics', {})
            results['per_type_metrics'] = per_type_metrics

        logger.info(f"Evaluation completed - Entity F1: {results['entity_f1_strict']:.4f}")

        return results

    def get_feature_importance(self, top_k: int = 50) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each label.

        Args:
            top_k: Number of top features to return per label

        Returns:
            Dictionary mapping labels to their top features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        feature_importance = {}

        # Get state features (transitions from features to labels)
        state_features = self.crf.state_features_

        for label in self.label_list:
            # Get features for this label
            label_features = []
            for feature_name, weight in state_features.items():
                if feature_name.endswith(f':{label}'):
                    feature_clean = feature_name.replace(f':{label}', '')
                    label_features.append((feature_clean, abs(weight)))

            # Sort by absolute weight and take top k
            label_features.sort(key=lambda x: x[1], reverse=True)
            feature_importance[label] = label_features[:top_k]

        return feature_importance

    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save the trained model.

        Args:
            save_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the CRF model
        model_path = save_path / "crf_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.crf, f)

        # Save metadata
        metadata = {
            'label_list': self.label_list,
            'training_stats': self.training_stats,
            'algorithm': self.algorithm,
            'c1': self.c1,
            'c2': self.c2,
            'max_iterations': self.max_iterations,
            'all_possible_transitions': self.all_possible_transitions
        }

        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save config.json for compatibility with prediction pipeline
        label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        id_to_label = {idx: label for idx, label in enumerate(self.label_list)}
        
        config = {
            'model_type': 'crf',
            'architecture': 'TraditionalCRF',
            'num_labels': len(self.label_list),
            'label2id': label_to_id,
            'id2label': id_to_label,
            'algorithm': self.algorithm,
            'c1': self.c1,
            'c2': self.c2,
            'max_iterations': self.max_iterations
        }

        config_path = save_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> 'TraditionalCRFModel':
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded CRF model
        """
        model_path = Path(model_path)

        # Load metadata
        metadata_path = model_path / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Create model instance
        model = cls(
            algorithm=metadata['algorithm'],
            c1=metadata['c1'],
            c2=metadata['c2'],
            max_iterations=metadata['max_iterations'],
            all_possible_transitions=metadata['all_possible_transitions']
        )

        # Load the CRF model
        crf_path = model_path / "crf_model.pkl"
        with open(crf_path, 'rb') as f:
            model.crf = pickle.load(f)

        # Restore metadata
        model.label_list = metadata['label_list']
        model.training_stats = metadata['training_stats']
        model.is_trained = True

        logger.info(f"Model loaded from {model_path}")
        return model


# Example usage and testing
if __name__ == "__main__":
    # Example training data
    sample_data = [
        {
            "tokens": ["O", "vereador", "João", "votou", "a", "favor", "."],
            "labels": ["O", "B-VOTANTE-FAVOR", "I-VOTANTE-FAVOR", "B-VOTACAO", "O", "O", "O"]
        },
        {
            "tokens": ["A", "proposta", "foi", "aprovada", "por", "unanimidade", "."],
            "labels": ["B-ASSUNTO", "I-ASSUNTO", "B-VOTACAO", "O", "O", "B-CONTABILIZACAO-UNANIMIDADE", "O"]
        }
    ]

    # Create and train model
    crf_model = TraditionalCRFModel()
    training_stats = crf_model.train(sample_data)

    # Evaluate
    results = crf_model.evaluate(sample_data)
    print(f"Results: {results}")

    # Feature importance
    importance = crf_model.get_feature_importance(top_k=10)
    print(f"Feature importance: {importance}")