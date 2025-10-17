#!/usr/bin/env python3
"""
BiLSTM + FastText Trainer for Voting NER

Trainer for BiLSTM models using FastText embeddings that integrates
with the existing training framework. Provides traditional baseline comparison
to transformer-based models.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from .models.bilstm_crf import BiLSTMFastTextVotIE, FastTextEmbedding, CharacterVocab
from src.evaluation.entity_metrics import EntityLevelEvaluator

logger = logging.getLogger(__name__)


class BiLSTMDataset(Dataset):
    """Dataset for BiLSTM training with FastText embeddings."""

    def __init__(
        self,
        sentences: List[List[str]],
        labels: List[List[str]],
        embedding_handler: FastTextEmbedding,
        char_vocab: Optional[CharacterVocab] = None,
        label2idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset.

        Args:
            sentences: List of tokenized sentences
            labels: List of label sequences
            embedding_handler: FastText embedding handler
            char_vocab: Character vocabulary (optional)
            label2idx: Label to index mapping
        """
        self.sentences = sentences
        self.labels = labels
        self.embedding_handler = embedding_handler
        self.char_vocab = char_vocab
        self.label2idx = label2idx or {}

        # Encode sentences and labels
        self.encoded_sentences = embedding_handler.encode_sentences(sentences)
        self.encoded_labels = self._encode_labels(labels)

        # Encode characters if char_vocab provided
        if char_vocab:
            self.char_sequences = [char_vocab.encode_words(sent) for sent in sentences]
        else:
            self.char_sequences = None

    def _encode_labels(self, labels: List[List[str]]) -> List[List[int]]:
        """Encode label sequences to indices."""
        encoded = []
        for label_seq in labels:
            encoded_seq = [self.label2idx.get(label, 0) for label in label_seq]
            encoded.append(encoded_seq)
        return encoded

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        sentence = self.encoded_sentences[idx]
        labels = self.encoded_labels[idx]

        # Use full sequences without truncation
        # Create attention mask
        attention_mask = [1] * len(sentence)

        # Padding will be done dynamically in collate function

        item = {
            'input_ids': torch.tensor(sentence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

        # Add character sequences if available
        if self.char_sequences:
            char_seq = self.char_sequences[idx]
            item['char_ids'] = torch.tensor(char_seq, dtype=torch.long)

        return item


def collate_fn(batch):
    """
    Custom collate function for dynamic padding.
    
    Handles variable-length sequences by padding them to the maximum length
    in the batch. Also handles character sequences if present.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences dynamically
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    result = {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded
    }
    
    # Handle character sequences if present
    if 'char_ids' in batch[0]:
        char_ids = [item['char_ids'] for item in batch]
        
        # Find maximum dimensions
        max_seq_len = max(char_seq.size(0) for char_seq in char_ids)
        max_char_len = max(char_seq.size(1) for char_seq in char_ids)
        
        char_ids_padded = []
        for char_seq in char_ids:
            # Get current dimensions
            curr_seq_len, curr_char_len = char_seq.size()
            
            # Pad sequence length dimension
            if curr_seq_len < max_seq_len:
                seq_pad = torch.zeros(max_seq_len - curr_seq_len, curr_char_len, dtype=torch.long)
                char_seq = torch.cat([char_seq, seq_pad], dim=0)
            
            # Pad character length dimension
            if curr_char_len < max_char_len:
                char_pad = torch.zeros(max_seq_len, max_char_len - curr_char_len, dtype=torch.long)
                char_seq = torch.cat([char_seq, char_pad], dim=1)
            
            char_ids_padded.append(char_seq)
        
        result['char_ids'] = torch.stack(char_ids_padded)
    
    return result


class BiLSTMTrainer:
    """
    Trainer for BiLSTM + FastText models.

    Integrates with the existing framework while providing
    traditional embedding-based baseline models.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_crf: bool = True,
        use_char_cnn: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 50,
        patience: int = 7,
        device: str = "auto",
        apply_bio_validation: bool = True,
        seed: int = 42
    ):
        """
        Initialize BiLSTM trainer.

        Args:
            config: Training configuration
            output_dir: Output directory
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_crf: Whether to use CRF layer
            use_char_cnn: Whether to use character CNN
            learning_rate: Learning rate
            weight_decay: L2 regularization strength (weight decay)
            batch_size: Batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            device: Device to use
            apply_bio_validation: Whether to apply BIO tag validation during inference
            seed: Random seed for reproducibility
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_crf = use_crf
        self.use_char_cnn = use_char_cnn

        # Training parameters (use config values if available)
        self.learning_rate = float(config.get('learning_rate', learning_rate))
        self.weight_decay = float(config.get('weight_decay', weight_decay))
        self.batch_size = int(config.get('batch_size', batch_size))
        self.max_epochs = int(config.get('epochs', max_epochs))
        self.patience = int(config.get('patience', patience))
        self.apply_bio_validation = apply_bio_validation
        self.seed = seed
        
        # Set seed for reproducibility
        self._set_seed()

        # Device setup with graceful fallback to CPU
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                self.device = torch.device(device)
                # Test if device is actually available by trying to create a tensor
                if device == "cuda":
                    torch.zeros(1).to(self.device)
            except (AssertionError, RuntimeError) as e:
                print(f"⚠️  Warning: Could not use device '{device}' ({e}). Falling back to CPU.")
                self.device = torch.device("cpu")

        # Components
        self.embedding_handler = None
        self.char_vocab = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.is_trained = False
        self.best_score = 0.0
        self.label2idx = {}
        self.idx2label = {}

        logger.info(f"Initialized BiLSTM trainer with device: {self.device}")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        import torch
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Set random seed to {self.seed} for reproducibility")

    def load_data_from_jsonl(self, file_path: Union[str, Path]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load data from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            Tuple of (sentences, labels)
        """
        file_path = Path(file_path)
        sentences = []
        labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line.strip())
                    if 'tokens' in example and 'labels' in example:
                        sentences.append(example['tokens'])
                        labels.append(example['labels'])

        logger.info(f"Loaded {len(sentences)} examples from {file_path}")
        return sentences, labels

    def build_label_vocabulary(self, all_labels: List[List[str]]) -> None:
        """Build label vocabulary."""
        label_set = set()
        for label_seq in all_labels:
            label_set.update(label_seq)

        self.label2idx = {label: idx for idx, label in enumerate(sorted(label_set))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        logger.info(f"Built label vocabulary: {len(self.label2idx)} labels")

    def prepare_embeddings(self, sentences: List[List[str]]) -> None:
        """Prepare FastText embeddings for Portuguese."""
        logger.info("Preparing embeddings...")

        # Initialize embedding handler
        self.embedding_handler = FastTextEmbedding(
            embedding_dim=self.embedding_dim
        )

        # Load pre-trained embeddings
        self.embedding_handler.load_pretrained_embeddings()

        # Build vocabulary from training data
        self.embedding_handler.build_vocabulary(sentences)

        # Create embedding matrix
        embedding_matrix = self.embedding_handler.create_embedding_matrix()

        # Character vocabulary if using char CNN
        if self.use_char_cnn:
            self.char_vocab = CharacterVocab()
            self.char_vocab.build_vocab(sentences)

        logger.info("Embeddings prepared successfully")

    def create_model(self) -> None:
        """Create BiLSTM model."""
        if self.embedding_handler is None:
            raise ValueError("Embeddings must be prepared before creating model")

        logger.info("Creating BiLSTM model...")

        # Model parameters
        model_params = {
            'vocab_size': self.embedding_handler.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_labels': len(self.label2idx),
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'use_crf': self.use_crf,
            'use_char_cnn': self.use_char_cnn,
            'pretrained_embeddings': self.embedding_handler.embeddings,
            'freeze_embeddings': False
        }

        if self.use_char_cnn and self.char_vocab:
            model_params['char_vocab_size'] = len(self.char_vocab.char2idx)

        # Create model
        self.model = BiLSTMFastTextVotIE(**model_params)
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")

    def create_data_loaders(
        self,
        train_sentences: List[List[str]],
        train_labels: List[List[str]],
        val_sentences: Optional[List[List[str]]] = None,
        val_labels: Optional[List[List[str]]] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create data loaders."""
        # Training dataset
        train_dataset = BiLSTMDataset(
            train_sentences, train_labels,
            self.embedding_handler, self.char_vocab, self.label2idx
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
        )

        # Validation dataset
        val_loader = None
        if val_sentences and val_labels:
            val_dataset = BiLSTMDataset(
                val_sentences, val_labels,
                self.embedding_handler, self.char_vocab, self.label2idx
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
            )

        return train_loader, val_loader

    def train(
        self,
        train_file: Union[str, Path],
        val_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Train the BiLSTM model.

        Args:
            train_file: Training data file
            val_file: Validation data file (optional)

        Returns:
            Training results
        """
        logger.info("Starting BiLSTM training...")
        start_time = time.time()

        # Load data
        train_sentences, train_labels = self.load_data_from_jsonl(train_file)

        val_sentences, val_labels = None, None
        if val_file:
            val_sentences, val_labels = self.load_data_from_jsonl(val_file)

        # Build vocabularies
        self.build_label_vocabulary(train_labels + (val_labels or []))
        self.prepare_embeddings(train_sentences)

        # Create model
        self.create_model()

        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_sentences, train_labels, val_sentences, val_labels
        )

        # Setup optimizer and scheduler
        logger.info(f"Training configuration:")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Weight decay (L2): {self.weight_decay}")
        logger.info(f"  Dropout: {self.dropout}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Max epochs: {self.max_epochs}")
        logger.info(f"  Patience: {self.patience}")
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

        # Training loop
        best_val_f1 = 0.0
        epochs_without_improvement = 0

        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }

        for epoch in range(self.max_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")

            # Training
            train_loss = self._train_epoch(train_loader)
            training_history['train_loss'].append(train_loss)

            # Validation
            if val_loader:
                val_loss, val_f1 = self._validate_epoch(val_loader)
                training_history['val_loss'].append(val_loss)
                training_history['val_f1'].append(val_f1)

                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

                # Learning rate scheduling
                self.scheduler.step(val_f1)

                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epochs_without_improvement = 0
                    self.best_score = val_f1
                    # Save best model
                    self._save_checkpoint("best_model")
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
            else:
                logger.info(f"Train Loss: {train_loss:.4f}")

        training_time = time.time() - start_time

        # Save final results
        results = {
            'best_eval_score': best_val_f1,
            'total_epochs': epoch + 1,
            'training_time': training_time,
            'training_history': training_history,
            'model_params': {
                'vocab_size': self.embedding_handler.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_labels': len(self.label2idx),
                'use_crf': self.use_crf,
                'use_char_cnn': self.use_char_cnn
            },
            'training_params': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'dropout': self.dropout,
                'batch_size': self.batch_size,
                'patience': self.patience
            }
        }



        # Save training results
        self._save_training_results(results)

        # Save vocabularies
        self.embedding_handler.save_vocabulary(self.output_dir)

        if self.char_vocab:
            with open(self.output_dir / "char_vocab.json", 'w') as f:
                json.dump({
                    'char2idx': self.char_vocab.char2idx,
                    'idx2char': self.char_vocab.idx2char,
                    'max_char_len': self.char_vocab.max_char_len
                }, f, indent=2)

        self.is_trained = True
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return results

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            outputs = self.model(**batch)
            loss = outputs['loss']

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch_device = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch_device)
                loss = outputs['loss']

                total_loss += loss.item()
                num_batches += 1

                # Get predictions
                predictions = self.model.decode(
                    batch_device['input_ids'],
                    batch_device['attention_mask'],
                    batch_device.get('char_ids')
                )

                # Get true labels
                labels = batch['labels'].cpu().numpy()
                attention_mask = batch['attention_mask'].cpu().numpy()

                for pred, label, mask in zip(predictions, labels, attention_mask):
                    # Get valid positions
                    valid_len = mask.sum()
                    pred_valid = pred[:valid_len]
                    label_valid = label[:valid_len]

                    # Remove padding tokens
                    label_valid = [l for l in label_valid if l != -100]

                    if len(pred_valid) == len(label_valid):
                        all_predictions.append(pred_valid)
                        all_labels.append(label_valid)

        avg_loss = total_loss / num_batches

        # Calculate F1 score using the common evaluator
        # Convert to label strings
        pred_labels = [[self.idx2label[p] for p in pred] for pred in all_predictions]
        true_labels = [[self.idx2label[l] for l in label] for label in all_labels]

        # Apply BIO validation to ensure valid tag sequences
        from src.data.dataset import validate_bio_sequence
        validated_pred_labels = []
        for i, pred_seq in enumerate(pred_labels):
            validated_seq = validate_bio_sequence(
                pred_seq,
                scheme='BIO',
                example_id=f"bilstm_{i}"
            )
            validated_pred_labels.append(validated_seq)

        # Use the entity-level evaluator
        evaluator = EntityLevelEvaluator()
        metrics = evaluator.compute_metrics(validated_pred_labels, true_labels)
        f1_score = metrics.get('entity_f1', 0.0)

        return avg_loss, f1_score

    def evaluate(self, test_file: Union[str, Path]) -> Dict[str, Any]:
        """Evaluate on test data using the common evaluator."""
        logger.info(f"Evaluating on {test_file}")

        # Load test data
        test_sentences, test_labels = self.load_data_from_jsonl(test_file)

        # Create dataset
        test_dataset = BiLSTMDataset(
            test_sentences, test_labels,
            self.embedding_handler, self.char_vocab, self.label2idx
        )

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        # Get predictions and labels
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                batch_device = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch_device)
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1

                # Get predictions
                predictions = self.model.decode(
                    batch_device['input_ids'],
                    batch_device['attention_mask'],
                    batch_device.get('char_ids')
                )

                # Get true labels
                labels = batch['labels'].cpu().numpy()
                attention_mask = batch['attention_mask'].cpu().numpy()

                for pred, label, mask in zip(predictions, labels, attention_mask):
                    # Get valid positions
                    valid_len = mask.sum()
                    pred_valid = pred[:valid_len]
                    label_valid = label[:valid_len]

                    # Remove padding tokens
                    label_valid = [l for l in label_valid if l != -100]

                    if len(pred_valid) == len(label_valid):
                        all_predictions.append(pred_valid)
                        all_labels.append(label_valid)

        # Convert to label strings
        pred_labels = [[self.idx2label[p] for p in pred] for pred in all_predictions]
        true_labels = [[self.idx2label[l] for l in label] for label in all_labels]

        # Apply BIO validation to ensure valid tag sequences
        from src.data.dataset import validate_bio_sequence
        validated_pred_labels = []
        for i, pred_seq in enumerate(pred_labels):
            validated_seq = validate_bio_sequence(
                pred_seq,
                scheme='BIO',
                example_id=f"bilstm_{i}"
            )
            validated_pred_labels.append(validated_seq)

        # Use the entity-level evaluator to get detailed metrics
        evaluator = EntityLevelEvaluator()
        metrics = evaluator.compute_metrics(validated_pred_labels, true_labels)

        # Create results in the same format as other models
        results = {
            'eval_loss': total_loss / num_batches,
            'entity_f1_strict': metrics.get('entity_f1', 0.0),
            'entity_precision': metrics.get('entity_precision', 0.0),
            'entity_recall': metrics.get('entity_recall', 0.0),
            'entity_accuracy': metrics.get('entity_accuracy', 0.0),
            'evaluation_method': 'seqeval_entity_level',
            'per_type_metrics': metrics.get('per_type_metrics', {}),
            'num_examples': len(test_sentences)
        }

        logger.info(f"Test F1: {results['entity_f1_strict']:.4f}")
        logger.info(f"Test Precision: {results['entity_precision']:.4f}")
        logger.info(f"Test Recall: {results['entity_recall']:.4f}")

        return results

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        # Save directly in output_dir like transformers, not in subdirectory
        self.output_dir.mkdir(exist_ok=True)

        # Save model
        self.model.save_pretrained(self.output_dir)

        # Save label mapping
        with open(self.output_dir / "label_mapping.json", 'w') as f:
            json.dump({
                'label2idx': self.label2idx,
                'idx2label': self.idx2label
            }, f, indent=2)

        logger.info(f"Checkpoint saved to {self.output_dir}")

    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results."""
        results_file = self.output_dir / "training_results.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training results saved to {results_file}")


def train_bilstm_fasttext(
    config: Dict[str, Any],
    train_file: Union[str, Path],
    val_file: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train BiLSTM + FastText model.

    Args:
        config: Training configuration
        train_file: Training data file
        val_file: Validation data file
        output_dir: Output directory
        **kwargs: Additional trainer arguments

    Returns:
        Training results with detailed evaluation metrics
    """
    if output_dir is None:
        output_dir = Path("models") / "bilstm_fasttext"

    # Create trainer
    trainer = BiLSTMTrainer(
        config=config,
        output_dir=output_dir,
        **kwargs
    )

    # Train model
    results = trainer.train(
        train_file=train_file,
        val_file=val_file
    )

    logger.info("BiLSTM + FastText training completed")

    return results