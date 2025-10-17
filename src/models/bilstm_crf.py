#!/usr/bin/env python3
"""
BiLSTM + FastText Model for Portuguese Voting NER

Implements a traditional BiLSTM-CRF model using FastText embeddings as a baseline. This provides insight into the performance gains from contextualized vs static embeddings.

Architecture:
    Input → FastText Embeddings → BiLSTM → Dropout → Linear → CRF → Output

Features:
- Facebook's pre-trained FastText embeddings (300d)
- Bidirectional LSTM for sequence modeling
- CRF layer for sequence consistency
- Character-level CNN for OOV handling
- Lightweight compared to transformer models

The model serves as a traditional NER baseline, demonstrating the effectiveness
of static embeddings with sequence modeling for Named Entity Recognition in
Portuguese voting documents.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import pickle
import json
from collections import defaultdict, Counter
import re
import fasttext
import fasttext.util

logger = logging.getLogger(__name__)


class FastTextEmbedding:
    """
    Handles loading pre-trained embeddings, vocabulary management,
    and OOV token handling for the BiLSTM model.
    """

    def __init__(
        self,
        embedding_dim: int = 300,
        max_vocab_size: int = 50000,
        min_freq: int = 2
    ):
        """
        Initialize embedding handler.

        Args:
            embedding_dim: Embedding dimension
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for vocabulary inclusion
        """
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.SPECIAL_TOKENS = [self.PAD_TOKEN, self.UNK_TOKEN]

        # Vocabulary and embeddings
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        self.pretrained_model = None

        # Statistics
        self.vocab_size = 0
        self.oov_count = 0
        self.total_tokens = 0

        logger.info(f"Initialized FastText embedding handler for Portuguese with dim {embedding_dim}")

    def _download_fasttext_embeddings(self, model_path: Path) -> bool:
        """
        Download Facebook's pre-trained FastText embeddings for Portuguese.
        
        Args:
            model_path: Path where to save the downloaded model
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            import urllib.request
            import gzip
            import shutil
            
            logger.info("📥 Downloading Facebook's pre-trained FastText embeddings for Portuguese...")
            logger.info("This may take a few minutes depending on your internet connection...")
                
            
            # Check available disk space
            try:
                import shutil as disk_util
                free_space = disk_util.disk_usage(model_path.parent).free
                required_space = 20 * 1024**3  # 20GB to be safe
                
                if free_space < required_space:
                    logger.warning(f"⚠️  Low disk space detected: {free_space / 1024**3:.1f}GB available")
                    logger.warning("⚠️  Recommended: At least 20GB free space")
                    
                    response = input("Continue download anyway? (y/N): ")
                    if response.lower() != 'y':
                        logger.info("Download cancelled by user")
                        return False
                        
            except Exception:
                pass  # Ignore disk space check errors

            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz"

            # Download compressed file
            compressed_path = model_path.with_suffix('.bin.gz')
            logger.info(f"Downloading from: {url}")
            logger.info(f"Saving to: {compressed_path}")
            
            # Try to use tqdm for better progress, fallback to simple callback
       
            from tqdm import tqdm
            
            class TqdmUpTo(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    return self.update(b * bsize - self.n)
            
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
                urllib.request.urlretrieve(url, compressed_path, reporthook=t.update_to)
                                    
            logger.info("✅ Download completed!")
            
            # Decompress the file
            logger.info("📦 Decompressing downloaded file...")
            logger.info("⏳ This may take a few minutes...")
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove compressed file to save space
            compressed_path.unlink()
            logger.info(f"✅ FastText embeddings ready at: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download FastText embeddings: {e}")
            logger.info("📋 Manual download options:")
            logger.info("   1. Download from: https://fasttext.cc/docs/en/crawl-vectors.html")
            logger.info("   2. Direct link: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz")
            logger.info(f"   3. Save as: {model_path}")
            logger.info("   4. Decompress the .gz file after downloading")
            return False

    def load_pretrained_embeddings(self) -> None:
        """Load pre-trained FastText embeddings using fasttext library."""
        try:
            # Get the path to the bilstm_fasttext folder
            bilstm_dir = Path(__file__).parent
            
            model_path = bilstm_dir / 'cc.pt.300.bin'
            logger.info(f"Loading Portuguese FastText embeddings from {model_path}...")

            # Check if file exists, if not try to download it
            if not model_path.exists():
                logger.warning(f"FastText model not found at {model_path}")
                logger.info("Attempting to download Facebook's pre-trained embeddings...")
                
                if not self._download_fasttext_embeddings(model_path):
                    raise FileNotFoundError("Could not download FastText model for Portuguese")

            # Load the FastText model
            logger.info("🔄 Loading FastText model (this may take a moment)...")
            self.pretrained_model = fasttext.load_model(str(model_path))
            self.embedding_dim = self.pretrained_model.get_dimension()
            logger.info(f"✅ Loaded FastText embeddings: dim={self.embedding_dim}")

        except Exception as e:
            logger.warning(f"Failed to load pre-trained FastText embeddings: {e}")
            logger.info("Will use random embeddings")
            self.pretrained_model = None

    def build_vocabulary(self, sentences: List[List[str]]) -> None:
        """
        Build vocabulary from training sentences.

        Args:
            sentences: List of tokenized sentences
        """
        logger.info("Building vocabulary from training data...")

        # Count word frequencies
        word_freq = Counter()
        for sentence in sentences:
            for word in sentence:
                word_freq[word.lower()] += 1
                self.total_tokens += 1

        # Add special tokens
        self.word2idx = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}

        # Add frequent words
        sorted_words = word_freq.most_common(self.max_vocab_size - len(self.SPECIAL_TOKENS))
        for word, freq in sorted_words:
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)

        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        logger.info(f"Built vocabulary: {self.vocab_size} words")
        logger.info(f"Total tokens: {self.total_tokens}")
        logger.info(f"Coverage: {len(sorted_words)} unique words from {len(word_freq)} total")

    def create_embedding_matrix(self) -> torch.Tensor:
        """
        Create embedding matrix with pre-trained vectors.

        Returns:
            Embedding matrix tensor
        """
        logger.info("Creating embedding matrix...")

        # Initialize random embeddings
        embedding_matrix = torch.randn(self.vocab_size, self.embedding_dim) * 0.1

        # Set PAD token to zeros
        embedding_matrix[self.word2idx[self.PAD_TOKEN]] = torch.zeros(self.embedding_dim)

        if self.pretrained_model is not None:
            # Fill with pre-trained vectors where available
            hits = 0
            for word, idx in self.word2idx.items():
                if word in self.SPECIAL_TOKENS:
                    continue

                try:
                    # FastText can handle any word, even OOV words
                    word_vector = self.pretrained_model.get_word_vector(word)
                    embedding_matrix[idx] = torch.tensor(word_vector)
                    hits += 1
                except Exception:
                    # Fallback to random initialization for any errors
                    pass

            coverage = hits / (self.vocab_size - len(self.SPECIAL_TOKENS)) * 100
            logger.info(f"Pre-trained embedding coverage: {hits}/{self.vocab_size-len(self.SPECIAL_TOKENS)} ({coverage:.1f}%)")
        else:
            logger.info("Using random embeddings (no pre-trained model)")

        self.embeddings = embedding_matrix
        return embedding_matrix

    def encode_sentences(self, sentences: List[List[str]]) -> List[List[int]]:
        """
        Convert sentences to token indices.

        Args:
            sentences: List of tokenized sentences

        Returns:
            List of encoded sentences
        """
        encoded = []
        for sentence in sentences:
            encoded_sentence = []
            for word in sentence:
                word_lower = word.lower()
                if word_lower in self.word2idx:
                    encoded_sentence.append(self.word2idx[word_lower])
                else:
                    encoded_sentence.append(self.word2idx[self.UNK_TOKEN])
                    self.oov_count += 1
            encoded.append(encoded_sentence)

        return encoded

    def save_vocabulary(self, save_path: Path) -> None:
        """Save vocabulary and embedding info."""
        vocab_info = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        }

        with open(save_path / "vocabulary.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, indent=2, ensure_ascii=False)

        # Save embedding matrix
        if self.embeddings is not None:
            torch.save(self.embeddings, save_path / "embeddings.pt")

        logger.info(f"Saved vocabulary to {save_path}")

    @classmethod
    def load_vocabulary(cls, load_path: Path) -> 'FastTextEmbedding':
        """Load saved vocabulary and embeddings."""
        with open(load_path / "vocabulary.json", 'r', encoding='utf-8') as f:
            vocab_info = json.load(f)

        # Create instance
        embedding_handler = cls(
            embedding_dim=vocab_info['embedding_dim']
        )

        # Load vocabulary
        embedding_handler.word2idx = vocab_info['word2idx']
        embedding_handler.idx2word = vocab_info['idx2word']
        embedding_handler.vocab_size = vocab_info['vocab_size']

        # Load embeddings if available
        embedding_path = load_path / "embeddings.pt"
        if embedding_path.exists():
            embedding_handler.embeddings = torch.load(embedding_path, map_location='cpu')

        logger.info(f"Loaded vocabulary from {load_path}")
        return embedding_handler


class CharCNN(nn.Module):
    """
    Character-level CNN for handling OOV words.

    Processes character sequences to create word-level representations
    that complement FastText embeddings for unknown words. Uses multiple
    convolutional filters with different kernel sizes to capture
    character-level patterns.
    """

    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 25,
        char_hidden_dim: int = 50,
        kernel_sizes: List[int] = [2, 3, 4],
        dropout: float = 0.2
    ):
        super().__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)

        # Convolutional layers for different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, char_hidden_dim, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_dim = len(kernel_sizes) * char_hidden_dim

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for character CNN.

        Args:
            char_ids: Character IDs tensor [batch_size, seq_len, max_char_len]

        Returns:
            Character-based word representations [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, max_char_len = char_ids.shape

        # Reshape for processing
        char_ids = char_ids.view(-1, max_char_len)  # [batch_size * seq_len, max_char_len]

        # Character embeddings
        char_embeds = self.char_embedding(char_ids)  # [batch_size * seq_len, max_char_len, char_embed_dim]
        char_embeds = char_embeds.transpose(1, 2)  # [batch_size * seq_len, char_embed_dim, max_char_len]

        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(char_embeds))  # [batch_size * seq_len, char_hidden_dim, conv_len]
            conv_out = torch.max(conv_out, dim=2)[0]  # [batch_size * seq_len, char_hidden_dim]
            conv_outputs.append(conv_out)

        # Concatenate outputs
        char_features = torch.cat(conv_outputs, dim=1)  # [batch_size * seq_len, output_dim]
        char_features = self.dropout(char_features)

        # Reshape back
        char_features = char_features.view(batch_size, seq_len, -1)

        return char_features


class BiLSTMFastTextVotIE(nn.Module):
    """
    BiLSTM model with FastText embeddings for VotIE baselines.

    Architecture:
        Input → [FastText Embeddings + Char CNN] → BiLSTM → Dropout → Linear → [CRF] → Output

    This model serves as a traditional baseline for comparison with BERT-based models,
    demonstrating the performance difference between static and contextualized embeddings
    for Vote Information Extraction in Portuguese voting documents.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of FastText embeddings (typically 300)
        hidden_dim: Hidden dimension of BiLSTM layers
        num_labels: Number of NER labels
        num_layers: Number of BiLSTM layers
        dropout: Dropout rate for regularization
        use_crf: Whether to use CRF layer for sequence tagging
        use_char_cnn: Whether to use character-level CNN
        char_vocab_size: Size of character vocabulary
        pretrained_embeddings: Pre-trained FastText embeddings
        freeze_embeddings: Whether to freeze embedding weights
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_crf: bool = True,
        use_char_cnn: bool = True,
        char_vocab_size: int = 100,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.use_crf = use_crf
        self.use_char_cnn = use_char_cnn

        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.word_embedding.weight.requires_grad = False

        # Character CNN
        if use_char_cnn:
            self.char_cnn = CharCNN(char_vocab_size)
            lstm_input_dim = embedding_dim + self.char_cnn.output_dim
        else:
            self.char_cnn = None
            lstm_input_dim = embedding_dim

        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.linear = nn.Linear(hidden_dim * 2, num_labels)  # *2 for bidirectional

        # CRF layer (optional)
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Word token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len] (optional)
            char_ids: Character IDs [batch_size, seq_len, max_char_len] (optional)

        Returns:
            Dictionary with loss and logits
        """
        batch_size, seq_len = input_ids.shape

        # Word embeddings
        word_embeds = self.word_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # Character embeddings
        if self.use_char_cnn and char_ids is not None:
            char_embeds = self.char_cnn(char_ids)  # [batch_size, seq_len, char_output_dim]
            # Concatenate word and character embeddings
            embeddings = torch.cat([word_embeds, char_embeds], dim=2)
        else:
            embeddings = word_embeds

        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)

        # Get sequence lengths for packing
        seq_lengths = attention_mask.sum(dim=1).cpu()

        # Pack sequences for efficiency
        packed_embeddings = pack_padded_sequence(
            embeddings, seq_lengths, batch_first=True, enforce_sorted=False
        )

        # BiLSTM
        packed_lstm_out, _ = self.bilstm(packed_embeddings)

        # Unpack sequences
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True, total_length=seq_len)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Linear projection
        logits = self.linear(lstm_out)  # [batch_size, seq_len, num_labels]

        # Calculate loss
        loss = None
        if labels is not None:
            if self.use_crf:
                # CRF loss - need to handle -100 ignore indices
                # CRF doesn't handle ignore_index like CrossEntropyLoss, so we need to mask
                valid_mask = (labels != -100) & (attention_mask.bool())

                # Replace -100 with 0 for CRF (will be masked out anyway)
                crf_labels = labels.clone()
                crf_labels[labels == -100] = 0

                log_likelihood = self.crf(logits, crf_labels, mask=valid_mask, reduction='mean')
                loss = -log_likelihood
            else:
                # Cross-entropy loss
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)

        return {
            'loss': loss,
            'logits': logits
        }

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Decode predictions from the model.

        Args:
            input_ids: Word token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            char_ids: Character IDs [batch_size, seq_len, max_char_len] (optional)

        Returns:
            List of predicted label sequences
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, char_ids=char_ids)
            logits = outputs['logits']

            if self.use_crf:
                # CRF decoding
                predictions = self.crf.decode(logits, mask=attention_mask.bool())
            else:
                # Argmax decoding
                predictions = torch.argmax(logits, dim=-1)
                # Apply mask and convert to list
                predictions = predictions.cpu().numpy()
                predictions = [
                    pred[:length].tolist()
                    for pred, length in zip(predictions, attention_mask.sum(dim=1).cpu().numpy())
                ]

        return predictions

    def save_pretrained(self, save_path: Union[str, Path]) -> None:
        """Save model to directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_labels': self.num_labels,
            'num_layers': self.num_layers,
            'use_crf': self.use_crf,
            'use_char_cnn': self.use_char_cnn,
            'char_vocab_size': self.char_cnn.char_embedding.weight.shape[0] if self.use_char_cnn else 100,
            'model_type': 'bilstm_fasttext'
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> 'BiLSTMFastTextVotIE':
        """Load model from directory."""
        model_path = Path(model_path)

        # Load config
        with open(model_path / "config.json", 'r') as f:
            config = json.load(f)

        # Create model
        model = cls(**{k: v for k, v in config.items() if k != 'model_type'})

        # Load weights
        model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location='cpu'))

        logger.info(f"Model loaded from {model_path}")
        return model


class CharacterVocab:
    """
    Character vocabulary helper for character-level CNN.
    
    Builds character-to-index mappings and handles character sequence encoding
    for the character-level CNN component of the BiLSTM model.
    """

    def __init__(self, max_char_len: int = 20):
        self.max_char_len = max_char_len
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2char = {0: '<PAD>', 1: '<UNK>'}

    def build_vocab(self, sentences: List[List[str]]) -> None:
        """Build character vocabulary from sentences."""
        chars = set()
        for sentence in sentences:
            for word in sentence:
                chars.update(word.lower())

        for char in sorted(chars):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

        logger.info(f"Built character vocabulary: {len(self.char2idx)} characters")

    def encode_words(self, words: List[str]) -> List[List[int]]:
        """Encode words as character sequences."""
        encoded = []
        for word in words:
            char_ids = []
            for char in word.lower()[:self.max_char_len]:
                char_ids.append(self.char2idx.get(char, self.char2idx['<UNK>']))

            # Pad to max length
            while len(char_ids) < self.max_char_len:
                char_ids.append(self.char2idx['<PAD>'])

            encoded.append(char_ids)

        return encoded


if __name__ == "__main__":
    # Example usage
    logger.setLevel(logging.INFO)

    # Test embedding handler
    embedding_handler = FastTextEmbedding()

    # Sample sentences
    sentences = [
        ["O", "vereador", "João", "votou", "a", "favor"],
        ["A", "proposta", "foi", "aprovada", "por", "unanimidade"]
    ]

    # Build vocabulary
    embedding_handler.build_vocabulary(sentences)

    # Create embedding matrix
    embedding_matrix = embedding_handler.create_embedding_matrix()

    # Create model
    model = BiLSTMFastTextVotIE(
        vocab_size=embedding_handler.vocab_size,
        embedding_dim=embedding_handler.embedding_dim,
        hidden_dim=128,
        num_labels=15,  # Voting NER labels
        pretrained_embeddings=embedding_matrix
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")