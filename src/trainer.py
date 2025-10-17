#!/usr/bin/env python
"""
Production-Ready VotIE Trainer

A trainer implementation for sequence labeling tasks, designed for reproducible research. Features comprehensive error handling and logging.

Key Features:
    - Paper-compliant training methodology
    - Comprehensive error handling (OOM, device failures, checkpoints)
    - Robust memory management and resource cleanup
    - Logging and monitoring
    - Type-safe parameter validation

Example:
    Basic usage for NER training:
    
    ```python
    from transformers import AutoTokenizer
    from .trainer import VotIETrainer
    from .model import BertimbauLinearVotIE
    
    # Initialize model and trainer
    model = BertimbauLinearVotIE("neuralmind/bert-large-portuguese-cased", num_labels=3)
    trainer = VotIETrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir="./results",
        id_to_label={0: "O", 1: "B-VOTE", 2: "I-VOTE"},
        num_train_epochs=10,
        learning_rate=2e-5
    )
    
    # Train
    results = trainer.train()
    
    # Evaluate on test set
    test_results = trainer.final_evaluate(test_dataloader, "test")
    ```

Technical Implementation:
    - AdamW optimizer with linear warmup scheduling
    - Gradient clipping for stability (max_grad_norm=1.0)
    - Early stopping with patience-based monitoring
    - Entity-level evaluation using seqeval metrics and event-level evaluation following proposed methodology on paper.
    - Automatic checkpoint management and cleanup
    - HuggingFace-compatible model saving format
"""

import gc
import json
import logging
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from safetensors.torch import save_file as save_safetensors
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from src.models.base import BaseVotIEModel
from src.data.dataset import VotIEDataset
from src.evaluation.entity_metrics import EntityLevelEvaluator


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility across all random number generators.
    
    This ensures deterministic behavior for:
    - Python's random module
    - NumPy random number generation
    - PyTorch CPU operations
    - PyTorch CUDA operations
    - cuDNN backend operations
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 Random seed set to {seed} for reproducibility")


# Configure logging
logger = logging.getLogger(__name__)

# Memory management constants
MEMORY_CLEANUP_THRESHOLD = 0.85  # Clean up when GPU memory usage exceeds 85%
OOM_BATCH_SIZE_REDUCTION = 0.75  # Reduce batch size by 25% on OOM


class VotIETrainer:
    """
    Trainer for transformer models.
    
    Implements industry best practices for robust, reproducible NER training following
    the methodology from Portuguese BERT paper. Features comprehensive error handling,
    memory management, and automatic recovery mechanisms.
    
    This trainer is designed for scientific publication and production deployment,
    with extensive validation, logging, and monitoring capabilities.
    
    Args:
        model: The transformer model to train (must inherit from BaseVotIEModel)
        train_dataset: Training dataset (VotIEDataset instance)
        eval_dataset: Evaluation/validation dataset (VotIEDataset instance)
        output_dir: Directory for saving models and logs
        id_to_label: Mapping from label IDs to label strings (e.g., {0: "O", 1: "B-PER"})
        learning_rate: Learning rate for all model parameters (default: 5e-5)
        num_train_epochs: Maximum number of training epochs (default: 10)
        train_batch_size: Training batch size (default: 16)
        eval_batch_size: Evaluation batch size (default: 32)
        warmup_proportion: Proportion of training for warmup (default: 0.1)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        patience: Early stopping patience in epochs (default: 3)
        logging_steps: Steps between log messages (default: 100)
        seed: Random seed for reproducibility (default: 42)
        base_model: Base model name for tokenizer creation (optional)
        tokenizer: Pre-initialized tokenizer (optional)
        device: CUDA device ID (-1 for CPU, >=0 for GPU) (default: -1)

        save_only_best_model: Only save the best model checkpoint (default: True)
        dataloader_num_workers: Number of workers for data loading (default: 0)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        
    Raises:
        ValueError: If parameters are invalid or inconsistent
        TypeError: If arguments have incorrect types
        RuntimeError: If initialization fails
        
    Example:
        ```python
        trainer = VotIETrainer(
            model=bert_ner_model,
            train_dataset=train_data,
            eval_dataset=dev_data,
            output_dir="./results/bert_linear",
            id_to_label={0: "O", 1: "B-VOTE", 2: "I-VOTE"},
            num_train_epochs=5,
            learning_rate=2e-5,
            patience=3
        )
        results = trainer.train()
        ```
    """
    
    def __init__(
        self,
        model: BaseVotIEModel,
        train_dataset: VotIEDataset,
        eval_dataset: VotIEDataset,
        output_dir: Union[str, Path],
        id_to_label: Dict[int, str],
        learning_rate: float = 5e-5,
        num_train_epochs: int = 10,
        train_batch_size: int = 16,
        eval_batch_size: int = 32,
        warmup_proportion: float = 0.1,
        max_grad_norm: float = 1.0,
        patience: int = 3,
        logging_steps: int = 100,
        seed: int = 42,
        base_model: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: int = -1,
        apply_bio_validation: bool = True,

        save_only_best_model: bool = True,
        no_save_models: bool = False,
        dataloader_num_workers: int = 0,
        pin_memory: bool = True
    ) -> None:
        # Validate inputs with comprehensive error messages
        self._validate_initialization_parameters(
            model, train_dataset, eval_dataset, output_dir, id_to_label,
            learning_rate, num_train_epochs,
            train_batch_size, eval_batch_size, warmup_proportion,
            max_grad_norm, patience,
            logging_steps, seed, device
        )
        
        # Core components
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.id_to_label = id_to_label
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.apply_bio_validation = apply_bio_validation

        # Initialize evaluator for entity-level evaluation
        try:
            self.evaluator = EntityLevelEvaluator(id_to_label)
            logger.info(f"✅ Initialized evaluator with {len(id_to_label)} labels")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EntityLevelEvaluator: {e}") from e
        
        # Training hyperparameters with validation
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.warmup_proportion = warmup_proportion
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.logging_steps = logging_steps
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Extended configuration
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Model saving configuration
        self.save_only_best_model = save_only_best_model
        self.no_save_models = no_save_models
        
        # Training state management
        self.global_step = 0
        self.best_eval_score = 0.0
        self.patience_counter = 0
        self.best_model_checkpoint: Optional[Path] = None
        self.training_interrupted = False
        self.oom_recovery_count = 0
        
        # Device setup with comprehensive error handling
        self.device = self._setup_device(device)
        
        # Memory management setup
        if self.device.type == "cuda":
            self._log_gpu_memory_status("Initial")
        else:
            self.scaler = None
            
        
        # Model initialization with error handling
        try:
            self.model.to(self.device)
            self._log_model_info()
        except Exception as e:
            raise RuntimeError(f"Failed to move model to device {self.device}: {e}") from e
        
        # Setup optimizer with validation
        try:
            self.optimizer = self._setup_optimizer()
            self.scheduler = None  # Will be set during training
        except Exception as e:
            raise RuntimeError(f"Failed to setup optimizer: {e}") from e
        
        # Final initialization log
        logger.info("🚀 VotIETrainer initialized successfully")
        self._log_configuration_summary()
    
    def _validate_initialization_parameters(
        self,
        model: BaseVotIEModel,
        train_dataset: VotIEDataset,
        eval_dataset: VotIEDataset,
        output_dir: Union[str, Path],
        id_to_label: Dict[int, str],
        learning_rate: float,
        num_train_epochs: int,
        train_batch_size: int,
        eval_batch_size: int,
        warmup_proportion: float,
        max_grad_norm: float,
        patience: int,
        logging_steps: int,
        seed: int,
        device: int
    ) -> None:
        """
        Comprehensive parameter validation with clear error messages.
        
        Raises:
            TypeError: If parameter types are incorrect
            ValueError: If parameter values are invalid
        """
        # Type validation
        if not isinstance(model, BaseVotIEModel):
            raise TypeError(f"model must be BaseVotIEModel instance, got {type(model)}")
        
        if not isinstance(train_dataset, VotIEDataset):
            raise TypeError(f"train_dataset must be VotIEDataset instance, got {type(train_dataset)}")
            
        if not isinstance(eval_dataset, VotIEDataset):
            raise TypeError(f"eval_dataset must be VotIEDataset instance, got {type(eval_dataset)}")
            
        if not isinstance(id_to_label, dict):
            raise TypeError(f"id_to_label must be dict, got {type(id_to_label)}")
        
        # Dataset validation
        if len(train_dataset) == 0:
            raise ValueError("train_dataset cannot be empty")
            
        if len(eval_dataset) == 0:
            raise ValueError("eval_dataset cannot be empty")
        
        # Label mapping validation
        if not id_to_label:
            raise ValueError("id_to_label cannot be empty")
            
        for label_id, label_str in id_to_label.items():
            if not isinstance(label_id, int):
                raise TypeError(f"Label ID must be int, got {type(label_id)} for {label_id}")
            if not isinstance(label_str, str):
                raise TypeError(f"Label string must be str, got {type(label_str)} for {label_str}")
        
        # Numeric parameter validation
        if not 0 < learning_rate < 1:
            raise ValueError(f"learning_rate must be in (0, 1), got {learning_rate}")
            
        if not 1 <= num_train_epochs <= 1000:
            raise ValueError(f"num_train_epochs must be in [1, 1000], got {num_train_epochs}")
            
        if not 1 <= train_batch_size <= 512:
            raise ValueError(f"train_batch_size must be in [1, 512], got {train_batch_size}")
            
        if not 1 <= eval_batch_size <= 512:
            raise ValueError(f"eval_batch_size must be in [1, 512], got {eval_batch_size}")
            
        if not 0 <= warmup_proportion <= 0.5:
            raise ValueError(f"warmup_proportion must be in [0, 0.5], got {warmup_proportion}")
            
        if not 0 < max_grad_norm <= 10:
            raise ValueError(f"max_grad_norm must be in (0, 10], got {max_grad_norm}")
            
        if not 1 <= patience <= 100:
            raise ValueError(f"patience must be in [1, 100], got {patience}")
        
        # Step validation
        for step_name, step_value in [
            ("logging_steps", logging_steps)
        ]:
            if not isinstance(step_value, int) or step_value < 1:
                raise ValueError(f"{step_name} must be positive integer, got {step_value}")
        
        # Device validation
        if not isinstance(device, int):
            raise TypeError(f"device must be int, got {type(device)}")
            
        if device >= 0 and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} requested but CUDA is not available")
            
        if device >= torch.cuda.device_count():
            raise ValueError(f"Device {device} not available. Available devices: 0-{torch.cuda.device_count()-1}")
        
        logger.info("✅ All initialization parameters validated successfully")

    def _setup_device(self, device: int) -> torch.device:
        """
        Setup and validate training device with comprehensive error handling.
        
        Args:
            device: Device ID (-1 for CPU, >=0 for CUDA)
            
        Returns:
            torch.device: Configured device
            
        Raises:
            RuntimeError: If device setup fails
        """
        try:
            if device >= 0 and torch.cuda.is_available():
                target_device = torch.device(f"cuda:{device}")
                # Validate device is accessible
                torch.cuda.set_device(device)
                
                # Log device information
                device_name = torch.cuda.get_device_name(device)
                memory_info = torch.cuda.get_device_properties(device)
                logger.info(f"🖥️  Using CUDA device {device}: {device_name}")
                logger.info(f"   Memory: {memory_info.total_memory / 1e9:.1f} GB")
                
                return target_device
            else:
                logger.info("🖥️  Using CPU device")
                return torch.device("cpu")
                
        except Exception as e:
            raise RuntimeError(f"Failed to setup device {device}: {e}") from e

    def _log_gpu_memory_status(self, context: str) -> None:
        """Log current GPU memory usage."""
        if self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                cached = torch.cuda.memory_reserved(self.device) / 1e9
                logger.info(f"🔧 {context} GPU memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            except Exception as e:
                logger.warning(f"Failed to log GPU memory status: {e}")

    def _log_model_info(self) -> None:
        """Log comprehensive model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model statistics:")
        logger.info(f"   Architecture: {self.model.__class__.__name__}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Frozen parameters: {total_params - trainable_params:,}")
        
        # Log model size estimate
        model_size_mb = total_params * 4 / 1e6  # Assuming fp32
        logger.info(f"   Estimated size: {model_size_mb:.1f} MB")

    def _log_configuration_summary(self) -> None:
        """Log training configuration summary."""
        logger.info("📋 Training Configuration:")
        logger.info(f"   Epochs: {self.num_train_epochs}")
        logger.info(f"   Batch sizes: train={self.train_batch_size}, eval={self.eval_batch_size}")
        logger.info(f"   Learning rate: {self.learning_rate}")
        logger.info(f"   Warmup proportion: {self.warmup_proportion}")
        logger.info(f"   Gradient clipping: {self.max_grad_norm}")
        logger.info(f"   Patience: {self.patience}")
        logger.info(f"   Output directory: {self.output_dir}")

    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer with single learning rate for all parameters."""
        # Collect all trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer_grouped_parameters = [
            {
                'params': trainable_params,
                'weight_decay': 0.01,
                'lr': self.learning_rate
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        
        logger.info(f"Optimizer setup - Learning Rate: {self.learning_rate} (uniform for all parameters)")
        
        return optimizer
    
    def _setup_scheduler(self, num_training_steps: int) -> LambdaLR:
        """Setup learning rate scheduler with warmup."""
        num_warmup_steps = int(num_training_steps * self.warmup_proportion)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Scheduler setup - Warmup steps: {num_warmup_steps}, Total steps: {num_training_steps}")
        return scheduler
    
    def _get_dataloader(self, dataset: VotIEDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create DataLoader with proper sampling."""
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        subtoken_mask = torch.stack([item['subtoken_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'subtoken_mask': subtoken_mask,
            'example_ids': [item['example_id'] for item in batch]
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        
        # Add progress bar for training
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                subtoken_mask=batch['subtoken_mask']
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            total_steps += 1
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # Update progress bar with current loss
            avg_loss = total_loss / total_steps
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                learning_rates = [group['lr'] for group in self.optimizer.param_groups]
                logger.info(
                    f"Step {self.global_step}: Loss = {avg_loss:.4f}, "
                    f"LR = {learning_rates}"
                )
                
        return {
            'train_loss': total_loss / total_steps if total_steps > 0 else 0.0,
            'steps': total_steps
        }
    
    def evaluate(self, dataloader: DataLoader, is_final_eval: bool = False, final_eval_prefix: str = "test") -> Dict[str, float]:
        """Evaluate the model using strict entity-level evaluation."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            # Add progress bar for evaluation
            progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    subtoken_mask=batch['subtoken_mask']
                )
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    total_steps += 1
                
                # Get predictions (BIO validation controlled by config)
                predictions = self.model.decode(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    subtoken_mask=batch['subtoken_mask'],
                    apply_bio_validation=self.apply_bio_validation,
                    id_to_label=self.id_to_label if self.apply_bio_validation else None
                )
                
                # Extract true labels (only for first subtokens)
                for i, (pred_seq, label_seq, mask) in enumerate(zip(
                    predictions, batch['labels'], batch['subtoken_mask']
                )):
                    # Get true labels for first subtokens only
                    true_labels = []
                    for j, (label, is_first) in enumerate(zip(label_seq, mask)):
                        if is_first and label.item() != -100:
                            true_labels.append(label.item())
                    
                    # Align predictions with true labels
                    if len(pred_seq) == len(true_labels):
                        all_predictions.append(pred_seq)
                        all_labels.append(true_labels)
                    else:
                        logger.warning(f"Length mismatch: pred={len(pred_seq)}, true={len(true_labels)}")
        
        eval_loss = total_loss / total_steps if total_steps > 0 else 0.0

        # Convert predictions and labels to strings for entity-level evaluation
        pred_label_strings = self.evaluator.convert_ids_to_labels(all_predictions)
        true_label_strings = self.evaluator.convert_ids_to_labels(all_labels)
                
        # Compute strict entity-level metrics
        entity_metrics = self.evaluator.compute_metrics(pred_label_strings, true_label_strings)
                
        
        eval_results = {
            'eval_loss': eval_loss,
            'entity_f1_strict': entity_metrics.get('entity_f1', 0.0),  # Use entity_f1 from EntityLevelEvaluator
            'entity_precision': entity_metrics.get('entity_precision', 0.0),
            'entity_recall': entity_metrics.get('entity_recall', 0.0),
            'entity_accuracy': entity_metrics.get('entity_accuracy', 0.0),
            'evaluation_method': 'seqeval',  # EntityLevelEvaluator uses seqeval
            'per_type_metrics': entity_metrics.get('per_type_metrics', {}),
            'num_examples': len(all_predictions)
        }
        
        return eval_results
      
    def save_model(self, save_path: Path, is_best: bool = False):
        """Save model in HuggingFace format with safetensors."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save training checkpoint in .pt format (only for resuming training)
        checkpoint_path = save_path / 'checkpoint.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_eval_score': self.best_eval_score,
        }, checkpoint_path)
        
        if is_best:
            # Save best model in HuggingFace
            self._save_huggingface_format(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def _save_huggingface_format(self, save_path: Path):
        """Save model in HuggingFace format with safetensors and tokenizer."""
        hf_dir = save_path
        hf_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights in safetensors format
        model_safetensors_path = hf_dir / "model.safetensors"
        save_safetensors(self.model.state_dict(), str(model_safetensors_path))
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(hf_dir))
            logger.info(f"Tokenizer saved to {hf_dir}")
        else:
            logger.warning("No tokenizer provided - creating tokenizer using base model name")
            # If no tokenizer provided, try to create one from base model
            if self.base_model:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                    tokenizer.save_pretrained(str(hf_dir))
                    logger.info(f"Tokenizer created and saved from {self.base_model} to {hf_dir}")
                except Exception as e:
                    logger.error(f"Failed to create tokenizer from {self.base_model}: {e}")
            elif hasattr(self.model, 'model_name'):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)
                    tokenizer.save_pretrained(str(hf_dir))
                    logger.info(f"Tokenizer created and saved from {self.model.model_name} to {hf_dir}")
                except Exception as e:
                    logger.error(f"Failed to create tokenizer from {self.model.model_name}: {e}")
            else:
                logger.warning("Cannot save tokenizer - no base model information available")
        
        # Create config.json with model configuration
        config = {
            "model_type": getattr(self.model, 'model_type', 'custom_ner'),
            "architecture": self.model.__class__.__name__,
            "num_labels": len(self.id_to_label),
            "id2label": self.id_to_label,
            "label2id": {label: idx for idx, label in self.id_to_label.items()},
            "torch_dtype": "float32"
        }
        
        # Add base model information (critical for loading)
        if self.base_model:
            config["base_model"] = self.base_model
        elif hasattr(self.model, 'model_name'):
            config["base_model"] = self.model.model_name
        
        # Add model-specific configurations from base model's config
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'config'):
            bert_config = self.model.bert.config
            config["hidden_size"] = getattr(bert_config, 'hidden_size', 768)
            config["num_attention_heads"] = getattr(bert_config, 'num_attention_heads', 12)
            config["num_hidden_layers"] = getattr(bert_config, 'num_hidden_layers', 12)
        elif hasattr(self.model, 'xlmr') and hasattr(self.model.xlmr, 'config'):
            xlmr_config = self.model.xlmr.config
            config["hidden_size"] = getattr(xlmr_config, 'hidden_size', 768)
            config["num_attention_heads"] = getattr(xlmr_config, 'num_attention_heads', 12)
            config["num_hidden_layers"] = getattr(xlmr_config, 'num_hidden_layers', 12)
        elif hasattr(self.model, 'spanbert') and hasattr(self.model.spanbert, 'config'):
            spanbert_config = self.model.spanbert.config
            config["hidden_size"] = getattr(spanbert_config, 'hidden_size', 768)
            config["num_attention_heads"] = getattr(spanbert_config, 'num_attention_heads', 12)
            config["num_hidden_layers"] = getattr(spanbert_config, 'num_hidden_layers', 12)
        elif hasattr(self.model, 'deberta') and hasattr(self.model.deberta, 'config'):
            deberta_config = self.model.deberta.config
            config["hidden_size"] = getattr(deberta_config, 'hidden_size', 768)
            config["num_attention_heads"] = getattr(deberta_config, 'num_attention_heads', 12)
            config["num_hidden_layers"] = getattr(deberta_config, 'num_hidden_layers', 12)
        
        # Save configuration
        config_path = hf_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save tokenizer config if available
        if hasattr(self.model, 'tokenizer'):
            tokenizer_config = {
                "tokenizer_class": self.model.tokenizer.__class__.__name__,
                "model_max_length": getattr(self.model.tokenizer, 'model_max_length', 512)
            }
            
            tokenizer_config_path = hf_dir / "tokenizer_config.json"
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, indent=2)
        
        # Create a simple README for the model
        readme_content = f"""# Vote NER Model

This model was trained for Portuguese vote identification Named Entity Recognition.

## Model Details
- Architecture: {self.model.__class__.__name__}
- Number of labels: {len(self.id_to_label)}
- Base model: {config.get('base_model', 'Unknown')}

## Labels
{chr(10).join([f"- {label}" for label in self.id_to_label.values()])}

## Usage
Load this model using the Vote NER framework or convert to standard HuggingFace format.

## Training Information
- Best F1 Score: {self.best_eval_score:.4f}
- Training completed at step: {self.global_step}
"""
        
        readme_path = hf_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Model saved in HuggingFace format to {hf_dir}")
        logger.info(f"  - Model weights (safetensors): {model_safetensors_path}")
        logger.info(f"  - Configuration: {config_path}")
        logger.info(f"  - README: {readme_path}")

    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results to JSON file with comprehensive metrics."""
        import json

        results_file = self.output_dir / "training_results.json"

        # Convert Path objects to strings for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, Path):
                serializable_results[key] = str(value)
            else:
                serializable_results[key] = value

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Training results saved to {results_file}")

    def _cleanup_checkpoints(self):
        """Clean up checkpoint directories after training to save space."""
        if self.no_save_models:
            logger.info("🧹 Cleaning up ALL temporary checkpoint directories (no_save_models=True)...")
        elif self.save_only_best_model:
            logger.info("🧹 Cleaning up temporary checkpoint directories (save_only_best_model=True)...")
        else:
            logger.info("🧹 Cleaning up checkpoint directories...")
        
        # Find all epoch directories
        epoch_dirs = list(self.output_dir.glob("epoch_*"))
        
        if epoch_dirs:
            total_size = 0
            cleaned_count = 0
            
            for epoch_dir in epoch_dirs:
                if epoch_dir.is_dir():
                    # Calculate directory size for logging
                    try:
                        for file_path in epoch_dir.rglob('*'):
                            if file_path.is_file():
                                total_size += file_path.stat().st_size
                    except:
                        pass
                    
                    # Remove the directory
                    try:
                        shutil.rmtree(epoch_dir)
                        logger.info(f"  🗑️  Removed checkpoint directory: {epoch_dir.name}")
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to remove {epoch_dir}: {e}")
            
            # Convert size to human readable format
            if total_size > 0:
                if total_size > 1024**3:  # GB
                    size_str = f"{total_size / (1024**3):.1f} GB"
                elif total_size > 1024**2:  # MB
                    size_str = f"{total_size / (1024**2):.1f} MB"
                else:  # KB
                    size_str = f"{total_size / 1024:.1f} KB"
                
                logger.info(f"✅ Cleaned up {cleaned_count} checkpoint directories (~{size_str} freed)")
                logger.info(f"📁 Only best model retained at: {self.output_dir / 'best_model'}")
            else:
                logger.info(f"✅ Cleaned up {cleaned_count} checkpoint directories")
        else:
            logger.info("ℹ️  No checkpoint directories found to clean up")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")

        # Setup data loaders
        train_dataloader = self._get_dataloader(self.train_dataset, self.train_batch_size, shuffle=True)
        eval_dataloader = self._get_dataloader(self.eval_dataset, self.eval_batch_size, shuffle=False)
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * self.num_train_epochs
        self.scheduler = self._setup_scheduler(total_steps)
        
        # Training history
        training_history = {
            'train_loss': [],
            'eval_loss': [],
            'entity_f1_strict': [],
            'learning_rates': []
        }
        
        # Initial evaluation
        eval_results = self.evaluate(eval_dataloader)
        logger.info(f"Initial evaluation: {eval_results}")
        
        start_time = time.time()
        
        # Add progress bar for epochs
        epoch_progress = tqdm(range(self.num_train_epochs), desc="Training Epochs")
        
        for epoch in epoch_progress:
            epoch_progress.set_description(f"Epoch {epoch + 1}/{self.num_train_epochs}")
            logger.info(f"Starting epoch {epoch + 1}/{self.num_train_epochs}")
            
            # Train epoch
            train_results = self.train_epoch(train_dataloader)
            
            # Evaluate
            eval_results = self.evaluate(eval_dataloader)
            
            # Update training history
            training_history['train_loss'].append(train_results['train_loss'])
            training_history['eval_loss'].append(eval_results['eval_loss'])
            training_history['entity_f1_strict'].append(eval_results['entity_f1_strict'])
            training_history['learning_rates'].append([group['lr'] for group in self.optimizer.param_groups])
            
            # Check for improvement (using strict entity F1 as main metric)
            current_score = eval_results['entity_f1_strict']
            is_best = current_score > self.best_eval_score
            
            if is_best:
                self.best_eval_score = current_score
                self.patience_counter = 0
                # Update best model checkpoint path
                epoch_dir = self.output_dir / f"epoch_{epoch + 1}"
                self.best_model_checkpoint = epoch_dir / 'checkpoint.pt'
                logger.info(f"New best score: {self.best_eval_score:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Save model 
            if not self.save_only_best_model or is_best:
                epoch_dir = self.output_dir / f"epoch_{epoch + 1}"
                self.save_model(epoch_dir, is_best=is_best)

            elif is_best:
                # For save_only_best_model=True, just track the best checkpoint path
                epoch_dir = self.output_dir / f"epoch_{epoch + 1}"
                self.best_model_checkpoint = epoch_dir / 'checkpoint.pt'
                # Save just the checkpoint data temporarily for later loading
                epoch_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = epoch_dir / 'checkpoint.pt'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'global_step': self.global_step,
                    'best_eval_score': self.best_eval_score,
                }, checkpoint_path)
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1} completed - "
                f"Train Loss: {train_results['train_loss']:.4f}, "
                f"Eval Loss: {eval_results['eval_loss']:.4f}, "
                f"Strict Entity F1: {eval_results['entity_f1_strict']:.4f}"
            )
            
            # Update epoch progress bar
            epoch_progress.set_postfix({
                'Loss': f"{train_results['train_loss']:.4f}",
                'Eval F1': f"{eval_results['entity_f1_strict']:.4f}",
                'Best': f"{self.best_eval_score:.4f}"
            })
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best evaluation score: {self.best_eval_score:.4f}")
        
        # Load best model before saving final model
        if self.best_model_checkpoint and self.best_model_checkpoint.exists():
            logger.info(f"Loading best model from {self.best_model_checkpoint}")
            checkpoint = torch.load(self.best_model_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Best model loaded with score: {checkpoint['best_eval_score']:.4f}")
        else:
            logger.warning("No best model checkpoint found, saving current model state")
        
        # Save final model in HuggingFace format directly in output_dir (no subdirectory)
        final_model_dir = self.output_dir
        
        if self.no_save_models:
            logger.info("Skipping final model save (no_save_models=True)")
            logger.info("🧹 Cleaning up ALL temporary checkpoints (no_save_models=True)")
            # Clean up all temporary checkpoints since we don't want to save anything
            self._cleanup_checkpoints()
        else:
            self._save_huggingface_format(final_model_dir)
            
            # Verify the best model was saved before cleanup
            if not (final_model_dir / "model.safetensors").exists():
                logger.error("❌ Best model was not saved properly - skipping checkpoint cleanup")
            else:
                logger.info(f"✅ Best model saved successfully to {final_model_dir}")
                # Clean up checkpoint directories to save space
                self._cleanup_checkpoints()

        # Save final results with comprehensive metadata
        final_results = {
            'best_eval_score': self.best_eval_score,
            'total_epochs': epoch + 1,
            'training_time': training_time,
            'training_history': training_history,
            'total_steps': self.global_step,
            'final_eval_results': eval_results,
            'best_model_path': str(final_model_dir),
            'model_params': {
                'base_model': self.base_model,
                'num_labels': len(self.id_to_label),
                'learning_rate': self.learning_rate,
                'train_batch_size': self.train_batch_size,
                'eval_batch_size': self.eval_batch_size,
                'warmup_proportion': self.warmup_proportion,
                'max_grad_norm': self.max_grad_norm,
                'patience': self.patience,
                'apply_bio_validation': self.apply_bio_validation
            }
        }

        # Save training results to JSON file
        self._save_training_results(final_results)

        return final_results