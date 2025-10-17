"""
Model utilities for parameter counting, checkpointing, and model information.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    Get model size information including memory usage.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_counts = count_parameters(model)
    
    # Calculate memory usage (rough estimate)
    total_params = param_counts['total_parameters']
    # Assume 4 bytes per float32 parameter
    memory_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        **param_counts,
        'memory_mb': memory_mb,
        'memory_gb': memory_mb / 1024
    }


def save_model_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                         checkpoint_path: str, epoch: int, loss: float, 
                         additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        loss: Current loss value
        additional_info: Additional information to save
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'model_info': get_model_size(model)
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, 
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to restore (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}, Loss: {checkpoint.get('loss', 'unknown')}")
    
    return checkpoint
