#!/usr/bin/env python3
"""
Device detection and selection utilities for PyTorch.

Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU with automatic detection.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_available_devices():
    """
    Get list of available devices.

    Returns:
        dict: Dictionary with device availability
    """
    devices = {
        'cuda': torch.cuda.is_available(),
        'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cpu': True  # Always available
    }

    return devices


def detect_best_device():
    """
    Automatically detect the best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_device(device_str: str = 'auto'):
    """
    Get torch device from string specification.

    Args:
        device_str: Device specification ('cuda', 'mps', 'cpu', 'auto', or 'cuda:N')

    Returns:
        torch.device: PyTorch device object
    """
    # Auto-detect
    if device_str == 'auto':
        device_str = detect_best_device()
        logger.info(f"Auto-detected device: {device_str}")

    # CUDA device
    if device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device('cpu')

        # Parse device ID if specified (e.g., 'cuda:1')
        if ':' in device_str:
            device_id = int(device_str.split(':')[1])
            if device_id >= torch.cuda.device_count():
                logger.warning(f"CUDA device {device_id} not available, using cuda:0")
                device_id = 0
            return torch.device(f'cuda:{device_id}')
        else:
            return torch.device('cuda:0')

    # MPS device (Apple Silicon)
    elif device_str == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            return torch.device('cpu')

    # CPU device
    elif device_str == 'cpu':
        return torch.device('cpu')

    # Unknown device string
    else:
        logger.warning(f"Unknown device '{device_str}', falling back to CPU")
        return torch.device('cpu')


def log_device_info(device: torch.device):
    """
    Log information about the selected device.

    Args:
        device: PyTorch device
    """
    logger.info("=" * 80)
    logger.info("Device Information")
    logger.info("=" * 80)

    if device.type == 'cuda':
        device_id = device.index if device.index is not None else 0
        device_name = torch.cuda.get_device_name(device_id)
        memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

        logger.info(f"Device Type:   CUDA (NVIDIA GPU)")
        logger.info(f"Device ID:     {device_id}")
        logger.info(f"Device Name:   {device_name}")
        logger.info(f"Total Memory:  {memory:.2f} GB")

    elif device.type == 'mps':
        logger.info(f"Device Type:   MPS (Apple Silicon GPU)")
        logger.info(f"Device Name:   Apple Metal Performance Shaders")
        logger.info(f"Note:          MPS provides GPU acceleration on Apple Silicon Macs")

    elif device.type == 'cpu':
        import platform
        logger.info(f"Device Type:   CPU")
        logger.info(f"Processor:     {platform.processor()}")
        logger.info(f"Note:          For better performance, use a GPU (CUDA or MPS)")

    logger.info("=" * 80)


def get_device_for_training(device_arg):
    """
    Convert device argument to device ID for training.

    Args:
        device_arg: Device string ('cuda', 'mps', 'cpu', 'auto')

    Returns:
        int: Device ID (-1 for CPU, -3 for MPS, >=0 for CUDA, -2 for auto)
    """
    if device_arg == 'auto':
        return -2  # Auto-detect marker

    elif device_arg == 'cpu':
        return -1

    elif device_arg == 'mps':
        return -3  # MPS marker

    elif device_arg == 'cuda' or device_arg.startswith('cuda:'):
        if ':' in device_arg:
            return int(device_arg.split(':')[1])
        else:
            return 0  # Default to cuda:0

    else:
        logger.warning(f"Unknown device '{device_arg}', using auto-detect")
        return -2


if __name__ == '__main__':
    # Test device detection
    print("Testing device detection...")

    devices = get_available_devices()
    print("\nAvailable devices:")
    print(f"  CUDA:       {devices['cuda']} ({devices['cuda_count']} devices)")
    print(f"  MPS:        {devices['mps']}")
    print(f"  CPU:        {devices['cpu']}")

    best = detect_best_device()
    print(f"\nBest device:  {best}")

    # Test device creation
    for device_str in ['auto', 'cuda', 'mps', 'cpu']:
        device = get_device(device_str)
        print(f"\n{device_str:10s} -> {device}")
