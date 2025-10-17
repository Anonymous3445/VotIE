#!/usr/bin/env python
"""
Simple unified config loader for all NER models.

Loads YAML configs directly with minimal processing.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing config
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_path(baseline_name: str, experiment_type: str = "main_experiment") -> Path:
    """
    Get default config path for a baseline.
    
    Args:
        baseline_name: Name of the baseline (e.g., 'bert_crf')
        experiment_type: Type of experiment ('main_experiment', 'cross_municipality', etc.)
        
    Returns:
        Path to config file
    """
    root_dir = Path(__file__).parent.parent.parent
    config_dir = root_dir / "configs" / experiment_type
    config_path = config_dir / f"{baseline_name}.yaml"
    
    return config_path


def load_default_config(baseline_name: str, experiment_type: str = "main_experiment") -> Dict[str, Any]:
    """
    Load default configuration for a baseline.
    
    Args:
        baseline_name: Name of the baseline
        experiment_type: Type of experiment
        
    Returns:
        Dictionary containing config
    """
    config_path = get_config_path(baseline_name, experiment_type)
    return load_config(str(config_path))
