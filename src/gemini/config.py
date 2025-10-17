"""
Configuration for Gemini Event Extraction

Simple configuration for Gemini API-based voting event extraction.
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration for Gemini event extraction"""

    # Gemini Model Configuration
    # Available models:
    # - "gemini-2.0-flash-exp"  (fast, experimental)
    # - "gemini-2.5-flash"      (fast, latest)
    # - "gemini-1.5-flash"      (fast, stable)
    # - "gemini-1.5-pro"        (high quality)
    api_key: str = None
    model_id: str = "gemini-2.5-flash"

    def __post_init__(self):
        """Load API key from environment if not set"""
        if not self.api_key:
            self.api_key = os.getenv('GOOGLE_API_KEY')
            
        if not self.api_key:
            raise ValueError(
                "No API key found. Set GOOGLE_API_KEY environment variable.\n"
                "Get your key at: https://aistudio.google.com/app/apikey"
            )


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config
