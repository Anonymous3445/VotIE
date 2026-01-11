"""
Configuration for Gemini span extraction.
"""

import os
from typing import Optional
from pathlib import Path

# Try to load from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly


class GeminiConfig:
    """Configuration for Gemini API extractor (supports both AI Studio and Vertex AI)."""

    def __init__(
        self,
        model_id: str = "gemini-1.5-flash-002",
        use_vertex_ai: bool = True,  # Default to Vertex AI
        project_id: Optional[str] = None,
        location: str = "us-central1",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,  # Gemini's maximum output tokens
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        """
        Initialize Gemini configuration.

        Args:
            model_id: Gemini model identifier
                - Vertex AI: "gemini-1.5-flash-002", "gemini-1.5-pro-002"
                - AI Studio: "gemini-1.5-flash", "gemini-1.5-pro"
            use_vertex_ai: Use Vertex AI (True) or AI Studio (False)
            project_id: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud location (default: us-central1)
            api_key: Google API key (required for AI Studio only)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum output tokens (default: 8192, Gemini's max)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for quota errors
            retry_delay: Delay between retries in seconds
        """
        # Check environment variables
        self.use_vertex_ai = use_vertex_ai or os.getenv("USE_VERTEX_AI", "true").lower() == "true"
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Validate configuration
        if self.use_vertex_ai:
            if not self.project_id:
                raise ValueError(
                    "Google Cloud project ID required for Vertex AI.\n"
                    "Set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter.\n"
                    "Find your project ID at: https://console.cloud.google.com/"
                )
        else:
            if not self.api_key:
                raise ValueError(
                    "Google API key required for AI Studio.\n"
                    "Set GOOGLE_API_KEY environment variable or pass api_key parameter.\n"
                    "Get your key from: https://aistudio.google.com/app/apikey"
                )

    def __repr__(self) -> str:
        if self.use_vertex_ai:
            return (
                f"GeminiConfig(vertex_ai=True, model={self.model_id}, "
                f"project={self.project_id}, location={self.location})"
            )
        else:
            return (
                f"GeminiConfig(vertex_ai=False, model={self.model_id})"
            )


# Default configuration
DEFAULT_CONFIG = GeminiConfig()
