"""Configuration management for the document-to-JSON-schema converter."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DOCUMENT_PATH = Path(os.getenv("DOCUMENT_PATH", "./documents"))
    OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "./output"))
    
    # Create necessary directories
    DOCUMENT_PATH.mkdir(exist_ok=True)
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    @staticmethod
    def validate():
        """Validate required configuration."""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
