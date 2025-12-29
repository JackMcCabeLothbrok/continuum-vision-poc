"""
Configuration settings using pydantic-settings pattern.
Mirrors the embedding proxy structure for consistency.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / '.env')


class Settings:
    """Application settings loaded from environment."""
    
    def __init__(self):
        self.resetdata_base_url = os.getenv(
            'RESETDATA_BASE_URL', 
            'https://models.au-syd.resetdata.ai/v1'
        )
        self.resetdata_api_key = os.getenv('RESETDATA_API_KEY', '')
        self.resetdata_vision_model = os.getenv(
            'RESETDATA_VISION_MODEL',
            'meta/llama-3.2-11b-vision-instruct:shared'
        )
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Derived paths
        self.project_root = PROJECT_ROOT
        self.test_data_dir = PROJECT_ROOT / 'test_data' / 'frames'
        self.results_dir = PROJECT_ROOT / 'evaluation' / 'results'
        self.prompts_dir = PROJECT_ROOT / 'prompts'
    
    def validate(self) -> bool:
        """Check required settings are present."""
        if not self.resetdata_api_key:
            return False
        return True


# Singleton instance
settings = Settings()
