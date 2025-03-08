import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CV Analysis System"
    
    # LLM settings
    LLM_PROVIDER: str = "anthropic"  # "anthropic" or "openai"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    
    # OCR settings
    TESSERACT_PATH: str = "C:\Program Files\Tesseract-OCR\/tesseract.exe"  # Default path
    
    # Storage settings
    STORAGE_DIR: str = "data/processed"
    TEMP_STORAGE_DIR: str = "data/temp"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
