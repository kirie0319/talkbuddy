from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ROOMS_FILE: str = "data/rooms.json" # supposed to be data/rooms.json?
    CLEANUP_INTERVAL: int = 3600
    DEBUG: bool = False

    @field_validator("OPENAI_API_KEY")
    def validate_openai_api_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError("OPENAI_API_KEY is not valid")
        return v      
    
    @field_validator("CLEANUP_INTERVAL")
    def validate_cleanup_interval(cls, v):
        if v < 60:
            raise ValueError("CLEANUP_INTERVAL is not valid")
        return v    
    
    # if Database is used, set Database url
    DATABASE_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # case sensitive is for not judging the case of the environment upper or lower.
        case_sensitive = False
        # Ignore extra fields from environment variables
        extra = "ignore"

try:
    settings = Settings()
except Exception as e:
    print(f"Failed to load settings: {e}")
    raise