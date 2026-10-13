"""
Configuration management module.
Uses Pydantic Settings for type validation and environment variable parsing.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Keys (Optional for now, will be required when connecting to LLMs)
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    anthropic_api_key: str = Field(default="", description="Anthropic API Key")

    # Environment Settings
    environment: str = Field(default="development", description="Execution environment")
    log_level: str = Field(default="INFO", description="Logging level")


# Instantiating a global settings object
settings = Settings()


if __name__ == "__main__":
    print("✅ Configuration Loaded Successfully!")
    print("Environment: {settings.environment}")
    print(f"Log Level: {settings.log_level}")
