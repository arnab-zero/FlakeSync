"""Configuration management using pydantic for validation."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model configuration."""
    checkpoint_path: str = Field(default="flakyXbert_augExp1.pth")
    device: str = Field(default="cuda")
    embedding_cache_path: str = Field(default="train_embeddings.pt")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        if v not in ['cuda', 'cpu', 'auto']:
            raise ValueError("device must be 'cuda', 'cpu', or 'auto'")
        return v


class LLMConfig(BaseModel):
    """LLM API configuration."""
    provider: str = Field(default="gemini")
    api_key: str = Field(default="AIzaSyCW8EbIYPATZnwjIl7UtuIJlXHyWY8fPSI")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        if v not in ['gemini', 'openai', 'anthropic']:
            raise ValueError("provider must be 'gemini', 'openai', or 'anthropic'")
        return v


class AnalysisConfig(BaseModel):
    """Analysis configuration."""
    interprocedural_depth: int = Field(default=3, ge=1, le=10)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=32)


class OutputConfig(BaseModel):
    """Output configuration."""
    directory: str = Field(default="output")
    generate_json: bool = Field(default=True)
    generate_text: bool = Field(default=True)
    generate_patches: bool = Field(default=True)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    file: Optional[str] = Field(default="flaky_detector.log")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return v.upper()


class Config(BaseModel):
    """Main configuration class."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    model_config = {'extra': 'forbid'}


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to YAML or JSON configuration file
        
    Returns:
        Config object with validated settings
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        # Use default configuration
        return Config()
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration based on file extension
    with open(config_file, 'r') as f:
        if config_file.suffix in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_file.suffix == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    # Validate and create Config object
    try:
        config = Config(**config_data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Config object to save
        output_path: Path to output file (YAML or JSON)
    """
    output_file = Path(output_path)
    config_dict = config.model_dump()
    
    with open(output_file, 'w') as f:
        if output_file.suffix in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False)
        elif output_file.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output file format: {output_file.suffix}")
