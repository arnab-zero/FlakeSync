"""Unit tests for configuration loading and validation."""

import pytest
import tempfile
import yaml
import json
from pathlib import Path

from src.utils.config import (
    Config,
    ModelConfig,
    LLMConfig,
    AnalysisConfig,
    OutputConfig,
    LoggingConfig,
    load_config,
    save_config
)


class TestModelConfig:
    """Test ModelConfig validation."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.checkpoint_path == "flakyXbert_augExp1.pth"
        assert config.device == "cuda"
        assert config.embedding_cache_path == "train_embeddings.pt"
    
    def test_valid_device(self):
        """Test valid device values."""
        for device in ['cuda', 'cpu', 'auto']:
            config = ModelConfig(device=device)
            assert config.device == device
    
    def test_invalid_device(self):
        """Test invalid device value raises error."""
        with pytest.raises(ValueError, match="device must be"):
            ModelConfig(device="invalid")


class TestLLMConfig:
    """Test LLMConfig validation."""
    
    def test_default_values(self):
        """Test default LLM configuration values."""
        config = LLMConfig()
        assert config.provider == "gemini"
        assert config.temperature == 0.2
        assert config.max_retries == 3
    
    def test_valid_provider(self):
        """Test valid provider values."""
        for provider in ['gemini', 'openai', 'anthropic']:
            config = LLMConfig(provider=provider)
            assert config.provider == provider
    
    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="provider must be"):
            LLMConfig(provider="invalid")
    
    def test_temperature_range(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=0.5)
        LLMConfig(temperature=1.0)
        
        # Invalid temperatures
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=1.1)
    
    def test_max_retries_range(self):
        """Test max_retries validation."""
        # Valid retries
        LLMConfig(max_retries=1)
        LLMConfig(max_retries=5)
        LLMConfig(max_retries=10)
        
        # Invalid retries
        with pytest.raises(ValueError):
            LLMConfig(max_retries=0)
        with pytest.raises(ValueError):
            LLMConfig(max_retries=11)


class TestAnalysisConfig:
    """Test AnalysisConfig validation."""
    
    def test_default_values(self):
        """Test default analysis configuration values."""
        config = AnalysisConfig()
        assert config.interprocedural_depth == 3
        assert config.confidence_threshold == 0.7
        assert config.parallel_processing is True
        assert config.max_workers == 4
    
    def test_depth_range(self):
        """Test interprocedural_depth validation."""
        # Valid depths
        AnalysisConfig(interprocedural_depth=1)
        AnalysisConfig(interprocedural_depth=5)
        AnalysisConfig(interprocedural_depth=10)
        
        # Invalid depths
        with pytest.raises(ValueError):
            AnalysisConfig(interprocedural_depth=0)
        with pytest.raises(ValueError):
            AnalysisConfig(interprocedural_depth=11)
    
    def test_confidence_range(self):
        """Test confidence_threshold validation."""
        # Valid thresholds
        AnalysisConfig(confidence_threshold=0.0)
        AnalysisConfig(confidence_threshold=0.5)
        AnalysisConfig(confidence_threshold=1.0)
        
        # Invalid thresholds
        with pytest.raises(ValueError):
            AnalysisConfig(confidence_threshold=-0.1)
        with pytest.raises(ValueError):
            AnalysisConfig(confidence_threshold=1.1)


class TestLoggingConfig:
    """Test LoggingConfig validation."""
    
    def test_default_values(self):
        """Test default logging configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.file == "flaky_detector.log"
    
    def test_valid_log_levels(self):
        """Test valid log level values."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            config = LoggingConfig(level=level)
            assert config.level == level
            
            # Test case insensitivity
            config = LoggingConfig(level=level.lower())
            assert config.level == level
    
    def test_invalid_log_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValueError, match="level must be one of"):
            LoggingConfig(level="INVALID")


class TestConfig:
    """Test main Config class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.analysis, AnalysisConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_nested_config(self):
        """Test nested configuration."""
        config = Config(
            model=ModelConfig(device="cpu"),
            llm=LLMConfig(provider="openai"),
            analysis=AnalysisConfig(interprocedural_depth=5)
        )
        assert config.model.device == "cpu"
        assert config.llm.provider == "openai"
        assert config.analysis.interprocedural_depth == 5
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            Config(unknown_field="value")


class TestLoadConfig:
    """Test configuration loading from files."""
    
    def test_load_default_config(self):
        """Test loading default configuration (no file)."""
        config = load_config()
        assert isinstance(config, Config)
        assert config.model.device == "cuda"
    
    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_config = {
                'model': {'device': 'cpu', 'checkpoint_path': 'test.pth'},
                'llm': {'provider': 'openai', 'temperature': 0.5},
                'analysis': {'interprocedural_depth': 5}
            }
            yaml.dump(yaml_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.model.device == "cpu"
            assert config.model.checkpoint_path == "test.pth"
            assert config.llm.provider == "openai"
            assert config.llm.temperature == 0.5
            assert config.analysis.interprocedural_depth == 5
        finally:
            Path(temp_path).unlink()
    
    def test_load_json_config(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_config = {
                'model': {'device': 'cpu'},
                'llm': {'provider': 'anthropic'},
                'logging': {'level': 'DEBUG'}
            }
            json.dump(json_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.model.device == "cpu"
            assert config.llm.provider == "anthropic"
            assert config.logging.level == "DEBUG"
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_invalid_format(self):
        """Test loading from unsupported format raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid config")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_config_data(self):
        """Test loading invalid configuration data raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_config = {
                'model': {'device': 'invalid_device'}
            }
            yaml.dump(yaml_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid configuration"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_partial_config(self):
        """Test loading partial configuration uses defaults for missing values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_config = {
                'model': {'device': 'cpu'}
                # Other sections missing
            }
            yaml.dump(yaml_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.model.device == "cpu"
            # Defaults should be used for other sections
            assert config.llm.provider == "gemini"
            assert config.analysis.interprocedural_depth == 3
        finally:
            Path(temp_path).unlink()


class TestSaveConfig:
    """Test configuration saving to files."""
    
    def test_save_yaml_config(self):
        """Test saving configuration to YAML file."""
        config = Config(
            model=ModelConfig(device="cpu"),
            llm=LLMConfig(provider="openai")
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            
            # Load and verify
            loaded_config = load_config(temp_path)
            assert loaded_config.model.device == "cpu"
            assert loaded_config.llm.provider == "openai"
        finally:
            Path(temp_path).unlink()
    
    def test_save_json_config(self):
        """Test saving configuration to JSON file."""
        config = Config(
            model=ModelConfig(device="cpu"),
            analysis=AnalysisConfig(interprocedural_depth=7)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            
            # Load and verify
            loaded_config = load_config(temp_path)
            assert loaded_config.model.device == "cpu"
            assert loaded_config.analysis.interprocedural_depth == 7
        finally:
            Path(temp_path).unlink()
    
    def test_save_invalid_format(self):
        """Test saving to unsupported format raises error."""
        config = Config()
        with pytest.raises(ValueError, match="Unsupported output file format"):
            save_config(config, "config.txt")
