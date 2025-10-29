"""
Test typed dataclass-based Config with LSP support.
"""

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tempfile
import yaml

from expmate.config import Config


@dataclass
class SimpleTypedConfig(Config):
    """Simple typed config for testing."""

    name: str
    value: int
    ratio: float


@dataclass
class TypedConfigWithDefaults(Config):
    """Typed config with default values."""

    required_field: str
    optional_field: Optional[int] = None
    default_field: str = "default_value"


class TestTypedConfig:
    """Test dataclass-based typed config functionality."""

    def test_simple_typed_config_from_dict(self):
        """Test creating typed config from dictionary."""
        config_data = {"name": "test", "value": 42, "ratio": 0.5}

        config = SimpleTypedConfig.from_dict(config_data)

        # Verify fields are set correctly
        assert config.name == "test"
        assert config.value == 42
        assert config.ratio == 0.5

        # Verify types
        assert isinstance(config.name, str)
        assert isinstance(config.value, int)
        assert isinstance(config.ratio, float)

    def test_typed_config_from_file(self, tmp_path):
        """Test creating typed config from YAML file."""
        config_path = tmp_path / "config.yaml"
        config_data = {"name": "experiment", "value": 100, "ratio": 0.75}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = SimpleTypedConfig.from_file(str(config_path))

        assert config.name == "experiment"
        assert config.value == 100
        assert config.ratio == 0.75

    def test_typed_config_with_defaults(self):
        """Test typed config with default values."""
        config_data = {"required_field": "test"}

        config = TypedConfigWithDefaults.from_dict(config_data)

        assert config.required_field == "test"
        assert config.optional_field is None
        assert config.default_field == "default_value"

    def test_typed_config_override_defaults(self):
        """Test overriding default values in typed config."""
        config_data = {
            "required_field": "test",
            "optional_field": 123,
            "default_field": "custom",
        }

        config = TypedConfigWithDefaults.from_dict(config_data)

        assert config.required_field == "test"
        assert config.optional_field == 123
        assert config.default_field == "custom"

    def test_typed_config_to_dict(self):
        """Test converting typed config back to dictionary."""
        config_data = {"name": "test", "value": 42, "ratio": 0.5}

        config = SimpleTypedConfig.from_dict(config_data)
        result_dict = config.to_dict()

        assert "name" in result_dict
        assert "value" in result_dict
        assert "ratio" in result_dict
        assert result_dict["name"] == "test"
        assert result_dict["value"] == 42
        assert result_dict["ratio"] == 0.5

    def test_typed_config_save(self, tmp_path):
        """Test saving typed config to file."""
        config_data = {"name": "test", "value": 42, "ratio": 0.5}

        config = SimpleTypedConfig.from_dict(config_data)
        save_path = tmp_path / "saved_config.yaml"

        config.save(save_path)

        assert save_path.exists()

        # Load and verify
        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == "test"
        assert loaded["value"] == 42
        assert loaded["ratio"] == 0.5

    def test_typed_config_hash(self):
        """Test generating hash from typed config."""
        config_data = {"name": "test", "value": 42, "ratio": 0.5}

        config = SimpleTypedConfig.from_dict(config_data)
        hash_val = config.hash()

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex digest length

    def test_typed_config_dict_access(self):
        """Test dict-style access on typed config."""
        config_data = {"name": "test", "value": 42, "ratio": 0.5}

        config = SimpleTypedConfig.from_dict(config_data)

        # Dict-style access should still work
        assert config["name"] == "test"
        assert config.get("value") == 42
        assert "ratio" in config

    def test_typed_config_with_overrides(self, tmp_path):
        """Test typed config with command-line style overrides."""
        config_path = tmp_path / "config.yaml"
        config_data = {"name": "test", "value": 42, "ratio": 0.5}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = SimpleTypedConfig.from_file(
            str(config_path), overrides=["value=100", "ratio=0.9"]
        )

        assert config.name == "test"
        assert config.value == 100
        assert config.ratio == 0.9

    def test_backward_compatibility_dynamic_config(self):
        """Test that dynamic Config still works (backward compatibility)."""
        config_data = {"name": "test", "nested": {"value": 42}}

        # Old-style dynamic config
        config = Config.from_dict(config_data)

        # Should work with dot notation
        assert config.name == "test"
        assert config.nested.value == 42

        # Should work with dict access
        assert config["name"] == "test"
        assert config["nested"]["value"] == 42

    def test_typed_and_dynamic_config_coexist(self, tmp_path):
        """Test that typed and dynamic configs can coexist."""
        config_path = tmp_path / "config.yaml"
        config_data = {"name": "test", "value": 42, "ratio": 0.5}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Dynamic config
        dynamic = Config.from_file(str(config_path))

        # Typed config from same file
        typed = SimpleTypedConfig.from_file(str(config_path))

        # Both should work
        assert dynamic.name == typed.name
        assert dynamic.value == typed.value
        assert dynamic.ratio == typed.ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
