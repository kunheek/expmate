import os
from pathlib import Path

import pytest

from expmate.config import (
    Config,
    deep_merge,
    load_config,
    parse_value,
    set_nested_value,
)


class TestParseValue:
    """Test value parsing."""

    def test_parse_bool_true(self):
        assert parse_value("true") is True
        assert parse_value("True") is True

    def test_parse_bool_false(self):
        assert parse_value("false") is False
        assert parse_value("False") is False

    def test_parse_int(self):
        assert parse_value("42") == 42
        assert parse_value("-10") == -10

    def test_parse_float(self):
        assert parse_value("3.14") == 3.14
        assert parse_value("-2.5") == -2.5

    def test_parse_string(self):
        assert parse_value("hello") == "hello"
        assert parse_value("test_value") == "test_value"

    def test_parse_list(self):
        assert parse_value("[1,2,3]") == [1, 2, 3]
        assert parse_value("[]") == []

    def test_parse_scientific_notation(self):
        assert parse_value("1e-5") == 1e-5
        assert parse_value("2.5e3") == 2.5e3


class TestDeepMerge:
    """Test deep dictionary merging."""

    def test_merge_simple(self):
        base = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        result = deep_merge(base, update)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        update = {"a": {"y": 3, "z": 4}, "c": 5}
        result = deep_merge(base, update)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}, "b": 3, "c": 5}

    def test_merge_deep_nested(self):
        base = {"a": {"b": {"c": 1}}}
        update = {"a": {"b": {"d": 2}}}
        result = deep_merge(base, update)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}


class TestSetNestedValue:
    """Test setting nested dictionary values."""

    def test_set_simple_key(self):
        d = {}
        set_nested_value(d, ["key"], "value")
        assert d == {"key": "value"}

    def test_set_nested_key(self):
        d = {}
        set_nested_value(d, ["a", "b", "c"], 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_set_existing_nested_key(self):
        d = {"a": {"b": 1}}
        set_nested_value(d, ["a", "c"], 2)
        assert d == {"a": {"b": 1, "c": 2}}


class TestLoadConfig:
    """Test config loading."""

    def test_load_from_file(self, config_file):
        config = load_config(str(config_file))
        assert config["model"]["name"] == "test_model"
        assert config["training"]["lr"] == 0.001
        assert config["seed"] == 42

    def test_load_from_dict(self):
        config_dict = {"a": 1, "b": {"c": 2}}
        config = load_config(config_dict)
        assert config == config_dict

    def test_load_with_overrides(self, config_file):
        overrides = ["training.lr=0.01", "training.epochs=20"]
        config = load_config(str(config_file), overrides)
        assert config["training"]["lr"] == 0.01
        assert config["training"]["epochs"] == 20

    def test_load_with_new_key_override(self, config_file):
        overrides = ["+experiment.debug=true"]
        config = load_config(str(config_file), overrides)
        assert config["experiment"]["debug"] is True

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_multiple_files(self, temp_dir):
        # Create two config files
        file1 = temp_dir / "config1.yaml"
        file1.write_text("a: 1\nb: 2")

        file2 = temp_dir / "config2.yaml"
        file2.write_text("b: 3\nc: 4")

        config = load_config([str(file1), str(file2)])
        assert config == {"a": 1, "b": 3, "c": 4}

    def test_invalid_override_format(self, config_file):
        with pytest.raises(ValueError, match="Invalid override format"):
            load_config(str(config_file), ["invalid_override"])


class TestConfig:
    """Test Config class."""

    def test_init_from_file(self, config_file):
        config = Config(str(config_file))
        assert config["model"]["name"] == "test_model"

    def test_dot_notation_access(self, config_file):
        config = Config(str(config_file))
        assert config.model.name == "test_model"
        assert config.training.lr == 0.001
        assert config.training.epochs == 10

    def test_dict_access(self, config_file):
        config = Config(str(config_file))
        assert config["model"]["name"] == "test_model"
        assert config["training"]["lr"] == 0.001

    def test_nested_string_access(self, config_file):
        config = Config(str(config_file))
        assert config["model.name"] == "test_model"
        assert config["training.lr"] == 0.001

    def test_get_with_default(self, config_file):
        config = Config(str(config_file))
        assert config.get("nonexistent.key", "default") == "default"
        assert config.get("model.name", "default") == "test_model"

    def test_to_dict(self, config_file):
        config = Config(str(config_file))
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model"]["name"] == "test_model"

    def test_contains(self, config_file):
        config = Config(str(config_file))
        assert "model" in config
        assert "training" in config
        assert "nonexistent" not in config

    def test_keys(self, config_file):
        config = Config(str(config_file))
        keys = list(config.keys())
        assert "model" in keys
        assert "training" in keys
        assert "data" in keys

    def test_values(self, config_file):
        config = Config(str(config_file))
        values = list(config.values())
        assert len(values) > 0

    def test_items(self, config_file):
        config = Config(str(config_file))
        items = list(config.items())
        assert ("seed", 42) in items

    def test_interpolation_cross_reference(self, advanced_config_file):
        config = Config(str(advanced_config_file))
        # Cross-references should be resolved
        assert config.model.name == "test_project_model"
        assert config.paths.output_dir == "/data/output"
        assert config.training.save_dir == "/data/output/checkpoints"

    def test_interpolation_env(self, config_file):
        os.environ["TEST_VAR"] = "test_value"

        # Create config with env interpolation
        temp_path = Path(config_file).parent / "env_config.yaml"
        temp_path.write_text("test_key: ${env:TEST_VAR}")

        config = Config(str(temp_path))
        assert config.test_key == "test_value"

    def test_interpolation_hostname(self, config_file):
        # Create config with hostname interpolation
        temp_path = Path(config_file).parent / "hostname_config.yaml"
        temp_path.write_text("host: ${hostname}")

        config = Config(str(temp_path))
        # Access via dict to get actual value
        host_value = config["host"]
        assert isinstance(host_value, str)
        assert len(host_value) > 0
        assert host_value != "${hostname}"  # Should be interpolated

    def test_interpolation_now(self, config_file):
        # Create config with timestamp interpolation
        temp_path = Path(config_file).parent / "time_config.yaml"
        temp_path.write_text("timestamp: ${now:%Y%m%d}")

        config = Config(str(temp_path))
        # Access via dict to get actual value
        timestamp = config["timestamp"]
        assert isinstance(timestamp, str)
        assert len(timestamp) == 8  # YYYYMMDD format
        assert timestamp != "${now:%Y%m%d}"  # Should be interpolated

    def test_save_config(self, config_file, temp_dir):
        config = Config(str(config_file))
        save_path = temp_dir / "saved_config.yaml"
        config.save(str(save_path))

        # Load and verify
        loaded = Config(str(save_path))
        assert loaded.model.name == config.model.name
        assert loaded.training.lr == config.training.lr

    def test_hash_generation(self, config_file):
        config = Config(str(config_file))
        hash1 = config.hash()

        # Same config should produce same hash
        config2 = Config(str(config_file))
        hash2 = config2.hash()
        assert hash1 == hash2

        # Modified config should produce different hash
        config3 = Config(str(config_file), overrides=["training.lr=0.01"])
        hash3 = config3.hash()
        assert hash1 != hash3

    def test_repr(self, config_file):
        config = Config(str(config_file))
        repr_str = repr(config)
        assert "Config" in repr_str

    def test_str(self, config_file):
        config = Config(str(config_file))
        str_output = str(config)
        assert "model" in str_output
        assert "training" in str_output
