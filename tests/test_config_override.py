"""Tests for Config.override() method."""

from dataclasses import dataclass

from expmate import Config


class TestConfigOverride:
    """Test suite for Config.override() method."""

    def test_override_dynamic_config(self):
        """Test override on dynamic Config."""
        config = Config.from_dict({"lr": 0.001, "batch_size": 32})
        new_config = config.override(["+lr=0.01", "batch_size=64"])

        # New config should have updated values
        assert new_config.lr == 0.01
        assert new_config.batch_size == 64

        # Original config should be unchanged
        assert config.lr == 0.001
        assert config.batch_size == 32

    def test_override_typed_config(self):
        """Test override on typed Config subclass."""

        @dataclass
        class TrainConfig(Config):
            lr: float = 0.001
            batch_size: int = 32
            epochs: int = 100

        config = TrainConfig.from_dict({"lr": 0.005})
        new_config = config.override(["lr=0.1", "+batch_size=128"])

        assert isinstance(new_config, TrainConfig)
        assert new_config.lr == 0.1
        assert new_config.batch_size == 128
        assert new_config.epochs == 100

        # Original unchanged
        assert config.lr == 0.005
        assert config.batch_size == 32

    def test_override_add_new_keys(self):
        """Test adding new keys with override."""
        config = Config.from_dict({"existing": "value"})
        new_config = config.override(["+new_key=123", "+name=test"])

        assert new_config.existing == "value"
        assert new_config.new_key == 123
        assert new_config.name == "test"

        # Original should not have new keys
        assert "new_key" not in config.config
        assert "name" not in config.config

    def test_override_with_type_hints(self):
        """Test override with type hints."""
        config = Config.from_dict({"value": "string"})
        new_config = config.override(["+num:int=42", "+pi:float=3.14"])

        assert new_config.num == 42
        assert isinstance(new_config.num, int)
        assert new_config.pi == 3.14
        assert isinstance(new_config.pi, float)

    def test_override_nested_values(self):
        """Test override of nested configuration values."""
        config = Config.from_dict(
            {"model": {"type": "resnet", "layers": 50}, "training": {"lr": 0.001}}
        )
        new_config = config.override(["model.layers=101", "training.lr=0.01"])

        assert new_config.model.layers == 101
        assert new_config.training.lr == 0.01
        assert new_config.model.type == "resnet"  # Unchanged

        # Original unchanged
        assert config.model.layers == 50
        assert config.training.lr == 0.001

    def test_override_empty_overrides(self):
        """Test override with empty list."""
        config = Config.from_dict({"key": "value"})
        new_config = config.override([])

        assert new_config.key == "value"
        assert new_config is not config  # Should be a new instance

    def test_override_preserves_type(self):
        """Test that override returns same type as original."""

        @dataclass
        class CustomConfig(Config):
            value: int = 42

        config = CustomConfig.from_dict({})
        new_config = config.override(["value=100"])

        assert type(new_config) is CustomConfig
        assert isinstance(new_config, CustomConfig)

    def test_override_with_lists(self):
        """Test override with list values."""
        config = Config.from_dict({"my_list": [1, 2, 3]})
        new_config = config.override(["+my_list=[4,5,6]"])

        assert new_config.my_list == [4, 5, 6]
        assert config.my_list == [1, 2, 3]  # Original unchanged

    def test_override_multiple_times(self):
        """Test chaining multiple overrides."""
        config = Config.from_dict({"a": 1, "b": 2})
        config2 = config.override(["a=10"])
        config3 = config2.override(["b=20", "+c=30"])

        assert config.a == 1
        assert config.b == 2
        assert config2.a == 10
        assert config2.b == 2
        assert config3.a == 10
        assert config3.b == 20
        assert config3.c == 30

    def test_override_repr_works(self):
        """Test that repr works correctly after override."""
        config = Config.from_dict({"key": "value"})
        new_config = config.override(["+new=123"])

        repr_str = repr(new_config)
        assert "Config" in repr_str
        assert "key" in repr_str or "new" in repr_str  # At least one key shown

    def test_override_empty_config(self):
        """Test override on empty config."""
        config = Config.from_dict({})
        new_config = config.override(["+key=value", "+num:int=42"])

        assert new_config.key == "value"
        assert new_config.num == 42
        assert repr(new_config)  # Should not crash

    def test_override_with_run_dir(self, tmp_path):
        """Test override with run_dir saves snapshot."""
        config = Config.from_dict({"key": "value"})
        config.override(["+new=123"], run_dir=tmp_path)

        # Check that snapshot was saved (as run.yaml, not config.yaml)
        snapshot_file = tmp_path / "run.yaml"
        assert snapshot_file.exists()

        # Verify content
        import yaml

        with open(snapshot_file) as f:
            saved = yaml.safe_load(f)
        assert saved["key"] == "value"
        assert saved["new"] == 123


class TestConfigRepr:
    """Test suite for Config __repr__ edge cases."""

    def test_repr_empty_config(self):
        """Test repr with empty config."""
        config = Config()
        assert repr(config) == "Config()"

    def test_repr_empty_from_dict(self):
        """Test repr with empty dict."""
        config = Config.from_dict({})
        assert repr(config) == "Config()"

    def test_repr_empty_with_overrides(self):
        """Test repr when starting from empty dict with overrides."""
        config = Config.from_dict({}, overrides=["+test=123"])
        repr_str = repr(config)
        assert "Config" in repr_str
        assert "test" in repr_str

    def test_repr_many_keys(self):
        """Test repr with many keys shows all up to threshold."""
        # 10 keys - should show all
        data = {f"key{i}": f"value{i}" for i in range(10)}
        config = Config.from_dict(data)
        repr_str = repr(config)

        # Should show all 10 keys
        assert "key0" in repr_str
        assert "key9" in repr_str
        assert "..." not in repr_str  # No truncation for 10 keys

        # 25 keys - should truncate
        data2 = {f"key{i}": f"value{i}" for i in range(25)}
        config2 = Config.from_dict(data2)
        repr_str2 = repr(config2)

        assert "..." in repr_str2
        assert "25 total" in repr_str2

    def test_repr_typed_config(self):
        """Test repr of typed config."""

        @dataclass
        class MyConfig(Config):
            name: str = "test"
            value: int = 42

        config = MyConfig.from_dict({})
        repr_str = repr(config)

        assert "MyConfig" in repr_str
        assert "name" in repr_str
        assert "value" in repr_str

    def test_repr_typed_config_with_missing_field(self):
        """Test repr when typed config is missing optional fields."""

        @dataclass
        class MyConfig(Config):
            required: str
            optional: int = 99

        config = MyConfig.from_dict({"required": "test"})
        repr_str = repr(config)

        assert "MyConfig" in repr_str
        assert "required" in repr_str
        assert "optional" in repr_str

    def test_repr_nested_config(self):
        """Test repr with nested data."""
        config = Config.from_dict(
            {"outer": {"inner": {"deep": "value"}}, "simple": 123}
        )
        repr_str = repr(config)

        assert "Config" in repr_str
        # Should show the nested structure
        assert "outer" in repr_str or "simple" in repr_str

    def test_repr_config_dict(self):
        """Test repr of ConfigDict (nested access)."""
        config = Config.from_dict({"level1": {"level2": {"level3": "value"}}})
        nested = config.level1
        repr_str = repr(nested)

        assert "ConfigDict" in repr_str
        assert "level1" in repr_str

    def test_str_vs_repr(self):
        """Test that __str__ and __repr__ are different."""
        config = Config.from_dict({"key": "value", "num": 42})

        str_output = str(config)
        repr_output = repr(config)

        # They should be different
        assert str_output != repr_output
        # str should be more verbose
        assert len(str_output) > len(repr_output)
