import pytest

from expmate.parser import ConfigArgumentParser, parse_value, str2bool


class TestStr2Bool:
    """Test string to boolean conversion."""

    def test_true_values(self):
        assert str2bool("true") is True
        assert str2bool("True") is True
        assert str2bool("yes") is True
        assert str2bool("y") is True
        assert str2bool("1") is True

    def test_false_values(self):
        assert str2bool("false") is False
        assert str2bool("False") is False
        assert str2bool("no") is False
        assert str2bool("n") is False
        assert str2bool("0") is False

    def test_invalid_value(self):
        with pytest.raises(Exception):
            str2bool("invalid")

    def test_whitespace_handling(self):
        assert str2bool("  true  ") is True
        assert str2bool("  false  ") is False


class TestParseValue:
    """Test value parsing in parser module."""

    def test_parse_bool(self):
        assert parse_value("true") is True
        assert parse_value("false") is False

    def test_parse_int(self):
        assert parse_value("42") == 42
        assert parse_value("-10") == -10

    def test_parse_float(self):
        assert parse_value("3.14") == 3.14
        assert parse_value("-2.5") == -2.5

    def test_parse_string(self):
        assert parse_value("hello") == "hello"


class TestConfigArgumentParser:
    """Test ConfigArgumentParser class."""

    def test_init(self):
        parser = ConfigArgumentParser()
        assert parser.parser is not None

    def test_init_with_config_path(self):
        parser = ConfigArgumentParser(config_path="config.yaml")
        assert parser.config_path == "config.yaml"

    def test_init_with_description(self):
        parser = ConfigArgumentParser(description="Test parser")
        assert parser.description == "Test parser"

    def test_parse_args_with_config_file(self, config_file):
        parser = ConfigArgumentParser()
        args = [str(config_file)]
        config = parser.parse_args(args)

        assert config.model.name == "test_model"
        assert config.training.lr == 0.001

    def test_parse_args_with_overrides(self, config_file):
        parser = ConfigArgumentParser()
        args = [str(config_file), "training.lr=0.01", "training.epochs=20"]
        config = parser.parse_args(args)

        assert config.training.lr == 0.01
        assert config.training.epochs == 20

    def test_parse_args_with_new_key(self, config_file):
        parser = ConfigArgumentParser()
        args = [str(config_file), "+experiment.debug=true"]
        config = parser.parse_args(args)

        # Access via dict to get actual value
        assert config["experiment"]["debug"] is True

    def test_parse_args_no_config_file(self):
        parser = ConfigArgumentParser()
        args = []

        with pytest.raises(ValueError, match="Config file required as first argument"):
            parser.parse_args(args)

    def test_parse_args_with_default_config_path(self, config_file):
        parser = ConfigArgumentParser(config_path=str(config_file))
        args = []
        config = parser.parse_args(args)

        assert config.model.name == "test_model"

    def test_parse_args_override_default_config_path(self, config_file, temp_dir):
        # Create another config file
        other_config = temp_dir / "other_config.yaml"
        other_config.write_text("model:\n  name: other_model\n")

        parser = ConfigArgumentParser(config_path=str(config_file))
        args = [str(other_config)]
        config = parser.parse_args(args)

        assert config.model.name == "other_model"

    def test_parse_args_mixed_overrides_and_args(self, config_file):
        parser = ConfigArgumentParser()
        args = [
            str(config_file),
            "training.lr=0.01",
            "model.hidden_dim=256",
            "+new_param=test",
        ]
        config = parser.parse_args(args)

        assert config.training.lr == 0.01
        assert config.model.hidden_dim == 256
        assert config.new_param == "test"
