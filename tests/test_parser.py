import pytest

from expmate.config import parse_value
from expmate.utils import str2bool


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
