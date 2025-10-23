import argparse
import json
import sys
from typing import Any, List, Optional

import yaml

from .config import Config


def str2bool(v: str) -> bool:
    """Convert string to boolean."""
    v = v.strip().lower()
    if isinstance(v, bool):
        return v
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_value(value: str) -> Any:
    """Parse string value to appropriate type.

    Supports:
    - Booleans: true/false (not 0/1 - those are treated as integers)
    - Integers: 42
    - Floats: 3.14
    - Strings: hello
    - Lists (JSON): [1,2,3] or ["a","b","c"]
    - Lists (space-separated): use parse_sequence() instead
    """
    # Try JSON parsing for lists/dicts first (e.g., [1,2,3] or {"key":"val"})
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try int (do this before boolean to handle 0/1 as numbers)
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try boolean (only for explicit true/false/yes/no, not 0/1)
    if value.lower() in ("true", "false", "yes", "no", "t", "f", "y", "n"):
        try:
            return str2bool(value)
        except argparse.ArgumentTypeError:
            pass

    # Return as string
    return value


def parse_sequence(values: List[str], type_hint: Optional[str] = None) -> List[Any]:
    """Parse a sequence of values with optional type hint.

    Args:
        values: List of string values to parse
        type_hint: Optional type hint ('int', 'float', 'str', 'bool')
                   If None, will auto-detect from first value

    Returns:
        List of parsed values

    Examples:
        parse_sequence(['1', '2', '3'], 'int') -> [1, 2, 3]
        parse_sequence(['1.0', '2.5'], 'float') -> [1.0, 2.5]
        parse_sequence(['a', 'b', 'c'], 'str') -> ['a', 'b', 'c']
        parse_sequence(['1', '2', '3'], None) -> [1, 2, 3]  # auto-detect
    """
    if not values:
        return []

    # If type hint provided, use it
    if type_hint:
        type_hint = type_hint.lower()
        if type_hint == "int":
            return [int(v) for v in values]
        elif type_hint == "float":
            return [float(v) for v in values]
        elif type_hint == "str":
            return values
        elif type_hint == "bool":
            return [str2bool(v) for v in values]
        else:
            raise ValueError(f"Unknown type hint: {type_hint}")

    # Auto-detect type from first value
    first_parsed = parse_value(values[0])

    result = [first_parsed]
    for v in values[1:]:
        parsed = parse_value(v)
        # Try to match the detected type
        if isinstance(first_parsed, int) and isinstance(parsed, (int, float)):
            result.append(int(parsed) if isinstance(parsed, float) else parsed)
        elif isinstance(first_parsed, float) and isinstance(parsed, (int, float)):
            result.append(float(parsed))
        elif isinstance(first_parsed, bool):
            result.append(str2bool(v) if isinstance(v, str) else bool(parsed))
        else:
            result.append(parsed)

    return result


class ConfigArgumentParser:
    """Argument parser that can modify and extend configs.

    Supports:
    - Hydra-style overrides: training.lr=0.001
    - Adding new keys: +new_key=value
    - List values: training.layers=[64,128,256]
    - Space-separated lists: training.gpus=0 1 2 3
    - Type hints for lists: training.ids:int=1 2 3
    """

    def __init__(self, config_path: Optional[str] = None, description: str = None):
        self.config_path = config_path
        self.description = description or "ExpMate Config Parser"
        self.parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Load config and override values
  python script.py config.yaml training.lr=0.001 model.depth=50

  # Add new config keys
  python script.py config.yaml +experiment.debug=true

  # Override with lists (JSON format)
  python script.py config.yaml training.layers=[64,128,256]

  # Override with lists (space-separated, auto-detect type)
  python script.py config.yaml training.gpus=0 1 2 3

  # Override with typed lists
  python script.py config.yaml training.ids:int=1 2 3
  python script.py config.yaml data.names:str=train val test

  # Show config schema
  python script.py config.yaml --show-config

  # Show config help
  python script.py config.yaml --config-help
            """,
        )

        # Add positional config file argument (first argument)
        self.parser.add_argument(
            "config",
            type=str,
            nargs="?",
            default=config_path,
            help="Path to YAML config file (required as first argument)",
        )

        # Add show-config argument
        self.parser.add_argument(
            "--show-config",
            action="store_true",
            help="Display current config structure and exit",
        )

        # Add config-help argument
        self.parser.add_argument(
            "--config-help",
            action="store_true",
            help="Display config schema with types and exit",
        )

    def _format_value_with_type(self, value: Any, indent: int = 0) -> str:
        """Format a config value with its type information."""
        prefix = "  " * indent

        if isinstance(value, dict):
            lines = [f"{prefix}{{"]
            for k, v in value.items():
                type_str = self._get_type_string(v)
                if isinstance(v, dict):
                    lines.append(f"{prefix}  {k}: {type_str}")
                    lines.append(self._format_value_with_type(v, indent + 1))
                elif isinstance(v, list):
                    list_type = self._get_list_type(v)
                    lines.append(f"{prefix}  {k}: {type_str} = {v}")
                else:
                    lines.append(f"{prefix}  {k}: {type_str} = {v}")
            lines.append(f"{prefix}}}")
            return "\n".join(lines)
        elif isinstance(value, list):
            list_type = self._get_list_type(value)
            return f"{prefix}[{list_type}] = {value}"
        else:
            return f"{prefix}{value}"

    def _get_type_string(self, value: Any) -> str:
        """Get a user-friendly type string."""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, list):
            return f"list[{self._get_list_type(value)}]"
        elif isinstance(value, dict):
            return "dict"
        else:
            return type(value).__name__

    def _get_list_type(self, lst: List[Any]) -> str:
        """Infer the type of list elements."""
        if not lst:
            return "Any"

        first_type = type(lst[0]).__name__
        if all(type(x).__name__ == first_type for x in lst):
            return first_type
        return "mixed"

    def _print_config_schema(self, config: Config):
        """Print config schema with types."""
        print("\n" + "=" * 70)
        print("Configuration Schema")
        print("=" * 70)
        print()
        print("Available configuration keys and their current values:")
        print()

        def print_nested(d: dict, prefix: str = ""):
            for key, value in sorted(d.items()):
                full_key = f"{prefix}.{key}" if prefix else key
                type_str = self._get_type_string(value)

                if isinstance(value, dict):
                    print(f"  {full_key}:")
                    print(f"    type: {type_str}")
                    print(f"    keys: {list(value.keys())}")
                    print()
                elif isinstance(value, list):
                    list_type = self._get_list_type(value)
                    print(f"  {full_key}:")
                    print(f"    type: list[{list_type}]")
                    print(f"    value: {value}")
                    print(f"    override: {full_key}=[...]  or  {full_key}:int=1 2 3")
                    print()
                else:
                    print(f"  {full_key}:")
                    print(f"    type: {type_str}")
                    print(f"    value: {value}")
                    print(f"    override: {full_key}=<value>")
                    print()

        print_nested(dict(config))

        print("=" * 70)
        print("Override examples:")
        print("  training.lr=0.001              # Override existing value")
        print("  +new_key=value                 # Add new configuration key")
        print("  training.layers=[64,128,256]   # List with JSON format")
        print("  training.gpus=0 1 2            # List with auto-type detection")
        print("  training.ids:int=1 2 3         # List with explicit type")
        print("=" * 70)
        print()

    def parse_args(self, args: Optional[List[str]] = None) -> Config:
        """Parse command line arguments and return modified config.

        Supports Hydra-style overrides:
            python script.py --config conf.yaml training.lr=0.001 model.depth=50
            python script.py --config conf.yaml +new_key=value  # Add new key
            python script.py --config conf.yaml training.gpus=0 1 2  # Space-separated list
            python script.py --config conf.yaml training.ids:int=1 2 3  # Typed list
        """
        if args is None:
            args = sys.argv[1:]

        # Separate config modifications from other args
        overrides = []
        other_args = []
        i = 0

        while i < len(args):
            arg = args[i]

            # Check for override patterns: key=value, key:type=value, or +key=value
            if "=" in arg and not arg.startswith("-"):
                # This is an override, check if next args are values (space-separated list)
                override_parts = arg.split("=", 1)
                key_part = override_parts[0]
                value_part = override_parts[1] if len(override_parts) > 1 else ""

                # Check if this is a typed list: key:type=value1 value2 value3
                if ":" in key_part and not value_part.startswith(("[", "{")):
                    # Typed list with space-separated values
                    key, type_hint = key_part.rsplit(":", 1)
                    values = [value_part] if value_part else []

                    # Collect following non-flag arguments as list values
                    j = i + 1
                    while (
                        j < len(args)
                        and not args[j].startswith("-")
                        and "=" not in args[j]
                    ):
                        values.append(args[j])
                        j += 1

                    if values:
                        parsed_list = parse_sequence(values, type_hint)
                        overrides.append(f"{key}={json.dumps(parsed_list)}")
                        i = j
                        continue

                # Check if this is a space-separated list (no brackets, followed by values)
                elif not value_part.startswith(("[", "{")) and i + 1 < len(args):
                    # Check if next args are values (not flags, not overrides)
                    next_arg = args[i + 1]
                    if not next_arg.startswith("-") and "=" not in next_arg:
                        # Collect all following values
                        values = [value_part] if value_part else []
                        j = i + 1
                        while (
                            j < len(args)
                            and not args[j].startswith("-")
                            and "=" not in args[j]
                        ):
                            values.append(args[j])
                            j += 1

                        # Parse as sequence (auto-detect type)
                        if len(values) > 1:
                            parsed_list = parse_sequence(values)
                            overrides.append(f"{key_part}={json.dumps(parsed_list)}")
                            i = j
                            continue

                # Regular override
                overrides.append(arg)
                i += 1
            else:
                other_args.append(arg)
                i += 1

        # Parse standard arguments
        parsed_args, remaining = self.parser.parse_known_args(other_args)

        # Add any remaining unknown args as potential overrides
        for arg in remaining:
            if "=" in arg:
                overrides.append(arg)

        # Determine config file (from positional argument or default)
        config_file = parsed_args.config

        # Handle --show-config and --config-help
        if parsed_args.show_config or parsed_args.config_help:
            if not config_file:
                print("Error: No config file specified as first argument.")
                sys.exit(1)

            # Load config
            config = Config(config_file, overrides)

            if parsed_args.config_help:
                self._print_config_schema(config)
            else:
                # Show config as YAML
                print("\n" + "=" * 70)
                print("Current Configuration")
                print("=" * 70)
                print()

                print(
                    yaml.dump(dict(config), default_flow_style=False, sort_keys=False)
                )
                print("=" * 70)
                print()

            sys.exit(0)

        if not config_file:
            raise ValueError(
                "Config file required as first argument: python script.py config.yaml [overrides...]"
            )

        # Load config with overrides
        return Config(config_file, overrides)


# Convenience function
def parse_config(
    config_path: Optional[str] = None, description: Optional[str] = None
) -> Config:
    """Parse command line arguments and return config.

    Args:
        config_path: Default config file path if not provided as first argument
        description: Description for help message

    Returns:
        Config object with parsed configuration

    Usage:
        # Basic usage (config as first argument)
        config = parse_config()

        # Command line examples:
        python script.py config.yaml training.lr=0.001 +new_option=value
        python script.py config.yaml training.gpus=0 1 2 3
        python script.py config.yaml training.ids:int=1 2 3
        python script.py config.yaml --show-config
        python script.py config.yaml --config-help
    """
    parser = ConfigArgumentParser(config_path, description)
    return parser.parse_args()
