import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import uuid
import warnings
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

# TypeVar for generic Config subclasses
ConfigT = TypeVar("ConfigT", bound="Config")

# Additional TypeVar for override_config to preserve type information
OverrideConfigT = TypeVar("OverrideConfigT", bound="Config")


def _preprocess_overrides(overrides: List[str]) -> List[str]:
    """Preprocess overrides to merge space-separated values into key=value format.

    Handles patterns like:
        ['+key:int', '128', '256', '512'] -> ['+key:int=[128,256,512]']
        ['+key:str', 'train', 'val'] -> ['+key:str=["train","val"]']
        ['+key=value'] -> ['+key=value']  # Already in correct format

    Args:
        overrides: Raw list of override strings from parse_known_args()

    Returns:
        Processed list where space-separated values are merged
    """
    if not overrides:
        return overrides

    processed = []
    i = 0

    while i < len(overrides):
        item = overrides[i]

        # Check if this is a key without value (no '=' sign)
        # and potentially followed by space-separated values
        if item.startswith("+") and "=" not in item:
            key = item
            values = []
            j = i + 1

            # Collect following non-option values
            while j < len(overrides) and not overrides[j].startswith(("+", "-")):
                values.append(overrides[j])
                j += 1

            if values:
                # Determine if we have a type hint
                if ":" in key:
                    # Extract type hint
                    key_part = key[1:]  # Remove '+'
                    base_key, type_hint = key_part.rsplit(":", 1)
                    type_hint = type_hint.lower()

                    # Convert values based on type hint
                    if len(values) == 1:
                        # Single value - just use as-is
                        processed.append(f"+{key_part}={values[0]}")
                    else:
                        # Multiple values - create JSON list with proper type
                        if type_hint in ("int", "float"):
                            # Numeric types - no quotes
                            json_list = "[" + ",".join(values) + "]"
                        else:
                            # String or other types - add quotes
                            json_list = '["' + '","'.join(values) + '"]'
                        processed.append(f"+{key_part}={json_list}")
                else:
                    # No type hint - auto-detect or create list
                    if len(values) == 1:
                        processed.append(f"{key}={values[0]}")
                    else:
                        # Multiple values - create JSON list
                        # Try to detect if all values are numeric
                        try:
                            # Try parsing as numbers
                            float_vals = [float(v) for v in values]
                            # Check if they're all integers
                            if all(v.is_integer() for v in float_vals):
                                json_list = "[" + ",".join(values) + "]"
                            else:
                                json_list = "[" + ",".join(values) + "]"
                        except ValueError:
                            # Not numeric - treat as strings
                            json_list = '["' + '","'.join(values) + '"]'
                        processed.append(f"{key}={json_list}")

                i = j
            else:
                # Key without following values - keep as-is (will error later)
                processed.append(item)
                i += 1
        else:
            # Already in key=value format or doesn't start with '+'
            processed.append(item)
            i += 1

    return processed


def load_config(
    config_input: Union[str, List[str], Dict[str, Any]], overrides: List[str] = None
) -> Dict[str, Any]:
    """Load and merge configuration from YAML files or dict, apply overrides.

    Args:
        config_input: Config file path(s) or dict
        overrides: List of key=value overrides

    Returns:
        dict: Loaded and merged configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config = {}

    # Handle different input types
    if isinstance(config_input, dict):
        config = config_input.copy()
    elif isinstance(config_input, str):
        # Single file path
        config_path = Path(config_input)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    elif isinstance(config_input, list):
        # List of file paths
        for path in config_input:
            config_path = Path(path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}
                    config = deep_merge(config, data)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
    else:
        raise ValueError(
            f"config_input must be a dict, string path, or list of paths, "
            f"got {type(config_input)}"
        )

    if overrides:
        # Preprocess overrides to handle space-separated values
        overrides = _preprocess_overrides(overrides)

        for override in overrides:
            # Handle +key=value syntax (add new key)
            is_new_key = override.startswith("+")
            if is_new_key:
                override = override[1:]

            if "=" not in override:
                raise ValueError(
                    f"Invalid override format '{override}'. Expected 'key=value'"
                )

            try:
                # Check for type hint: key:type=value
                key_part, value = override.split("=", 1)
                has_type_hint = ":" in key_part

                if has_type_hint:
                    # Type hint present - extract key without type
                    key = key_part.rsplit(":", 1)[0]
                    # Always set the value directly when type hint is present
                    set_nested_value(config, key.split("."), parse_value(value))
                else:
                    # No type hint - preserve original type if key exists
                    key = key_part
                    parsed_value = parse_value(value)

                    # Use type preservation unless it's a new key
                    if is_new_key:
                        set_nested_value(config, key.split("."), parsed_value)
                    else:
                        set_nested_value_with_type_preservation(
                            config, key.split("."), parsed_value, force_type=True
                        )
            except Exception as e:
                raise ValueError(f"Failed to apply override '{override}': {e}") from e

    return config


def override_config(
    config: OverrideConfigT, unknown_args: List[str]
) -> OverrideConfigT:
    """Override config with unknown args from ArgumentParser.

    Only processes arguments starting with "+". Arguments with "--" or "-"
    are left for the built-in ArgumentParser.

    Supports both formats:
    - Key-value format: +key=value, +nested.key=value
    - Separate format: +key value
    - Sequence format: +key value1 value2 value3

    The "+" prefix is required for all config overrides.

    Args:
        config: Config instance to override
        unknown_args: List of unknown arguments from ArgumentParser.parse_known_args()

    Returns:
        The same config instance with overrides applied (modifies in place)

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> args, unknown = parser.parse_known_args(['+lr=0.01', '+new_key=value'])
        >>> config = Config.from_file("config.yaml")
        >>> config = override_config(config, unknown)

        Supports:
        - +key=value  → Set config key (adds or overrides)
        - +key value  → Set config key with space-separated value
        - +key value1 value2 value3  → Set config key with list of values
        - +nested.key=value  → Set nested config key
        - +key:int=42  → Set with type hint
        - +key:int 1 2 3  → Set list with type hint
    """
    # Warn if there are arguments starting with "-" or "--"
    dash_args = [
        arg for arg in unknown_args if arg.startswith("-") and not arg.startswith("+")
    ]
    if dash_args:
        warnings.warn(
            f"Found arguments with '-' or '--' prefix: {dash_args}. "
            "These will be ignored by override_config(). "
            "Use '+' prefix for config overrides (e.g., +key=value).",
            UserWarning,
            stacklevel=2,
        )

    overrides = []
    i = 0

    while i < len(unknown_args):
        arg = unknown_args[i]

        # Only process arguments starting with "+"
        if not arg.startswith("+"):
            i += 1
            continue

        # Remove the "+" prefix
        arg = arg[1:]

        # Check if it's key=value format
        if "=" in arg:
            overrides.append(arg)
            i += 1
        else:
            # It's key-value(s) separated by space: +key value [value2 value3 ...]
            # Collect all following non-option values
            key = arg
            values = []
            j = i + 1

            # Collect all following values until we hit another option or end
            while j < len(unknown_args) and not unknown_args[j].startswith(("+", "-")):
                values.append(unknown_args[j])
                j += 1

            if values:
                # Check if we have a type hint
                if ":" in key:
                    # Keep type hint for processing later
                    if len(values) == 1:
                        overrides.append(f"{key}={values[0]}")
                    else:
                        # Multiple values - create a list with type hint
                        overrides.append(f"{key}={json.dumps(values)}")
                else:
                    # No type hint
                    if len(values) == 1:
                        overrides.append(f"{key}={values[0]}")
                    else:
                        # Multiple values - create a list (auto-detect type)
                        overrides.append(f"{key}={json.dumps(values)}")
                i = j
            else:
                # Standalone key without value - skip it
                i += 1

    # Apply overrides
    data = config._data  # Access internal data dict

    for override in overrides:
        if "=" not in override:
            continue

        try:
            # Check for type hint: key:type=value
            key_part, value = override.split("=", 1)
            has_type_hint = ":" in key_part

            if has_type_hint:
                # Extract type hint
                key, type_hint = key_part.rsplit(":", 1)
                type_hint = type_hint.lower()

                # Parse value (might be JSON list or single value)
                parsed_value = parse_value(value)

                # Apply type conversion if it's a list
                if isinstance(parsed_value, list):
                    if type_hint == "int":
                        parsed_value = [int(v) for v in parsed_value]
                    elif type_hint == "float":
                        parsed_value = [float(v) for v in parsed_value]
                    elif type_hint == "str":
                        parsed_value = [str(v) for v in parsed_value]
                    elif type_hint == "bool":
                        parsed_value = [
                            v.lower() in ("true", "1", "yes")
                            if isinstance(v, str)
                            else bool(v)
                            for v in parsed_value
                        ]
                elif type_hint in ("int", "float", "str", "bool"):
                    # Single value with type hint
                    if type_hint == "int":
                        parsed_value = int(parsed_value)
                    elif type_hint == "float":
                        parsed_value = float(parsed_value)
                    elif type_hint == "str":
                        parsed_value = str(parsed_value)
                    elif type_hint == "bool":
                        parsed_value = str(parsed_value).lower() in ("true", "1", "yes")

                set_nested_value(data, key.split("."), parsed_value)
            else:
                # No type hint - preserve original type if key exists
                key = key_part
                parsed_value = parse_value(value)

                # Auto-detect type for lists of strings
                if isinstance(parsed_value, list) and all(
                    isinstance(v, str) for v in parsed_value
                ):
                    # Try to convert each element
                    typed_list = []
                    for v in parsed_value:
                        typed_list.append(parse_value(v))
                    parsed_value = typed_list

                # Check if key exists in config
                existing_value = get_nested_value(data, key.split("."))
                if existing_value is not None:
                    # Key exists - preserve type
                    set_nested_value_with_type_preservation(
                        data, key.split("."), parsed_value, force_type=True
                    )
                else:
                    # New key - just set it
                    set_nested_value(data, key.split("."), parsed_value)

            # Also set as attribute if it's a dataclass field (for typed configs)
            # This ensures typed configs get their fields updated properly
            if (
                hasattr(config.__class__, "__dataclass_fields__")
                and key in config.__class__.__dataclass_fields__
            ):
                object.__setattr__(config, key, parsed_value)

        except Exception as e:
            raise ValueError(f"Failed to apply override '{override}': {e}") from e

    # After all overrides are applied, reconstruct nested dataclass instances
    # This ensures nested configs like +model.dim=512 work properly
    if hasattr(config.__class__, "__dataclass_fields__"):
        field_types = get_type_hints(config.__class__)
        for field_name, field_type in field_types.items():
            # Check if this field is a nested dataclass
            if field_name in data and hasattr(field_type, "__dataclass_fields__"):
                # Reconstruct the nested dataclass from the updated dict
                nested_instance = dict_to_dataclass(field_type, data[field_name])
                object.__setattr__(config, field_name, nested_instance)

    return config


def deep_merge(base: Dict, update: Dict) -> Dict:
    """Deep merge two dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def get_nested_value(d: Dict, keys: List[str]) -> Any:
    """Get a nested value from a dict using dot notation.

    Returns None if key doesn't exist.
    """
    try:
        val = d
        for key in keys:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return None


def set_nested_value(d: Dict, keys: List[str], value: Any):
    """Set a nested value in a dict using dot notation."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def set_nested_value_with_type_preservation(
    d: Dict, keys: List[str], value: Any, force_type: bool = False
):
    """Set a nested value in a dict, preserving the original type if it exists.

    Args:
        d: Dictionary to modify
        keys: List of keys forming the path (e.g., ['training', 'lr'])
        value: New value to set
        force_type: If True, force the type to match the original
    """
    # Get the original value if it exists
    original_value = get_nested_value(d, keys)

    # If original exists and we should preserve type
    if original_value is not None and force_type:
        # If original is a list and new value is not, wrap it
        if isinstance(original_value, list) and not isinstance(value, list):
            value = [value]
        # If original is not a list but new value is a single-element list, unwrap
        elif (
            not isinstance(original_value, list)
            and isinstance(value, list)
            and len(value) == 1
        ):
            value = value[0]

    # Set the value
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def parse_value(value: str) -> Any:
    """Parse string value to appropriate type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try to parse as JSON (handles lists, dicts, etc.)
    if value.startswith(("[", "{")) or value.startswith('"'):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def interpolate_config(config: Dict) -> Dict:
    """Interpolate variables in config with resolvers.

    Supported resolvers:
        ${env:VAR_NAME}         - Environment variable
        ${env:VAR_NAME,default} - Environment variable with default
        ${now:%Y%m%d_%H%M%S}    - Current timestamp with format
        ${git_sha}              - Current git commit SHA
        ${git_sha:short}        - Short git commit SHA (7 chars)
        ${hostname}             - Current hostname
        ${uuid4}                - Random UUID4
        ${config:other.key}     - Reference another config key

    Examples:
        run_dir: "runs/${now:%Y%m%d_%H%M%S}"
        api_key: "${env:API_KEY,default_key}"
        model_path: "${config:paths.base}/model.pt"
    """

    def resolve_value(value, config_dict):
        """Recursively resolve a single value."""
        if not isinstance(value, str):
            return value

        # Pattern: ${resolver:arg1,arg2,...}
        pattern = r"\$\{([^}]+)\}"

        def replacer(match):
            expr = match.group(1)

            # If no colon, check if it's a known resolver or config reference
            if ":" not in expr:
                # Check if it's a known resolver without args
                if expr == "hostname":
                    return socket.gethostname()
                if expr == "git_sha":
                    try:
                        result = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        return result.stdout.strip()
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        return "unknown"
                elif expr == "uuid4":
                    return str(uuid.uuid4())

                # Otherwise, try as config reference
                keys = expr.split(".")
                val = config_dict
                try:
                    for key in keys:
                        val = val[key]
                    return str(val)
                except (KeyError, TypeError):
                    # If not found as config reference, return as is
                    return match.group(0)

            parts = expr.split(":", 1)
            resolver = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            # Environment variable
            if resolver == "env":
                args_list = args.split(",", 1)
                var_name = args_list[0]
                default = args_list[1] if len(args_list) > 1 else None
                result = os.getenv(var_name, default)
                if result is None:
                    raise ValueError(
                        f"Environment variable '{var_name}' not found "
                        f"and no default provided"
                    )
                return result

            # Timestamp
            if resolver == "now":
                fmt = args if args else "%Y%m%d_%H%M%S"
                return datetime.datetime.now().strftime(fmt)

            # Git SHA
            if resolver == "git_sha":
                try:
                    sha = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ).strip()
                    if args == "short":
                        return sha[:7]
                    return sha
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return "unknown"

            # Hostname
            elif resolver == "hostname":
                return socket.gethostname()

            # UUID
            elif resolver == "uuid4":
                return str(uuid.uuid4())

            # Config reference
            elif resolver == "config":
                keys = args.split(".")
                val = config_dict
                for key in keys:
                    val = val[key]
                return str(val)

            else:
                raise ValueError(f"Unknown resolver: {resolver}")

        # Keep replacing until no more patterns found (handles nested references)
        max_iterations = 10
        for _ in range(max_iterations):
            new_value = re.sub(pattern, replacer, value)
            if new_value == value:
                break
            value = new_value

        return value

    def interpolate_recursive(obj, config_dict):
        """Recursively interpolate all strings in nested structures."""
        if isinstance(obj, dict):
            return {k: interpolate_recursive(v, config_dict) for k, v in obj.items()}
        if isinstance(obj, list):
            return [interpolate_recursive(item, config_dict) for item in obj]
        if isinstance(obj, str):
            return resolve_value(obj, config_dict)
        return obj

    return interpolate_recursive(config, config)


def save_config_snapshot(config: Dict, run_dir: Path, warn_if_exists: bool = True):
    """Save resolved config and its hash.

    Args:
        config: Configuration dictionary to save
        run_dir: Directory to save config snapshot
        warn_if_exists: If True, print warning if run_dir already exists
    """
    if run_dir.exists() and warn_if_exists:
        warnings.warn(f"Run directory already exists: {run_dir}", stacklevel=2)

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run.yaml"
    snapshot_file = run_dir / "run.hash"

    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        config_str = yaml.dump(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        with open(snapshot_file, "w") as f:
            f.write(config_hash)
    except Exception as e:
        raise OSError(f"Failed to save config snapshot: {e}") from e


def dict_to_dataclass(cls, data: Dict[str, Any]):
    """Convert a dictionary to a dataclass instance recursively.

    Args:
        cls: The dataclass type to instantiate
        data: Dictionary with config data

    Returns:
        Instance of the dataclass with nested dataclasses created
    """
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    field_types = get_type_hints(cls)
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            # Check if field has a default value
            field_obj = cls.__dataclass_fields__.get(field_name)
            if field_obj and field_obj.default is not MISSING:
                continue
            elif field_obj and field_obj.default_factory is not MISSING:
                continue
            else:
                # Field is required but missing
                continue

        value = data[field_name]

        # Handle Optional types
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Check if it's Optional (Union[X, None])
            if type(None) in args:
                # Get the non-None type
                field_type = next(arg for arg in args if arg is not type(None))
                origin = get_origin(field_type)

        # Recursively convert nested dicts to dataclasses
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[field_name] = dict_to_dataclass(field_type, value)
        # Handle lists of dataclasses
        elif origin is list and isinstance(value, list):
            args = get_args(field_type)
            if args and hasattr(args[0], "__dataclass_fields__"):
                kwargs[field_name] = [
                    dict_to_dataclass(args[0], item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                kwargs[field_name] = value
        # Handle dicts
        elif origin is dict:
            kwargs[field_name] = value
        else:
            kwargs[field_name] = value

    return cls(**kwargs)


@dataclass
class ConfigDict:
    """Helper class for nested dictionary access with dot notation.

    Provides LSP-friendly access to nested config values.
    """

    _data: Dict[str, Any] = field(repr=False)
    _parent_config: "Config" = field(repr=False)
    _path: List[str] = field(default_factory=list, repr=False)

    def __getattr__(self, name: str) -> Any:
        """Enable chained dot notation."""
        data = object.__getattribute__(self, "_data")

        if name in data:
            value = data[name]
            if isinstance(value, dict):
                parent = object.__getattribute__(self, "_parent_config")
                path = object.__getattribute__(self, "_path")
                return ConfigDict(value, parent, path + [name])
            return value

        raise AttributeError(f"Config has no key '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Support bracket notation."""
        data = object.__getattribute__(self, "_data")
        return data[key]

    def __setitem__(self, key: str, value: Any):
        """Support setting with brackets."""
        data = object.__getattribute__(self, "_data")
        data[key] = value

    def __setattr__(self, name: str, value: Any):
        """Support setting with dot notation."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            data = object.__getattribute__(self, "_data")
            data[name] = value

    def __repr__(self) -> str:
        path = ".".join(object.__getattribute__(self, "_path"))
        data = object.__getattribute__(self, "_data")
        return f"ConfigDict({path}={data!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to regular dictionary."""
        return dict(object.__getattribute__(self, "_data"))


@dataclass
class Config:
    """Base configuration class with dynamic attribute support and LSP-friendly typing.

    This class can be used in two ways:
    1. As a base class - define your own dataclass with typed fields for full LSP support
    2. As a dynamic config - access any config key with bracket or dot notation

    Example with typed subclass:
        @dataclass
        class MyConfig(Config):
            learning_rate: float
            batch_size: int
            model_name: str = "resnet50"

        config = MyConfig.from_file("config.yaml")
        lr = config.learning_rate  # Full LSP autocomplete!

    Example with dynamic access:
        config = Config.from_file("config.yaml")
        lr = config["training.lr"]
        batch_size = config.training.batch_size
    """

    # Internal fields - not part of config data
    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _raw_config: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __init__(
        self,
        config_input: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        overrides: Optional[List[str]] = None,
        run_dir: Optional[Path] = None,
    ):
        """Initialize Config (backward compatible with old API).

        Args:
            config_input: Config file path(s) or dict (optional for subclasses)
            overrides: List of key=value overrides
            run_dir: Directory to save config snapshot
        """
        # For typed dataclass subclasses initialized via dict_to_dataclass,
        # config_input will be None
        if config_input is None:
            object.__setattr__(self, "_data", {})
            object.__setattr__(self, "_raw_config", {})
            return

        # Normal initialization from file/dict
        raw_config = load_config(config_input, overrides)
        interpolated = interpolate_config(raw_config)

        object.__setattr__(self, "_raw_config", raw_config)
        object.__setattr__(self, "_data", interpolated)

        if run_dir:
            save_config_snapshot(interpolated, run_dir)

    @classmethod
    def from_dict(
        cls: Type[ConfigT],
        config_data: Dict[str, Any],
        overrides: Optional[List[str]] = None,
        run_dir: Optional[Path] = None,
    ) -> ConfigT:
        """Create Config from dictionary.

        Args:
            config_data: Configuration dictionary
            overrides: List of key=value overrides
            run_dir: Directory to save config snapshot

        Returns:
            Config instance of the appropriate type
        """
        raw_config = load_config(config_data, overrides)
        interpolated = interpolate_config(raw_config)

        if run_dir:
            save_config_snapshot(interpolated, run_dir)

        # If cls is the base Config class, use dynamic dict
        if cls is Config:
            instance = cls()
            object.__setattr__(instance, "_raw_config", raw_config)
            object.__setattr__(instance, "_data", interpolated)
            return instance

        # Otherwise, it's a typed subclass - populate fields
        instance = dict_to_dataclass(cls, interpolated)
        object.__setattr__(instance, "_raw_config", raw_config)
        object.__setattr__(instance, "_data", interpolated)
        return instance

    @classmethod
    def from_file(
        cls: Type[ConfigT],
        config_input: Union[str, List[str]],
        overrides: Optional[List[str]] = None,
        run_dir: Optional[Path] = None,
    ) -> ConfigT:
        """Create Config from YAML file(s).

        Args:
            config_input: Config file path or list of paths
            overrides: List of key=value overrides
            run_dir: Directory to save config snapshot

        Returns:
            Config instance of the appropriate type
        """
        raw_config = load_config(config_input, overrides)
        interpolated = interpolate_config(raw_config)

        if run_dir:
            save_config_snapshot(interpolated, run_dir)

        # If cls is the base Config class, use dynamic dict
        if cls is Config:
            instance = cls()
            object.__setattr__(instance, "_raw_config", raw_config)
            object.__setattr__(instance, "_data", interpolated)
            return instance

        # Otherwise, it's a typed subclass - populate fields
        instance = dict_to_dataclass(cls, interpolated)
        object.__setattr__(instance, "_raw_config", raw_config)
        object.__setattr__(instance, "_data", interpolated)
        return instance

    def override(
        self: ConfigT, overrides: List[str], run_dir: Optional[Path] = None
    ) -> ConfigT:
        """Create a new config instance with overrides applied.

        This method creates a new Config instance with the specified overrides
        applied to the current configuration. The original config is not modified.

        Args:
            overrides: List of key=value overrides
            run_dir: Optional directory to save the new config snapshot

        Returns:
            New Config instance with overrides applied

        Example:
            config = Config.from_file("config.yaml")
            new_config = config.override(["+lr=0.01", "batch_size=64"])
        """
        # Get current data and merge with overrides (deep copy to avoid mutation)
        current_data = deepcopy(object.__getattribute__(self, "_data"))
        raw_config = load_config(current_data, overrides)
        interpolated = interpolate_config(raw_config)

        if run_dir:
            save_config_snapshot(interpolated, run_dir)

        # Create new instance of the same type
        cls = type(self)

        # If cls is the base Config class, use dynamic dict
        if cls is Config:
            instance = cls()
            object.__setattr__(instance, "_raw_config", raw_config)
            object.__setattr__(instance, "_data", interpolated)
            return instance

        # Otherwise, it's a typed subclass - populate fields
        instance = dict_to_dataclass(cls, interpolated)
        object.__setattr__(instance, "_raw_config", raw_config)
        object.__setattr__(instance, "_data", interpolated)
        return instance

    @property
    def raw_config(self) -> Dict[str, Any]:
        """Get raw config before interpolation."""
        return object.__getattribute__(self, "_raw_config")

    @property
    def config(self) -> Dict[str, Any]:
        """Get full interpolated config as dict."""
        return object.__getattribute__(self, "_data")

    def __getitem__(self, key: str) -> Any:
        """Support dict-style access: config['key'] or config['nested.key']"""
        data = object.__getattribute__(self, "_data")
        if isinstance(key, str) and "." in key:
            keys = key.split(".")
            value = data
            for k in keys:
                value = value[k]
            return value
        return data[key]

    def __setitem__(self, key: str, value: Any):
        """Support dict-style setting."""
        data = object.__getattribute__(self, "_data")
        if isinstance(key, str) and "." in key:
            keys = key.split(".")
            d = data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        else:
            data[key] = value

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        if isinstance(key, str) and "." in key:
            try:
                self[key]
                return True
            except KeyError:
                return False
        data = object.__getattribute__(self, "_data")
        return key in data

    def __getattr__(self, name: str) -> Any:
        """Enable dot notation for dynamic config access.

        Only called if attribute doesn't exist as a dataclass field.
        """
        # Try to get from _data
        try:
            data = object.__getattribute__(self, "_data")
            if name in data:
                value = data[name]
                # If it's a dict, wrap it in a ConfigDict for chaining
                if isinstance(value, dict):
                    return ConfigDict(value, self, [name])
                return value
        except AttributeError:
            pass

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Handle attribute setting for both dataclass fields and dynamic config."""
        # Check if it's a dataclass field
        if (
            hasattr(self.__class__, "__dataclass_fields__")
            and name in self.__class__.__dataclass_fields__
        ):
            object.__setattr__(self, name, value)
        # Internal attributes
        elif name.startswith("_"):
            object.__setattr__(self, name, value)
        # Dynamic config key
        else:
            data = object.__getattribute__(self, "_data")
            data[name] = value

    def __repr__(self) -> str:
        """Return representation of Config."""
        data = object.__getattribute__(self, "_data")
        if not data:
            return f"{self.__class__.__name__}()"

        # For typed dataclasses, show field values
        if hasattr(self.__class__, "__dataclass_fields__"):
            field_names = [
                f.name for f in fields(self.__class__) if not f.name.startswith("_")
            ]
            if field_names:
                field_strs = []
                for name in field_names:
                    try:
                        value = getattr(self, name)
                        field_strs.append(f"{name}={value!r}")
                    except AttributeError:
                        pass
                if field_strs:
                    return f"{self.__class__.__name__}({', '.join(field_strs)})"

        # For dynamic configs, show all keys (or reasonable limit)
        MAX_REPR_KEYS = 20  # Show up to 20 keys before truncating
        keys = list(data.keys())

        if len(keys) <= MAX_REPR_KEYS:
            # Show all keys
            key_str = ", ".join(f"{k}={data[k]!r}" for k in keys)
        else:
            # Truncate if too many
            shown_keys = keys[:MAX_REPR_KEYS]
            key_str = ", ".join(f"{k}={data[k]!r}" for k in shown_keys)
            key_str += f", ... ({len(data)} total)"

        return f"{self.__class__.__name__}({key_str})"

    def __str__(self) -> str:
        """Return string representation of Config."""
        data = object.__getattribute__(self, "_data")

        # For typed dataclasses, show field values even if _data is empty
        if hasattr(self.__class__, "__dataclass_fields__"):
            field_names = [
                f.name for f in fields(self.__class__) if not f.name.startswith("_")
            ]
            if field_names:
                lines = [f"{self.__class__.__name__}:"]
                for name in field_names:
                    try:
                        value = getattr(self, name)
                        lines.append(f"  {name}: {self._format_value(value, indent=2)}")
                    except AttributeError:
                        pass
                return "\n".join(lines)

        # For dynamic configs, check if data is empty
        if not data:
            return f"{self.__class__.__name__}()"

        lines = [f"{self.__class__.__name__}:"]
        for key, value in data.items():
            lines.append(f"  {key}: {self._format_value(value, indent=2)}")
        return "\n".join(lines)

    def _format_value(self, value, indent=0):
        """Format a value for display, handling nested structures."""
        indent_str = " " * indent

        # Handle nested Config objects
        if isinstance(value, Config):
            # Get the fields to display
            if hasattr(value.__class__, "__dataclass_fields__"):
                field_names = [
                    f.name
                    for f in fields(value.__class__)
                    if not f.name.startswith("_")
                ]
                if field_names:
                    lines = [f"{value.__class__.__name__}"]
                    for name in field_names:
                        try:
                            field_value = getattr(value, name)
                            formatted = self._format_value(field_value, indent + 2)
                            # Check if formatted value is multiline
                            if "\n" in formatted:
                                # For multiline nested values, first line is inline with key
                                formatted_lines = formatted.split("\n")
                                lines.append(
                                    f"{indent_str}  {name}: {formatted_lines[0]}"
                                )
                                # Rest of lines are indented
                                for line in formatted_lines[1:]:
                                    if line:
                                        lines.append(f"{indent_str}  {line}")
                            else:
                                lines.append(f"{indent_str}  {name}: {formatted}")
                        except AttributeError:
                            pass
                    return "\n".join(lines)
            # Fallback for dynamic configs
            data = object.__getattribute__(value, "_data")
            if data:
                lines = [f"{value.__class__.__name__}"]
                for k, v in data.items():
                    formatted = self._format_value(v, indent + 2)
                    if "\n" in formatted:
                        formatted_lines = formatted.split("\n")
                        lines.append(f"{indent_str}  {k}: {formatted_lines[0]}")
                        for line in formatted_lines[1:]:
                            if line:
                                lines.append(f"{indent_str}  {line}")
                    else:
                        lines.append(f"{indent_str}  {k}: {formatted}")
                return "\n".join(lines)
            return f"{value.__class__.__name__}()"

        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = ["{"]
            for k, v in value.items():
                formatted = self._format_value(v, indent + 2)
                if "\n" in formatted:
                    lines.append(f"{indent_str}  {k}:")
                    for line in formatted.split("\n"):
                        lines.append(f"{indent_str}    {line}")
                else:
                    lines.append(f"{indent_str}  {k}: {formatted}")
            lines.append(f"{indent_str}}}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return "[]"
            if len(value) <= 3 and all(
                not isinstance(v, (dict, list, Config)) for v in value
            ):
                return f"[{', '.join(repr(v) for v in value)}]"
            lines = ["["]
            for item in value:
                formatted = self._format_value(item, indent + 2)
                if "\n" in formatted:
                    for line in formatted.split("\n"):
                        lines.append(f"{indent_str}  {line}")
                else:
                    lines.append(f"{indent_str}  {formatted}")
            lines.append(f"{indent_str}]")
            return "\n".join(lines)
        return repr(value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Return config keys."""
        return object.__getattribute__(self, "_data").keys()

    def items(self):
        """Return config items."""
        return object.__getattribute__(self, "_data").items()

    def values(self):
        """Return config values."""
        return object.__getattribute__(self, "_data").values()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        # If it's a typed dataclass, convert fields to dict
        if hasattr(self.__class__, "__dataclass_fields__"):
            result = {}
            for field_obj in fields(self.__class__):
                if not field_obj.name.startswith("_"):
                    value = getattr(self, field_obj.name)
                    if hasattr(value, "to_dict"):
                        result[field_obj.name] = value.to_dict()
                    else:
                        result[field_obj.name] = value
            # Add any dynamic keys from _data
            data = object.__getattribute__(self, "_data")
            for key, value in data.items():
                if key not in result:
                    result[key] = value
            return result

        # Otherwise, return _data
        return dict(object.__getattribute__(self, "_data"))

    def save(self, filepath: Union[str, Path]):
        """Save config to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def hash(self) -> str:
        """Generate SHA256 hash of config."""
        config_str = yaml.dump(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
