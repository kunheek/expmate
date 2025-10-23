import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


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
                has_type_hint = ":" in key_part and not is_new_key

                if has_type_hint:
                    # Type hint present - don't preserve original type
                    key = key_part.rsplit(":", 1)[0]
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


class DotAccessor:
    """Helper class for dot notation access to nested config values.

    Automatically resolves the value when accessed, making config.training.lr
    work directly without needing to call it like config.training.lr().
    """

    def __init__(self, config, path_parts):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_path_parts", path_parts)
        object.__setattr__(self, "_value", None)
        object.__setattr__(self, "_resolved", False)

    def _resolve(self):
        """Resolve the actual value from the config."""
        if not object.__getattribute__(self, "_resolved"):
            path = ".".join(object.__getattribute__(self, "_path_parts"))
            config = object.__getattribute__(self, "_config")
            try:
                value = config[path]
                object.__setattr__(self, "_value", value)
                object.__setattr__(self, "_resolved", True)
            except KeyError as e:
                raise AttributeError(f"Config has no key '{path}'") from e
        return object.__getattribute__(self, "_value")

    def __getattr__(self, name):
        """Allow chaining: config.training.lr"""
        # First try to resolve and see if it's a dict
        try:
            value = self._resolve()
            if isinstance(value, dict) and name in value:
                # Continue chaining
                path_parts = object.__getattribute__(self, "_path_parts")
                config = object.__getattribute__(self, "_config")
                return DotAccessor(config, path_parts + [name])
        except (KeyError, AttributeError):
            pass

        # Otherwise, extend the path
        path_parts = object.__getattribute__(self, "_path_parts")
        config = object.__getattribute__(self, "_config")
        return DotAccessor(config, path_parts + [name])

    def __getitem__(self, key):
        """Allow indexing: config.training['lr']"""
        value = self._resolve()
        if isinstance(value, (dict, list, tuple)):
            return value[key]
        raise TypeError(f"'{type(value).__name__}' object is not subscriptable")

    def __setattr__(self, name, value):
        """Support setting: config.training.lr = value"""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            path_parts = object.__getattribute__(self, "_path_parts")
            config = object.__getattribute__(self, "_config")
            path = ".".join(path_parts + [name])
            config[path] = value

    def __setitem__(self, key, value):
        """Support setting with indexing: config.training['lr'] = value"""
        resolved = self._resolve()
        if isinstance(resolved, dict):
            resolved[key] = value
        else:
            raise TypeError(
                f"'{type(resolved).__name__}' object does not support item assignment"
            )

    # Automatic conversion methods - resolve value when used in operations
    def __int__(self):
        return int(self._resolve())

    def __float__(self):
        return float(self._resolve())

    def __bool__(self):
        return bool(self._resolve())

    def __index__(self):
        """Support using in range(), indexing, etc."""
        value = self._resolve()
        if isinstance(value, int):
            return value
        return int(value)

    def __str__(self):
        return str(self._resolve())

    def __repr__(self):
        value = self._resolve()
        path = ".".join(object.__getattribute__(self, "_path_parts"))
        return f"<Config[{path}]={value!r}>"

    def __eq__(self, other):
        return self._resolve() == other

    def __ne__(self, other):
        return self._resolve() != other

    def __lt__(self, other):
        return self._resolve() < other

    def __le__(self, other):
        return self._resolve() <= other

    def __gt__(self, other):
        return self._resolve() > other

    def __ge__(self, other):
        return self._resolve() >= other

    def __add__(self, other):
        return self._resolve() + other

    def __radd__(self, other):
        return other + self._resolve()

    def __sub__(self, other):
        return self._resolve() - other

    def __rsub__(self, other):
        return other - self._resolve()

    def __mul__(self, other):
        return self._resolve() * other

    def __rmul__(self, other):
        return other * self._resolve()

    def __truediv__(self, other):
        return self._resolve() / other

    def __rtruediv__(self, other):
        return other / self._resolve()

    def __floordiv__(self, other):
        return self._resolve() // other

    def __rfloordiv__(self, other):
        return other // self._resolve()

    def __mod__(self, other):
        return self._resolve() % other

    def __rmod__(self, other):
        return other % self._resolve()

    def __pow__(self, other):
        return self._resolve() ** other

    def __rpow__(self, other):
        return other ** self._resolve()

    def __iter__(self):
        """Support iteration for lists/dicts"""
        return iter(self._resolve())

    def __len__(self):
        """Support len() for lists/dicts/strings"""
        return len(self._resolve())

    def __contains__(self, item):
        """Support 'in' operator"""
        return item in self._resolve()


class Config:
    """Configuration container with dot notation access and dict-like interface.

    Supports both attribute-style (config.training.lr) and dict-style
    (config['training']['lr'] or config['training.lr']) access.
    """

    def __init__(
        self,
        config_input: Union[str, List[str], Dict[str, Any]],
        overrides: List[str] = None,
        run_dir: Optional[Path] = None,
    ):
        object.__setattr__(self, "_raw_config", load_config(config_input, overrides))
        object.__setattr__(
            self,
            "_data",
            interpolate_config(object.__getattribute__(self, "_raw_config")),
        )
        if run_dir:
            save_config_snapshot(object.__getattribute__(self, "_data"), run_dir)

    @property
    def raw_config(self):
        return object.__getattribute__(self, "_raw_config")

    @property
    def config(self):
        return object.__getattribute__(self, "_data")

    @property
    def data(self):
        """For compatibility with dict-like interface."""
        return object.__getattribute__(self, "_data")

    def __repr__(self):
        """Human-friendly representation of the config."""
        lines = ["Config:"]
        data = object.__getattribute__(self, "_data")
        for key, value in data.items():
            lines.append(f"  {key}: {self._format_value(value, indent=2)}")
        return "\n".join(lines)

    def _format_value(self, value, indent=0):
        """Format a value for display, handling nested structures."""
        indent_str = " " * indent
        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = ["{"]
            for k, v in value.items():
                lines.append(f"{indent_str}  {k}: {self._format_value(v, indent + 2)}")
            lines.append(f"{indent_str}}}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return "[]"
            if len(value) <= 3 and all(not isinstance(v, (dict, list)) for v in value):
                return f"[{', '.join(repr(v) for v in value)}]"
            lines = ["["]
            for item in value:
                lines.append(f"{indent_str}  {self._format_value(item, indent + 2)}")
            lines.append(f"{indent_str}]")
            return "\n".join(lines)
        return repr(value)

    def __getitem__(self, key):
        """Support dict-style access: config['key'] or config['nested.key']"""
        data = object.__getattribute__(self, "_data")
        if isinstance(key, str) and "." in key:
            # Support dot notation in string keys: config['training.lr']
            keys = key.split(".")
            value = data
            for k in keys:
                value = value[k]
            return value
        return data[key]

    def __setitem__(self, key, value):
        """Support dict-style setting.

        Examples: config['key'] = value or config['nested.key'] = value
        """
        data = object.__getattribute__(self, "_data")
        if isinstance(key, str) and "." in key:
            # Support dot notation in string keys: config['training.lr'] = 0.01
            keys = key.split(".")
            d = data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        else:
            data[key] = value

    def __contains__(self, key):
        """Support 'in' operator: 'key' in config or 'nested.key' in config"""
        if isinstance(key, str) and "." in key:
            try:
                self[key]
                return True
            except KeyError:
                return False
        else:
            data = object.__getattribute__(self, "_data")
            return key in data

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getattribute__(self, name):
        """Intercept attribute access to provide DotAccessor for config keys.

        This enables attribute-style access: config.training.lr
        """
        # Always use object.__getattribute__ for internal attributes and methods
        if name.startswith("_") or name in (
            "raw_config",
            "config",
            "keys",
            "items",
            "values",
            "get",
            "pop",
            "update",
            "clear",
            "setdefault",
            "popitem",
            "copy",
            "fromkeys",
            "to_dict",
            "save",
            "hash",
        ):
            return object.__getattribute__(self, name)

        # Special case: 'data' - check if it's a config key first
        if name == "data":
            try:
                config_data = object.__getattribute__(self, "_data")
                if "data" in config_data:
                    # 'data' is a config key, return DotAccessor
                    return DotAccessor(self, ["data"])
            except AttributeError:
                pass
            # Not a config key, return the property
            return object.__getattribute__(self, name)

        # Check if it's a config key
        try:
            data = object.__getattribute__(self, "_data")
            if name in data:
                # Return DotAccessor for config keys
                return DotAccessor(self, [name])
        except AttributeError:
            pass

        # Not found, raise AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        """Enable dot notation setting: config.training = value"""
        # Internal attributes use object.__setattr__
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            # Set as config value
            data = object.__getattribute__(self, "_data")
            data[name] = value

    # Dict-like interface methods
    def keys(self):
        """Return config keys."""
        data = object.__getattribute__(self, "_data")
        return data.keys()

    def items(self):
        """Return config items."""
        data = object.__getattribute__(self, "_data")
        return data.items()

    def values(self):
        """Return config values."""
        data = object.__getattribute__(self, "_data")
        return data.values()

    def update(self, other):
        """Update config with another dict."""
        data = object.__getattribute__(self, "_data")
        data.update(other)

    def pop(self, key, *args):
        """Remove and return config value."""
        data = object.__getattribute__(self, "_data")
        return data.pop(key, *args)

    def clear(self):
        """Clear all config values."""
        data = object.__getattribute__(self, "_data")
        data.clear()

    def setdefault(self, key, default=None):
        """Set default value if key doesn't exist."""
        data = object.__getattribute__(self, "_data")
        return data.setdefault(key, default)

    def copy(self):
        """Return a shallow copy of the config data."""
        data = object.__getattribute__(self, "_data")
        return data.copy()

    def __len__(self):
        """Return number of top-level config keys."""
        data = object.__getattribute__(self, "_data")
        return len(data)

    def __iter__(self):
        """Iterate over top-level config keys."""
        data = object.__getattribute__(self, "_data")
        return iter(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a regular dictionary.

        Returns:
            dict: Dictionary representation of the config
        """
        data = object.__getattribute__(self, "_data")
        return dict(data)

    def save(self, filepath: str):
        """Save config to a YAML file.

        Args:
            filepath: Path to save the config file
        """
        data = object.__getattribute__(self, "_data")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def hash(self) -> str:
        """Generate a hash of the config for reproducibility.

        Returns:
            str: SHA256 hash of the config
        """
        data = object.__getattribute__(self, "_data")
        config_str = yaml.dump(data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
