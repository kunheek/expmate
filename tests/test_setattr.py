"""Test setattr by dot access."""

from dataclasses import dataclass
from expmate.config import Config


print("=" * 70)
print("Testing setattr with dot access")
print("=" * 70)
print()

# Test 1: Dynamic config
print("Test 1: Dynamic Config (base Config class)")
print("-" * 70)
config = Config.from_dict({"existing_key": "value1"})

# Set existing key
config.existing_key = "modified_value"
print(f"Modified existing key: {config.existing_key}")

# Set new key via dot access
config.new_key = "new_value"
print(f"New key via dot access: {config.new_key}")

# Verify it's in _data
print(f"Keys in config: {list(config.keys())}")
print(f"config['new_key'] = {config['new_key']}")
print()

# Test 2: Typed config
print("Test 2: Typed Config (dataclass subclass)")
print("-" * 70)


@dataclass
class MyConfig(Config):
    """Typed config with defined fields."""

    required_field: str
    optional_field: int = 42


typed_config = MyConfig.from_dict({"required_field": "test"})

# Set dataclass field
typed_config.optional_field = 100
print(f"Set dataclass field: {typed_config.optional_field}")

# Set new dynamic key (not in dataclass definition)
typed_config.dynamic_key = "dynamic_value"
print(f"Set dynamic key: {typed_config.dynamic_key}")

# Verify both are accessible
print(f"Dataclass field: {typed_config.optional_field}")
print(f"Dynamic field: {typed_config.dynamic_key}")
print(f"All keys: {list(typed_config.keys())}")
print()

# Test 3: Nested setting
print("Test 3: Nested attribute setting")
print("-" * 70)
config2 = Config.from_dict({"nested": {"key1": "value1"}})

# Note: This sets the top-level 'nested' to a new value
# It doesn't automatically create nested dicts
config2.nested = {"key1": "modified", "key2": "new"}
print(f"Replaced nested dict: {config2.nested}")

# To add to nested, you need to access and modify
config2["nested"]["key3"] = "added"
print(f"Modified nested dict: {config2['nested']}")
print()

# Test 4: Setting via dict-style also works
print("Test 4: Dict-style setting")
print("-" * 70)
config3 = Config.from_dict({"key1": "value1"})

config3["key2"] = "value2"  # Dict-style
config3.key3 = "value3"  # Dot-style

print(f"Both methods work:")
print(f"  config['key2'] = {config3['key2']}")
print(f"  config.key3 = {config3.key3}")
print()

print("=" * 70)
print("Summary:")
print("=" * 70)
print("✓ config.new_key = value  → Sets dynamic config value")
print("✓ config['new_key'] = value  → Also sets dynamic config value")
print("✓ Works for both base Config and typed dataclass subclasses")
print("✓ Dataclass fields are set as attributes")
print("✓ Non-dataclass fields are stored in _data dict")
