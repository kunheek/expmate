#!/usr/bin/env python3
"""Test script for the new dataclass-based Config implementation."""

from dataclasses import dataclass
from pathlib import Path
from expmate.config import Config

# Test 1: Dynamic config access
print("Test 1: Dynamic config access")
print("-" * 50)
# Get path relative to this test file
test_dir = Path(__file__).parent
config_path = test_dir.parent / "examples" / "conf" / "default.yaml"
config1 = Config.from_file(str(config_path))
print(f"Config loaded: {type(config1)}")
print(f"Keys: {list(config1.keys())}")
print()

# Test 2: Typed config with dataclass
print("Test 2: Typed config with dataclass")
print("-" * 50)


@dataclass
class TrainingConfig(Config):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "adam"


@dataclass
class MyTypedConfig(Config):
    name: str
    seed: int
    training: dict  # We'll use dict here since TrainingConfig is also a Config


# Try loading with the base Config first
config2 = Config.from_file(str(config_path))
print(f"Dynamic config type: {type(config2)}")
print(f"Config data: {config2.to_dict()}")
print()

# Test 3: Accessing values with dot notation
print("Test 3: Dot notation access")
print("-" * 50)
try:
    if "training" in config2:
        print(f"Has training: Yes")
        training = config2.training
        print(f"Training config: {training}")
        print(f"Training type: {type(training)}")
except Exception as e:
    print(f"Error accessing training: {e}")
print()

# Test 4: Dict-style access
print("Test 4: Dict-style access")
print("-" * 50)
try:
    if "name" in config2:
        name = config2["name"]
        print(f"Config name (dict-style): {name}")
except Exception as e:
    print(f"Error with dict access: {e}")
print()

# Test 5: Save and hash
print("Test 5: Save and hash")
print("-" * 50)
try:
    config_hash = config2.hash()
    print(f"Config hash: {config_hash[:16]}...")

    # Save to temp file
    temp_file = Path("/tmp/test_config.yaml")
    config2.save(temp_file)
    print(f"Saved config to: {temp_file}")
    print(f"File exists: {temp_file.exists()}")
except Exception as e:
    print(f"Error saving: {e}")
print()

print("All tests completed!")
