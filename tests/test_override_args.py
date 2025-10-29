"""Test override_config with ArgumentParser unknown_args."""

import argparse
import warnings
from dataclasses import dataclass
from expmate.config import Config, override_config


print("=" * 70)
print("Testing override_config with ArgumentParser")
print("Only + prefix is used for config overrides")
print("=" * 70)
print()

# Test 1: Basic override with +key=value format
print("Test 1: Basic override with +key=value format")
print("-" * 70)
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(["+lr=0.01", "+batch_size=64"])

config = Config.from_dict({"lr": 0.001, "batch_size": 32, "epochs": 100})
print(f"Before: lr={config.lr}, batch_size={config.batch_size}")

config = override_config(config, unknown)
print(f"After:  lr={config.lr}, batch_size={config.batch_size}")
print(f"Types:  lr={type(config.lr)}, batch_size={type(config.batch_size)}")
print()

# Test 2: Adding new keys with +new_key=value
print("Test 2: Adding new keys with +new_key=value")
print("-" * 70)
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(["+new_param=hello", "+another=42"])

config = Config.from_dict({"existing": "value"})
print(f"Before: keys={list(config.keys())}")

config = override_config(config, unknown)
print(f"After:  keys={list(config.keys())}")
print(f"Values: new_param={config.new_param}, another={config.another}")
print()

# Test 3: Space-separated format +key value
print("Test 3: Space-separated format +key value")
print("-" * 70)
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(["+optimizer", "adam", "+weight_decay", "0.01"])

config = Config.from_dict({"lr": 0.001})
print(f"Before: keys={list(config.keys())}")

config = override_config(config, unknown)
print(f"After:  keys={list(config.keys())}")
print(f"Values: optimizer={config.optimizer}, weight_decay={config.weight_decay}")
print()

# Test 4: Mixed formats (= and space-separated)
print("Test 4: Mixed formats (= and space-separated)")
print("-" * 70)
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(
    ["+lr=0.005", "+new_key=123", "+another", "value", "+epochs", "200"]
)

config = Config.from_dict({"lr": 0.001, "batch_size": 32})
print(f"Before: {config.to_dict()}")

config = override_config(config, unknown)
print(f"After:  {config.to_dict()}")
print()

# Test 5: -- and - prefixed args are IGNORED (left for ArgumentParser)
print("Test 5: -- and - prefixed args are ignored")
print("-" * 70)
parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("-b", "--batch-size", type=int, default=32)

args, unknown = parser.parse_known_args(
    [
        "--learning-rate=0.01",  # Parsed by ArgumentParser
        "-b",
        "64",  # Parsed by ArgumentParser
        "+lr=0.005",  # Processed by override_config
        "+epochs=200",  # Processed by override_config
    ]
)

config = Config.from_dict({"lr": 0.001, "epochs": 100})
print(
    f"ArgumentParser args: learning_rate={args.learning_rate}, batch_size={args.batch_size}"
)
print(f"Before override: lr={config.lr}, epochs={config.epochs}")

config = override_config(config, unknown)
print(f"After override:  lr={config.lr}, epochs={config.epochs}")
print(
    f"Note: --learning-rate and -b were handled by ArgumentParser, not override_config"
)
print()

# Test 5b: Warning for -- and - prefixed args
print("Test 5b: Warning is issued for -- and - prefixed args")
print("-" * 70)

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args(["--ignored", "-x", "value", "+lr=0.01"])

    config = Config.from_dict({"lr": 0.001})
    config = override_config(config, unknown)

    # Check if warning was issued
    if len(w) > 0:
        print(f"✓ Warning issued: {w[0].message}")
        print(
            f"  Detected args: {[arg for arg in unknown if arg.startswith('-') and not arg.startswith('+')]}"
        )
    else:
        print("✗ No warning issued")
print()

# Test 6: Nested key override
print("Test 6: Nested key override with dot notation")
print("-" * 70)
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(["+model.hidden_dim=256", "+model.dropout=0.1"])

config = Config.from_dict({"model": {"hidden_dim": 128, "num_layers": 3}})
print(f"Before: model={config.model}")

config = override_config(config, unknown)
print(f"After:  model={config['model']}")
print()

# Test 7: Type hints
print("Test 7: Type hints with +key:type=value")
print("-" * 70)
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(["+lr:float=0.01", "+batch_size:int=64"])

config = Config.from_dict({"lr": "0.001", "batch_size": "32"})
print(f"Before: lr={config.lr} (type={type(config.lr).__name__})")
print(
    f"        batch_size={config.batch_size} (type={type(config.batch_size).__name__})"
)

config = override_config(config, unknown)
print(f"After:  lr={config.lr} (type={type(config.lr).__name__})")
print(
    f"        batch_size={config.batch_size} (type={type(config.batch_size).__name__})"
)
print()

# Test 7b: Type hints with space-separated format
print("Test 7b: Type hints with +key:type value (space-separated)")
print("-" * 70)
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(
    ["+lr:float", "0.05", "+batch_size:int", "128", "+name:str", "experiment"]
)

config = Config.from_dict({"lr": "0.001", "batch_size": "32"})
print(f"Before: lr={config.lr} (type={type(config.lr).__name__})")
print(
    f"        batch_size={config.batch_size} (type={type(config.batch_size).__name__})"
)

config = override_config(config, unknown)
print(f"After:  lr={config.lr} (type={type(config.lr).__name__})")
print(
    f"        batch_size={config.batch_size} (type={type(config.batch_size).__name__})"
)
print(f"        name={config.name} (type={type(config.name).__name__})")
print()

# Test 8: Works with typed configs
print("Test 8: Override typed dataclass config")
print("-" * 70)


@dataclass
class TrainingConfig(Config):
    """Typed training config."""

    lr: float
    batch_size: int
    epochs: int = 100


parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(
    ["+lr=0.005", "+batch_size=128", "+new_field=dynamic"]
)

config = TrainingConfig.from_dict({"lr": 0.001, "batch_size": 32})
print(f"Before: lr={config.lr}, batch_size={config.batch_size}, epochs={config.epochs}")

config = override_config(config, unknown)
print(f"After:  lr={config.lr}, batch_size={config.batch_size}, epochs={config.epochs}")
print(f"Dynamic: new_field={config.new_field}")
print()

print("=" * 70)
print("Test 9: Sequence Support")
print("=" * 70)
print("Testing: +layers 64 128 256 +ids:int 1 2 3 +lr 0.01")

config = Config.from_dict({})
args = ["+layers", "64", "128", "256", "+ids:int", "1", "2", "3", "+lr", "0.01"]
config = override_config(config, args)

print(f"layers (auto-detect): {config.layers} (type: {type(config.layers).__name__})")
print(f"ids (typed int):      {config.ids} (type: {type(config.ids).__name__})")
print(f"lr (single value):    {config.lr} (type: {type(config.lr).__name__})")

# Verify the values
assert config.layers == [64, 128, 256], f"Expected [64, 128, 256], got {config.layers}"
assert config.ids == [1, 2, 3], f"Expected [1, 2, 3], got {config.ids}"
assert config.lr == 0.01, f"Expected 0.01, got {config.lr}"
print("✓ All sequence tests passed!")
print()

print("=" * 70)
print("Summary of Supported Formats:")
print("=" * 70)
print("✓ +key=value           → Set config key (add or override)")
print("✓ +key value           → Set with space-separated value")
print("✓ +key val1 val2 val3  → Set sequence/list (multiple values)")
print("✓ +nested.key=value    → Set nested config key")
print("✓ +key:type=value      → Set with type hint (= format)")
print("✓ +key:type value      → Set with type hint (space-separated)")
print("✓ +key:type v1 v2 v3   → Set typed sequence/list")
print()
print("Note: Arguments with -- or - prefix are left for ArgumentParser")
print("      Only + prefix is used for config overrides")
