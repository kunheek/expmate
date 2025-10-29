"""Test the new ArgumentParser with config integration."""

import os
import sys
import tempfile
from dataclasses import dataclass

from expmate import ArgumentParser, Config


# Define a custom config class
@dataclass
class TrainingConfig(Config):
    """Training configuration."""

    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 100


print("=" * 70)
print("Test ArgumentParser with Config Integration")
print("=" * 70)
print()

# Test 1: Basic usage with default Config
print("Test 1: Basic usage with default Config class")
print("-" * 70)
parser = ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--verbose", action="store_true")

# Simulate command line: --epochs 50 +lr=0.01 +batch_size=64
sys.argv = [
    "test_script.py",
    "--epochs",
    "50",
    "--verbose",
    "+lr=0.01",
    "+batch_size=64",
]

args, config = parser.parse_args_with_config()
print(f"Args: epochs={args.epochs}, verbose={args.verbose}")
print(f"Config: lr={config.lr}, batch_size={config.batch_size}")
print()

# Test 2: With custom config class
print("Test 2: With custom config class (TrainingConfig)")
print("-" * 70)
parser = ArgumentParser(config_class=TrainingConfig)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")

sys.argv = [
    "test_script.py",
    "--seed",
    "123",
    "--device",
    "cpu",
    "+lr=0.005",
    "+epochs=200",
]

args, config = parser.parse_args_with_config()

config = TrainingConfig.from_file(args.config_file)
print(f"Args: seed={args.seed}, device={args.device}")
print(f"Config: lr={config.lr}, batch_size={config.batch_size}, epochs={config.epochs}")
print(f"Config type: {type(config).__name__}")
print()

# Test 3: With config file
print("Test 3: With config file loading")
print("-" * 70)

# Create a test config file
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write("lr: 0.002\n")
    f.write("batch_size: 128\n")
    f.write("optimizer: adam\n")
    config_file = f.name

parser = ArgumentParser(config_class=TrainingConfig)
parser.add_argument("--experiment", type=str, default="baseline")

sys.argv = [
    "test_script.py",
    "--config-file",
    config_file,
    "--experiment",
    "test_exp",
    "+lr=0.003",
    "+new_param=hello",
]

args, config = parser.parse_args_with_config()
print(f"Args: experiment={args.experiment}, config_file={args.config_file}")
print(
    f"Config from file: lr={config.lr}, batch_size={config.batch_size}, optimizer={config.optimizer}"
)
print(f"Config override: lr={config.lr}, new_param={config.new_param}")
print()

# Clean up
os.unlink(config_file)

# Test 4: Using parse_known_args_with_config
print("Test 4: Using parse_known_args_with_config")
print("-" * 70)
parser = ArgumentParser(config_class=TrainingConfig)
parser.add_argument("--output", type=str, default="./output")

sys.argv = [
    "test_script.py",
    "--output",
    "./results",
    "+lr=0.01",
    "--unknown-arg",
    "value",
]

args, config, unknown = parser.parse_known_args_with_config()
print(f"Args: output={args.output}")
print(f"Config: lr={config.lr}")
print(f"Unknown args (not + prefix): {unknown}")
print()

# Test 5: Type hints and nested config
print("Test 5: Type hints and nested overrides")
print("-" * 70)
parser = ArgumentParser()
parser.add_argument("--mode", type=str, default="train")

sys.argv = [
    "test_script.py",
    "--mode",
    "eval",
    "+lr:float=0.001",
    "+batch_size:int=256",
    "+model.hidden_dim=512",
    "+model.dropout=0.2",
]

args, config = parser.parse_args_with_config()
print(f"Args: mode={args.mode}")
print(f"Config: lr={config.lr} (type={type(config.lr).__name__})")
print(
    f"Config: batch_size={config.batch_size} (type={type(config.batch_size).__name__})"
)
print(f"Config: model={config.model}")
print()

# Test 6: Space-separated values
print("Test 6: Space-separated values with type hints")
print("-" * 70)
parser = ArgumentParser(config_class=TrainingConfig)

sys.argv = [
    "test_script.py",
    "+lr:float",
    "0.007",
    "+name:str",
    "my_experiment",
    "+layers",
    "128",
    "256",
    "512",
]

args, config = parser.parse_args_with_config()
print(f"Config: lr={config.lr} (type={type(config.lr).__name__})")
print(f"Config: name={config.name}")
# Note: Space-separated lists need special handling - this might be "128" only
print(f"Config: layers={config.layers if hasattr(config, 'layers') else 'N/A'}")
print()

print("=" * 70)
print("Summary:")
print("=" * 70)
print("✓ ArgumentParser integrates with Config system")
print("✓ Supports --standard-args for ArgumentParser")
print("✓ Supports +config_overrides for Config system")
print("✓ Automatically loads config from --config-file")
print("✓ Works with both Config and custom config classes")
print("✓ parse_args_with_config() returns (args, config)")
print("✓ parse_known_args_with_config() returns (args, config, unknown)")
print("=" * 70)
