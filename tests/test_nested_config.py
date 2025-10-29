#!/usr/bin/env python3
"""Test nested dataclass configs with override_config."""

from dataclasses import dataclass, field
from expmate import Config, override_config


@dataclass
class ModelConfig(Config):
    """Model configuration."""

    dim: int = 128
    layers: int = 4


@dataclass
class OptimizerConfig(Config):
    """Optimizer configuration."""

    name: str = "adam"
    lr: float = 0.001
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class MainConfig(Config):
    """Main configuration with nested configs."""

    seed: int = 42
    epochs: int = 100
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


print("=" * 70)
print("Test 1: Nested Config Defaults")
print("=" * 70)
config = MainConfig.from_dict({})
print(f"seed: {config.seed}")
print(f"model.dim: {config.model.dim}")
print(f"model.layers: {config.model.layers}")
print(f"optimizer.name: {config.optimizer.name}")
print(f"optimizer.lr: {config.optimizer.lr}")
print(f"optimizer.betas: {config.optimizer.betas}")
assert config.seed == 42
assert config.model.dim == 128
assert config.optimizer.lr == 0.001
print("✓ Defaults work correctly")
print()

print("=" * 70)
print("Test 2: Override Nested Int Fields")
print("=" * 70)
config = MainConfig.from_dict({})
args = ["+model.dim=256", "+model.layers=12"]
config = override_config(config, args)
print(f"model.dim: {config.model.dim}")
print(f"model.layers: {config.model.layers}")
assert config.model.dim == 256
assert config.model.layers == 12
print("✓ Nested int overrides work")
print()

print("=" * 70)
print("Test 3: Override Nested String and Float Fields")
print("=" * 70)
config = MainConfig.from_dict({})
args = ["+optimizer.name=sgd", "+optimizer.lr=0.01"]
config = override_config(config, args)
print(f"optimizer.name: {config.optimizer.name}")
print(f"optimizer.lr: {config.optimizer.lr}")
assert config.optimizer.name == "sgd"
assert config.optimizer.lr == 0.01
print("✓ Nested string/float overrides work")
print()

print("=" * 70)
print("Test 4: Override Nested List Fields")
print("=" * 70)
config = MainConfig.from_dict({})
args = ["+optimizer.betas:float", "0.95", "0.999"]
config = override_config(config, args)
print(f"optimizer.betas: {config.optimizer.betas}")
assert config.optimizer.betas == [0.95, 0.999]
print("✓ Nested list overrides work")
print()

print("=" * 70)
print("Test 5: Mix Top-Level and Nested Overrides")
print("=" * 70)
config = MainConfig.from_dict({})
args = [
    "+seed=999",
    "+epochs=200",
    "+model.dim=512",
    "+optimizer.lr=0.0001",
]
config = override_config(config, args)
print(f"seed: {config.seed}")
print(f"epochs: {config.epochs}")
print(f"model.dim: {config.model.dim}")
print(f"optimizer.lr: {config.optimizer.lr}")
assert config.seed == 999
assert config.epochs == 200
assert config.model.dim == 512
assert config.optimizer.lr == 0.0001
print("✓ Mixed overrides work")
print()

print("=" * 70)
print("Test 6: Verify Types")
print("=" * 70)
config = MainConfig.from_dict({})
args = [
    "+model.dim=256",
    "+optimizer.lr=0.01",
    "+optimizer.betas:float",
    "0.95",
    "0.999",
]
config = override_config(config, args)
assert isinstance(config.model, ModelConfig)
assert isinstance(config.optimizer, OptimizerConfig)
assert isinstance(config.model.dim, int)
assert isinstance(config.optimizer.lr, float)
assert isinstance(config.optimizer.betas, list)
assert all(isinstance(x, float) for x in config.optimizer.betas)
print(f"model type: {type(config.model).__name__}")
print(f"optimizer type: {type(config.optimizer).__name__}")
print(f"model.dim type: {type(config.model.dim).__name__}")
print(f"optimizer.lr type: {type(config.optimizer.lr).__name__}")
print(f"optimizer.betas type: {type(config.optimizer.betas).__name__}")
print("✓ All types are correct")
print()

print("=" * 70)
print("All nested config tests passed! ✅")
print("=" * 70)
