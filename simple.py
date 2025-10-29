from dataclasses import dataclass
from expmate import Config


@dataclass
class MyConfig(Config):
    lr: float = 0.001
    batch_size: int = 32


config = MyConfig.from_dict({})
print(config)

config = config.override(["+lr=0.01", "batch_size=64"])
print(config)
