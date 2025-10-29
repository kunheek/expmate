# API Reference: Config

## Config Class

A dictionary-like configuration object with attribute access.

```python
from expmate import Config

config = Config({'model': {'hidden_dim': 256}})
print(config.model.hidden_dim)  # 256
```

## load_config()

Load configuration from YAML files or dictionaries.

```python
from expmate.config import load_config

config = load_config('config.yaml')
config = load_config('config.yaml', overrides=['training.lr=0.01'])
```

**Parameters:**
- `config_input` (str|list|dict): Config file path(s) or dict
- `overrides` (list): List of key=value overrides

**Returns:** Dictionary with configuration

See the [source code](https://github.com/kunheek/expmate/blob/main/src/expmate/config.py) for full API details.
