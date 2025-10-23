# API Reference: Parser

## parse_config()

Parse configuration from command-line arguments.

```python
from expmate import parse_config

# Automatically parses: python script.py config.yaml key=value
config = parse_config()
```

## ConfigArgumentParser

Argument parser with config support.

```python
from expmate import ConfigArgumentParser

parser = ConfigArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
config = parser.parse_config()
```

## parse_value()

Parse string value to appropriate type.

```python
from expmate.parser import parse_value

parse_value('42')      # 42 (int)
parse_value('3.14')    # 3.14 (float)
parse_value('true')    # True (bool)
parse_value('[1,2,3]') # [1, 2, 3] (list)
```

See the [source code](https://github.com/kunheek/expmate/blob/main/src/expmate/parser.py) for full API details.
