# CLI Tools

ExpMate provides command-line tools for experiment management and analysis.

## Installation

CLI tools are included with the base installation:

```bash
pip install expmate
```

Verify installation:

```bash
expmate --help
```

## Commands

### Compare Runs

Compare multiple experiment runs and their metrics.

#### Basic Usage

```bash
# Compare all runs in a directory
expmate compare runs/exp_*

# Compare specific runs
expmate compare runs/exp1 runs/exp2 runs/exp3
```

#### Options

```bash
# Compare specific metrics
expmate compare runs/exp_* --metrics loss accuracy

# Show configuration differences
expmate compare runs/exp_* --show-config

# Show git information
expmate compare runs/exp_* --show-git

# Export to CSV
expmate compare runs/exp_* --output results.csv

# Pretty print table
expmate compare runs/exp_* --format table
```

#### Example Output

```
┌──────────────────────┬────────────┬─────────────┬──────────────┐
│ run_id               │ best_loss  │ best_acc    │ final_loss   │
├──────────────────────┼────────────┼─────────────┼──────────────┤
│ exp_20250123_100000  │ 0.234      │ 0.956       │ 0.245        │
│ exp_20250123_110000  │ 0.189      │ 0.967       │ 0.201        │
│ exp_20250123_120000  │ 0.156      │ 0.978       │ 0.167        │
└──────────────────────┴────────────┴─────────────┴──────────────┘
```

### Visualize Metrics

Plot training curves and compare experiments.

#### Basic Usage

```bash
# Plot metrics for a single run
expmate viz runs/exp1

# Plot specific metrics
expmate viz runs/exp1 --metrics loss accuracy

# Compare multiple runs
expmate viz runs/exp1 runs/exp2 runs/exp3 --metrics loss
```

#### Options

```bash
# Specify output file
expmate viz runs/exp1 --output metrics.png

# Choose plot style
expmate viz runs/exp1 --style seaborn
expmate viz runs/exp1 --style ggplot

# Filter by split
expmate viz runs/exp1 --split train
expmate viz runs/exp1 --split val

# Smooth curves
expmate viz runs/exp1 --smooth 0.9

# Set axis limits
expmate viz runs/exp1 --ylim 0 1
```

#### Example

```bash
# Create comparison plot
expmate viz \
    runs/exp_lr_0.001 \
    runs/exp_lr_0.01 \
    runs/exp_lr_0.1 \
    --metrics loss accuracy \
    --style seaborn \
    --output learning_rate_comparison.png
```

### Hyperparameter Sweeps

Run systematic hyperparameter searches with grid search.

#### Basic Grid Search

```bash
# Basic sweep - creates all combinations
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --sweep "training.lr=[0.001,0.01,0.1]" \
          "model.hidden_dim=[128,256,512]"
```

This creates 3 × 3 = 9 runs with all combinations of the parameters.

The `{config}` placeholder is replaced with the path to each generated config file.

#### With Distributed Training

Use `torchrun` in the command template:

```bash
expmate sweep "torchrun --nproc_per_node=4 train.py {config}" \
  --config config.yaml \
  --sweep "training.lr=[0.001,0.01,0.1]"
```

#### Sweep Options

```bash
# Custom sweep name
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --name my_lr_sweep \
  --sweep "training.lr=[0.001,0.01,0.1]"

# Custom runs directory
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --runs-dir sweeps/experiments \
  --sweep "training.lr=[0.001,0.01,0.1]"

# Dry run (preview commands without running)
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --sweep "training.lr=[0.001,0.01,0.1]" \
  --dry-run
```

#### Multiple Parameters

Sweep over multiple parameters:

```bash
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --sweep \
    "training.lr=[0.0001,0.001,0.01]" \
    "training.weight_decay=[0,0.0001,0.001]" \
    "model.dropout=[0.1,0.2,0.3]"
```

This creates 3 × 3 × 3 = 27 runs.

#### Parameter Value Types

The sweep command automatically detects types:

```bash
# Floats
--sweep "training.lr=[0.001,0.01,0.1]"

# Integers
--sweep "model.hidden_dim=[128,256,512]"

# Strings
--sweep "model.activation=[relu,gelu,silu]"

# Booleans
--sweep "training.use_amp=[true,false]"

# Mixed types
--sweep "training.batch_size=[16,32,64]" \
        "optimizer.name=[adam,sgd,adamw]"
```

#### Example Workflow

```bash
# 1. Run hyperparameter sweep
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --name lr_wd_sweep \
  --runs-dir sweeps/lr_wd \
  --sweep \
    "training.lr=[0.0001,0.001,0.01]" \
    "training.weight_decay=[0,0.0001,0.001]"

# 2. Compare results
expmate compare sweeps/lr_wd/exp_* --output sweep_results.csv

# 3. Visualize best runs
expmate viz \
    sweeps/lr_wd/exp_000 \
    sweeps/lr_wd/exp_004 \
    sweeps/lr_wd/exp_008 \
    --metrics val_loss val_accuracy
```

#### Output Structure

Each sweep creates a directory structure:

```
sweeps/lr_wd_sweep_20250123_143022/
├── sweep_info.json           # Sweep configuration and metadata
├── exp_000/
│   ├── config.yaml           # Generated config for this run
│   ├── run.yaml              # Full config after execution
│   ├── exp.log               # Logs
│   └── metrics.csv           # Metrics
├── exp_001/
│   └── ...
└── exp_008/
    └── ...
```

## Python API

You can also use CLI functionality from Python:

### Generate Sweep Configurations

```python
from expmate.cli.sweep import generate_sweep_configs

# Generate all configurations for a grid search
configs = generate_sweep_configs(
    base_config={'model': {'depth': 18}, 'training': {'epochs': 100}},
    sweep_params={
        'training.lr': [0.001, 0.01, 0.1],
        'model.hidden_dim': [128, 256, 512]
    }
)

# Returns list of 9 configurations with all combinations
for i, config in enumerate(configs):
    print(f"Config {i}: lr={config['training']['lr']}, "
          f"hidden_dim={config['model']['hidden_dim']}")
```

### Run Sweep

```python
from expmate.cli.sweep import run_sweep

# Run the sweep
run_sweep(
    command_template="python train.py {config}",
    sweep_params={
        'training.lr': [0.001, 0.01, 0.1],
        'model.hidden_dim': [128, 256, 512]
    },
    base_config_file='config.yaml',
    sweep_name='my_sweep',
    runs_dir='sweeps',
    dry_run=False  # Set True to preview without running
)
```

## Advanced Usage

### Custom Analysis Scripts

Combine CLI tools with custom analysis:

```python
import pandas as pd
from expmate.cli.compare import load_run_info

# Load all runs
runs = [load_run_info(f'runs/{d}') for d in os.listdir('runs')]

# Filter runs by config
high_lr_runs = [r for r in runs if r['config']['training']['lr'] > 0.01]

# Analyze
df = pd.DataFrame([{
    'run_id': r['run_id'],
    'lr': r['config']['training']['lr'],
    'best_loss': r['best_metrics']['loss']['value']
} for r in high_lr_runs])

print(df.sort_values('best_loss'))
```

### Integration with Other Tools

```bash
# Export to pandas for analysis
expmate compare runs/exp_* --output results.csv
python analyze.py results.csv

# Generate plots for reports
expmate viz runs/exp_* --output figure1.png
expmate viz runs/exp_* --metrics accuracy --output figure2.png

# Create summary report
expmate compare runs/exp_* > results.txt
cat results.txt
```

## Tips

1. **Use glob patterns** to match multiple runs: `runs/exp_*`
2. **Export to CSV** for custom analysis: `--output results.csv`
3. **Run sweeps in parallel** to save time: `--parallel 4`
4. **Use dry run** to verify commands: `--dry-run`
5. **Combine tools** for comprehensive analysis

## See Also

- [Configuration Management](configuration.md)
- [Experiment Logging](logging.md)
- [Examples: Hyperparameter Sweeps](../examples/sweeps.md)
