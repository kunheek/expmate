# CLI Tools

ExpMate provides command-line tools for experiment management and analysis.

## Installation

CLI tools are available after installing ExpMate:

```bash
pip install expmate[viz]  # For visualization features
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

Run systematic hyperparameter searches.

#### Grid Search

```bash
expmate sweep config.yaml \
    --param training.lr 0.001 0.01 0.1 \
    --param model.hidden_dim 128 256 512 \
    --script train.py \
    --runs-dir sweeps/grid
```

This creates 3 × 3 = 9 runs with all combinations.

#### Random Search

```bash
expmate sweep config.yaml \
    --param training.lr 0.001:0.1:log \
    --param training.dropout 0.1:0.5 \
    --param model.hidden_dim 128:512:int \
    --num-samples 20 \
    --mode random \
    --script train.py
```

Parameter syntax:
- `min:max`: Uniform distribution
- `min:max:log`: Log-uniform distribution
- `min:max:int`: Integer uniform distribution

#### Options

```bash
# Parallel execution
expmate sweep config.yaml \
    --param training.lr 0.001 0.01 0.1 \
    --script train.py \
    --parallel 4  # Run 4 experiments in parallel

# GPU assignment
expmate sweep config.yaml \
    --param training.lr 0.001 0.01 0.1 \
    --script train.py \
    --gpus 0,1,2,3  # Distribute across GPUs

# Resume interrupted sweep
expmate sweep config.yaml \
    --param training.lr 0.001 0.01 0.1 \
    --script train.py \
    --resume sweeps/grid

# Dry run (print commands without executing)
expmate sweep config.yaml \
    --param training.lr 0.001 0.01 0.1 \
    --script train.py \
    --dry-run
```

#### Example Workflow

```bash
# 1. Run hyperparameter sweep
expmate sweep config.yaml \
    --param training.lr 0.0001 0.001 0.01 \
    --param training.weight_decay 0 0.0001 0.001 \
    --script train.py \
    --runs-dir sweeps/lr_wd \
    --parallel 3

# 2. Compare results
expmate compare sweeps/lr_wd/exp_* --output sweep_results.csv

# 3. Visualize best runs
expmate viz \
    sweeps/lr_wd/exp_001 \
    sweeps/lr_wd/exp_004 \
    sweeps/lr_wd/exp_007 \
    --metrics val_loss val_accuracy
```

## Python API

You can also use CLI functionality from Python:

### Compare Runs

```python
from expmate.cli.compare import compare_runs

df = compare_runs(
    run_dirs=['runs/exp1', 'runs/exp2', 'runs/exp3'],
    metrics=['loss', 'accuracy'],
    show_config=True
)
print(df)
```

### Visualize Metrics

```python
from expmate.cli.visualize import plot_metrics

plot_metrics(
    run_dirs=['runs/exp1', 'runs/exp2'],
    metrics=['loss', 'accuracy'],
    output_file='comparison.png',
    style='seaborn'
)
```

### Generate Sweep

```python
from expmate.cli.sweep import generate_sweep_configs, run_sweep

# Generate configurations
configs = generate_sweep_configs(
    base_config_file='config.yaml',
    param_grid={
        'training.lr': [0.001, 0.01, 0.1],
        'model.hidden_dim': [128, 256, 512]
    }
)

# Run sweep
run_sweep(
    script='train.py',
    configs=configs,
    runs_dir='sweeps/grid',
    parallel=4
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
