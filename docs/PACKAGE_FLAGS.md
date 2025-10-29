# ExpMate Package-Level Configuration Flags

All flags can be configured either programmatically or via environment variables.

## Available Flags

### 1. `expmate.debug` (default: `False`)
Enable debug mode and debug-level logging.

**Use cases:**
- Development and debugging
- Verbose logging for troubleshooting

**Configuration:**
```python
import expmate
expmate.debug = True  # Enable debug mode
```
```bash
export EM_DEBUG=1  # Enable via environment
```

---

### 2. `expmate.timer` (default: `True`)
Control whether `logger.timer()` performs timing measurements.

**Use cases:**
- Disable in production to eliminate profiling overhead
- Enable during development to track performance

**Configuration:**
```python
import expmate
expmate.timer = False  # Disable profiling
```
```bash
export EM_TIMER=0  # Disable via environment
```

---

### 3. `expmate.log_level` (default: `"INFO"`)
Default log level for all `ExperimentLogger` instances.

**Options:** `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`

**Use cases:**
- Set to `"WARNING"` in production for quieter logs
- Set to `"DEBUG"` during development

**Configuration:**
```python
import expmate
expmate.log_level = "WARNING"  # Only warnings and errors
```
```bash
export EM_LOG_LEVEL=WARNING  # Set via environment
```

---

### 4. `expmate.verbose` (default: `True`)
Control console output for loggers.

**Use cases:**
- Disable for cleaner output when running many experiments
- Enable for interactive debugging

**Configuration:**
```python
import expmate
expmate.verbose = False  # Disable console output
```
```bash
export EM_VERBOSE=0  # Disable via environment
```

---

### 5. `expmate.track_metrics` (default: `True`)
Control whether to log metrics to CSV files.

**Use cases:**
- Disable for faster iteration when metrics aren't needed
- Reduce I/O during quick experiments

**Configuration:**
```python
import expmate
expmate.track_metrics = False  # Skip metrics CSV
```
```bash
export EM_TRACK_METRICS=0  # Disable via environment
```

---

### 6. `expmate.track_git` (default: `True`)
Control whether to collect git repository information.

**Use cases:**
- Disable in CI/CD where git operations are slow
- Skip when not using version control

**Configuration:**
```python
import expmate
expmate.track_git = False  # Skip git info
```
```bash
export EM_TRACK_GIT=0  # Disable via environment
```

---

### 7. `expmate.save_checkpoints` (default: `True`)
Control whether to save model checkpoints (PyTorch).

**Use cases:**
- Disable for quick experiments or testing
- Skip checkpoint I/O during debugging

**Configuration:**
```python
import expmate
expmate.save_checkpoints = False  # Skip checkpoint saving
```
```bash
export EM_SAVE_CHECKPOINTS=0  # Disable via environment
```

---

### 8. `expmate.force_single_process` (default: `False`)
Force single-process mode even in distributed environments.

**Use cases:**
- Debug distributed code without multiple processes
- Run DDP code in single-process mode

**Configuration:**
```python
import expmate
expmate.force_single_process = True  # Force single process
```
```bash
export EM_FORCE_SINGLE=1  # Enable via environment
```

---

## Usage Patterns

### Production Configuration
Minimize overhead and noise:
```python
import expmate

expmate.timer = False            # No timing overhead
expmate.log_level = "WARNING"    # Only warnings/errors
expmate.verbose = False          # No console output
```

Or via environment:
```bash
export EM_TIMER=0
export EM_LOG_LEVEL=WARNING
export EM_VERBOSE=0
```

### Development Configuration
Maximum visibility and debugging:
```python
import expmate

expmate.debug = True             # Debug logging
expmate.log_level = "DEBUG"      # All log messages
expmate.timer = True             # Track performance
```

Or via environment:
```bash
export EM_DEBUG=1
export EM_LOG_LEVEL=DEBUG
export EM_TIMER=1
```

### Quick Experimentation
Skip I/O for faster iteration:
```python
import expmate

expmate.track_metrics = False    # No CSV writing
expmate.save_checkpoints = False # No checkpoint I/O
expmate.track_git = False        # No git operations
```

Or via environment:
```bash
export EM_TRACK_METRICS=0
export EM_SAVE_CHECKPOINTS=0
export EM_TRACK_GIT=0
```

### Runtime Toggle
Enable/disable features dynamically:
```python
import expmate

# Start with profiling disabled
expmate.timer = False

# ... fast training loop ...

# Enable timing for critical section
expmate.timer = True
with logger.timer('critical_section'):
    important_code()

# Disable again
expmate.timer = False
```

---

## Environment Variable Reference

| Flag | Environment Variable | Default |
|------|---------------------|---------|
| `debug` | `EM_DEBUG` | `0` (False) |
| `timer` | `EM_TIMER` | `1` (True) |
| `log_level` | `EM_LOG_LEVEL` | `INFO` |
| `verbose` | `EM_VERBOSE` | `1` (True) |
| `track_metrics` | `EM_TRACK_METRICS` | `1` (True) |
| `track_git` | `EM_TRACK_GIT` | `1` (True) |
| `save_checkpoints` | `EM_SAVE_CHECKPOINTS` | `1` (True) |
| `force_single_process` | `EM_FORCE_SINGLE` | `0` (False) |

---

## Testing

Run comprehensive tests:
```bash
python test_package_flags.py
```

Check current flag values:
```python
import expmate

print(f"debug: {expmate.debug}")
print(f"timer: {expmate.timer}")
print(f"log_level: {expmate.log_level}")
print(f"verbose: {expmate.verbose}")
print(f"track_metrics: {expmate.track_metrics}")
print(f"track_git: {expmate.track_git}")
print(f"save_checkpoints: {expmate.save_checkpoints}")
print(f"force_single_process: {expmate.force_single_process}")
```
