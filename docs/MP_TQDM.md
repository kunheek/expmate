# Progress Bars in Distributed Training (mp_tqdm)

When running distributed training across multiple GPUs, showing progress bars from all processes creates cluttered, overlapping output. ExpMate provides `mp_tqdm`, a wrapper around tqdm that only displays progress bars on `local_rank==0` of each node.

## Overview

`mp_tqdm` automatically:
- Shows progress bars **only on local_rank 0** (master process per node)
- Hides progress bars on all other processes
- Falls back gracefully if tqdm is not installed
- Supports all standard tqdm features and parameters
- Works seamlessly with both single and multi-GPU setups

## Installation

```bash
# Install tqdm (optional but recommended)
pip install tqdm

# Or install with expmate extras
pip install expmate[tracking]  # Includes tqdm
```

## Basic Usage

### Simple Progress Bar

```python
from expmate.torch import mp_tqdm

# In distributed training - only shows on local_rank 0
for batch in mp_tqdm(dataloader, desc="Training"):
    loss = train_step(batch)
```

### Nested Progress Bars

Track both epochs and batches:

```python
from expmate.torch import mp_tqdm

for epoch in mp_tqdm(range(epochs), desc="Epochs"):
    for batch in mp_tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
        loss = train_step(batch)
```

### Context Manager for Manual Updates

```python
from expmate.torch import mp_tqdm

with mp_tqdm(total=total_steps, desc="Processing") as pbar:
    for i in range(total_steps):
        result = process_item(i)
        pbar.update(1)
        pbar.set_postfix(loss=result)
```

## Complete Example

```python
#!/usr/bin/env python3
from expmate.torch import mp_tqdm, setup_ddp, cleanup_ddp
import torch
from torch.utils.data import DataLoader

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    # Progress bar only shows on local_rank 0
    for data, target in mp_tqdm(dataloader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Setup DDP
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = MyModel().to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None
        )
    
    # Training loop with progress tracking
    for epoch in mp_tqdm(range(num_epochs), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # This also prints only on rank 0
        if rank == 0:
            print(f"Epoch {epoch}: loss={train_loss:.4f}")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
```

**Output (only on local_rank 0):**
```
Epochs:   0%|                                      | 0/10 [00:00<?, ?it/s]
Training: 100%|████████████████████| 782/782 [00:15<00:00, 51.2it/s]
Epoch 0: loss=0.4523
Epochs:  10%|████                                  | 1/10 [00:15<02:15, 15.0s/it]
```

**Output on other processes:**
```
Epoch 0: loss=0.4523
Epoch 1: loss=0.3821
...
```

## Advanced Features

### Manual Control

Override automatic behavior:

```python
from expmate.torch import mp_tqdm

# Force enable on all ranks
for batch in mp_tqdm(dataloader, disable=False):
    process(batch)

# Force disable even on rank 0
for batch in mp_tqdm(dataloader, disable=True):
    process(batch)
```

### Custom Formatting

All tqdm parameters are supported:

```python
from expmate.torch import mp_tqdm

for batch in mp_tqdm(
    dataloader,
    desc="Processing",
    unit="batches",
    ncols=100,
    leave=True,
    colour="green",
):
    process(batch)
```

### Dynamic Postfix

Update progress bar with current metrics:

```python
from expmate.torch import mp_tqdm

with mp_tqdm(dataloader, desc="Training") as pbar:
    for batch in pbar:
        loss = train_step(batch)
        pbar.set_postfix(loss=f"{loss:.4f}", lr=optimizer.param_groups[0]['lr'])
```

## Behavior

### Single-GPU or CPU
- `local_rank` is always 0
- Progress bars are shown

### Multi-GPU (same node)
- Only `local_rank==0` shows progress bars
- Other ranks run silently

### Multi-Node Multi-GPU
- Each node shows progress on its `local_rank==0`
- Useful for monitoring per-node progress

## Fallback Behavior

If tqdm is not installed:
```python
from expmate.torch import mp_tqdm

# Returns the iterable directly, no progress bar
for item in mp_tqdm(data):
    process(item)  # Works fine, just no progress bar
```

## Comparison with Regular tqdm

| Feature | Regular tqdm | mp_tqdm |
|---------|-------------|---------|
| Single process | ✅ Shows bar | ✅ Shows bar |
| Multi-GPU (all ranks) | ⚠️ Cluttered output | ✅ Only local_rank 0 |
| Auto-detection | ❌ No | ✅ Yes |
| Fallback if not installed | ❌ ImportError | ✅ Graceful |
| All tqdm features | ✅ Yes | ✅ Yes |

## Best Practices

### 1. Use for Training Loops

```python
# ✓ Good: Clear, single progress bar per node
for epoch in mp_tqdm(range(epochs)):
    for batch in mp_tqdm(train_loader, leave=False):
        train(batch)
```

### 2. Combine with Rank-Aware Logging

```python
from expmate.torch import mp_tqdm, mp_print

for epoch in mp_tqdm(range(epochs)):
    loss = train_epoch()
    mp_print(f"Epoch {epoch}: loss={loss:.4f}")  # Also only rank 0
```

### 3. Use `leave=False` for Inner Loops

```python
# Outer loop: keep progress bar
for epoch in mp_tqdm(range(epochs), desc="Epochs"):
    # Inner loop: remove after completion
    for batch in mp_tqdm(train_loader, desc="Batches", leave=False):
        train(batch)
```

### 4. Disable for Debugging

```python
# During debugging, show on all ranks
DEBUG = True

for batch in mp_tqdm(dataloader, disable=DEBUG or None):
    train(batch)
```

## Troubleshooting

### Progress bar shows on all ranks

**Issue**: `LOCAL_RANK` environment variable not set  
**Solution**: Use `torchrun` or set `LOCAL_RANK` manually

```bash
# Correct
torchrun --nproc_per_node=4 train.py

# Incorrect (missing env vars)
python train.py  # In multi-GPU setup
```

### Progress bar not showing at all

**Issue**: tqdm not installed  
**Solution**: Install tqdm

```bash
pip install tqdm
```

### Progress bar disappears too quickly

**Issue**: `leave=False` on outer loop  
**Solution**: Use `leave=True` (default) for outer loops

```python
# ✓ Correct
for epoch in mp_tqdm(range(epochs)):  # leave=True by default
    for batch in mp_tqdm(dataloader, leave=False):  # Clean up after
        train(batch)
```

## See Also

- [Distributed Training Guide](guide/distributed.md)
- [Examples: DDP Training](examples/ddp.md)
