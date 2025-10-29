#!/usr/bin/env python3
"""Test backward compatibility of profile() method."""

import time
from expmate import ExperimentLogger

logger = ExperimentLogger("runs/test")

# Test profile() backward compatibility
with logger.profile("test") as r:
    time.sleep(0.01)

print(f"✓ Backward compat works: {r['elapsed']:.4f}s")
print("profile() method still works (deprecated, use timer() instead)")
