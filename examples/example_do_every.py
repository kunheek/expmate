"""Example demonstrating the do_every context manager for rate-limited execution.

The do_every context manager allows you to execute code at specified intervals:
- every=N: Execute every N iterations
- seconds=N: Execute every N seconds
- key: Optional unique identifier for multiple rate limiters

This is useful for:
- Checkpointing at intervals
- Validation runs
- Logging expensive metrics
- GPU memory monitoring
- Any operation that should be rate-limited
"""

import time
from pathlib import Path

import expmate


def main():
    # Setup logger
    logger = expmate.ExperimentLogger(run_dir=Path("runs/do_every_demo"))

    print("\n=== Example 1: Iteration-based rate limiting ===")
    print("Checkpoint every 100 iterations")

    for step in range(500):
        # Simulate training
        loss = 1.0 / (step + 1)

        # Save checkpoint every 100 iterations
        with logger.do_every(every=100) as should_execute:
            if should_execute:
                logger.info(f"💾 Saving checkpoint at step {step}")
                # save_checkpoint(model, f"checkpoint_{step}.pt")

        # Log every 50 iterations
        with logger.do_every(every=50, key="metrics") as should_execute:
            if should_execute:
                logger.info(f"Step {step}: loss={loss:.4f}")

    print("\n=== Example 2: Time-based rate limiting ===")
    print("Validate every 2 seconds")

    start_time = time.time()
    step = 0

    while time.time() - start_time < 10:
        # Simulate training
        step += 1
        time.sleep(0.1)  # Simulate work

        # Validate every 2 seconds
        with logger.do_every(seconds=2.0, key="validation") as should_execute:
            if should_execute:
                elapsed = time.time() - start_time
                logger.info(f"🔍 Running validation at {elapsed:.1f}s (step {step})")
                # val_loss = validate(model, val_loader)

        # Monitor GPU every 1 second
        with logger.do_every(seconds=1.0, key="gpu_monitor") as should_execute:
            if should_execute:
                elapsed = time.time() - start_time
                logger.info(f"📊 GPU stats at {elapsed:.1f}s")
                # log_gpu_stats()

    print("\n=== Example 3: Multiple rate limiters ===")
    print("Different operations at different rates")

    for epoch in range(3):
        for step in range(100):
            # Quick logging every 10 steps
            with logger.do_every(every=10, key="train_log") as should_execute:
                if should_execute:
                    logger.info(f"Epoch {epoch}, Step {step}/{100}")

            # Validation every 50 steps
            with logger.do_every(every=50, key="validation") as should_execute:
                if should_execute:
                    logger.info(f"  🔍 Validation at epoch {epoch}, step {step}")

        # End of epoch checkpoint
        logger.info(f"💾 End of epoch {epoch} checkpoint")

    print("\n=== Comparison: log_every vs do_every ===")

    # log_every: For rate-limiting logging specifically
    for step in range(20):
        with logger.log_every(every=5):
            logger.info(f"log_every: Step {step}")  # Console output suppressed unless step % 5 == 0

    print()

    # do_every: For rate-limiting any code execution
    for step in range(20):
        with logger.do_every(every=5) as should_execute:
            if should_execute:
                logger.info(f"do_every: Step {step}")  # Code only runs when should_execute is True

    logger.info("\n✅ do_every examples completed!")
    logger.close()


if __name__ == "__main__":
    main()
