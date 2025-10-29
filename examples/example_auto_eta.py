#!/usr/bin/env python3
"""Example demonstrating automatic ETA estimation in log_metric."""

import time
import tempfile
from expmate import ExperimentLogger


def main():
    """Demo automatic time estimation when logging metrics."""
    print("\n" + "=" * 70)
    print("Automatic ETA Estimation Demo")
    print("=" * 70 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir)

        total_epochs = 100
        print(f"Training for {total_epochs} epochs...\n")

        for epoch in range(1, total_epochs + 1):
            # Simulate training
            time.sleep(0.05)  # Simulate work
            train_loss = 1.0 / (epoch + 1)  # Fake decreasing loss

            # Log metric WITH automatic ETA
            # Just pass total_steps and it will automatically estimate remaining time!
            logger.log_metric(
                step=epoch,
                split="train",
                name="loss",
                value=train_loss,
                total_steps=total_epochs,  # This enables automatic ETA
            )

            # Validation every 10 epochs
            if epoch % 10 == 0:
                time.sleep(0.03)  # Simulate validation
                val_loss = 0.9 / (epoch + 1)

                logger.log_metric(
                    step=epoch,
                    split="val",
                    name="loss",
                    value=val_loss,
                    total_steps=total_epochs,
                )

        print("\n" + "=" * 70)
        print("✅ Training complete!")
        print("=" * 70)
        print("\nKey Features:")
        print("  • Automatic ETA calculation based on progress")
        print("  • Progress percentage tracking")
        print("  • Elapsed time formatting (human-readable)")
        print("  • Per-metric ETA tracking")
        print("\nUsage:")
        print("  logger.log_metric(step, split, name, value, total_steps=N)")
        print("=" * 70 + "\n")

        logger.close()


if __name__ == "__main__":
    main()
