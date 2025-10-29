#!/usr/bin/env python3
"""
Simple Linear Regression Example
==================================

This example demonstrates the core features of ExpMate:
- Typed configuration with dataclasses and LSP support
- Experiment logging with colorful console output
- Metrics tracking and best model selection
- Hierarchical stage tracking
- Simple timing with timer()

Run with:
    python 01_linear_regression.py examples/conf/linear_regression.yaml
    python 01_linear_regression.py examples/conf/linear_regression.yaml +model.lr=0.01 +training.epochs=200
"""

import argparse
from dataclasses import dataclass, field

import numpy as np

from expmate import Config, ExperimentLogger, set_seed


# Configuration dataclasses
@dataclass
class DataConfig(Config):
    """Data configuration"""

    n_samples: int = 1000
    noise: float = 0.1
    train_split: float = 0.8


@dataclass
class ModelConfig(Config):
    """Model configuration"""

    lr: float = 0.01


@dataclass
class TrainingConfig(Config):
    """Training configuration"""

    epochs: int = 100


@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""

    run_id: str = "linear_reg_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Generate synthetic data
def generate_data(n_samples=1000, noise=0.1, seed=42):
    """Generate linear regression data: y = 2x + 1 + noise"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 1)
    y = 2 * X + 1 + noise * rng.randn(n_samples, 1)
    return X, y


# Simple linear regression model
class LinearRegression:
    """Simple linear regression with gradient descent"""

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, epochs=100):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        # Training loop
        losses = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)

            # Backward pass (gradients)
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            yield epoch, loss

    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias

    def score(self, X, y):
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Linear regression with ExpMate")
    parser.add_argument(
        "--config",
        default="examples/conf/linear_regression.yaml",
        help="Config file path",
    )
    args, overrides = parser.parse_known_args()

    # Load config with optional overrides
    config = ExperimentConfig.from_file(args.config, overrides=overrides)

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Create experiment logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"🚀 Starting experiment: {config.run_id}")
    logger.info(f"Configuration: {config.to_dict()}")

    # Generate data
    with logger.stage("data_preparation"):
        with logger.timer("generate_data"):
            X, y = generate_data(
                n_samples=config.data.n_samples,
                noise=config.data.noise,
                seed=config.seed,
            )

        # Split into train/validation
        split_idx = int(config.data.train_split * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")

    # Create model
    model = LinearRegression(learning_rate=config.model.lr)
    logger.info(f"Model: LinearRegression(lr={config.model.lr})")

    # Training loop
    logger.info(f"Training for {config.training.epochs} epochs...")

    with logger.stage("training"):
        for epoch, loss in model.fit(X_train, y_train, epochs=config.training.epochs):
            # Log training metrics with automatic ETA estimation
            logger.log_metric(
                step=epoch,
                split="train",
                name="loss",
                value=float(loss),
                total_steps=config.training.epochs,  # Enables automatic ETA!
            )

            # Validate every 10 epochs
            if epoch % 10 == 0 or epoch == config.training.epochs - 1:
                # Compute validation metrics
                val_pred = model.predict(X_val)
                val_loss = np.mean((val_pred - y_val) ** 2)
                val_r2 = model.score(X_val, y_val)

                # Log validation metrics (also with ETA)
                logger.log_metric(
                    step=epoch,
                    split="val",
                    name="loss",
                    value=float(val_loss),
                    total_steps=config.training.epochs,
                )
                logger.log_metric(
                    step=epoch,
                    split="val",
                    name="r2_score",
                    value=float(val_r2),
                    total_steps=config.training.epochs,
                )

                # Log progress (with rate limiting)
                with logger.log_every(every=1):
                    logger.info(
                        f"Epoch {epoch:3d}/{config.training.epochs}: "
                        f"train_loss={loss:.6f}, val_loss={val_loss:.6f}, val_r2={val_r2:.4f}"
                    )

    # Final evaluation
    with logger.stage("evaluation"):
        train_r2 = model.score(X_train, y_train)
        val_r2 = model.score(X_val, y_val)

        logger.info(f"\n{'=' * 60}")
        logger.info("Final Results:")
        logger.info(f"  Training R²:   {train_r2:.4f}")
        logger.info(f"  Validation R²: {val_r2:.4f}")
        logger.info(f"  Learned weights: {model.weights.flatten()}")
        logger.info(f"  Learned bias:    {model.bias:.4f}")
        logger.info(f"  True weights:    [2.0]")
        logger.info(f"  True bias:       1.0")
        logger.info(f"{'=' * 60}\n")

        # Get best metrics
        best_val_loss = logger.get_best_metric("loss", split="val")
        logger.info(
            f"✨ Best validation loss: {best_val_loss['value']:.6f} at epoch {best_val_loss['step']}"
        )

    logger.info(f"📁 Results saved to: {logger.run_dir}")
    logger.info("✅ Experiment completed!")


if __name__ == "__main__":
    main()
