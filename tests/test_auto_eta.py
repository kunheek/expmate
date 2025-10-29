#!/usr/bin/env python3
"""Test automatic ETA estimation in log_metric."""

import time
import tempfile

import expmate
from expmate import ExperimentLogger


def test_automatic_eta_basic():
    """Test basic automatic ETA estimation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Log metrics with total_steps
        for step in range(1, 11):
            time.sleep(0.01)  # Small delay to simulate work
            logger.log_metric(
                step=step,
                split="train",
                name="loss",
                value=1.0 / step,
                total_steps=10,
            )

        # Check that ETA tracking was initialized
        assert "train/loss" in logger._eta_tracking
        tracking = logger._eta_tracking["train/loss"]

        assert "start_time" in tracking
        assert "start_step" in tracking
        assert "total_steps" in tracking
        assert tracking["total_steps"] == 10
        assert tracking["start_step"] == 1

        logger.close()


def test_automatic_eta_custom_key():
    """Test ETA with custom key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Use custom eta_key
        for step in range(1, 6):
            logger.log_metric(
                step=step,
                split="train",
                name="loss",
                value=1.0 / step,
                total_steps=5,
                eta_key="custom_key",
            )

        # Check that custom key was used
        assert "custom_key" in logger._eta_tracking
        assert "train/loss" not in logger._eta_tracking

        logger.close()


def test_automatic_eta_multiple_metrics():
    """Test ETA tracking for multiple different metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Log train loss
        for step in range(1, 6):
            logger.log_metric(
                step=step,
                split="train",
                name="loss",
                value=1.0 / step,
                total_steps=5,
            )

        # Log validation accuracy
        for step in range(1, 4):
            logger.log_metric(
                step=step,
                split="val",
                name="accuracy",
                value=0.5 + step * 0.1,
                total_steps=3,
            )

        # Both should have separate tracking
        assert "train/loss" in logger._eta_tracking
        assert "val/accuracy" in logger._eta_tracking

        assert logger._eta_tracking["train/loss"]["total_steps"] == 5
        assert logger._eta_tracking["val/accuracy"]["total_steps"] == 3

        logger.close()


def test_automatic_eta_without_total_steps():
    """Test that metrics work fine without total_steps (no ETA)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Log without total_steps - should work normally
        for step in range(1, 6):
            logger.log_metric(
                step=step,
                split="train",
                name="loss",
                value=1.0 / step,
            )

        # No ETA tracking should be initialized
        assert len(logger._eta_tracking) == 0

        logger.close()


def test_format_time():
    """Test the _format_time helper method."""
    logger = ExperimentLogger(run_dir=tempfile.mkdtemp(), console_output=False)

    # Test various time formats
    assert logger._format_time(0) == "0s"
    assert logger._format_time(30) == "30s"
    assert logger._format_time(60) == "1m"
    assert logger._format_time(90) == "1m 30s"
    assert logger._format_time(3600) == "1h"
    assert logger._format_time(3661) == "1h 1m 1s"
    assert logger._format_time(86400) == "1d"
    assert logger._format_time(90061) == "1d 1h 1m 1s"
    assert logger._format_time(-10) == "0s"  # Negative time

    logger.close()


def test_eta_disabled_with_track_metrics_false():
    """Test that ETA respects track_metrics flag."""
    original = expmate.track_metrics

    try:
        expmate.track_metrics = False

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

            # Log metrics - should be skipped
            for step in range(1, 6):
                logger.log_metric(
                    step=step,
                    split="train",
                    name="loss",
                    value=1.0 / step,
                    total_steps=5,
                )

            # No ETA tracking should happen
            assert len(logger._eta_tracking) == 0

            logger.close()

    finally:
        expmate.track_metrics = original


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing Automatic ETA Estimation")
    print("=" * 70 + "\n")

    # Save original state
    original_track_metrics = expmate.track_metrics

    try:
        test_automatic_eta_basic()
        print("✓ test_automatic_eta_basic passed")

        test_automatic_eta_custom_key()
        print("✓ test_automatic_eta_custom_key passed")

        test_automatic_eta_multiple_metrics()
        print("✓ test_automatic_eta_multiple_metrics passed")

        test_automatic_eta_without_total_steps()
        print("✓ test_automatic_eta_without_total_steps passed")

        test_format_time()
        print("✓ test_format_time passed")

        test_eta_disabled_with_track_metrics_false()
        print("✓ test_eta_disabled_with_track_metrics_false passed")

        print("\n" + "=" * 70)
        print("All tests passed! ✅")
        print("=" * 70 + "\n")

    finally:
        # Restore original state
        expmate.track_metrics = original_track_metrics
