#!/usr/bin/env python3
"""
Tests for logger.stage() and logger.log_every() context managers.
"""

import time
import tempfile
from pathlib import Path

from expmate import ExperimentLogger


def test_stage_basic():
    """Test basic stage functionality."""
    print("Test 1: Basic Stage")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir)

        with logger.stage("training") as info:
            time.sleep(0.01)

        assert info["name"] == "training"
        assert info["elapsed"] > 0.009
        assert len(info["metadata"]) == 0

    print("✓ Basic stage works")


def test_stage_with_metadata():
    """Test stage with metadata."""
    print("Test 2: Stage with Metadata")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir)

        with logger.stage("epoch", epoch=5, lr=0.001) as info:
            time.sleep(0.01)

        assert info["metadata"]["epoch"] == 5
        assert info["metadata"]["lr"] == 0.001

    print("✓ Stage with metadata works")


def test_stage_nested():
    """Test nested stages."""
    print("Test 3: Nested Stages")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir)

        with logger.stage("epoch"):
            with logger.stage("train") as train_info:
                time.sleep(0.01)

            with logger.stage("validation") as val_info:
                time.sleep(0.01)

        # Both nested stages should have completed
        assert train_info["elapsed"] > 0.009
        assert val_info["elapsed"] > 0.009

    print("✓ Nested stages work")


def test_log_every_iteration():
    """Test iteration-based rate limiting."""
    print("Test 4: log_every - Iteration")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        logged_count = 0
        for i in range(20):
            with logger.log_every(every=5) as should_log:
                if should_log:
                    logged_count += 1
                logger.info(f"Step {i}")

        # Should log on iterations 0, 5, 10, 15 = 4 times
        assert logged_count == 4

    print("✓ Iteration-based rate limiting works")


def test_log_every_time():
    """Test time-based rate limiting."""
    print("Test 5: log_every - Time")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        start = time.time()
        logged_count = 0

        while time.time() - start < 1.5:
            with logger.log_every(seconds=0.5) as should_log:
                if should_log:
                    logged_count += 1
                logger.info("Message")
            time.sleep(0.1)

        # Should log ~3 times (at 0s, 0.5s, 1.0s, 1.5s)
        assert logged_count >= 3 and logged_count <= 4

    print("✓ Time-based rate limiting works")


def test_log_every_multiple_keys():
    """Test multiple independent rate limiters."""
    print("Test 6: log_every - Multiple Keys")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        loss_count = 0
        metrics_count = 0

        for i in range(30):
            with logger.log_every(every=5, key="loss") as should_log:
                if should_log:
                    loss_count += 1

            with logger.log_every(every=10, key="metrics") as should_log:
                if should_log:
                    metrics_count += 1

        # loss: 0,5,10,15,20,25 = 6 times
        # metrics: 0,10,20 = 3 times
        assert loss_count == 6
        assert metrics_count == 3

    print("✓ Multiple rate limiters work independently")


def test_stage_and_log_every_combined():
    """Test using stage and log_every together."""
    print("Test 7: Combined Stage + log_every")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        with logger.stage("training") as stage_info:
            for step in range(20):
                with logger.log_every(every=10):
                    logger.info(f"Step {step}")
                time.sleep(0.001)

        # Stage should have tracked total time
        assert stage_info["elapsed"] > 0.015

    print("✓ Combined stage + log_every works")


def main():
    print()
    print("=" * 60)
    print("Testing logger.stage() and logger.rate_limit()")
    print("=" * 60)
    print()

    test_stage_basic()
    test_stage_with_metadata()
    test_stage_nested()
    test_log_every_iteration()
    test_log_every_time()
    test_log_every_multiple_keys()
    test_stage_and_log_every_combined()

    print()
    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
