#!/usr/bin/env python3
"""Test that logger.timer() can be disabled with expmate.timer flag."""

import time
import tempfile
from pathlib import Path

import expmate
from expmate import ExperimentLogger


def test_profile_enabled():
    """Test profiling when enabled (default)."""
    print("=" * 70)
    print("Test 1: Profiling Enabled (default)")
    print("=" * 70)

    # Ensure profiling is enabled
    expmate.timer = True

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Profile a section
        with logger.timer("test_section", log_result=False) as result:
            time.sleep(0.1)  # Sleep for 100ms

        # Check that timing was recorded
        assert result["elapsed"] > 0.09, f"Expected ~0.1s, got {result['elapsed']}"
        print(f"✓ Profiling worked: {result['elapsed']:.4f}s")

        logger.close()

    print()


def test_profile_disabled():
    """Test profiling when disabled."""
    print("=" * 70)
    print("Test 2: Profiling Disabled")
    print("=" * 70)

    # Disable profiling
    expmate.timer = False

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Profile a section - should be skipped
        with logger.timer("test_section", log_result=False) as result:
            time.sleep(0.1)  # Sleep for 100ms

        # Check that timing was NOT recorded (should be 0.0)
        assert result["elapsed"] == 0.0, (
            f"Expected 0.0 (disabled), got {result['elapsed']}"
        )
        print(f"✓ Profiling was skipped: {result['elapsed']}s (no overhead)")

        logger.close()

    print()


def test_profile_toggle():
    """Test toggling profiling on/off."""
    print("=" * 70)
    print("Test 3: Toggling Profiling On/Off")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Enable profiling
        expmate.timer = True
        with logger.timer("enabled_section", log_result=False) as result1:
            time.sleep(0.05)
        assert result1["elapsed"] > 0.04
        print(f"✓ Enabled: {result1['elapsed']:.4f}s")

        # Disable profiling
        expmate.timer = False
        with logger.timer("disabled_section", log_result=False) as result2:
            time.sleep(0.05)
        assert result2["elapsed"] == 0.0
        print(f"✓ Disabled: {result2['elapsed']}s (skipped)")

        # Re-enable profiling
        expmate.timer = True
        with logger.timer("re_enabled_section", log_result=False) as result3:
            time.sleep(0.05)
        assert result3["elapsed"] > 0.04
        print(f"✓ Re-enabled: {result3['elapsed']:.4f}s")

        logger.close()

    print()


def test_profile_with_logging():
    """Test that log messages are also skipped when disabled."""
    print("=" * 70)
    print("Test 4: Log Messages When Profiling Disabled")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        logger = ExperimentLogger(run_dir=run_dir, console_output=False)

        # Disable profiling
        expmate.timer = False

        # Profile with logging enabled
        with logger.timer("test_with_logging", log_result=True):
            time.sleep(0.05)

        # Check that no profile log was written
        log_file = run_dir / "exp.log"
        log_content = log_file.read_text()

        # Should not contain profile message
        assert "Profile [test_with_logging]" not in log_content
        print("✓ No profile log message written when disabled")

        logger.close()

    print()

    print()


if __name__ == "__main__":
    print()
    print("Testing logger.timer() with expmate.timer flag")
    print("=" * 70)
    print()

    # Save original state
    original_timer = expmate.timer

    try:
        test_profile_enabled()
        test_profile_disabled()
        test_profile_toggle()
        test_profile_with_logging()

        print("=" * 70)
        print("All tests passed! ✅")
        print("=" * 70)
        print()
        print("Usage:")
        print("  # Enable profiling (default)")
        print("  expmate.timer = True")
        print("  # or set environment variable: EM_TIMER=1")
        print()
        print("  # Disable profiling for production")
        print("  expmate.timer = False")
        print("  # or set environment variable: EM_TIMER=0")
        print("=" * 70)

    finally:
        # Restore original state
        expmate.timer = original_timer
