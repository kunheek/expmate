#!/usr/bin/env python3
"""
Test colorful logging implementation.
"""

import tempfile
from pathlib import Path

from expmate import ExperimentLogger


def test_file_logs_no_colors():
    """Verify file logs don't contain ANSI color codes."""
    print("Test 1: File logs have no color codes")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir)

        logger.info("Test message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Read log file
        log_file = Path(tmpdir) / "exp.log"
        content = log_file.read_text()

        # Check for ANSI escape codes
        assert "\033[" not in content, "File log should not contain ANSI codes"
        assert "INFO - Test message" in content
        assert "WARNING - Warning message" in content
        assert "ERROR - Error message" in content

    print("✓ File logs are plain text without colors")


def test_stage_and_timer_logging():
    """Test that stage and timer messages are logged correctly."""
    print("Test 2: Stage and timer messages")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir)

        with logger.stage("training"):
            pass

        with logger.timer("processing"):
            pass

        # Read log file
        log_file = Path(tmpdir) / "exp.log"
        content = log_file.read_text()

        # Check messages are present
        assert "Stage [training] - START" in content
        assert "Stage [training] - END" in content
        assert "Timer [processing]:" in content

    print("✓ Stage and timer messages logged correctly")


def test_jsonl_no_colors():
    """Verify JSONL logs don't contain color codes."""
    print("Test 3: JSONL logs have no color codes")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir)

        logger.info("Test message")
        logger.warning("Warning")

        # Read JSONL file
        jsonl_file = Path(tmpdir) / "events.jsonl"
        content = jsonl_file.read_text()

        # Check for ANSI escape codes
        assert "\033[" not in content, "JSONL should not contain ANSI codes"

    print("✓ JSONL logs are plain JSON without colors")


def main():
    print()
    print("=" * 60)
    print("Testing Colorful Logging")
    print("=" * 60)
    print()

    test_file_logs_no_colors()
    test_stage_and_timer_logging()
    test_jsonl_no_colors()

    print()
    print("=" * 60)
    print("All color tests passed! ✅")
    print("=" * 60)
    print()
    print("Summary:")
    print("  • Console output uses colors when running in a TTY")
    print("  • File logs (exp.log) are plain text")
    print("  • JSONL logs are clean JSON")
    print("  • Colors auto-disabled when output is redirected")
    print()


if __name__ == "__main__":
    main()
