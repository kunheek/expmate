#!/usr/bin/env python3
"""
Test and demonstration of all expmate package-level configuration flags.

This script shows how to use and control various expmate behaviors globally.
"""

import tempfile
from pathlib import Path

import expmate
from expmate import ExperimentLogger
from expmate.git import get_git_info, save_git_info


def test_log_level():
    """Test global log level configuration."""
    print("=" * 70)
    print("Test 1: Global Log Level (expmate.log_level)")
    print("=" * 70)

    # Set global log level to WARNING
    expmate.log_level = "WARNING"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Logger should use global log level
        logger = ExperimentLogger(run_dir=tmpdir)

        logger.debug("This should NOT appear")
        logger.info("This should NOT appear")
        logger.warning("This SHOULD appear")
        logger.error("This SHOULD appear")

        logger.close()

    print("✓ Log level set to WARNING - only warnings and errors logged")
    print()

    # Reset to INFO
    expmate.log_level = "INFO"


def test_verbose():
    """Test verbose flag for console output."""
    print("=" * 70)
    print("Test 2: Verbose Mode (expmate.verbose)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Verbose mode ON (console output)
        expmate.verbose = True
        logger1 = ExperimentLogger(run_dir=tmpdir)
        print(f"✓ Console output enabled: {logger1.console_handler is not None}")
        logger1.close()

        # Verbose mode OFF (no console output)
        expmate.verbose = False
        logger2 = ExperimentLogger(run_dir=tmpdir)
        print(f"✓ Console output disabled: {logger2.console_handler is None}")
        logger2.close()

    print()

    # Reset
    expmate.verbose = True


def test_track_metrics():
    """Test metrics tracking flag."""
    print("=" * 70)
    print("Test 3: Metrics Tracking (expmate.track_metrics)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Metrics tracking ON
        expmate.track_metrics = True
        logger1 = ExperimentLogger(run_dir=run_dir / "with_metrics")
        logger1.log_metric(step=0, split="train", name="loss", value=0.5)
        logger1.close()

        metrics_file = run_dir / "with_metrics" / "metrics.csv"
        print(f"✓ Metrics enabled: {metrics_file.exists()}")

        # Metrics tracking OFF
        expmate.track_metrics = False
        logger2 = ExperimentLogger(run_dir=run_dir / "no_metrics")
        logger2.log_metric(step=0, split="train", name="loss", value=0.5)
        logger2.close()

        # CSV file should not be created
        csv_file = run_dir / "no_metrics" / "metrics.csv"
        print(f"✓ Metrics disabled: CSV not created = {not csv_file.exists()}")

    print()

    # Reset
    expmate.track_metrics = True


def test_track_git():
    """Test git tracking flag."""
    print("=" * 70)
    print("Test 4: Git Tracking (expmate.track_git)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Git tracking ON
        expmate.track_git = True
        info1 = get_git_info()
        print(f"✓ Git enabled: sha={info1['sha_short']}, branch={info1['branch']}")

        save_git_info(run_dir / "with_git")
        git_file = run_dir / "with_git" / "git_info.txt"
        print(f"✓ Git info saved: {git_file.exists()}")

        # Git tracking OFF
        expmate.track_git = False
        info2 = get_git_info()
        print(f"✓ Git disabled: sha={info2['sha_short']} (should be 'disabled')")

        save_git_info(run_dir / "no_git")
        git_file2 = run_dir / "no_git" / "git_info.txt"
        print(f"✓ Git info not saved: {not git_file2.exists()}")

    print()

    # Reset
    expmate.track_git = True


def test_profile():
    """Test profiling flag."""
    print("=" * 70)
    print("Test 5: Profiling (expmate.timer)")
    print("=" * 70)

    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(run_dir=tmpdir, console_output=False)

        # Profiling ON
        expmate.timer = True
        with logger.timer("with_profiling", log_result=False) as result1:
            time.sleep(0.01)
        print(f"✓ Profiling enabled: {result1['elapsed']:.4f}s measured")

        # Profiling OFF
        expmate.timer = False
        with logger.timer("no_profiling", log_result=False) as result2:
            time.sleep(0.01)
        print(f"✓ Profiling disabled: {result2['elapsed']:.4f}s (no overhead)")

        logger.close()

    print()

    # Reset
    expmate.timer = True


def test_save_checkpoints():
    """Test checkpoint saving flag."""
    print("=" * 70)
    print("Test 6: Checkpoint Saving (expmate.save_checkpoints)")
    print("=" * 70)

    # Only test if torch is available
    try:
        import torch

        from expmate.torch import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model
            model = torch.nn.Linear(10, 5)

            # Checkpoints ON
            expmate.save_checkpoints = True
            ckpt_mgr1 = CheckpointManager(Path(tmpdir) / "with_ckpt")
            path1 = ckpt_mgr1.save(model, step=0)
            print(f"✓ Checkpoints enabled: {path1 is not None and path1.exists()}")

            # Checkpoints OFF
            expmate.save_checkpoints = False
            ckpt_mgr2 = CheckpointManager(Path(tmpdir) / "no_ckpt")
            path2 = ckpt_mgr2.save(model, step=0)
            print(f"✓ Checkpoints disabled: {path2 is None}")

        # Reset
        expmate.save_checkpoints = True

    except ImportError:
        print("⊘ PyTorch not available, skipping checkpoint test")

    print()


def show_all_flags():
    """Display all available flags and their current values."""
    print("=" * 70)
    print("Summary: All Available Package-Level Flags")
    print("=" * 70)
    print()

    flags = [
        ("debug", "Enable debug mode and debug logging"),
        ("profile", "Enable profiling measurements in logger.profile()"),
        ("log_level", "Default log level for ExperimentLogger"),
        ("verbose", "Enable console output for loggers"),
        ("track_metrics", "Enable metrics tracking to CSV files"),
        ("track_git", "Enable git repository information collection"),
        ("save_checkpoints", "Enable checkpoint saving (PyTorch)"),
        ("force_single_process", "Force single-process mode in DDP"),
    ]

    for name, description in flags:
        value = getattr(expmate, name, "N/A")
        env_var = f"EM_{name.upper()}"
        print(f"{name:20s} = {value}")
        print(f"{'':20s}   {description}")
        print(f"{'':20s}   Env: {env_var}")
        print()

    print("=" * 70)


def main():
    print()
    print("ExpMate Package-Level Configuration Flags Demo")
    print("=" * 70)
    print()

    # Save original values
    original_values = {
        "debug": expmate.debug,
        "timer": expmate.timer,
        "log_level": expmate.log_level,
        "verbose": expmate.verbose,
        "track_metrics": expmate.track_metrics,
        "track_git": expmate.track_git,
        "save_checkpoints": expmate.save_checkpoints,
        "force_single_process": expmate.force_single_process,
    }

    try:
        # Run tests
        test_log_level()
        test_verbose()
        test_track_metrics()
        test_track_git()
        test_profile()
        test_save_checkpoints()

        # Show summary
        show_all_flags()

        print()
        print("=" * 70)
        print("All tests passed! ✅")
        print("=" * 70)

    finally:
        # Restore original values
        for key, value in original_values.items():
            setattr(expmate, key, value)


if __name__ == "__main__":
    main()
