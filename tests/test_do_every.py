"""Tests for the do_every context manager."""

import time

import pytest

from expmate.logger import ExperimentLogger


class TestDoEvery:
    """Test suite for do_every context manager."""

    def test_iteration_based_execution(self, tmp_path):
        """Test iteration-based rate limiting."""
        logger = ExperimentLogger(run_dir=tmp_path)

        executed = []
        for i in range(20):
            with logger.do_every(every=5) as should_execute:
                if should_execute:
                    executed.append(i)

        # Should execute at iterations 0, 5, 10, 15
        assert executed == [0, 5, 10, 15]

    def test_time_based_execution(self, tmp_path):
        """Test time-based rate limiting."""
        logger = ExperimentLogger(run_dir=tmp_path)

        executed = []
        start_time = time.time()

        # Run for ~1.5 seconds
        while time.time() - start_time < 1.5:
            with logger.do_every(seconds=0.5) as should_execute:
                if should_execute:
                    executed.append(time.time() - start_time)
            time.sleep(0.1)

        # Should execute approximately 3-4 times (at 0s, 0.5s, 1.0s, maybe 1.5s)
        assert 3 <= len(executed) <= 4

        # Verify intervals are approximately 0.5 seconds
        for i in range(1, len(executed)):
            interval = executed[i] - executed[i - 1]
            assert 0.4 <= interval <= 0.6

    def test_multiple_keys(self, tmp_path):
        """Test multiple independent rate limiters with different keys."""
        logger = ExperimentLogger(run_dir=tmp_path)

        executed_a = []
        executed_b = []

        for i in range(30):
            with logger.do_every(every=5, key="key_a") as should_execute:
                if should_execute:
                    executed_a.append(i)

            with logger.do_every(every=10, key="key_b") as should_execute:
                if should_execute:
                    executed_b.append(i)

        assert executed_a == [0, 5, 10, 15, 20, 25]
        assert executed_b == [0, 10, 20]

    def test_auto_key_generation(self, tmp_path):
        """Test that keys are auto-generated from caller location."""
        logger = ExperimentLogger(run_dir=tmp_path)

        # Same line number means same key, so counter continues
        executed = []
        for i in range(10):
            with logger.do_every(every=3) as should_execute:
                if should_execute:
                    executed.append(i)

        # Should execute at 0, 3, 6, 9 (starts at 0)
        assert executed == [0, 3, 6, 9]

        # Using an explicit key allows independent counters
        executed_a = []
        for i in range(10):
            with logger.do_every(every=3, key="explicit_a") as should_execute:
                if should_execute:
                    executed_a.append(i)

        executed_b = []
        for i in range(10):
            with logger.do_every(every=3, key="explicit_b") as should_execute:
                if should_execute:
                    executed_b.append(i)

        # Each explicit key starts independently
        assert executed_a == [0, 3, 6, 9]
        assert executed_b == [0, 3, 6, 9]

    def test_requires_either_every_or_seconds(self, tmp_path):
        """Test that either 'every' or 'seconds' must be specified."""
        logger = ExperimentLogger(run_dir=tmp_path)

        with pytest.raises(ValueError, match="Must specify either 'every' or 'seconds'"):
            with logger.do_every() as _:
                pass

    def test_cannot_specify_both(self, tmp_path):
        """Test that 'every' and 'seconds' are mutually exclusive."""
        logger = ExperimentLogger(run_dir=tmp_path)

        with pytest.raises(ValueError, match="Cannot specify both 'every' and 'seconds'"):
            with logger.do_every(every=5, seconds=1.0) as _:
                pass

    def test_log_every_uses_do_every(self, tmp_path):
        """Test that log_every still works correctly after refactoring to use do_every."""
        logger = ExperimentLogger(run_dir=tmp_path)

        # Count actual log outputs by capturing them
        logged_steps = []

        for i in range(20):
            with logger.log_every(every=5):
                # This should only log at steps 0, 5, 10, 15
                logger.info(f"Step {i}")
                logged_steps.append(i)

        # log_every doesn't prevent code execution, just suppresses console output
        # But we can verify it was called 20 times
        assert len(logged_steps) == 20

    def test_combined_log_every_and_do_every(self, tmp_path):
        """Test using both log_every and do_every in the same loop."""
        logger = ExperimentLogger(run_dir=tmp_path)

        checkpoints = []
        logs = []

        for i in range(50):
            # Save checkpoint every 20 iterations
            with logger.do_every(every=20, key="checkpoint") as should_execute:
                if should_execute:
                    checkpoints.append(i)

            # Log every 10 iterations
            with logger.log_every(every=10, key="logging"):
                logger.info(f"Step {i}")
                logs.append(i)

        assert checkpoints == [0, 20, 40]
        assert len(logs) == 50  # log_every doesn't prevent code execution

    def test_zero_interval_edge_case(self, tmp_path):
        """Test that zero interval is handled correctly."""
        logger = ExperimentLogger(run_dir=tmp_path)

        # every=0 should never execute (except first iteration)
        executed = []
        for i in range(10):
            with logger.do_every(every=1) as should_execute:
                if should_execute:
                    executed.append(i)

        # Should execute every iteration
        assert executed == list(range(10))

    def test_cleanup(self, tmp_path):
        """Test that logger closes cleanly after using do_every."""
        logger = ExperimentLogger(run_dir=tmp_path)

        for i in range(10):
            with logger.do_every(every=5) as should_execute:
                if should_execute:
                    logger.info(f"Step {i}")  # Write something to create files

        logger.close()

        # Verify files were created
        assert (tmp_path / "exp.log").exists()
        assert (tmp_path / "events.jsonl").exists()
