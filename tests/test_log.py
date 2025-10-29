import json
import time

from expmate.logger import ExperimentLogger


class TestExperimentLogger:
    """Test ExperimentLogger class."""

    def test_init_creates_run_dir(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        assert run_dir.exists()
        assert logger.run_dir == run_dir

    def test_init_with_run_id(self, temp_dir):
        logger = ExperimentLogger(run_dir=temp_dir, run_id="exp_001")
        assert logger.run_id == "exp_001"

    def test_log_writes_to_file(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))
        logger.log("Test message")

        events_file = run_dir / "events.jsonl"
        assert events_file.exists()

        with open(events_file) as f:
            line = f.readline()
            event = json.loads(line)
            assert event["message"] == "Test message"
            assert "timestamp" in event

    def test_log_metric(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        logger.log_metric(step=0, split="train", name="loss", value=0.5)
        logger.log_metric(step=1, split="train", name="loss", value=0.3)

        metrics_file = run_dir / "metrics.csv"
        assert metrics_file.exists()

        # Read and verify CSV
        import csv

        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["name"] == "loss"
            assert float(rows[0]["value"]) == 0.5
            assert float(rows[1]["value"]) == 0.3

    def test_log_metrics_multiple_types(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        logger.log_metric(step=0, split="train", name="loss", value=0.5)
        logger.log_metric(step=0, split="train", name="accuracy", value=0.8)
        logger.log_metric(step=0, split="val", name="loss", value=0.6)

        metrics_file = run_dir / "metrics.csv"
        with open(metrics_file) as f:
            import csv

            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3

    def test_track_best_metric_min(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        # Log decreasing loss (best should be tracked)
        logger.log_metric(step=0, split="val", name="loss", value=1.0)
        logger.log_metric(step=1, split="val", name="loss", value=0.8)
        logger.log_metric(step=2, split="val", name="loss", value=0.6)
        logger.log_metric(step=3, split="val", name="loss", value=0.7)  # Worse

        best_file = run_dir / "best.json"
        assert best_file.exists()

        with open(best_file) as f:
            best = json.load(f)
            assert "val/loss" in best
            assert best["val/loss"]["value"] == 0.6
            assert best["val/loss"]["step"] == 2

    def test_track_best_metric_max(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        # Log increasing accuracy (best should be tracked)
        logger.log_metric(step=0, split="val", name="accuracy", value=0.5)
        logger.log_metric(step=1, split="val", name="accuracy", value=0.7)
        logger.log_metric(step=2, split="val", name="accuracy", value=0.9)
        logger.log_metric(step=3, split="val", name="accuracy", value=0.8)  # Worse

        best_file = run_dir / "best.json"
        with open(best_file) as f:
            best = json.load(f)
            assert "val/accuracy" in best
            assert best["val/accuracy"]["value"] == 0.9
            assert best["val/accuracy"]["step"] == 2

    def test_profile_context_manager(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        with logger.profile("test_section"):
            time.sleep(0.01)  # Small delay

        events_file = run_dir / "events.jsonl"
        with open(events_file) as f:
            lines = f.readlines()
            # Should have at least one event (profile result)
            assert len(lines) >= 1

            # Check for profile events
            events = [json.loads(line) for line in lines]
            profile_events = [
                e for e in events if "section" in e and e["section"] == "test_section"
            ]
            assert len(profile_events) >= 1
            assert profile_events[0]["elapsed"] > 0

    def test_rank_aware_logging(self, temp_dir):
        run_dir = temp_dir / "test_run"

        # Rank 0 logger
        logger0 = ExperimentLogger(run_dir=str(run_dir), rank=0)
        logger0.log("Message from rank 0")

        # Rank 1 logger
        logger1 = ExperimentLogger(run_dir=str(run_dir), rank=1)
        logger1.log("Message from rank 1")

        # Check separate files exist
        events_file_0 = run_dir / "events.jsonl"
        events_file_1 = run_dir / "events_rank1.jsonl"

        assert events_file_0.exists()
        assert events_file_1.exists()

    def test_log_level_filtering(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir), log_level="WARNING")

        logger.log("Debug message", level="DEBUG")
        logger.log("Info message", level="INFO")
        logger.log("Warning message", level="WARNING")
        logger.log("Error message", level="ERROR")

        events_file = run_dir / "events.jsonl"
        with open(events_file) as f:
            lines = f.readlines()
            events = [json.loads(line) for line in lines]

            # Should only have WARNING and ERROR
            assert len(events) == 2
            assert any(e["message"] == "Warning message" for e in events)
            assert any(e["message"] == "Error message" for e in events)

    def test_close(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))
        logger.log("Test message")
        logger.close()

        # Files should exist after close
        assert (run_dir / "events.jsonl").exists()

    def test_context_manager(self, temp_dir):
        run_dir = temp_dir / "test_run"

        with ExperimentLogger(run_dir=str(run_dir)) as logger:
            logger.log("Test message")

        # Files should exist after context exit
        assert (run_dir / "events.jsonl").exists()

    def test_log_dict(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        data = {"key1": "value1", "key2": 42, "key3": 3.14}
        logger.log_dict(data)

        events_file = run_dir / "events.jsonl"
        with open(events_file) as f:
            event = json.loads(f.readline())
            assert event["key1"] == "value1"
            assert event["key2"] == 42
            assert event["key3"] == 3.14

    def test_save_metadata(self, temp_dir):
        run_dir = temp_dir / "test_run"
        logger = ExperimentLogger(run_dir=str(run_dir))

        metadata = {"experiment": "test", "version": "1.0"}
        logger.save_metadata(metadata, "metadata.json")

        metadata_file = run_dir / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            loaded = json.load(f)
            assert loaded == metadata
