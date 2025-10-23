import pytest

try:
    from expmate.torch import mp

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


class TestMPUtilities:
    """Test multiprocessing utility functions."""

    def test_get_rank_not_initialized(self):
        # When not in distributed mode, should return 0
        rank = mp.get_rank()
        assert rank == 0

    def test_get_local_rank_not_initialized(self):
        local_rank = mp.get_local_rank()
        assert local_rank == 0

    def test_get_world_size_not_initialized(self):
        world_size = mp.get_world_size()
        assert world_size == 1

    def test_is_main_process_not_initialized(self):
        assert mp.is_main_process() is True

    def test_main_process_only_decorator(self):
        call_count = []

        @mp.main_process_only
        def test_func():
            call_count.append(1)
            return "executed"

        result = test_func()
        assert len(call_count) == 1
        assert result == "executed"

    def test_mp_print(self, capsys):
        mp.mp_print("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_reduce_dict_single_process(self):
        input_dict = {"loss": 1.5, "accuracy": 0.8}
        result = mp.reduce_dict(input_dict)

        # In single process, should return same values
        assert result["loss"] == 1.5
        assert result["accuracy"] == 0.8

    def test_barrier_single_process(self):
        # Should not raise error in single process
        mp.barrier()

    def test_create_shared_run_dir(self, temp_dir):
        run_dir = mp.create_shared_run_dir(base_dir=str(temp_dir), run_id="test_run")

        assert run_dir.exists()
        assert "test_run" in str(run_dir)

    def test_create_shared_run_dir_with_timestamp(self, temp_dir):
        run_dir = mp.create_shared_run_dir(base_dir=str(temp_dir))

        assert run_dir.exists()
        # Should have timestamp in name
        assert len(run_dir.name) > 0
