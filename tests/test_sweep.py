"""Tests for sweep module."""

from expmate.cli.sweep import generate_sweep_configs, run_sweep


class TestGenerateSweepConfigs:
    """Test sweep configuration generation."""

    def test_generate_single_param(self):
        base_config = {"model": {"hidden_dim": 64}}
        sweep_params = {"model.hidden_dim": [128, 256, 512]}

        configs = generate_sweep_configs(base_config, sweep_params)

        assert len(configs) == 3
        assert configs[0]["model"]["hidden_dim"] == 128
        assert configs[1]["model"]["hidden_dim"] == 256
        assert configs[2]["model"]["hidden_dim"] == 512

    def test_generate_multiple_params(self):
        base_config = {"training": {"lr": 0.001, "batch_size": 32}}
        sweep_params = {"training.lr": [0.001, 0.01], "training.batch_size": [16, 32]}

        configs = generate_sweep_configs(base_config, sweep_params)

        # Should generate 2 x 2 = 4 configs
        assert len(configs) == 4

    def test_generate_nested_params(self):
        base_config = {"model": {"encoder": {"dim": 64}}}
        sweep_params = {"model.encoder.dim": [128, 256]}

        configs = generate_sweep_configs(base_config, sweep_params)

        assert len(configs) == 2
        assert configs[0]["model"]["encoder"]["dim"] == 128
        assert configs[1]["model"]["encoder"]["dim"] == 256

    def test_generate_preserves_base_config(self):
        base_config = {"seed": 42, "training": {"epochs": 10}}
        sweep_params = {"training.lr": [0.001, 0.01]}

        configs = generate_sweep_configs(base_config, sweep_params)

        # All configs should have seed and epochs preserved
        for config in configs:
            assert config["seed"] == 42
            assert config["training"]["epochs"] == 10


class TestRunSweep:
    """Test sweep execution."""

    def test_run_sweep_dry_run(self, temp_dir, capsys):
        sweep_params = {"lr": [0.001, 0.01]}

        run_sweep(
            command_template="echo {config}",
            sweep_params=sweep_params,
            runs_dir=str(temp_dir),
            sweep_name="test_sweep",
            dry_run=True,
        )

        captured = capsys.readouterr()
        # Should print commands but not execute
        assert "echo" in captured.out

    def test_run_sweep_creates_directory(self, temp_dir):
        sweep_params = {"lr": [0.001]}

        run_sweep(
            command_template="echo test",
            sweep_params=sweep_params,
            runs_dir=str(temp_dir),
            sweep_name="test_sweep",
            dry_run=True,
        )

        sweep_dir = temp_dir / "test_sweep"
        assert sweep_dir.exists()
