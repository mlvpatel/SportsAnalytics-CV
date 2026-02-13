"""
Unit tests for the configuration module.

Author: Malav Patel
"""

import os
import tempfile


class TestConfig:
    """Tests for config loading and defaults."""

    def test_default_config(self):
        from src.utils.config import AppConfig

        config = AppConfig()
        assert config.model.path == "models/best.pt"
        assert config.model.confidence == 0.5
        assert config.speed.frame_window == 5
        assert config.court.width == 68.0

    def test_load_config_from_yaml(self):
        from src.utils.config import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  path: custom.pt\n  confidence: 0.8\n  device: cpu\n")
            f.flush()

            config = load_config(f.name)
            assert config.model.path == "custom.pt"
            assert config.model.confidence == 0.8
            assert config.model.device == "cpu"

        os.unlink(f.name)

    def test_load_config_missing_file(self):
        from src.utils.config import load_config

        config = load_config("/nonexistent/config.yaml")
        # Should use defaults
        assert config.model.path == "models/best.pt"

    def test_env_var_override(self):
        from src.utils.config import load_config

        os.environ["MODEL_PATH"] = "override.pt"
        os.environ["DEVICE"] = "cpu"
        try:
            config = load_config("/nonexistent/config.yaml")
            assert config.model.path == "override.pt"
            assert config.model.device == "cpu"
        finally:
            del os.environ["MODEL_PATH"]
            del os.environ["DEVICE"]
