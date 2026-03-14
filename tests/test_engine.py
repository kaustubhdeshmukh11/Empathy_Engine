"""Integration tests for the Empathy Engine pipeline."""

import os
import pytest
from pathlib import Path

from empathy_engine.engine import EmpathyEngine
from empathy_engine.tts_engine import OfflineTTSEngine


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Create engine with offline TTS for testing (no Azure key needed)."""
    output_dir = str(tmp_path_factory.mktemp("test_output"))
    return EmpathyEngine(output_dir=output_dir, tts_engine=OfflineTTSEngine())


class TestEmpathyEngine:
    """End-to-end integration tests."""

    def test_process_returns_required_keys(self, engine):
        result = engine.process("Hello world")
        required_keys = ["text", "emotion", "confidence", "intensity", "valence",
                         "voice_config", "audio_path", "tts_engine"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_process_generates_audio_file(self, engine):
        result = engine.process("I feel great today!")
        assert os.path.exists(result["audio_path"]), "Audio file was not created"
        assert os.path.getsize(result["audio_path"]) > 0, "Audio file is empty"

    def test_process_happy_text(self, engine):
        result = engine.process("This is wonderful! I'm so excited!")
        assert result["emotion"] == "joy"
        assert result["voice_config"]["rate"] > 175  # Should be faster than base

    def test_process_sad_text(self, engine):
        result = engine.process("I'm really heartbroken and devastated by the loss.")
        assert result["emotion"] in ("sadness", "fear")
        assert result["voice_config"]["rate"] <= 175  # Should be slower

    def test_process_custom_filename(self, engine):
        result = engine.process("Test text", output_filename="custom_test.wav")
        assert result["audio_path"].endswith("custom_test.wav")

    def test_multiple_calls_different_files(self, engine):
        r1 = engine.process("Happy text!")
        r2 = engine.process("Sad text.")
        assert r1["audio_path"] != r2["audio_path"]

    def test_voice_config_structure(self, engine):
        result = engine.process("Neutral test sentence.")
        vc = result["voice_config"]
        assert "rate" in vc
        assert "pitch_delta" in vc
        assert "volume" in vc
        assert "azure_style" in vc
        assert "style_degree" in vc
        assert "emphasis" in vc
