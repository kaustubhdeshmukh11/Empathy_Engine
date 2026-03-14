"""Tests for the VoiceModulator module."""

import pytest
from empathy_engine.emotion_detector import EmotionResult
from empathy_engine.voice_modulator import VoiceModulator, VoiceConfig


@pytest.fixture
def modulator():
    return VoiceModulator()


def make_emotion(emotion: str, intensity: float = 0.5, confidence: float = 0.9) -> EmotionResult:
    """Helper to create test EmotionResults."""
    return EmotionResult(
        emotion=emotion,
        confidence=confidence,
        intensity=intensity,
        valence=0.5 if emotion in ("joy", "surprise") else -0.5 if emotion in ("sadness", "anger") else 0.0,
        all_scores={emotion: confidence},
    )


class TestVoiceModulator:
    """Test emotion-to-voice parameter mapping."""

    def test_joy_increases_rate(self, modulator):
        config = modulator.modulate(make_emotion("joy", intensity=0.8))
        assert config.rate > VoiceModulator.BASE_RATE

    def test_joy_increases_pitch(self, modulator):
        config = modulator.modulate(make_emotion("joy", intensity=0.8))
        assert config.pitch_delta > 0

    def test_sadness_decreases_rate(self, modulator):
        config = modulator.modulate(make_emotion("sadness", intensity=0.8))
        assert config.rate < VoiceModulator.BASE_RATE

    def test_sadness_decreases_pitch(self, modulator):
        config = modulator.modulate(make_emotion("sadness", intensity=0.8))
        assert config.pitch_delta < 0

    def test_anger_increases_volume(self, modulator):
        config = modulator.modulate(make_emotion("anger", intensity=0.8))
        assert config.volume > VoiceModulator.BASE_VOLUME

    def test_neutral_no_change(self, modulator):
        config = modulator.modulate(make_emotion("neutral", intensity=0.0))
        assert config.rate == VoiceModulator.BASE_RATE
        assert config.pitch_delta == 0
        assert config.volume == VoiceModulator.BASE_VOLUME

    def test_intensity_scaling_low(self, modulator):
        """Low intensity should produce smaller deltas."""
        low = modulator.modulate(make_emotion("joy", intensity=0.1))
        high = modulator.modulate(make_emotion("joy", intensity=0.9))
        assert high.rate > low.rate
        assert high.pitch_delta > low.pitch_delta

    def test_intensity_scaling_proportional(self, modulator):
        """Mid-intensity should be between low and high."""
        low = modulator.modulate(make_emotion("joy", intensity=0.2))
        mid = modulator.modulate(make_emotion("joy", intensity=0.5))
        high = modulator.modulate(make_emotion("joy", intensity=0.9))
        assert low.rate <= mid.rate <= high.rate

    def test_azure_style_mapping(self, modulator):
        """Each emotion should map to the correct Azure style."""
        assert modulator.modulate(make_emotion("joy")).azure_style == "cheerful"
        assert modulator.modulate(make_emotion("anger")).azure_style == "angry"
        assert modulator.modulate(make_emotion("sadness")).azure_style == "sad"
        assert modulator.modulate(make_emotion("fear")).azure_style == "fearful"
        assert modulator.modulate(make_emotion("neutral")).azure_style == "default"

    def test_style_degree_scaling(self, modulator):
        """Higher intensity should produce higher style degree."""
        low = modulator.modulate(make_emotion("joy", intensity=0.1))
        high = modulator.modulate(make_emotion("joy", intensity=0.9))
        assert high.style_degree > low.style_degree

    def test_volume_clamping(self, modulator):
        """Volume should never exceed 1.0 or drop below 0.1."""
        # Very intense anger (high volume boost)
        config = modulator.modulate(make_emotion("anger", intensity=1.0))
        assert config.volume <= 1.0
        # Very intense fear (volume drop)
        config = modulator.modulate(make_emotion("fear", intensity=1.0))
        assert config.volume >= 0.1

    def test_rate_clamping(self, modulator):
        """Rate should stay within sane bounds."""
        fast = modulator.modulate(make_emotion("surprise", intensity=1.0))
        assert 80 <= fast.rate <= 350
        slow = modulator.modulate(make_emotion("sadness", intensity=1.0))
        assert 80 <= slow.rate <= 350

    def test_unknown_emotion_defaults_to_neutral(self, modulator):
        """Unknown emotion label should produce neutral config."""
        config = modulator.modulate(make_emotion("unknown_emotion"))
        assert config.rate == VoiceModulator.BASE_RATE
        assert config.azure_style == "default"
