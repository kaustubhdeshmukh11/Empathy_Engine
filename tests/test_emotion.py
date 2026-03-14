"""Tests for the EmotionDetector module."""

import pytest
from empathy_engine.emotion_detector import EmotionDetector, EmotionResult


@pytest.fixture(scope="module")
def detector():
    """Shared detector instance (model loads once)."""
    return EmotionDetector()


class TestEmotionResult:
    """Test the EmotionResult dataclass."""

    def test_creation(self):
        result = EmotionResult(
            emotion="joy",
            confidence=0.95,
            intensity=0.8,
            valence=0.75,
            all_scores={"joy": 0.95, "neutral": 0.05},
        )
        assert result.emotion == "joy"
        assert result.confidence == 0.95
        assert result.intensity == 0.8


class TestEmotionDetector:
    """Test emotion detection on known texts."""

    def test_positive_text(self, detector):
        result = detector.detect("I am so happy and excited today! This is the best day ever!")
        assert result.emotion == "joy"
        assert result.confidence > 0.3
        assert result.intensity > 0.3
        assert result.valence > 0

    def test_negative_text(self, detector):
        result = detector.detect("I am devastated and heartbroken. This is terrible news.")
        assert result.emotion in ("sadness", "anger", "disgust", "fear")
        assert result.confidence > 0.3
        assert result.valence < 0

    def test_neutral_text(self, detector):
        result = detector.detect("The meeting is scheduled for tomorrow at 3 PM.")
        assert result.emotion == "neutral"
        assert result.intensity < 0.5

    def test_anger_text(self, detector):
        result = detector.detect("I am furious! This is absolutely unacceptable and outrageous!")
        assert result.emotion == "anger"
        assert result.confidence > 0.3
        assert result.intensity > 0.3

    def test_surprise_text(self, detector):
        result = detector.detect("What?! I can't believe this actually happened! No way!")
        assert result.emotion in ("surprise", "joy")
        assert result.confidence > 0.2

    def test_all_scores_populated(self, detector):
        result = detector.detect("I feel great today.")
        assert len(result.all_scores) >= 5  # Should have multiple emotion scores

    def test_intensity_scaling(self, detector):
        """Mild text should have lower intensity than emphatic text."""
        mild = detector.detect("This is nice.")
        strong = detector.detect("THIS IS ABSOLUTELY INCREDIBLE!! BEST THING EVER!!!")
        assert strong.intensity > mild.intensity

    def test_empty_text_handling(self, detector):
        """Should not crash on minimal input."""
        result = detector.detect("ok")
        assert result.emotion is not None
        assert 0 <= result.intensity <= 1
