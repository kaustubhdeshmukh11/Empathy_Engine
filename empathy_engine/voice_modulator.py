"""
Voice Modulation Module

Maps detected emotions to vocal parameter configurations (rate, pitch, volume)
with intensity-based scaling. Generates SSML markup for Azure TTS and prosody
parameters for the offline fallback engine.
"""

from dataclasses import dataclass
from typing import Optional

try:
    from empathy_engine.emotion_detector import EmotionResult
except ImportError:
    from .emotion_detector import EmotionResult


@dataclass
class VoiceConfig:
    """Configuration for vocal parameters."""
    rate: int              # Words per minute (base ~175)
    pitch_delta: int       # Pitch shift in Hz-like units (-50 to +50)
    volume: float          # Volume level (0.0-1.0)
    azure_style: str       # Azure TTS speaking style name
    style_degree: float    # Azure style intensity (0.01-2.0)
    emphasis: str          # SSML emphasis level (strong, moderate, reduced, none)


class VoiceModulator:
    """
    Maps emotion detection results to voice parameter configurations.
    All parameter deltas scale linearly with emotion intensity.
    """

    # Base vocal parameters
    BASE_RATE = 165     # WPM (Lowered from 175)
    BASE_PITCH = 0       # Delta from default
    BASE_VOLUME = 0.95  # 0.0-1.0 (increased for louder output)

    # Emotion-to-voice mapping: (rate_delta_range, pitch_delta_range, volume_delta_range, azure_style, emphasis)
    # Each range is (min_delta, max_delta) — interpolated by intensity
    # Azure styles chosen for en-US-DavisNeural compatibility:
    #   Supported: angry, cheerful, excited, friendly, hopeful,
    #              shouting, terrified, unfriendly, whispering, sad
    EMOTION_MAP = {
        "joy": {
            "rate": (10, 30),
            "pitch": (15, 40),
            "volume": (0.05, 0.15),
            "azure_style": "cheerful",
            "emphasis": "strong",
        },
        "surprise": {
            "rate": (15, 35),
            "pitch": (20, 50),
            "volume": (0.05, 0.15),
            "azure_style": "excited",
            "emphasis": "strong",
        },
        "anger": {
            "rate": (5, 20),
            "pitch": (-5, -15),
            "volume": (0.10, 0.20),
            "azure_style": "angry",
            "emphasis": "strong",
        },
        "fear": {
            "rate": (10, 25),
            "pitch": (10, 25),
            "volume": (-0.10, -0.20),
            "azure_style": "terrified",     # DavisNeural native style
            "emphasis": "reduced",
        },
        "sadness": {
            "rate": (-15, -35),
            "pitch": (-10, -30),
            "volume": (-0.05, -0.15),
            "azure_style": "sad",
            "emphasis": "reduced",
        },
        "disgust": {
            "rate": (-5, -15),
            "pitch": (-5, -15),
            "volume": (0.00, 0.10),
            "azure_style": "unfriendly",    # DavisNeural native style
            "emphasis": "moderate",
        },
        "neutral": {
            "rate": (0, 0),
            "pitch": (0, 0),
            "volume": (0.0, 0.0),
            "azure_style": "friendly",      # DavisNeural warm baseline
            "emphasis": "none",
        },
    }

    def _lerp(self, low: float, high: float, t: float) -> float:
        """Linear interpolation between low and high by factor t (0.0-1.0)."""
        return low + (high - low) * t

    def modulate(self, emotion_result: EmotionResult) -> VoiceConfig:
        """
        Generate a VoiceConfig from an EmotionResult.
        
        Parameter deltas are scaled by emotion intensity:
        - Low intensity (0.1-0.3): Subtle changes
        - Medium intensity (0.4-0.6): Noticeable changes
        - High intensity (0.7-1.0): Dramatic changes
        """
        emotion = emotion_result.emotion.lower()
        intensity = emotion_result.intensity

        # Look up the mapping, default to neutral
        mapping = self.EMOTION_MAP.get(emotion, self.EMOTION_MAP["neutral"])

        # Interpolate deltas based on intensity
        rate_delta = self._lerp(mapping["rate"][0], mapping["rate"][1], intensity)
        pitch_delta = self._lerp(mapping["pitch"][0], mapping["pitch"][1], intensity)
        volume_delta = self._lerp(mapping["volume"][0], mapping["volume"][1], intensity)

        # Compute final values with clamping
        final_rate = max(80, min(350, int(self.BASE_RATE + rate_delta)))
        final_pitch = int(pitch_delta)
        final_volume = max(0.1, min(1.0, round(self.BASE_VOLUME + volume_delta, 3)))

        # Azure style degree: map intensity 0-1 to 0.5-2.0 range
        style_degree = round(0.5 + intensity * 1.5, 2)
        style_degree = max(0.01, min(2.0, style_degree))

        return VoiceConfig(
            rate=final_rate,
            pitch_delta=final_pitch,
            volume=final_volume,
            azure_style=mapping["azure_style"],
            style_degree=style_degree,
            emphasis=mapping["emphasis"],
        )
