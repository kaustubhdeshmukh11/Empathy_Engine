"""
Emotion Detection Module

Combines VADER sentiment analysis (for intensity/valence scoring) with a
HuggingFace DistilRoBERTa model (for 7-class granular emotion classification).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class EmotionResult:
    """Result of emotion detection on a piece of text."""
    emotion: str              # Primary emotion label (joy, anger, sadness, etc.)
    confidence: float         # Confidence score for the primary emotion (0.0-1.0)
    intensity: float          # Emotional intensity derived from VADER (0.0-1.0)
    valence: float            # Sentiment valence from VADER (-1.0 to 1.0)
    all_scores: Dict[str, float] = field(default_factory=dict)  # All emotion scores


class EmotionDetector:
    """
    Dual-model emotion detection:
    1. VADER — fast, rule-based sentiment → intensity & valence
    2. DistilRoBERTa — transformer-based 7-class emotion classification
    """

    # Mapping from model labels to our canonical emotion set
    EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        self._transformer_pipeline = None  # Lazy-loaded

    def _load_transformer(self):
        """Lazy-load the transformer pipeline to reduce startup time."""
        if self._transformer_pipeline is None:
            from transformers import pipeline
            print("  [EmotionDetector] Loading emotion classification model (first run may download ~300MB)...")
            self._transformer_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,  # Return all class scores
                truncation=True,
            )
            print("  [EmotionDetector] Model loaded successfully.")
        return self._transformer_pipeline

    def detect(self, text: str) -> EmotionResult:
        """
        Analyze input text and return an EmotionResult with:
        - emotion: primary emotion label
        - confidence: transformer confidence for that label
        - intensity: 0.0-1.0 from VADER abs(compound)
        - valence: -1.0 to 1.0 from VADER compound score
        - all_scores: full distribution from transformer
        """
        # --- VADER Analysis ---
        vader_scores = self._vader.polarity_scores(text)
        valence = vader_scores["compound"]          # -1.0 to 1.0
        intensity = min(abs(valence) * 1.25, 1.0)   # Scale up slightly, cap at 1.0

        # --- Transformer Analysis ---
        pipe = self._load_transformer()
        results = pipe(text)[0]  # List of {label, score} dicts

        # Build scores dict and find top emotion
        all_scores = {}
        top_emotion = "neutral"
        top_score = 0.0

        for item in results:
            label = item["label"].lower()
            score = item["score"]
            all_scores[label] = score
            if score > top_score:
                top_score = score
                top_emotion = label

        # If VADER shows very low emotion but transformer is unsure, bias toward neutral
        if intensity < 0.1 and top_score < 0.4:
            top_emotion = "neutral"
            top_score = all_scores.get("neutral", top_score)

        return EmotionResult(
            emotion=top_emotion,
            confidence=round(top_score, 4),
            intensity=round(intensity, 4),
            valence=round(valence, 4),
            all_scores={k: round(v, 4) for k, v in all_scores.items()},
        )
