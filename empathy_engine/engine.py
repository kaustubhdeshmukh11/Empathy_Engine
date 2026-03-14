"""
Empathy Engine Orchestrator

Ties together emotion detection, voice modulation, and TTS synthesis
into a single pipeline. Supports:
- Single-sentence mode (short text → one emotion)
- Segmented mode (long text → per-sentence emotion with multi-block SSML)

Rich intermediate step logging with reasoning at every stage.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from empathy_engine.emotion_detector import EmotionDetector, EmotionResult
    from empathy_engine.voice_modulator import VoiceModulator, VoiceConfig
    from empathy_engine.tts_engine import get_tts_engine, BaseTTSEngine
except ImportError:
    from .emotion_detector import EmotionDetector, EmotionResult
    from .voice_modulator import VoiceModulator, VoiceConfig
    from .tts_engine import get_tts_engine, BaseTTSEngine


# ── ANSI colours ──
_R = "\033[0m"
_B = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[96m"
_YEL = "\033[93m"
_GRN = "\033[92m"
_RED = "\033[91m"
_MAG = "\033[95m"
_BLU = "\033[94m"

EMOTION_COLORS = {
    "joy": _YEL, "surprise": _MAG, "anger": _RED,
    "fear": "\033[35m", "sadness": _BLU, "disgust": _GRN, "neutral": _DIM,
}
EMOTION_ICONS = {
    "joy": "😊", "surprise": "😲", "anger": "😡",
    "fear": "😰", "sadness": "😢", "disgust": "🤢", "neutral": "😐",
}
EMOTION_REASONING = {
    "joy":      "Joy → faster, brighter — excitement and positivity",
    "surprise": "Surprise → rapid, higher pitch — burst of astonishment",
    "anger":    "Anger → forceful, louder, lower — commanding and tense",
    "fear":     "Fear → quick, hushed — anxious whispered urgency",
    "sadness":  "Sadness → slower, quieter, lower — subdued weight of grief",
    "disgust":  "Disgust → slower, flatter — disdain and contempt",
    "neutral":  "Neutral → no modulation, natural baseline",
}


def _div(char: str = "─", w: int = 62) -> str:
    return _DIM + char * w + _R


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation boundaries.
    Handles ., !, ?, and ... with basic abbreviation guards.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences if sentences else [text.strip()]


class EmpathyEngine:
    """
    The Empathy Engine: Text → Emotion → Voice Modulation → Expressive TTS.

    For multi-sentence inputs, each sentence is individually analyzed and
    gets its own emotion style in a single Azure SSML document.
    """

    def __init__(self, output_dir: str = "output", tts_engine: Optional[BaseTTSEngine] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{_B}{_CYAN}🧠 Initializing The Empathy Engine...{_R}")
        self.detector = EmotionDetector()
        self.modulator = VoiceModulator()
        self.tts = tts_engine or get_tts_engine()
        print(f"{_GRN}✅ Empathy Engine ready!{_R}  TTS: {_B}{self.tts.engine_name}{_R}\n")

    # ── Verbose Logging ──────────────────────────────────────

    def _log_sentence_analysis(
        self, idx: int, total: int, sentence: str,
        emo: EmotionResult, vc: VoiceConfig,
    ) -> None:
        """Log the emotion + modulation for one sentence."""
        color = EMOTION_COLORS.get(emo.emotion, _DIM)
        icon = EMOTION_ICONS.get(emo.emotion, "❓")
        reason = EMOTION_REASONING.get(emo.emotion, "")

        # Intensity mini-bar
        bar_len = int(emo.intensity * 16)
        bar = "█" * bar_len + "░" * (16 - bar_len)

        print(f"\n  {_B}[Sentence {idx}/{total}]{_R}")
        print(f"  {_DIM}\"{_CYAN}{sentence}{_DIM}\"{_R}")
        print(f"  Emotion   : {color}{_B}{icon} {emo.emotion.upper()}{_R}  "
              f"({emo.confidence:.1%})")
        print(f"  Intensity : [{color}{bar}{_R}] {emo.intensity:.0%}   "
              f"Valence: {_YEL if emo.valence >= 0 else _BLU}"
              f"{emo.valence:+.2f}{_R}")

        # Top-3 scores
        top3 = sorted(emo.all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        scores_str = "  ".join(
            f"{EMOTION_COLORS.get(l, _DIM)}{l}={s:.0%}{_R}" for l, s in top3
        )
        print(f"  Scores    : {scores_str}")

        # Modulation
        print(f"  Reasoning : {_DIM}{reason}{_R}")
        print(f"  Voice     : rate={vc.rate} WPM  pitch={vc.pitch_delta:+d}Hz  "
              f"vol={vc.volume:.2f}  "
              f"style={color}{vc.azure_style}({vc.style_degree}){_R}")

    # ── Core Pipeline ────────────────────────────────────────

    def process(
        self, text: str,
        output_filename: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline: text → sentence split → per-sentence emotion → SSML → audio.

        Returns a dict with:
          text, emotion (dominant), confidence, intensity, valence,
          all_scores, voice_config (dominant), audio_path, tts_engine,
          segments (list of per-sentence details)
        """
        sentences = _split_sentences(text)
        is_multi = len(sentences) > 1

        if verbose:
            print(f"\n{_div('═')}")
            print(f"{_B}  INPUT TEXT{_R}  ({len(sentences)} sentence{'s' if is_multi else ''})")
            print(f"  \"{_CYAN}{text}{_R}\"")
            print(_div('═'))

            if is_multi:
                print(f"\n{_B}  ╔══ SENTENCE-LEVEL SEGMENTATION ══╗{_R}")
            else:
                print(f"\n{_B}  STEP 1–2 — EMOTION DETECTION & VOICE MODULATION{_R}")
                print(_div())

        # Analyze each sentence
        segment_results = []
        for i, sentence in enumerate(sentences, 1):
            emo = self.detector.detect(sentence)
            vc = self.modulator.modulate(emo)
            segment_results.append((sentence, emo, vc))

            if verbose:
                self._log_sentence_analysis(i, len(sentences), sentence, emo, vc)

        if verbose and is_multi:
            print(f"\n{_B}  ╚══ END SEGMENTATION ══╝{_R}")

        # Determine dominant emotion (highest intensity × confidence)
        best_seg = max(segment_results, key=lambda s: s[1].intensity * s[1].confidence)
        dominant_emo = best_seg[1]
        dominant_vc = best_seg[2]

        # Synthesize
        if output_filename is None:
            output_filename = f"empathy_{uuid.uuid4().hex[:8]}.wav"
        output_path = str(self.output_dir / output_filename)

        if verbose:
            print(f"\n{_div()}")
            print(f"{_B}  STEP 3 — TTS SYNTHESIS{_R}")
            print(_div())
            if is_multi:
                print(f"  Mode      : {_GRN}Segmented SSML{_R} "
                      f"({len(sentences)} blocks, per-sentence emotions)")
            else:
                print(f"  Mode      : Single-block SSML")

        # Use segmented synthesis if multi-sentence
        if is_multi and len(segment_results) > 1:
            segments_for_tts = [(s, vc) for s, _, vc in segment_results]
            audio_path = self.tts.synthesize_segmented(segments_for_tts, output_path)
        else:
            audio_path = self.tts.synthesize(text, dominant_vc, output_path)

        if verbose:
            print(f"  Engine    : {_B}{self.tts.engine_name}{_R}")
            print(f"  Output    : {_GRN}{audio_path}{_R}")
            print(_div())
            print()

        # Build segments list for API response
        segments_data = []
        for sentence, emo, vc in segment_results:
            segments_data.append({
                "sentence": sentence,
                "emotion": emo.emotion,
                "confidence": emo.confidence,
                "intensity": emo.intensity,
                "valence": emo.valence,
                "voice_config": {
                    "rate": vc.rate, "pitch_delta": vc.pitch_delta,
                    "volume": vc.volume, "azure_style": vc.azure_style,
                    "style_degree": vc.style_degree, "emphasis": vc.emphasis,
                },
            })

        return {
            "text": text,
            "emotion": dominant_emo.emotion,
            "confidence": dominant_emo.confidence,
            "intensity": dominant_emo.intensity,
            "valence": dominant_emo.valence,
            "all_scores": dominant_emo.all_scores,
            "voice_config": {
                "rate": dominant_vc.rate, "pitch_delta": dominant_vc.pitch_delta,
                "volume": dominant_vc.volume, "azure_style": dominant_vc.azure_style,
                "style_degree": dominant_vc.style_degree,
                "emphasis": dominant_vc.emphasis,
            },
            "audio_path": audio_path,
            "tts_engine": self.tts.engine_name,
            "segments": segments_data,
            "is_segmented": is_multi,
        }
