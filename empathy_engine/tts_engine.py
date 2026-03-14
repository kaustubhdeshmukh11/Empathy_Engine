"""
TTS Engine Module

Dual-engine TTS with automatic engine selection:
- AzureTTSEngine: Uses Azure Cognitive Services Speech SDK with full SSML
  including <mstts:express-as> for native emotion styles and <prosody> for
  rate/pitch/volume modulation. Supports multi-sentence segmented SSML.
- OfflineTTSEngine: Uses pyttsx3 (SAPI5 on Windows) with basic prosody control.
"""

import os
import html
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

try:
    from empathy_engine.voice_modulator import VoiceConfig
except ImportError:
    from .voice_modulator import VoiceConfig


class BaseTTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str, voice_config: VoiceConfig, output_path: str) -> str:
        """Synthesize a single text block. Returns the output file path."""
        pass

    def synthesize_segmented(
        self,
        segments: List[Tuple[str, VoiceConfig]],
        output_path: str,
    ) -> str:
        """
        Synthesize multiple sentence segments, each with its own voice config.
        Default implementation just concatenates text and uses first config.
        Subclasses (Azure) override this with proper multi-block SSML.
        """
        # Fallback: combine text, use the first segment's config
        full_text = " ".join(seg[0] for seg in segments)
        config = segments[0][1] if segments else VoiceConfig(
            rate=165, pitch_delta=0, volume=0.95,
            azure_style="default", style_degree=1.0, emphasis="none",
        )
        return self.synthesize(full_text, config, output_path)

    @property
    @abstractmethod
    def engine_name(self) -> str:
        pass


# ──────────────────────────────────────────────────────────────
#  Azure TTS
# ──────────────────────────────────────────────────────────────
class AzureTTSEngine(BaseTTSEngine):
    """
    Azure Cognitive Services TTS with full SSML support.

    Supports:
    - Single-block SSML: one emotion for the whole text
    - Multi-block SSML:  per-sentence emotion styles with natural transitions
    """

    # Change VOICE_NAME to switch voice:
    #   Female: en-US-JennyNeural, en-US-AriaNeural, en-US-JaneNeural
    #   Male:   en-US-GuyNeural, en-US-DavisNeural, en-US-TonyNeural
    VOICE_NAME = "en-US-DavisNeural"

    def __init__(self, speech_key: str, speech_region: str):
        import azure.cognitiveservices.speech as speechsdk
        self._speechsdk = speechsdk
        self._speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region,
        )
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )

    @property
    def engine_name(self) -> str:
        return "Azure Cognitive Services TTS"

    # ── SSML helpers ──────────────────────────────────────────

    def _prosody_attrs(self, config: VoiceConfig) -> dict:
        """Convert a VoiceConfig into clamped SSML prosody attribute strings."""
        BASE_WPM = 165
        raw_rate_pct = int(((config.rate - BASE_WPM) / BASE_WPM) * 100)
        rate_pct = max(-30, min(30, raw_rate_pct))

        pitch_hz = max(-50, min(50, config.pitch_delta))

        raw_vol_pct = int((config.volume - 0.95) / 0.95 * 100)
        vol_pct = max(-20, min(20, raw_vol_pct))

        return {
            "rate": f"{rate_pct:+d}%",
            "pitch": f"{pitch_hz:+d}Hz",
            "volume": f"{vol_pct:+d}%",
        }

    def _sentence_ssml(self, text: str, config: VoiceConfig) -> str:
        """Build one <mstts:express-as> + <prosody> block for a sentence."""
        p = self._prosody_attrs(config)
        safe = html.escape(text)

        if config.azure_style and config.azure_style != "default":
            return (
                f'    <mstts:express-as style="{config.azure_style}" '
                f'styledegree="{config.style_degree}">\n'
                f'      <prosody rate="{p["rate"]}" pitch="{p["pitch"]}" '
                f'volume="{p["volume"]}">\n'
                f'        {safe}\n'
                f'      </prosody>\n'
                f'    </mstts:express-as>'
            )
        else:
            return (
                f'    <prosody rate="{p["rate"]}" pitch="{p["pitch"]}" '
                f'volume="{p["volume"]}">\n'
                f'      {safe}\n'
                f'    </prosody>'
            )

    def _build_ssml(self, text: str, config: VoiceConfig) -> str:
        """Build SSML for a single text block (backwards-compatible)."""
        body = self._sentence_ssml(text, config)
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"\n'
            f'    xmlns:mstts="https://www.w3.org/2001/mstts"\n'
            f'    xml:lang="en-US">\n'
            f'  <voice name="{self.VOICE_NAME}">\n'
            f'{body}\n'
            f'  </voice>\n'
            f'</speak>'
        )

    def _build_segmented_ssml(
        self, segments: List[Tuple[str, VoiceConfig]]
    ) -> str:
        """
        Build multi-sentence SSML for natural emotional transitions.

        Strategy for coherence — same voice, different feeling:
        - USE express-as:  Azure's neural style system naturally morphs
          tone/tempo/energy without changing the speaker's identity.
        - ZERO OUT pitch:  Pitch shifts are the #1 cause of sounding like
          a different person. Removed entirely from segmented mode.
        - SUBTLE rate:     Capped at ±12% — enough to feel faster/slower
          without jarring acoustic jumps.
        - SUBTLE volume:   Capped at ±8% — barely perceptible shift.
        - NO explicit breaks: Punctuation (. ! ?) already gives Azure a
          natural breath point between sentences.
        """
        BASE_WPM = 165
        blocks = []
        for sentence, config in segments:
            safe = html.escape(sentence)

            # Rate — widen clamp to ±25% for more tempo expression
            raw_rate = int(((config.rate - BASE_WPM) / BASE_WPM) * 100)
            rate_pct = max(-25, min(25, raw_rate))
            rate_str = f"{rate_pct:+d}%"

            # Volume — widen clamp to ±15%
            raw_vol = int((config.volume - 0.95) / 0.95 * 100)
            vol_pct = max(-15, min(15, raw_vol))
            vol_str = f"{vol_pct:+d}%"

            # Pitch — very tight clamp to prevent voice character breaking
            # but still allow subtle emotional bounce (0Hz was too flat)
            pitch_hz = max(-10, min(10, config.pitch_delta))
            pitch_str = f"{pitch_hz:+d}Hz"

            if config.azure_style and config.azure_style != "default":
                # Allow styledegree up to 2.0 (Azure max) for maximum expression
                blocks.append(
                    f'    <mstts:express-as style="{config.azure_style}" '
                    f'styledegree="{min(config.style_degree, 2.0)}">\n'
                    f'      <prosody rate="{rate_str}" pitch="{pitch_str}" '
                    f'volume="{vol_str}">{safe}</prosody>\n'
                    f'    </mstts:express-as>'
                )
            else:
                blocks.append(
                    f'    <prosody rate="{rate_str}" pitch="{pitch_str}" '
                    f'volume="{vol_str}">{safe}</prosody>'
                )

        body = "\n".join(blocks)
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"\n'
            f'    xmlns:mstts="https://www.w3.org/2001/mstts"\n'
            f'    xml:lang="en-US">\n'
            f'  <voice name="{self.VOICE_NAME}">\n'
            f'{body}\n'
            f'  </voice>\n'
            f'</speak>'
        )

    # ── Synthesis ─────────────────────────────────────────────

    def _do_synthesis(self, ssml: str, output_path: str) -> str:
        """Run the Azure SDK synthesis for an SSML string."""
        speechsdk = self._speechsdk
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
        )
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_path
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise RuntimeError(
                f"Azure TTS canceled: {cancellation.reason}. "
                f"Details: {cancellation.error_details}"
            )
        else:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")

    def synthesize(self, text: str, voice_config: VoiceConfig, output_path: str) -> str:
        """Synthesize a single text block."""
        ssml = self._build_ssml(text, voice_config)
        return self._do_synthesis(ssml, output_path)

    def synthesize_segmented(
        self,
        segments: List[Tuple[str, VoiceConfig]],
        output_path: str,
    ) -> str:
        """Synthesize multi-sentence text with per-sentence emotion styles."""
        ssml = self._build_segmented_ssml(segments)
        return self._do_synthesis(ssml, output_path)


# ──────────────────────────────────────────────────────────────
#  Offline pyttsx3 fallback
# ──────────────────────────────────────────────────────────────
class OfflineTTSEngine(BaseTTSEngine):
    """
    Offline TTS using pyttsx3 (SAPI5 on Windows).
    Supports rate and volume modulation.
    """

    def __init__(self):
        import pyttsx3
        self._pyttsx3 = pyttsx3
        engine = pyttsx3.init()
        self._default_rate = engine.getProperty("rate")
        self._default_volume = engine.getProperty("volume")
        engine.stop()

    @property
    def engine_name(self) -> str:
        return "pyttsx3 (Offline / SAPI5)"

    def synthesize(self, text: str, voice_config: VoiceConfig, output_path: str) -> str:
        engine = self._pyttsx3.init()
        try:
            engine.setProperty("rate", voice_config.rate)
            engine.setProperty("volume", voice_config.volume)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        finally:
            engine.stop()
        return output_path


# ──────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────
def get_tts_engine() -> BaseTTSEngine:
    """Return AzureTTSEngine if credentials exist, else pyttsx3 fallback."""
    from dotenv import load_dotenv
    load_dotenv()

    speech_key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    speech_region = os.environ.get("AZURE_SPEECH_REGION", "").strip()

    if speech_key and speech_region and speech_key != "your_azure_speech_key_here":
        try:
            engine = AzureTTSEngine(speech_key, speech_region)
            print(f"  [TTS] Using Azure Cognitive Services TTS (region: {speech_region})")
            return engine
        except Exception as e:
            print(f"  [TTS] Azure TTS init failed ({e}), falling back to offline engine.")

    print("  [TTS] Using pyttsx3 offline engine (set AZURE_SPEECH_KEY for better quality)")
    return OfflineTTSEngine()
