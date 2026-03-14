"""
TTS Engine Module — Dual-Strategy SSML Architecture

Two distinct SSML strategies depending on context:

SINGLE-SENTENCE MODE (max expressiveness):
  - Full <mstts:express-as> with uncapped styledegree (up to 2.0)
  - Full <prosody> with wide clamps: ±30% rate, ±50Hz pitch, ±20% volume
  - Azure handles one emotion block — no continuity concerns
  → Result: Punchy, dramatic, theatrical delivery

SEGMENTED MODE (multi-sentence continuity):
  - <mstts:express-as> carries all the emotional weight (styledegree up to 2.0)
  - <prosody> is kept minimal: ±8% rate, ±5Hz pitch, ±6% volume
  - Azure naturally transitions between express-as blocks within one voice
  → Result: Same speaker, shifting emotions, smooth continuous flow
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
#  Azure TTS — Dual-Strategy SSML
# ──────────────────────────────────────────────────────────────
class AzureTTSEngine(BaseTTSEngine):
    """
    Azure Cognitive Services TTS with dual-strategy SSML.

    Voice: en-US-DavisNeural (male)
    Supported express-as styles: angry, cheerful, excited, friendly,
        hopeful, shouting, terrified, unfriendly, whispering, sad

    Strategy 1 — SINGLE SENTENCE (maximum expressiveness):
        Full prosody + full express-as. Go theatrical.

    Strategy 2 — SEGMENTED (multi-sentence continuity):
        Express-as carries emotion, prosody is whisper-quiet.
        Same speaker, shifting feelings.
    """

    # ── Voice Selection ──────────────────────────────────────
    # DavisNeural: best male voice for express-as style support
    # Alternative: en-US-JennyNeural (female, also great style support)
    VOICE_NAME = "en-US-DavisNeural"

    BASE_WPM = 165
    BASE_VOL = 0.95

    def __init__(self, speech_key: str, speech_region: str):
        import azure.cognitiveservices.speech as speechsdk
        self._speechsdk = speechsdk
        self._speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region,
        )
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )

    @property
    def engine_name(self) -> str:
        return "Azure Cognitive Services TTS"

    # ══════════════════════════════════════════════════════════
    #  STRATEGY 1: Single-Sentence — Maximum Expressiveness
    # ══════════════════════════════════════════════════════════

    def _build_single_ssml(self, text: str, config: VoiceConfig) -> str:
        """
        Build SSML for a SINGLE sentence/text block.

        Goes all-out with expressiveness:
        - Full pitch range (±50Hz) for dramatic vocal shifts
        - Full rate range (±30%) for speed variation
        - Full volume range (±20%) for dynamic loudness
        - Express-as with uncapped styledegree (up to 2.0)

        No continuity concerns — this is one emotion, one shot.
        """
        safe = html.escape(text)

        # Wide clamps — let the modulator's full range through
        raw_rate = int(((config.rate - self.BASE_WPM) / self.BASE_WPM) * 100)
        rate_pct = max(-30, min(30, raw_rate))

        pitch_hz = max(-50, min(50, config.pitch_delta))

        raw_vol = int((config.volume - self.BASE_VOL) / self.BASE_VOL * 100)
        vol_pct = max(-20, min(20, raw_vol))

        rate_str = f"{rate_pct:+d}%"
        pitch_str = f"{pitch_hz:+d}Hz"
        vol_str = f"{vol_pct:+d}%"

        # Build the inner content
        if config.azure_style and config.azure_style != "default":
            body = (
                f'    <mstts:express-as style="{config.azure_style}" '
                f'styledegree="{config.style_degree}">\n'
                f'      <prosody rate="{rate_str}" pitch="{pitch_str}" '
                f'volume="{vol_str}">\n'
                f'        {safe}\n'
                f'      </prosody>\n'
                f'    </mstts:express-as>'
            )
        else:
            body = (
                f'    <prosody rate="{rate_str}" pitch="{pitch_str}" '
                f'volume="{vol_str}">\n'
                f'      {safe}\n'
                f'    </prosody>'
            )

        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"\n'
            f'    xmlns:mstts="https://www.w3.org/2001/mstts"\n'
            f'    xml:lang="en-US">\n'
            f'  <voice name="{self.VOICE_NAME}">\n'
            f'{body}\n'
            f'  </voice>\n'
            f'</speak>'
        )

    # ══════════════════════════════════════════════════════════
    #  STRATEGY 2: Segmented — Continuity-First
    # ══════════════════════════════════════════════════════════

    def _build_segmented_ssml(
        self, segments: List[Tuple[str, VoiceConfig]]
    ) -> str:
        """
        Build multi-sentence SSML prioritizing voice continuity.

        KEY INSIGHT: Azure's express-as changes the voice CHARACTER on
        every supported voice — cheerful Davis sounds like a different
        person than sad Davis. This is great for single sentences but
        breaks multi-sentence continuity.

        Solution: Use ONLY <prosody> for segmented mode. The emotion
        comes through rate, pitch, and volume shifts — subtle enough
        to keep the same speaker, expressive enough to feel the mood:

        - rate:    ±20% — clearly faster/slower
        - pitch:   ±15Hz — noticeable but same throat
        - volume:  ±12% — louder/quieter without jarring
        - No express-as: Same voice character throughout
        - No breaks: Natural punctuation handles pacing
        """
        blocks = []
        for sentence, config in segments:
            safe = html.escape(sentence)

            # Moderate prosody — expressive but same speaker
            raw_rate = int(((config.rate - self.BASE_WPM) / self.BASE_WPM) * 100)
            rate_pct = max(-20, min(20, raw_rate))
            rate_str = f"{rate_pct:+d}%"

            pitch_hz = max(-15, min(15, config.pitch_delta))
            pitch_str = f"{pitch_hz:+d}Hz"

            raw_vol = int((config.volume - self.BASE_VOL) / self.BASE_VOL * 100)
            vol_pct = max(-12, min(12, raw_vol))
            vol_str = f"{vol_pct:+d}%"

            # Prosody-only — no express-as, same voice throughout
            blocks.append(
                f'    <prosody rate="{rate_str}" pitch="{pitch_str}" '
                f'volume="{vol_str}">{safe}</prosody>'
            )

        body = "\n".join(blocks)
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"\n'
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
        """Synthesize a single text block — MAXIMUM expressiveness."""
        ssml = self._build_single_ssml(text, voice_config)
        return self._do_synthesis(ssml, output_path)

    def synthesize_segmented(
        self,
        segments: List[Tuple[str, VoiceConfig]],
        output_path: str,
    ) -> str:
        """Synthesize multi-sentence text — CONTINUITY-first strategy."""
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
