"""
CLI Interface for The Empathy Engine

Interactive command-line loop: enter text → see detected emotion → hear the result.
"""

import argparse
import os
import sys
from pathlib import Path


def print_banner():
    """Print a nice startup banner."""
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🧠  T H E   E M P A T H Y   E N G I N E  🎙️             ║
║                                                              ║
║    Emotionally Intelligent Text-to-Speech                    ║
║    ─────────────────────────────────────                     ║
║    Type text → Detect emotion → Generate expressive speech   ║
║                                                              ║
║    Commands:                                                 ║
║      Type any text and press Enter to synthesize             ║
║      Type 'quit' or 'exit' to stop                           ║
║      Type 'demo' to run sample emotional texts               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


DEMO_TEXTS = [
    "I just got promoted at work! This is the best day of my life!",
    "I can't believe they cancelled the event. I'm so disappointed.",
    "The weather forecast for tomorrow is partly cloudy with a high of 72 degrees.",
    "What?! You're telling me we won the lottery?!",
    "I'm terrified of what might happen if we don't act now.",
    "This is absolutely disgusting behavior. I refuse to tolerate it.",
    "I'm furious about the way they treated us. This is unacceptable!",
]


def print_result(result: dict):
    """Pretty-print the processing result."""
    # Emotion colors (ANSI)
    emotion_colors = {
        "joy": "\033[93m",       # Yellow
        "surprise": "\033[95m",  # Magenta
        "anger": "\033[91m",     # Red
        "fear": "\033[35m",      # Purple
        "sadness": "\033[94m",   # Blue
        "disgust": "\033[32m",   # Green
        "neutral": "\033[37m",   # White
    }
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"

    emotion = result["emotion"]
    color = emotion_colors.get(emotion, "\033[37m")

    # Intensity bar
    intensity = result["intensity"]
    bar_len = int(intensity * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)

    print(f"\n  {bold}── Analysis Result ──{reset}")
    print(f"  {dim}Emotion:{reset}     {color}{bold}{emotion.upper()}{reset}  ({result['confidence']:.1%} confidence)")
    print(f"  {dim}Intensity:{reset}   [{bar}] {intensity:.1%}")
    print(f"  {dim}Valence:{reset}     {result['valence']:+.2f}")
    print()
    vc = result["voice_config"]
    print(f"  {bold}── Voice Config ──{reset}")
    print(f"  {dim}Rate:{reset}        {vc['rate']} WPM")
    print(f"  {dim}Pitch Δ:{reset}     {vc['pitch_delta']:+d} Hz")
    print(f"  {dim}Volume:{reset}      {vc['volume']:.2f}")
    print(f"  {dim}Azure Style:{reset} {vc['azure_style']} (degree: {vc['style_degree']})")
    print(f"  {dim}Emphasis:{reset}    {vc['emphasis']}")
    print()
    print(f"  {bold}── Output ──{reset}")
    print(f"  {dim}TTS Engine:{reset}  {result['tts_engine']}")
    print(f"  {dim}Audio File:{reset}  {result['audio_path']}")
    print(f"  {'─' * 50}")


def run_cli(output_dir: str = "output"):
    """Run the interactive CLI."""
    print_banner()

    from empathy_engine.engine import EmpathyEngine
    engine = EmpathyEngine(output_dir=output_dir)

    print(f"\n  Ready! Output directory: {os.path.abspath(output_dir)}\n")

    while True:
        try:
            text = input("  📝 Enter text (or 'quit'/'demo'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  👋 Goodbye!")
            break

        if not text:
            continue

        if text.lower() in ("quit", "exit", "q"):
            print("\n  👋 Goodbye!")
            break

        if text.lower() == "demo":
            print(f"\n  🎬 Running demo with {len(DEMO_TEXTS)} sample texts...\n")
            for i, demo_text in enumerate(DEMO_TEXTS, 1):
                print(f"  [{i}/{len(DEMO_TEXTS)}] \"{demo_text}\"")
                try:
                    result = engine.process(demo_text)
                    print_result(result)
                except Exception as e:
                    print(f"  ❌ Error: {e}\n")
            continue

        try:
            result = engine.process(text)
            print_result(result)
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="The Empathy Engine - CLI")
    parser.add_argument("--output-dir", default="output", help="Directory for output audio files")
    args = parser.parse_args()
    run_cli(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
