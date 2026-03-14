"""
The Empathy Engine — Main Entry Point

Usage:
    python main.py --mode cli       Run the interactive CLI
    python main.py --mode web       Start the FastAPI web server
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="🧠 The Empathy Engine — Emotionally Intelligent Text-to-Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode cli                  Interactive CLI
  python main.py --mode web                  Web UI at http://127.0.0.1:8000
  python main.py --mode web --port 3000      Custom port
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Run mode: 'cli' for interactive terminal, 'web' for FastAPI server (default: cli)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for web server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output audio files (default: output)",
    )

    args = parser.parse_args()

    if args.mode == "cli":
        from cli import run_cli
        run_cli(output_dir=args.output_dir)
    elif args.mode == "web":
        import uvicorn
        print(f"\n🧠 The Empathy Engine — Web Interface")
        print(f"   Starting at http://{args.host}:{args.port}")
        print(f"   API docs at http://{args.host}:{args.port}/docs")
        print(f"   Press Ctrl+C to stop\n")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
