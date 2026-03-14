"""
FastAPI Web Interface for The Empathy Engine

Provides:
- GET /        → Beautiful web UI
- POST /synthesize → API endpoint (JSON in, JSON + audio out) 
- GET /audio/{filename} → Serve generated audio files
"""

import os
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from empathy_engine.engine import EmpathyEngine


# --- App Setup ---
app = FastAPI(
    title="The Empathy Engine",
    description="Emotionally Intelligent Text-to-Speech",
    version="1.0.0",
)

# Output directory for audio files
OUTPUT_DIR = Path("demos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")

# Engine (initialized on first request for faster startup)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = EmpathyEngine(output_dir=str(OUTPUT_DIR))
    return _engine


# --- Models ---
class SynthesizeRequest(BaseModel):
    text: str


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Process text and return emotion analysis + audio."""
    if not req.text or not req.text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Text cannot be empty"},
        )

    if len(req.text) > 5000:
        return JSONResponse(
            status_code=400,
            content={"error": "Text must be under 5000 characters"},
        )

    engine = get_engine()
    filename = f"empathy_{uuid.uuid4().hex[:8]}.wav"

    try:
        result = engine.process(req.text.strip(), output_filename=filename)
        # Replace local path with URL
        result["audio_url"] = f"/audio/{filename}"
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Synthesis failed: {str(e)}"},
        )


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve a generated audio file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Audio file not found"})
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "The Empathy Engine", "version": "1.0.0"}
