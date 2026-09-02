"""
main.py
-------
VoiceSentinel backend — FastAPI serves both:
  1. The REST API (/api/...) used by the frontend
  2. The static frontend (index.html) at "/"

Run with:
    python main.py

Then open http://localhost:8000 in your browser.
"""

import base64
import io
import os

import librosa.display
import matplotlib

matplotlib.use("Agg")  # headless backend, no GUI window needed
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from core import audit_log, db, risk_engine
from core.audio_utils import SAMPLE_RATE, bytes_to_array, mel_spectrogram_db
from core.feature_extraction import pitch_jitter_spectral_flatness
from core.report import generate_incident_report
from core.speaker_verifier import compute_embedding, verify as speaker_verify
from core.spoof_detector import analyze as spoof_analyze

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = FastAPI(title="VoiceSentinel API")

# Not strictly needed since the frontend is served from the same origin,
# but harmless and useful if you ever split them onto different ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

@app.get("/ping")
def ping():
    return {"status": "ok"}


def _spectrogram_png_base64(y, sr) -> str:
    S_db = mel_spectrogram_db(y, sr)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/api/enroll")
async def enroll(username: str = Form(...), file: UploadFile = File(...)):
    username = username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")
    raw = await file.read()
    try:
        y = bytes_to_array(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {exc}")

    embedding = compute_embedding(y, SAMPLE_RATE)
    db.save_enrollment(username, embedding)
    duration = len(y) / SAMPLE_RATE
    return {
        "status": "ok",
        "username": username,
        "duration_seconds": round(duration, 2),
        "short_clip_warning": duration < 2,
    }


@app.get("/api/users")
async def get_users():
    return {"users": db.list_users()}


@app.post("/api/analyze")
async def analyze(username: str = Form(""), file: UploadFile = File(...)):
    raw = await file.read()
    try:
        y = bytes_to_array(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {exc}")

    spoof_result = spoof_analyze(y, SAMPLE_RATE)

    speaker_similarity = None
    username = username.strip()
    if username:
        enrolled_emb = db.get_embedding(username)
        if enrolled_emb is not None:
            emb = compute_embedding(y, SAMPLE_RATE)
            speaker_similarity = speaker_verify(emb, enrolled_emb)["similarity"]

    risk = risk_engine.compute_risk(spoof_result["confidence"], speaker_similarity)
    jitter, flatness = pitch_jitter_spectral_flatness(y, SAMPLE_RATE)
    spectrogram_b64 = _spectrogram_png_base64(y, SAMPLE_RATE)

    entry = audit_log.record_detection(
        username=username if username else "unknown",
        decision=risk["decision"],
        risk_score=risk["risk_score"],
        spoof_confidence=spoof_result["confidence"],
        speaker_similarity=speaker_similarity,
        detector_mode=spoof_result["mode"],
    )

    return {
        "audit_id": entry["id"],
        "decision": risk["decision"],
        "risk_score": risk["risk_score"],
        "spoof_confidence": spoof_result["confidence"],
        "speaker_similarity": speaker_similarity,
        "detector_mode": spoof_result["mode"],
        "jitter": jitter,
        "flatness": flatness,
        "spectrogram_png_base64": spectrogram_b64,
    }


@app.get("/api/audit-log")
async def get_audit_log():
    return {"rows": db.get_all_audit_rows()}


@app.get("/api/verify-chain")
async def verify_chain():
    return audit_log.verify_chain()


@app.get("/api/report/{audit_id}")
async def get_report(audit_id: int):
    entry = db.get_audit_row(audit_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Audit entry not found.")
    pdf_bytes = generate_incident_report(entry)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="incident_report_{audit_id}.pdf"'},
    )


# --- Serve the frontend last, so it doesn't shadow the /api/* routes ---
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
