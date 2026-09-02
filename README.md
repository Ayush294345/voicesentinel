# VoiceSentinel (HTML + FastAPI version)

Same detection engine as the Streamlit build, now with a plain HTML/JS
frontend and a FastAPI backend — no Streamlit, no React/Node.js required.
One Python command runs everything.

## Project structure

```
voicesentinel_web/
├── backend/
│   ├── main.py                # FastAPI app — serves the API AND the frontend
│   ├── core/                  # same detection engine as the Streamlit build
│   │   ├── audio_utils.py
│   │   ├── feature_extraction.py
│   │   ├── spoof_detector.py
│   │   ├── speaker_verifier.py
│   │   ├── risk_engine.py
│   │   ├── audit_log.py
│   │   ├── db.py
│   │   └── report.py
│   ├── scripts/
│   │   ├── make_demo_data.py
│   │   └── train_baseline.py
│   ├── data/                  # training clips + sqlite db (created on first run)
│   ├── models/                # trained model saved here
│   └── requirements.txt
└── frontend/
    └── index.html             # single-file HTML/CSS/JS app, no build step
```

## 1. Setup (VS Code, Windows)

```powershell
cd voicesentinel_web/backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

(macOS/Linux: `source venv/bin/activate` instead of the `.\venv\...` line.)

## 2. Generate demo data & train the model

```powershell
python scripts/make_demo_data.py
python scripts/train_baseline.py
```

(Same as the Streamlit build — see the note in the module docstrings:
this demo data is a synthetic stand-in so the pipeline runs immediately.
Swap in real labeled audio in `data/real/` and `data/fake/`, then re-run
`train_baseline.py`, to get real detection accuracy.)

## 3. Run the app

```powershell
python main.py
```

Then open **http://localhost:8000** in your browser. That's it — one
process serves both the API and the web page, no separate frontend
server needed.

## 4. Using it

1. **Enroll Voice tab** — type a name, click the mic button to record
   (browser will ask mic permission), click again to stop, then **Save
   Voice Profile**. (Or use the file upload box instead of the mic.)
2. **Live Detection tab** — pick the enrolled name, record/upload a new
   clip, click **Analyze**. You'll get a risk score, spectrogram, and a
   Legitimate/Attack Detected verdict. An attack verdict shows a
   **Download Incident Report (PDF)** button.
3. **Dashboard / Audit Log tab** — table + charts of everything analyzed
   so far, plus a **Verify Audit Trail** button that recomputes the
   SHA-256 hash chain end-to-end to confirm nothing was tampered with.

## Notes on the pieces that are simplified/simulated

Same as the Streamlit build — see the docstrings in `core/spoof_detector.py`
and `core/speaker_verifier.py`:
- Spoof detection model is trained on **synthetic demo data** by default —
  proves the pipeline works, not a real-world accuracy claim until you
  swap in real labeled audio (e.g. ASVspoof).
- Speaker verification uses a lightweight MFCC-based voiceprint (cosine
  similarity), not a trained deep embedding model like resemblyzer —
  fine for a demo, weaker than a deep model at rejecting real impostors.
- The audit log's "tamper-evident" claim is a real SHA-256 hash chain,
  not literally a blockchain.

## Recording notes

The mic recorder in `index.html` encodes real `.wav` audio directly in
the browser using the Web Audio API (no browser-codec dependency), so it
works reliably across Chrome/Edge/Firefox without needing ffmpeg on the
frontend side. If mic permission is denied or unavailable, the file
upload box next to it is a full fallback — record on your phone and
upload the `.wav`.
