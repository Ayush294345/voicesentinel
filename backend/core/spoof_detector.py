"""
spoof_detector.py
------------------
Answers: "is this voice synthetic at all?"

Primary path: a trained GradientBoosting classifier (see
scripts/train_baseline.py) over the MFCC/spectral/pitch features in
feature_extraction.py.

Fallback path: if no trained model is found (e.g. you haven't run the
training script yet), falls back to a simple pitch-jitter /
spectral-flatness heuristic so the app never crashes — this is clearly
labeled "fallback heuristic mode" in the UI, never presented as the real
model.

NOTE ON HONESTY: the shipped model is trained on synthetic demo data
(scripts/make_demo_data.py generates crude sine-wave "fake" audio vs.
more natural "real" audio) purely so the whole pipeline works out of the
box. It proves the plumbing, not real-world deepfake detection accuracy.
Swap in a real labeled dataset (ASVspoof, "In the Wild", or your own
recordings + open TTS-generated pairs) and re-run
scripts/train_baseline.py before trusting this on real cloned voices.
"""

import os

import joblib
import numpy as np

from core.feature_extraction import extract_features, pitch_jitter_spectral_flatness

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.pkl")

_model = None
_model_load_attempted = False


def _get_model():
    global _model, _model_load_attempted
    if not _model_load_attempted:
        _model_load_attempted = True
        if os.path.exists(MODEL_PATH):
            try:
                _model = joblib.load(MODEL_PATH)
            except Exception:
                _model = None
    return _model


def analyze(y: np.ndarray, sr: int) -> dict:
    """
    Returns a normalized dict:
        {"is_spoof": bool, "confidence": float (0-1), "mode": "model" | "fallback"}
    `confidence` is the probability the model assigns to "fake/cloned".
    """
    model = _get_model()

    if model is not None:
        try:
            feats = extract_features(y, sr=sr)
            proba = model.predict_proba([feats])[0]
            confidence = float(proba[1])  # class 1 = fake/cloned
            return {"is_spoof": confidence >= 0.5, "confidence": confidence, "mode": "model"}
        except Exception:
            pass  # fall through to heuristic below

    # --- Fallback heuristic: unnaturally low pitch jitter + very flat
    # spectrum are simple, honest (if weak) proxies for synthetic vocoders.
    jitter, flatness = pitch_jitter_spectral_flatness(y, sr=sr)
    # Heuristic scoring, tuned loosely — treat as a rough signal only.
    score = 0.0
    if jitter < 1.5:
        score += 0.5
    if flatness > 0.15:
        score += 0.5
    return {"is_spoof": score >= 0.5, "confidence": float(score), "mode": "fallback"}
