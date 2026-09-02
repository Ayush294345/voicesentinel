"""
feature_extraction.py
----------------------
Turns a raw audio waveform into a fixed-length numeric feature vector
used by the spoof-detection classifier ("real human voice" vs
"synthetic/cloned voice").

Features (kept lightweight so this runs fast on CPU, no GPU needed):
  1. MFCCs + their delta - overall timbre/spectral shape.
  2. Spectral contrast - peak-vs-valley energy across frequency bands;
     synthetic vocoders often smooth this out unnaturally.
  3. Zero-crossing rate - cheap proxy for noisiness/breathiness.
  4. Pitch (F0) statistics + jitter - real speech has natural pitch
     "wobble"; many cloned voices are unnaturally smooth or unnaturally
     jittery depending on the vocoder.

This is a from-scratch baseline, not a production anti-spoofing model.
See README.md for how to upgrade to a pretrained embedding model
(SpeechBrain / wav2vec2) once you have real labeled training data
(e.g. ASVspoof).
"""

import numpy as np
import librosa

from core.audio_utils import SAMPLE_RATE


def extract_features(path_or_array, sr: int = SAMPLE_RATE) -> np.ndarray:
    if isinstance(path_or_array, str):
        y, _ = librosa.load(path_or_array, sr=sr, mono=True)
    else:
        y = path_or_array

    # Guard against silence / too-short clips
    if y is None or len(y) < sr * 0.3:
        pad = np.zeros(int(sr * 0.3), dtype=np.float32)
        if y is not None and len(y) > 0:
            pad[: len(y)] = y
        y = pad

    # --- 1. MFCC + delta ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    mfcc_feats = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1), delta.std(axis=1)]
    )

    # --- 2. Spectral contrast ---
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_feats = np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)])

    # --- 3. Zero-crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feats = np.array([zcr.mean(), zcr.std()])

    # --- 4. Pitch (F0) statistics via pYIN ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
        if len(f0_voiced) > 1:
            f0_mean = np.nanmean(f0_voiced)
            f0_std = np.nanstd(f0_voiced)
            jitter = np.nanmean(np.abs(np.diff(f0_voiced))) if len(f0_voiced) > 2 else 0.0
        else:
            f0_mean, f0_std, jitter = 0.0, 0.0, 0.0
    except Exception:
        f0_mean, f0_std, jitter = 0.0, 0.0, 0.0

    pitch_feats = np.array([f0_mean, f0_std, jitter])

    features = np.concatenate([mfcc_feats, contrast_feats, zcr_feats, pitch_feats])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def pitch_jitter_spectral_flatness(y: np.ndarray, sr: int = SAMPLE_RATE):
    """Cheap heuristic signals, used only by the fallback detector and the
    explainability panel (no model required)."""
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
        jitter = float(np.nanmean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 2 else 0.0
    except Exception:
        jitter = 0.0
    return jitter, flatness
