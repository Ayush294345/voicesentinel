"""
speaker_verifier.py
--------------------
Answers a different question from spoof_detector.py: "is this the SAME
PERSON's voice as the one enrolled?" The risk engine combines both.

Implementation note (read before trusting this on real impostors):
This uses a lightweight, dependency-free "voiceprint" — the mean/std of
MFCCs over the clip, L2-normalized — compared by cosine similarity. It
is a real, working signal, but it is a much weaker speaker embedding
than a trained deep model like resemblyzer or SpeechBrain's
spkrec-ecapa-voxceleb. Those give materially better impostor rejection
but pull in torch + multi-hundred-MB model downloads. This trade-off was
made deliberately to keep `pip install` fast and reliable on a laptop.
Swap in resemblyzer's VoiceEncoder().embed_utterance(wav) here later for
a real accuracy jump if you have time/bandwidth for the heavier install.
"""

import numpy as np

from core.feature_extraction import extract_features

SIMILARITY_THRESHOLD = 0.75  # tune this live while testing


def compute_embedding(y: np.ndarray, sr: int) -> np.ndarray:
    """A compact voiceprint derived from the same MFCC pipeline as the
    spoof detector, L2-normalized so cosine similarity is well-behaved."""
    feats = extract_features(y, sr=sr)
    norm = np.linalg.norm(feats)
    if norm < 1e-8:
        return feats
    return feats / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def verify(embedding: np.ndarray, enrolled_embedding: np.ndarray) -> dict:
    similarity = cosine_similarity(embedding, enrolled_embedding)
    return {
        "similarity": similarity,
        "same_speaker": similarity >= SIMILARITY_THRESHOLD,
    }
