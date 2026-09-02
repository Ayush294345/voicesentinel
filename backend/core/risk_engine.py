"""
risk_engine.py
---------------
Combines the two independent signals into a single 0-100 risk score and
a Legitimate / Attack Detected decision.

    risk_score = 0.6 * spoof_confidence * 100 + 0.4 * (1 - speaker_similarity) * 100

If no speaker is enrolled/selected, we fall back to weighting purely on
the spoof-detector signal (speaker check contributes nothing rather than
penalizing arbitrarily).

Thresholds (tune live while testing with real vs synthetic clips):
    risk_score < 35   -> Legitimate
    risk_score >= 35  -> Attack Detected
"""

from typing import Optional

RISK_THRESHOLD = 35.0


def compute_risk(spoof_confidence: float, speaker_similarity: Optional[float]) -> dict:
    if speaker_similarity is None:
        risk_score = spoof_confidence * 100
    else:
        risk_score = 0.6 * spoof_confidence * 100 + 0.4 * (1 - speaker_similarity) * 100
    risk_score = max(0.0, min(100.0, risk_score))

    decision = "Attack Detected" if risk_score >= RISK_THRESHOLD else "Legitimate"
    return {"risk_score": risk_score, "decision": decision}
