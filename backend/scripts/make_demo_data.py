"""
make_demo_data.py
------------------
Generates a small synthetic "real vs fake" audio dataset so the ENTIRE
pipeline (train -> serve -> score) runs immediately, with zero external
downloads.

IMPORTANT — read this:
This synthetic data only proves the plumbing works (feature extraction
-> training -> app -> live score). The "fake" class here is a crude
stand-in (locked-pitch tones + robotic amplitude modulation) for an
actual AI voice clone — it is NOT a real deepfake detection benchmark.

For a model that actually detects real AI-cloned voices, replace this
step with a real labeled dataset, e.g.:
  - ASVspoof 2019/2021 (https://www.asvspoof.org/) - registration required
  - "In the Wild" deepfake audio dataset
  - Your own recordings + samples from an open TTS tool (Coqui TTS,
    Tortoise-TTS), used ONLY to build your own labeled training pairs.

Just drop your own .wav files into data/real/ and data/fake/ and re-run
train_baseline.py — no other code changes needed.

Usage:
    python scripts/make_demo_data.py
"""

import os

import numpy as np
import soundfile as sf

SR = 16000
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_REAL = os.path.join(ROOT, "data", "real")
OUT_FAKE = os.path.join(ROOT, "data", "fake")
N_PER_CLASS = 40
DURATION = 2.0  # seconds


def make_real_like(rng, duration=DURATION, sr=SR):
    """Simulates natural speech-like audio: a wandering fundamental
    frequency (vibrato/jitter) with harmonics and soft noise."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0_base = rng.uniform(110, 220)
    wobble = 6 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
    jitter = rng.normal(0, 3, size=t.shape).cumsum() * 0.02
    f0 = f0_base + wobble + jitter

    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = np.zeros_like(t)
    for k in range(1, 6):
        detune = rng.normal(0, 0.003)
        signal += (1.0 / k) * np.sin(k * phase * (1 + detune))

    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t)
    noise = rng.normal(0, 0.02, size=t.shape)
    signal = signal * envelope + noise
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.7
    return signal.astype(np.float32)


def make_fake_like(rng, duration=DURATION, sr=SR):
    """Simulates crude synthetic-voice artifacts: unnaturally STABLE
    pitch (a common TTS/vocoder tell) + robotic step-wise amplitude."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0_base = rng.uniform(110, 220)
    f0 = np.full_like(t, f0_base) + rng.normal(0, 0.2, size=t.shape)

    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = np.zeros_like(t)
    for k in range(1, 6):
        signal += (1.0 / k) * np.sin(k * phase)

    step_rate = 20
    steps = np.floor(t * step_rate)
    envelope = 0.5 + 0.3 * (np.sin(2 * np.pi * steps / step_rate) > 0)
    signal = signal * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.7
    return signal.astype(np.float32)


def main():
    rng = np.random.default_rng(42)
    os.makedirs(OUT_REAL, exist_ok=True)
    os.makedirs(OUT_FAKE, exist_ok=True)

    for i in range(N_PER_CLASS):
        sf.write(os.path.join(OUT_REAL, f"real_{i:03d}.wav"), make_real_like(rng), SR)
        sf.write(os.path.join(OUT_FAKE, f"fake_{i:03d}.wav"), make_fake_like(rng), SR)

    print(f"Wrote {N_PER_CLASS} synthetic 'real' clips to {OUT_REAL}")
    print(f"Wrote {N_PER_CLASS} synthetic 'fake' clips to {OUT_FAKE}")
    print("\nThis is DEMO data only — swap in real labeled audio before")
    print("trusting this model's accuracy claims. See the module docstring.")


if __name__ == "__main__":
    main()
