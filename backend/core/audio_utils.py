"""
audio_utils.py
--------------
Helpers for turning raw bytes (from the Streamlit mic recorder or a file
uploader) into a mono 16kHz numpy array, plus a spectrogram plot for the
explainability panel.
"""

import io

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def bytes_to_array(raw_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode WAV/most common audio bytes into a mono float32 array at `sr`."""
    y, file_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix to mono
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def load_wav_file(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def duration_seconds(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    return len(y) / float(sr)


def mel_spectrogram_db(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a mel-spectrogram in dB, ready for plotting."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    return librosa.power_to_db(S, ref=np.max)
