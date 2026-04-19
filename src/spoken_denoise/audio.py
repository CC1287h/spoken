from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a mono wav file as float32."""
    samples, sample_rate = librosa.load(path, sr=target_sr, mono=True)
    return samples.astype(np.float32, copy=False), int(sample_rate)


def save_audio(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Save audio after clipping to the valid float wav range."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    sf.write(output_path, clipped, sample_rate)


def match_length(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Trim or zero-pad candidate to match reference length."""
    target_length = len(reference)
    if len(candidate) == target_length:
        return candidate
    if len(candidate) > target_length:
        return candidate[:target_length]
    return np.pad(candidate, (0, target_length - len(candidate)))


def align_signals(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim all signals to the shortest length for fair metric calculation."""
    min_length = min(len(signal) for signal in signals)
    return tuple(np.asarray(signal[:min_length], dtype=np.float32) for signal in signals)


def peak_normalize(samples: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Normalize only when a signal would otherwise clip."""
    max_abs = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if max_abs <= peak or max_abs == 0.0:
        return samples.astype(np.float32, copy=False)
    return (samples / max_abs * peak).astype(np.float32)
