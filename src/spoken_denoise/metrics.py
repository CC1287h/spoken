from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np

from spoken_denoise.audio import align_signals

try:
    from pesq import pesq as pesq_score
except Exception:  # pragma: no cover - optional dependency
    pesq_score = None

try:
    from pystoi import stoi as stoi_score
except Exception:  # pragma: no cover - optional dependency
    stoi_score = None


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    clean, estimate = align_signals(clean, estimate)
    noise = clean - estimate
    signal_power = float(np.sum(clean**2))
    noise_power = float(np.sum(noise**2))
    if noise_power <= 1e-12:
        return 99.0
    return 10.0 * math.log10((signal_power + 1e-12) / (noise_power + 1e-12))


def mae(clean: np.ndarray, estimate: np.ndarray) -> float:
    clean, estimate = align_signals(clean, estimate)
    return float(np.mean(np.abs(clean - estimate)))


def mse(clean: np.ndarray, estimate: np.ndarray) -> float:
    clean, estimate = align_signals(clean, estimate)
    return float(np.mean((clean - estimate) ** 2))


def rmse(clean: np.ndarray, estimate: np.ndarray) -> float:
    return float(math.sqrt(mse(clean, estimate)))


def _resample_for_perceptual_metric(
    clean: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    target_rate: int = 16000,
) -> tuple[np.ndarray, np.ndarray, int]:
    clean, estimate = align_signals(clean, estimate)
    if sample_rate != target_rate:
        clean = librosa.resample(clean, orig_sr=sample_rate, target_sr=target_rate)
        estimate = librosa.resample(estimate, orig_sr=sample_rate, target_sr=target_rate)
    clean, estimate = align_signals(clean, estimate)
    return clean.astype(np.float32), estimate.astype(np.float32), target_rate


def pesq_wb(clean: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float | None:
    if pesq_score is None:
        return None
    try:
        clean_16k, estimate_16k, metric_rate = _resample_for_perceptual_metric(clean, estimate, sample_rate)
        return float(pesq_score(metric_rate, clean_16k, estimate_16k, "wb"))
    except Exception:
        return None


def stoi(clean: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float | None:
    if stoi_score is None:
        return None
    try:
        clean_16k, estimate_16k, metric_rate = _resample_for_perceptual_metric(clean, estimate, sample_rate)
        return float(stoi_score(clean_16k, estimate_16k, metric_rate, extended=False))
    except Exception:
        return None


def evaluate_signals(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    clean, noisy, enhanced = align_signals(clean, noisy, enhanced)
    noisy_snr = snr_db(clean, noisy)
    enhanced_snr = snr_db(clean, enhanced)

    return {
        "noisy_snr": noisy_snr,
        "enhanced_snr": enhanced_snr,
        "snr_improvement": enhanced_snr - noisy_snr,
        "mae": mae(clean, enhanced),
        "mse": mse(clean, enhanced),
        "rmse": rmse(clean, enhanced),
        "pesq": pesq_wb(clean, enhanced, sample_rate),
        "stoi": stoi(clean, enhanced, sample_rate),
    }


def evaluate_noisy_baseline(clean: np.ndarray, noisy: np.ndarray, sample_rate: int) -> dict[str, Any]:
    clean, noisy = align_signals(clean, noisy)
    noisy_snr = snr_db(clean, noisy)
    return {
        "noisy_snr": noisy_snr,
        "enhanced_snr": noisy_snr,
        "snr_improvement": 0.0,
        "mae": mae(clean, noisy),
        "mse": mse(clean, noisy),
        "rmse": rmse(clean, noisy),
        "pesq": pesq_wb(clean, noisy, sample_rate),
        "stoi": stoi(clean, noisy, sample_rate),
    }
