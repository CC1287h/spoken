from __future__ import annotations

import numpy as np
from scipy import signal

from spoken_denoise.audio import match_length, peak_normalize


def spectral_subtraction(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    noise_frames: int = 6,
    alpha: float = 1.5,
    beta: float = 0.02,
) -> np.ndarray:
    """Classical magnitude spectral subtraction."""
    _, _, spectrum = signal.stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(spectrum)
    phase = np.exp(1j * np.angle(spectrum))
    frame_count = max(1, min(noise_frames, magnitude.shape[1]))
    noise_profile = np.mean(magnitude[:, :frame_count], axis=1, keepdims=True)

    enhanced_magnitude = magnitude - alpha * noise_profile
    floor = beta * magnitude
    enhanced_magnitude = np.maximum(enhanced_magnitude, floor)

    _, enhanced = signal.istft(
        enhanced_magnitude * phase,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
    )
    enhanced = match_length(samples, enhanced.astype(np.float32))
    return peak_normalize(enhanced)
