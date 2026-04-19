from __future__ import annotations

import numpy as np
from scipy import signal

from spoken_denoise.audio import match_length, peak_normalize


def frequency_masking(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    noise_frames: int = 6,
    threshold: float = 1.0,
    mask_floor: float = 0.05,
) -> np.ndarray:
    """Suppress low-SNR time-frequency bins with a soft spectral mask."""
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
    noise_power = np.mean(magnitude[:, :frame_count] ** 2, axis=1, keepdims=True)
    speech_power = np.maximum(magnitude**2 - noise_power, 0.0)

    ratio = speech_power / (noise_power + 1e-12)
    mask = ratio / (ratio + threshold + 1e-12)
    mask = np.maximum(mask, mask_floor)

    _, enhanced = signal.istft(
        magnitude * mask * phase,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
    )
    enhanced = match_length(samples, enhanced.astype(np.float32))
    return peak_normalize(enhanced)
