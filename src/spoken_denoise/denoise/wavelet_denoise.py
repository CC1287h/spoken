from __future__ import annotations

import numpy as np
import pywt

from spoken_denoise.audio import match_length, peak_normalize


def wavelet_denoise(
    samples: np.ndarray,
    sample_rate: int,
    wavelet: str = "db4",
    level: int | None = None,
    threshold_scale: float = 1.0,
    mode: str = "soft",
) -> np.ndarray:
    """Wavelet threshold denoising using a robust noise estimate."""
    del sample_rate
    coeffs = pywt.wavedec(samples, wavelet=wavelet, mode="symmetric", level=level)
    if len(coeffs) <= 1:
        return samples.astype(np.float32, copy=False)

    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(max(len(samples), 2)))

    denoised_coeffs = [coeffs[0]]
    denoised_coeffs.extend(pywt.threshold(c, value=threshold, mode=mode) for c in coeffs[1:])
    enhanced = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode="symmetric")
    enhanced = match_length(samples, enhanced.astype(np.float32))
    return peak_normalize(enhanced)
