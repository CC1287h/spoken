from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spoken_denoise.denoise import frequency_masking, spectral_subtraction, wavelet_denoise

DenoiseFunction = Callable[..., np.ndarray]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    function: DenoiseFunction
    default_params: dict
    search_grid: list[dict]


METHODS: dict[str, MethodSpec] = {
    "spectral_subtraction": MethodSpec(
        name="spectral_subtraction",
        function=spectral_subtraction,
        default_params={"alpha": 2.0, "beta": 0.02, "noise_frames": 6},
        search_grid=[
            {"alpha": 1.0, "beta": 0.02, "noise_frames": 6},
            {"alpha": 1.5, "beta": 0.02, "noise_frames": 6},
            {"alpha": 2.0, "beta": 0.02, "noise_frames": 6},
        ],
    ),
    "wavelet_denoise": MethodSpec(
        name="wavelet_denoise",
        function=wavelet_denoise,
        default_params={"wavelet": "db4", "threshold_scale": 1.0, "mode": "soft"},
        search_grid=[
            {"wavelet": "db4", "threshold_scale": 0.8, "mode": "soft"},
            {"wavelet": "db4", "threshold_scale": 1.0, "mode": "soft"},
            {"wavelet": "sym8", "threshold_scale": 1.0, "mode": "soft"},
            {"wavelet": "sym8", "threshold_scale": 1.2, "mode": "hard"},
        ],
    ),
    "frequency_masking": MethodSpec(
        name="frequency_masking",
        function=frequency_masking,
        default_params={"threshold": 2.0, "mask_floor": 0.05, "noise_frames": 6},
        search_grid=[
            {"threshold": 0.5, "mask_floor": 0.05, "noise_frames": 6},
            {"threshold": 1.0, "mask_floor": 0.05, "noise_frames": 6},
            {"threshold": 2.0, "mask_floor": 0.05, "noise_frames": 6},
        ],
    ),
}


def get_methods(selected: list[str] | None = None) -> list[MethodSpec]:
    if not selected or selected == ["all"]:
        return list(METHODS.values())
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return [METHODS[name] for name in selected]
