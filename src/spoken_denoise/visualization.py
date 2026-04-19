from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signals = [("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, y=0.98)

    for index, (axis, (label, samples)) in enumerate(zip(axes, signals)):
        times = np.arange(len(samples)) / sample_rate
        axis.plot(times, samples, linewidth=0.7)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if index < len(signals) - 1:
            axis.tick_params(labelbottom=False)
            axis.set_xlabel("")

    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.24)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_spectrogram_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    title: str,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signals = [("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, sharey=True)
    fig.suptitle(title, y=0.98)

    for index, (axis, (label, samples)) in enumerate(zip(axes, signals)):
        spectrum = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
        db = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)
        image = librosa.display.specshow(db, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="hz", ax=axis)
        axis.set_title(label)
        if index < len(signals) - 1:
            axis.tick_params(labelbottom=False)
            axis.set_xlabel("")
        else:
            axis.set_xlabel("Time (s)")

    fig.subplots_adjust(top=0.89, bottom=0.08, right=0.88, hspace=0.20)
    fig.colorbar(image, ax=axes, format="%+2.0f dB", fraction=0.035, pad=0.06)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
