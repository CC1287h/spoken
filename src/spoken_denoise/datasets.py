from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioPair:
    name: str
    noisy_path: Path
    clean_path: Path


def find_audio_pairs(noisy_dir: str | Path, clean_dir: str | Path) -> list[AudioPair]:
    """Pair noisy and clean wav files by identical relative file names."""
    noisy_root = Path(noisy_dir)
    clean_root = Path(clean_dir)
    if not noisy_root.exists():
        raise FileNotFoundError(f"Noisy directory does not exist: {noisy_root}")
    if not clean_root.exists():
        raise FileNotFoundError(f"Clean directory does not exist: {clean_root}")

    pairs: list[AudioPair] = []
    for noisy_path in sorted(noisy_root.rglob("*.wav")):
        relative = noisy_path.relative_to(noisy_root)
        clean_path = clean_root / relative
        if clean_path.exists():
            pairs.append(AudioPair(name=relative.as_posix(), noisy_path=noisy_path, clean_path=clean_path))

    if not pairs:
        raise RuntimeError(
            f"No matched wav pairs found. Expected identical file names under {noisy_root} and {clean_root}."
        )
    return pairs
