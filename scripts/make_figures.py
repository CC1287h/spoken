from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from spoken_denoise.audio import align_signals, load_audio
from spoken_denoise.datasets import find_audio_pairs
from spoken_denoise.visualization import plot_spectrogram_comparison, plot_waveform_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create waveform and spectrogram comparison figures.")
    parser.add_argument("--noisy-dir", default="data/noisy_testset_wav")
    parser.add_argument("--clean-dir", default="data/clean_testset_wav")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--methods", nargs="+", default=None, help="Output method directories to plot.")
    parser.add_argument("--files", nargs="+", default=None, help="Specific wav file names to visualize, such as p232_003.wav.")
    parser.add_argument("--num-files", type=int, default=3)
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Sample rate used for loading audio before visualization. Defaults to 16000 Hz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = find_audio_pairs(args.noisy_dir, args.clean_dir)
    if args.files:
        selected = set(args.files)
        pairs = [pair for pair in pairs if Path(pair.name).name in selected]
    else:
        pairs = pairs[: args.num_files]
    output_root = Path(args.output_dir)
    if args.methods:
        method_dirs = [output_root / method for method in args.methods]
    else:
        method_dirs = sorted(path for path in output_root.iterdir() if path.is_dir()) if output_root.exists() else []

    figures_root = Path(args.figures_dir)
    for pair in pairs:
        noisy, sample_rate = load_audio(pair.noisy_path, target_sr=args.target_sr)
        clean, _ = load_audio(pair.clean_path, target_sr=sample_rate)
        clean, noisy = align_signals(clean, noisy)

        for method_dir in method_dirs:
            enhanced_path = method_dir / pair.name
            if not enhanced_path.exists():
                continue
            enhanced, _ = load_audio(enhanced_path, target_sr=sample_rate)
            clean_eval, noisy_eval, enhanced_eval = align_signals(clean, noisy, enhanced)
            safe_name = pair.name.replace("/", "_").replace("\\", "_").removesuffix(".wav")
            title = f"{method_dir.name}: {pair.name}"
            plot_waveform_comparison(
                clean_eval,
                noisy_eval,
                enhanced_eval,
                sample_rate,
                figures_root / "waveform_comparison" / method_dir.name / f"{safe_name}.png",
                title,
            )
            plot_spectrogram_comparison(
                clean_eval,
                noisy_eval,
                enhanced_eval,
                sample_rate,
                figures_root / "spectrogram_comparison" / method_dir.name / f"{safe_name}.png",
                title,
            )

    print(f"Saved figures to {figures_root}")


if __name__ == "__main__":
    main()
