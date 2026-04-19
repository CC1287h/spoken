from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from _bootstrap import add_src_to_path

add_src_to_path()

from spoken_denoise.audio import align_signals, load_audio
from spoken_denoise.datasets import find_audio_pairs
from spoken_denoise.metrics import evaluate_noisy_baseline, evaluate_signals
from spoken_denoise.results_io import safe_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing enhanced wav outputs against the clean test set.")
    parser.add_argument("--noisy-dir", default="data/noisy_testset_wav")
    parser.add_argument("--clean-dir", default="data/clean_testset_wav")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = find_audio_pairs(args.noisy_dir, args.clean_dir)
    if args.limit:
        pairs = pairs[: args.limit]

    output_root = Path(args.output_dir)
    method_dirs = sorted(path for path in output_root.iterdir() if path.is_dir()) if output_root.exists() else []
    rows: list[dict] = []

    for pair in tqdm(pairs, desc="Evaluating"):
        noisy, sample_rate = load_audio(pair.noisy_path, target_sr=args.target_sr)
        clean, _ = load_audio(pair.clean_path, target_sr=sample_rate)
        clean, noisy = align_signals(clean, noisy)
        rows.append({"file": pair.name, "method": "noisy_input", **evaluate_noisy_baseline(clean, noisy, sample_rate)})

        for method_dir in method_dirs:
            enhanced_path = method_dir / pair.name
            if not enhanced_path.exists():
                continue
            enhanced, _ = load_audio(enhanced_path, target_sr=sample_rate)
            clean_eval, noisy_eval, enhanced_eval = align_signals(clean, noisy, enhanced)
            rows.append(
                {
                    "file": pair.name,
                    "method": method_dir.name,
                    **evaluate_signals(clean_eval, noisy_eval, enhanced_eval, sample_rate),
                }
            )

    detail = pd.DataFrame(rows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    detail_path = results_dir / "reevaluated_results.csv"
    summary_path = results_dir / "reevaluated_summary.csv"
    detail_path = safe_to_csv(detail, detail_path, index=False)

    metric_cols = ["noisy_snr", "enhanced_snr", "snr_improvement", "mae", "mse", "rmse", "pesq", "stoi"]
    summary = detail.groupby("method", as_index=False)[metric_cols].mean(numeric_only=True)
    summary_path = safe_to_csv(summary, summary_path, index=False)
    print(f"Saved detailed metrics to {detail_path}")
    print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
