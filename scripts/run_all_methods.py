from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from _bootstrap import add_src_to_path

add_src_to_path()

from spoken_denoise.audio import align_signals, load_audio, save_audio
from spoken_denoise.datasets import find_audio_pairs
from spoken_denoise.methods import get_methods
from spoken_denoise.metrics import evaluate_noisy_baseline, evaluate_signals
from spoken_denoise.results_io import safe_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical denoising methods on the paired test set.")
    parser.add_argument("--noisy-dir", default="data/noisy_testset_wav", help="Directory containing noisy wav files.")
    parser.add_argument("--clean-dir", default="data/clean_testset_wav", help="Directory containing clean wav files.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for enhanced wav files.")
    parser.add_argument("--results-dir", default="results", help="Directory for csv metrics.")
    parser.add_argument("--methods", nargs="+", default=["all"], help="Methods to run, or 'all'.")
    parser.add_argument("--target-sr", type=int, default=None, help="Optional sample rate for loading all files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = find_audio_pairs(args.noisy_dir, args.clean_dir)
    if args.limit:
        pairs = pairs[: args.limit]

    methods = get_methods(args.methods)
    output_root = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pair in tqdm(pairs, desc="Files"):
        noisy, sample_rate = load_audio(pair.noisy_path, target_sr=args.target_sr)
        clean, clean_rate = load_audio(pair.clean_path, target_sr=sample_rate)
        if clean_rate != sample_rate:
            raise RuntimeError(f"Sample-rate mismatch after loading {pair.name}: {sample_rate} vs {clean_rate}")
        clean, noisy = align_signals(clean, noisy)

        baseline = evaluate_noisy_baseline(clean, noisy, sample_rate)
        rows.append({"file": pair.name, "method": "noisy_input", "params": "{}", **baseline})

        for method in methods:
            enhanced = method.function(noisy, sample_rate, **method.default_params)
            clean_eval, noisy_eval, enhanced_eval = align_signals(clean, noisy, enhanced)
            metrics = evaluate_signals(clean_eval, noisy_eval, enhanced_eval, sample_rate)

            output_path = output_root / method.name / pair.name
            save_audio(output_path, enhanced_eval, sample_rate)
            rows.append(
                {
                    "file": pair.name,
                    "method": method.name,
                    "params": json.dumps(method.default_params, sort_keys=True),
                    **metrics,
                }
            )

    detail = pd.DataFrame(rows)
    detail_path = results_dir / "final_results.csv"
    summary_path = results_dir / "final_summary.csv"
    detail_path = safe_to_csv(detail, detail_path, index=False)

    metric_cols = ["noisy_snr", "enhanced_snr", "snr_improvement", "mae", "mse", "rmse", "pesq", "stoi"]
    summary = detail.groupby("method", as_index=False)[metric_cols].mean(numeric_only=True)
    summary_path = safe_to_csv(summary, summary_path, index=False)
    print(f"Saved detailed metrics to {detail_path}")
    print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
