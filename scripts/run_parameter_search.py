from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from _bootstrap import add_src_to_path

add_src_to_path()

from spoken_denoise.audio import align_signals, load_audio
from spoken_denoise.datasets import find_audio_pairs
from spoken_denoise.methods import get_methods
from spoken_denoise.metrics import evaluate_signals
from spoken_denoise.results_io import safe_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search simple parameter grids for the classical methods.")
    parser.add_argument("--noisy-dir", default="data/noisy_testset_wav")
    parser.add_argument("--clean-dir", default="data/clean_testset_wav")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = find_audio_pairs(args.noisy_dir, args.clean_dir)
    if args.limit:
        pairs = pairs[: args.limit]

    rows: list[dict] = []
    for method in get_methods(args.methods):
        for config_index, params in enumerate(method.search_grid):
            file_rows: list[dict] = []
            for pair in tqdm(pairs, desc=f"{method.name} config {config_index}"):
                noisy, sample_rate = load_audio(pair.noisy_path, target_sr=args.target_sr)
                clean, _ = load_audio(pair.clean_path, target_sr=sample_rate)
                clean, noisy = align_signals(clean, noisy)
                enhanced = method.function(noisy, sample_rate, **params)
                clean, noisy, enhanced = align_signals(clean, noisy, enhanced)
                file_rows.append(evaluate_signals(clean, noisy, enhanced, sample_rate))

            frame = pd.DataFrame(file_rows)
            means = frame.mean(numeric_only=True).to_dict()
            rows.append(
                {
                    "method": method.name,
                    "config_index": config_index,
                    "params": json.dumps(params, sort_keys=True),
                    **means,
                }
            )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "parameter_search.csv"
    output_path = safe_to_csv(pd.DataFrame(rows), output_path, index=False)
    print(f"Saved parameter search results to {output_path}")


if __name__ == "__main__":
    main()
