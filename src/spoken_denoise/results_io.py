from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def safe_to_csv(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write csv, falling back to a timestamped file when the target is locked."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(output_path, index=index)
        return output_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        frame.to_csv(fallback, index=index)
        return fallback
