"""Parquet caching shared by the two source loaders.

Both source datasets are expensive to fetch and immutable once pinned, so every
loader writes one Parquet file keyed by the parameters that identify the dataset,
plus an optional JSON sidecar recording run provenance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def cache_path(cache_dir: Path, prefix: str, key_parts: Mapping[str, object]) -> Path:
    """`<cache_dir>/<prefix>-<digest>.parquet`, stable across runs and orderings."""
    payload = json.dumps(key_parts, sort_keys=True, default=str)
    digest = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
    return Path(cache_dir) / f"{prefix}-{digest}.parquet"


def sidecar_path(path: Path) -> Path:
    return Path(path).with_suffix(".provenance.json")


def read_cached(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if Path(path).exists() else None


def write_cached(
    path: Path, frame: pd.DataFrame, provenance: Mapping[str, object] | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    if provenance is not None:
        sidecar_path(path).write_text(json.dumps(provenance, indent=2, default=str))
