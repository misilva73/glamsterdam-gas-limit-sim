"""Parquet caching shared by the two source loaders.

Both source datasets are expensive to fetch and immutable once pinned, so every
loader writes one Parquet entry keyed by the parameters that identify the dataset,
plus an optional JSON sidecar recording run provenance.

An entry is either a single Parquet *file* (written whole by `write_cached`) or a
*directory* of `part-NNNNN.parquet` parts (streamed by `begin_parts` /
`write_part` / `commit_parts`, for a dataset too large to hold twice). Both are
read back by `read_cached` and both key off the same path, so a caller never has
to know which shape it got.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

PART_GLOB = "part-*.parquet"


def cache_path(cache_dir: Path, prefix: str, key_parts: Mapping[str, object]) -> Path:
    """`<cache_dir>/<prefix>-<digest>.parquet`, stable across runs and orderings."""
    payload = json.dumps(key_parts, sort_keys=True, default=str)
    digest = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
    return Path(cache_dir) / f"{prefix}-{digest}.parquet"


def sidecar_path(path: Path) -> Path:
    return Path(path).with_suffix(".provenance.json")


def read_cached(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return _read_parts(path) if path.is_dir() else pd.read_parquet(path)


def _read_parts(path: Path) -> pd.DataFrame:
    """Read a streamed entry back as one frame, in part order.

    The parts are handed to Arrow as an explicit sorted list rather than letting it
    discover the directory, because the parts are written in ascending block order
    and concatenating them in any other order would silently unsort the trace.

    `self_destruct`/`split_blocks` matter at this size: the combined frame is tens
    of GB, and the default conversion would hold the Arrow table and the pandas
    copy at once -- the very doubling the streamed write exists to avoid.
    """
    import pyarrow.dataset as ds

    parts = sorted(path.glob(PART_GLOB))
    if not parts:
        raise ValueError(f"cache entry {path} is a directory with no parquet parts")
    table = ds.dataset(parts, format="parquet").to_table()
    return table.to_pandas(self_destruct=True, split_blocks=True)


def write_cached(
    path: Path, frame: pd.DataFrame, provenance: Mapping[str, object] | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    if provenance is not None:
        write_sidecar(path, provenance)


def write_sidecar(path: Path, provenance: Mapping[str, object]) -> None:
    sidecar_path(path).write_text(json.dumps(provenance, indent=2, default=str))


# --- Streamed writes ---------------------------------------------------------------


def begin_parts(path: Path) -> Path:
    """Open a staging directory for a streamed write.

    Staging is separate from the final path so that an interrupted fetch cannot
    leave a short entry that the next run would read back as if it were complete.
    """
    staging = _staging_path(path)
    discard_parts(staging)
    staging.mkdir(parents=True)
    return staging


def write_part(staging: Path, index: int, frame: pd.DataFrame) -> None:
    """Write one part. Zero-padded so lexical order is block order."""
    frame.to_parquet(Path(staging) / f"part-{index:05d}.parquet", index=False)


def commit_parts(
    staging: Path, path: Path, provenance: Mapping[str, object] | None = None
) -> None:
    """Publish a staged write, replacing whatever shape the entry had before."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    Path(staging).replace(path)
    if provenance is not None:
        write_sidecar(path, provenance)


def discard_parts(staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)


def _staging_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".partial")
