"""Xatu block headers and replay-range resolution.

Two queries only: resolve the replayed week's execution block range from
`canonical_beacon_block FINAL` (slot times are the natural handle on "a week"), then
pull the matching headers from `canonical_execution_block FINAL`. Both are pinned to
one network and bounded, because Xatu holds every network's full history and an
unbounded scan is expensive for everyone.

Headers supply only two things the simulation needs: the counterfactual starting
base fee (from the actual parent of the first source cohort) and the historical
gas_used/gas_limit series used as an observed-behaviour reference.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from config import SimConfig
from data import cache
from data.load_tx_gas_results import as_int64, block_chunks
from schemas import BLOCK_HEADER_COLUMNS

XATU_HOST = "clickhouse.xatu.ethpandaops.io"
XATU_PORT = 443
XATU_DATABASE = "default"

BEACON_TABLE = "canonical_beacon_block"
EXECUTION_TABLE = "canonical_execution_block"


# --- Connection -------------------------------------------------------------------


def read_secrets(cfg: SimConfig, required: Sequence[str] = ()) -> dict:
    path = Path(cfg.secrets_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; it must hold Xatu credentials as "
            '{"xatu_username": ..., "xatu_password": ...}'
        )
    secrets = json.loads(path.read_text())
    missing = [key for key in required if not secrets.get(key)]
    if missing:
        raise KeyError(f"{path} is missing required key(s): {', '.join(missing)}")
    return secrets


def xatu_client(cfg: SimConfig):
    import clickhouse_connect

    secrets = read_secrets(cfg, required=("xatu_username", "xatu_password"))
    return clickhouse_connect.get_client(
        host=XATU_HOST,
        port=XATU_PORT,
        secure=True,
        database=XATU_DATABASE,
        username=secrets["xatu_username"],
        password=secrets["xatu_password"],
    )


# --- SQL (pure builders, asserted on in tests) ------------------------------------


def beacon_block_range_sql(cfg: SimConfig, start: dt.datetime, end: dt.datetime) -> str:
    """Min/max execution block number over a half-open slot-time window.

    This is the one query that cannot pre-filter on block number -- it is what
    discovers the range -- so the bounded slot-time window is its bound.
    """
    return (
        "SELECT\n"
        "    min(execution_payload_block_number) AS first_block,\n"
        "    max(execution_payload_block_number) AS last_block,\n"
        "    count() AS slot_count\n"
        f"FROM {BEACON_TABLE} FINAL\n"
        f"WHERE meta_network_name = {_quote(cfg.meta_network_name)}\n"
        f"  AND slot_start_date_time >= {_datetime(start)}\n"
        f"  AND slot_start_date_time < {_datetime(end)}\n"
        "  AND execution_payload_block_number > 0"
    )


def execution_block_headers_sql(cfg: SimConfig, first_block: int, last_block: int) -> str:
    columns = ",\n    ".join(BLOCK_HEADER_COLUMNS)
    return (
        f"SELECT\n    {columns}\n"
        f"FROM {EXECUTION_TABLE} FINAL\n"
        f"WHERE meta_network_name = {_quote(cfg.meta_network_name)}\n"
        f"  AND block_number >= {int(first_block)}\n"
        f"  AND block_number <= {int(last_block)}\n"
        "ORDER BY block_number"
    )


# --- Range resolution -------------------------------------------------------------


def resolve_block_range(
    cfg: SimConfig, start: dt.datetime, end: dt.datetime
) -> tuple[int, int]:
    """Execution block range covering slots in `[start, end)`."""
    if end <= start:
        raise ValueError(f"empty slot-time window: {start} .. {end}")
    frame = xatu_client(cfg).query_df(beacon_block_range_sql(cfg, start, end))
    if frame.empty or int(frame["slot_count"].iloc[0]) == 0:
        raise ValueError(
            f"no {cfg.meta_network_name} beacon blocks with an execution payload in "
            f"{start} .. {end}"
        )
    row = frame.iloc[0]
    return int(row["first_block"]), int(row["last_block"])


# --- Headers ----------------------------------------------------------------------


def fetch_block_headers(cfg: SimConfig, first_block: int, last_block: int) -> pd.DataFrame:
    """Cached `schemas.BLOCK_HEADER_COLUMNS` for an inclusive block range."""
    if last_block < first_block:
        raise ValueError(f"empty block range: {(first_block, last_block)}")
    path = block_headers_cache_path(cfg, first_block, last_block)
    cached = cache.read_cached(path)
    if cached is not None:
        return cached

    headers = _normalize_headers(fetch_block_headers_uncached(cfg, first_block, last_block))
    cache.write_cached(path, headers, _headers_provenance(cfg, first_block, last_block, headers))
    return headers


def block_headers_cache_path(cfg: SimConfig, first_block: int, last_block: int) -> Path:
    return cache.cache_path(
        cfg.cache_dir,
        "block_headers",
        {
            "table": EXECUTION_TABLE,
            "network": cfg.meta_network_name,
            "first_block": int(first_block),
            "last_block": int(last_block),
        },
    )


def fetch_block_headers_uncached(
    cfg: SimConfig, first_block: int, last_block: int
) -> pd.DataFrame:
    """Uncached fetch.

    The only place this module pulls headers, so it is also the seam the cache
    tests spy on and the offline tests replace.
    """
    client = xatu_client(cfg)
    chunks = [
        client.query_df(execution_block_headers_sql(cfg, lo, hi))
        for lo, hi in block_chunks(first_block, last_block)
    ]
    non_empty = [c for c in chunks if not c.empty]
    if not non_empty:
        return pd.DataFrame(columns=list(BLOCK_HEADER_COLUMNS))
    return pd.concat(non_empty, ignore_index=True)


def _normalize_headers(headers: pd.DataFrame) -> pd.DataFrame:
    missing = set(BLOCK_HEADER_COLUMNS) - set(headers.columns)
    if missing:
        raise ValueError(f"header frame missing columns: {sorted(missing)}")
    out = headers.loc[:, list(BLOCK_HEADER_COLUMNS)].copy()
    for column in BLOCK_HEADER_COLUMNS:
        out[column] = as_int64(out[column], column)
    return (
        out.drop_duplicates("block_number", keep="last")
        .sort_values("block_number", ignore_index=True)
    )


def _headers_provenance(
    cfg: SimConfig, first_block: int, last_block: int, headers: pd.DataFrame
) -> dict:
    requested = last_block - first_block + 1
    return {
        "table": EXECUTION_TABLE,
        "network": cfg.meta_network_name,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_block_range": [int(first_block), int(last_block)],
        "row_count": int(len(headers)),
        "missing_block_count": int(requested - len(headers)),
    }


# --- Derived initial state --------------------------------------------------------


def starting_base_fee(headers: pd.DataFrame, start_block: int) -> int:
    """Base fee of the ACTUAL parent of `start_block`.

    EIP-1559 makes a block's base fee a function of its parent, so the
    counterfactual run must start from the parent header, not the start block itself.
    """
    parent = int(start_block) - 1
    row = headers.loc[headers["block_number"] == parent, "base_fee_per_gas"]
    if row.empty:
        raise ValueError(
            f"parent block {parent} of start block {start_block} "
            "is missing from the header frame; extend the fetched range by one block"
        )
    return int(row.iloc[0])


def _quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _datetime(value: dt.datetime) -> str:
    stamp = value.astimezone(dt.timezone.utc) if value.tzinfo else value
    return f"toDateTime('{stamp.strftime('%Y-%m-%d %H:%M:%S')}', 'UTC')"
