"""Load the per-transaction Glamsterdam replay rows and derive gas dimensions.

The source is ClickHouse `gas_analysis_tx_gas_result FINAL`, pinned to exactly one
dataset: one `analysis_config_hash`, one `schedule_config_hash`, one `chain_id`.
Mixing two replay configurations in one frame would silently average incomparable
gas schedules, so it is an error rather than a warning.

`fetch_tx_gas_results` is the single seam that touches the network, so tests stub
it with synthetic source-table rows rather than the module carrying an offline mode.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_RESCUE_MULTIPLIER,
    LEGACY_TX_TYPES,
    SimConfig,
    TxInclusionPolicy,
)
from data import cache
from schemas import (
    TX_GAS_RESULT_COLUMNS,
    TX_GAS_RESULT_FIELDS,
    TX_GAS_RESULT_PROVENANCE_FIELDS,
)

CLICKHOUSE_DATABASE = "gas_analysis"
CLICKHOUSE_TABLE = f"{CLICKHOUSE_DATABASE}.gas_analysis_tx_gas_result"
# Database-qualified: the client connects to `default`, but the replay tables live
# in their own `gas_analysis` database.

# A replayed week is ~50,400 blocks; pull it in slices so no single query has to
# materialise the whole week server-side.
BLOCK_CHUNK = 5_000

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1

# Dataset identity: pinned in the query, kept in the frame so the guard below can
# still fire after a cache reload.
DATASET_KEY_COLUMNS = ("analysis_config_hash", "schedule_config_hash", "chain_id")

_GAS_INT_FIELDS = (
    "block_number",
    "tx_index",
    "tx_type",
    "tx_gas_limit",
    "baseline_gas_used",
    "baseline_total_gas_spent",
    "schedule_gas_used",
    "schedule_total_gas_spent",
    "schedule_gas_refunded",
    "schedule_floor_gas",
    "schedule_state_gas_spent",
)
# Nullable upstream ("null for execution-only schedules which have no intrinsic
# opinion"), and unused by the simulation, so it is kept as a nullable Int64
# rather than forced through the strict integer gate.
_NULLABLE_GAS_FIELDS = ("schedule_intrinsic_gas",)
_WEI_FIELDS = ("max_fee_per_gas", "max_priority_fee_per_gas")
_FLAG_FIELDS = ("baseline_success", "schedule_success")

EXCLUSION_REASONS = (
    "baseline_only_failure",
    "schedule_gas_rescuable",
    "schedule_halted_regardless",
    "both_failed",
)

RESCUE_SWEEP_CEILING = 9.7429
"""Top rung of the producer's gas-limit multiplier sweep.

`min_multiplier_to_succeed` piles up at this value, so rows sitting on it needed
at least the top rung and their true requirement is grid-censored.
"""


# --- Public entry point -----------------------------------------------------------


def load_tx_gas_results(cfg: SimConfig) -> pd.DataFrame:
    """Fetch (or reload from cache) the pinned replay rows, dtype-clean and derived."""
    path = tx_gas_results_cache_path(cfg)
    cached = cache.read_cached(path)
    if cached is not None:
        return assert_single_dataset(cached, cfg)

    raw = fetch_tx_gas_results(cfg)
    frame = prepare_tx_gas_results(raw, cfg)
    cache.write_cached(path, frame, tx_gas_results_provenance(raw, frame, cfg))
    return frame


def tx_gas_results_cache_path(cfg: SimConfig) -> Path:
    return cache.cache_path(
        cfg.cache_dir,
        "tx_gas_results",
        {
            "table": CLICKHOUSE_TABLE,
            "analysis_config_hash": cfg.analysis_config_hash,
            "schedule_name": cfg.schedule_name,
            "chain_id": cfg.chain_id,
            "block_range": list(cfg.source_block_range) if cfg.source_block_range else None,
        },
    )


def fetch_tx_gas_results(cfg: SimConfig) -> pd.DataFrame:
    """Uncached fetch in the *source table* shape.

    The only place this module talks to ClickHouse, so it is also the seam the
    cache tests spy on and the offline tests replace.
    """
    return _fetch_clickhouse(cfg, *require_block_range(cfg))


def prepare_tx_gas_results(raw: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    frame = normalize_dtypes(raw)
    frame = assert_single_dataset(frame, cfg)
    frame = derive_gas_dimensions(frame)
    return frame.sort_values(["block_number", "tx_index"], ignore_index=True)[
        _contract_columns(frame)
    ]


def require_block_range(cfg: SimConfig) -> tuple[int, int]:
    if not cfg.source_block_range:
        raise ValueError(
            "source_block_range is required; unbounded scans of the replay table "
            "are not allowed"
        )
    first_block, last_block = (int(b) for b in cfg.source_block_range)
    if last_block < first_block:
        raise ValueError(f"empty source_block_range: {(first_block, last_block)}")
    return first_block, last_block


# --- ClickHouse -------------------------------------------------------------------


def clickhouse_tx_gas_results_sql(cfg: SimConfig, first_block: int, last_block: int) -> str:
    """Pure SQL builder, so the query shape is testable without the table existing."""
    if not cfg.analysis_config_hash:
        raise ValueError("analysis_config_hash must be pinned for ClickHouse input")
    columns = ",\n    ".join(TX_GAS_RESULT_FIELDS + TX_GAS_RESULT_PROVENANCE_FIELDS)
    conditions = [
        f"analysis_config_hash = {_quote(cfg.analysis_config_hash)}",
        f"chain_id = {int(cfg.chain_id)}",
        f"schedule_name = {_quote(cfg.schedule_name)}",
        f"block_number >= {int(first_block)}",
        f"block_number <= {int(last_block)}",
    ]
    if cfg.schedule_config_hash:
        conditions.append(f"schedule_config_hash = {_quote(cfg.schedule_config_hash)}")
    return (
        f"SELECT\n    analysis_config_hash,\n    chain_id,\n    schedule_name,\n    {columns}\n"
        f"FROM {CLICKHOUSE_TABLE} FINAL\n"
        "WHERE " + "\n  AND ".join(conditions) + "\n"
        "ORDER BY block_number, tx_index"
    )


def clickhouse_client(cfg: SimConfig):
    """Client for the gas-analysis cluster.

    The replay table is expected on the same ethpandaops ClickHouse as Xatu, so the
    `xatu_*` credentials are the default; override per-key with `gas_analysis_*`
    entries in `secrets.json` if it lands on a separate cluster.
    """
    import clickhouse_connect

    from data.fetch_blocks import XATU_HOST, read_secrets

    secrets = read_secrets(cfg, required=("xatu_username", "xatu_password"))
    return clickhouse_connect.get_client(
        host=secrets.get("gas_analysis_host", XATU_HOST),
        port=int(secrets.get("gas_analysis_port", 443)),
        secure=True,
        database=secrets.get("gas_analysis_database", "default"),
        username=secrets.get("gas_analysis_username", secrets["xatu_username"]),
        password=secrets.get("gas_analysis_password", secrets["xatu_password"]),
    )


def _fetch_clickhouse(cfg: SimConfig, first_block: int, last_block: int) -> pd.DataFrame:
    client = clickhouse_client(cfg)
    chunks = [
        client.query_df(clickhouse_tx_gas_results_sql(cfg, lo, hi))
        for lo, hi in block_chunks(first_block, last_block)
    ]
    frame = _concat_chunks(chunks)
    if frame.empty:
        raise ValueError(_empty_result_message(client, cfg, first_block, last_block))
    return frame


def _empty_result_message(client, cfg: SimConfig, first_block: int, last_block: int) -> str:
    """An empty result is almost always a mis-pinned hash or schedule name.

    Reporting what the table actually holds turns a silent empty simulation into a
    one-line fix -- the live data uses schedule_name 'amsterdam', not
    'glamsterdam-v1'.
    """
    lines = [
        f"no rows for analysis_config_hash={cfg.analysis_config_hash!r} "
        f"schedule_name={cfg.schedule_name!r} chain_id={cfg.chain_id} "
        f"blocks {first_block}-{last_block}."
    ]
    try:
        available = client.query_df(
            f"SELECT chain_id, analysis_config_hash, schedule_name, count() AS rows, "
            f"min(block_number) AS first_block, max(block_number) AS last_block "
            f"FROM {CLICKHOUSE_TABLE} FINAL WHERE chain_id = {int(cfg.chain_id)} "
            f"GROUP BY 1, 2, 3 ORDER BY rows DESC LIMIT 20"
        )
        lines.append(f"Available datasets:\n{available.to_string(index=False)}")
    except Exception as exc:  # noqa: BLE001 - diagnostics must not mask the real error
        lines.append(f"(could not list available datasets: {exc})")
    return "\n".join(lines)


def block_chunks(
    first_block: int, last_block: int, chunk: int = BLOCK_CHUNK
) -> Iterator[tuple[int, int]]:
    for lo in range(first_block, last_block + 1, chunk):
        yield lo, min(lo + chunk - 1, last_block)


# --- Dataset identity guard -------------------------------------------------------


def assert_single_dataset(frame: pd.DataFrame, cfg: SimConfig | None = None) -> pd.DataFrame:
    """Refuse frames that mix replay configurations, chains, or gas schedules."""
    for column in DATASET_KEY_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.unique(frame[column].dropna())
        if len(values) > 1:
            raise ValueError(
                f"{column} must be pinned to exactly one value; frame mixes "
                f"{sorted(map(str, values))}"
            )
    if cfg is None:
        return frame
    expected = {
        "analysis_config_hash": cfg.analysis_config_hash,
        "schedule_config_hash": cfg.schedule_config_hash,
        "chain_id": cfg.chain_id,
    }
    for column, want in expected.items():
        if want is None or column not in frame.columns or frame.empty:
            continue
        got = frame[column].iloc[0]
        if str(got) != str(want):
            raise ValueError(f"{column} mismatch: config pins {want!r}, data has {got!r}")
    return frame


# --- Dtypes -----------------------------------------------------------------------


def as_int64(values, name: str = "value") -> pd.Series:
    """Exact conversion of UInt256 / decimal-string integers to int64.

    `astype('int64')` wraps silently on overflow, which for a wei fee would turn a
    nonsense value into a plausible one; raise instead.
    """
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError(f"{name} contains nulls; expected an integer for every row")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("int64")  # ClickHouse writes the success flags as Bool
    if pd.api.types.is_signed_integer_dtype(series.dtype) and series.dtype.itemsize <= 8:
        return series.astype("int64")
    if pd.api.types.is_unsigned_integer_dtype(series.dtype) and series.dtype.itemsize <= 8:
        if int(series.max()) > INT64_MAX:
            raise OverflowError(f"{name} exceeds int64: max {int(series.max())}")
        return series.astype("int64")

    ints = series.map(_exact_int)
    out_of_range = ints[(ints > INT64_MAX) | (ints < INT64_MIN)]
    if not out_of_range.empty:
        raise OverflowError(
            f"{name} does not fit in int64 for {len(out_of_range)} row(s); "
            f"first offending value {out_of_range.iloc[0]} at index {out_of_range.index[0]}"
        )
    return pd.Series(ints.to_numpy(dtype=np.int64), index=series.index, name=series.name)


def _exact_int(value) -> int:
    if isinstance(value, bool):
        raise TypeError(f"unexpected bool where an integer was expected: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return int(value.strip(), 10)  # ClickHouse ships UInt256 fees as decimal strings
    if isinstance(value, Decimal):
        if value != value.to_integral_value():
            raise ValueError(f"non-integral decimal: {value}")
        return int(value)
    if isinstance(value, float):
        if not float(value).is_integer():
            raise ValueError(f"non-integral float: {value}")
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__} to an exact integer: {value!r}")


def normalize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    out = _decode_byte_strings(frame.copy())
    out = _resolve_absent_priority_fee(out)
    for field in _GAS_INT_FIELDS + _WEI_FIELDS:
        if field in out.columns:
            out[field] = as_int64(out[field], field)
    for field in _NULLABLE_GAS_FIELDS:
        if field in out.columns:
            out[field] = pd.to_numeric(out[field], errors="coerce").astype("Int64")
    for field in _FLAG_FIELDS:
        if field in out.columns:
            out[field] = as_int64(out[field], field).astype("int8")
    if "min_multiplier_to_succeed" in out.columns:
        out["min_multiplier_to_succeed"] = pd.to_numeric(
            out["min_multiplier_to_succeed"], errors="coerce"
        ).astype("float64")
    # Provenance is optional upstream; keep the declared shape so downstream code
    # can reference the columns unconditionally.
    for field in TX_GAS_RESULT_PROVENANCE_FIELDS:
        if field not in out.columns:
            out[field] = pd.NA
    return out


def _decode_byte_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """ClickHouse `FixedString` columns arrive as bytes; compare them as text.

    Left as bytes, a pinned `analysis_config_hash` would never equal the string in
    the config and the dataset guard would misfire on every real query.
    """
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        non_null = frame[column].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], bytes):
            frame[column] = frame[column].map(
                lambda v: v.decode() if isinstance(v, bytes) else v
            )
    return frame


def _resolve_absent_priority_fee(frame: pd.DataFrame) -> pd.DataFrame:
    """Legacy/access-list rows carry no priority cap, so the producer writes null.

    Those rows are priced off `max_fee_per_gas` (a normalized gas price) by the
    engine's legacy tip rule, so the value is never read and 0 is a safe fill. A
    null on any *other* envelope type would silently zero a real bid, so that is an
    error rather than a fill.
    """
    if "max_priority_fee_per_gas" not in frame.columns:
        return frame
    absent = frame["max_priority_fee_per_gas"].isna()
    if not absent.any():
        return frame
    if "tx_type" in frame.columns:
        unexpected = absent & ~frame["tx_type"].astype("int64").isin(LEGACY_TX_TYPES)
        if unexpected.any():
            offending = sorted(frame.loc[unexpected, "tx_type"].unique())
            raise ValueError(
                "max_priority_fee_per_gas is null on non-legacy transaction "
                f"type(s) {offending}; refusing to treat a missing bid as zero"
            )
    frame["max_priority_fee_per_gas"] = frame["max_priority_fee_per_gas"].fillna(0)
    return frame


def _contract_columns(frame: pd.DataFrame) -> list[str]:
    declared = [c for c in TX_GAS_RESULT_COLUMNS if c in frame.columns]
    extra = [c for c in frame.columns if c not in declared]
    return declared + extra


# --- Derivation and filtering -----------------------------------------------------


def derive_gas_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the two block-capacity dimensions.

    `schedule_state_gas_spent` is a component of `schedule_total_gas_spent`, not an
    extra charge, so execution gas is the remainder -- floored by the calldata floor,
    which can exceed that remainder and then binds the execution dimension.
    """
    out = frame.copy()
    out["state_gas"] = as_int64(out["schedule_state_gas_spent"], "schedule_state_gas_spent")
    out["execution_gas"] = np.maximum(
        as_int64(out["schedule_total_gas_spent"], "schedule_total_gas_spent") - out["state_gas"],
        as_int64(out["schedule_floor_gas"], "schedule_floor_gas"),
    ).astype("int64")
    return out


def split_simulatable(
    frame: pd.DataFrame,
    policy: TxInclusionPolicy = "all",
    max_rescue_multiplier: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into rows the simulator places and rows it only reports.

    Three policies, giving materially different answers:

    `all` (default) simulates every row, failures included. A failed transaction
    still occupies block space and still pays, so it is real demand. Its gas
    figures come from whichever replay run the producer recorded -- the rescued run
    where one succeeded, the original otherwise.

    `gas_rescued` keeps successes plus rows that a larger gas limit would rescue,
    dropping only transactions that halted for a non-gas reason.

    `successful_only` is the strict rule: `baseline_success == 1 AND
    schedule_success == 1`. It models senders never adjusting their gas limits,
    and on live data discards ~90% of all state gas.

    `max_rescue_multiplier` bounds how much extra limit a sender is assumed to
    grant. Note `min_multiplier_to_succeed` is grid-censored at
    `RESCUE_SWEEP_CEILING`, so rows sitting there needed *at least* that much.
    """
    if policy == "all":
        usable = pd.Series(True, index=frame.index)
    else:
        usable = (frame["baseline_success"] == 1) & (frame["schedule_success"] == 1)
        if policy == "gas_rescued":
            usable |= (frame["baseline_success"] == 1) & _is_gas_rescuable(frame)
    if max_rescue_multiplier is not None:
        within_budget = (
            frame["min_multiplier_to_succeed"].fillna(DEFAULT_RESCUE_MULTIPLIER)
            <= max_rescue_multiplier
        )
        usable &= (frame["schedule_success"] == 1) | within_budget

    selected = frame.loc[usable].reset_index(drop=True)
    excluded = frame.loc[~usable].copy()
    excluded["exclusion_reason"] = exclusion_reason(excluded)
    return selected, excluded.reset_index(drop=True)


def exclusion_reason(frame: pd.DataFrame) -> pd.Series:
    """Classify why a row is not simulatable under the strict original-limit filter.

    A schedule failure splits in two, and the distinction dominates the analysis:
    `schedule_gas_rescuable` rows succeeded once the replay raised the gas limit,
    so their recorded gas is a real measurement of what the transaction costs given
    an adequate limit -- they are censored by the sender's signed limit, not broken.
    `schedule_halted_regardless` rows failed at every swept multiplier.
    """
    baseline_failed = frame["baseline_success"] != 1
    schedule_failed = frame["schedule_success"] != 1
    rescuable = schedule_failed & _is_gas_rescuable(frame)
    return pd.Series(
        np.select(
            [
                baseline_failed & schedule_failed,
                baseline_failed,
                rescuable,
                schedule_failed,
            ],
            [
                "both_failed",
                "baseline_only_failure",
                "schedule_gas_rescuable",
                "schedule_halted_regardless",
            ],
            default="simulatable",
        ),
        index=frame.index,
        dtype="object",
    )


def _is_gas_rescuable(frame: pd.DataFrame) -> pd.Series:
    """True where the replay found a gas-limit multiplier that made the tx succeed."""
    if "min_multiplier_to_succeed" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["min_multiplier_to_succeed"].notna()


def excluded_summary(
    excluded: pd.DataFrame, simulatable: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Counts and gas by exclusion reason.

    `*_share` is of the excluded set. Pass `simulatable` to also get
    `*_share_of_all`, which is the figure that says how much of the dataset the
    filter actually removed -- the within-excluded shares always sum to 1 and so
    cannot answer that on their own.
    """
    columns = ["tx_count", "execution_gas", "state_gas", "schedule_gas_used"]
    if excluded.empty:
        empty = pd.DataFrame(
            {c: pd.Series(dtype="int64") for c in columns},
            index=pd.Index([], name="exclusion_reason"),
        )
        return empty.assign(**{f"{c}_share": pd.Series(dtype="float64") for c in columns})

    reason = (
        excluded["exclusion_reason"]
        if "exclusion_reason" in excluded.columns
        else exclusion_reason(excluded)
    )
    summary = (
        excluded.assign(exclusion_reason=reason, tx_count=1)
        .groupby("exclusion_reason")[columns]
        .sum()
        .reindex(EXCLUSION_REASONS, fill_value=0)
        .astype("int64")
    )
    shares = summary.div(summary.sum()).fillna(0.0).add_suffix("_share")
    parts = [summary, shares]
    if simulatable is not None:
        kept = simulatable.assign(tx_count=1)[columns].sum()
        parts.append(
            summary.div(summary.sum() + kept).fillna(0.0).add_suffix("_share_of_all")
        )
    return pd.concat(parts, axis=1)


def block_gas_used(selected: pd.DataFrame) -> int:
    """Header-equivalent gas for a set of transactions: the binding dimension."""
    if selected.empty:
        return 0
    return int(max(selected["execution_gas"].sum(), selected["state_gas"].sum()))


# --- Provenance -------------------------------------------------------------------


def tx_gas_results_provenance(
    raw: pd.DataFrame, frame: pd.DataFrame, cfg: SimConfig
) -> dict:
    """What was fetched, from where, under which pinned dataset."""
    blocks = frame["block_number"]
    return {
        "table": CLICKHOUSE_TABLE,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_config_hash": _single(frame, "analysis_config_hash", cfg.analysis_config_hash),
        "schedule_config_hash": _single(frame, "schedule_config_hash", cfg.schedule_config_hash),
        "schedule_name": cfg.schedule_name,
        "chain_id": _single(frame, "chain_id", cfg.chain_id),
        "requested_block_range": list(cfg.source_block_range) if cfg.source_block_range else None,
        "observed_block_range": [int(blocks.min()), int(blocks.max())] if len(blocks) else None,
        "row_count": int(len(frame)),
        "block_count": int(blocks.nunique()),
        "simulatable_row_count": int(
            ((frame["baseline_success"] == 1) & (frame["schedule_success"] == 1)).sum()
        ),
        "block_hash_coverage": _coverage(raw, "block_hash"),
        "block_timestamp_coverage": _timestamp_coverage(raw),
        "producer_schema_version": _distinct(raw, "producer_schema_version"),
        "producer_git_commit": _distinct(raw, "producer_git_commit"),
        "replay_semantics": _distinct(raw, "replay_semantics"),
    }


def _single(frame: pd.DataFrame, column: str, fallback=None):
    if column not in frame.columns or frame.empty:
        return fallback
    values = pd.unique(frame[column].dropna())
    return None if len(values) == 0 else _plain(values[0])


def _distinct(frame: pd.DataFrame, column: str) -> list | None:
    if column not in frame.columns:
        return None
    return sorted({_plain(v) for v in pd.unique(frame[column].dropna())}, key=str)


def _coverage(frame: pd.DataFrame, column: str) -> dict:
    if column not in frame.columns:
        return {"present": False}
    values = frame[column]
    return {
        "present": True,
        "distinct": int(values.nunique(dropna=True)),
        "null_count": int(values.isna().sum()),
    }


def _timestamp_coverage(frame: pd.DataFrame) -> dict:
    coverage = _coverage(frame, "block_timestamp")
    if not coverage["present"]:
        return coverage
    stamps = pd.to_datetime(frame["block_timestamp"], utc=True, errors="coerce")
    if stamps.notna().any():
        coverage |= {"min": stamps.min().isoformat(), "max": stamps.max().isoformat()}
    return coverage


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def _concat_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [c for c in chunks if not c.empty]
    if not non_empty:
        return pd.DataFrame(columns=list(TX_GAS_RESULT_COLUMNS))
    return pd.concat(non_empty, ignore_index=True)


def _quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"
