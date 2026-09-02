"""Synthetic stand-ins for the two source datasets. Test fixtures only.

The simulation itself reads ClickHouse and nothing else; this generator exists so
the suite needs no credentials and no network. Tests inject it at the two fetch
seams (`tests.conftest.offline_data`), so it is never reachable from a real run.

The frames are in the *source-table* shape (no derived columns, unfiltered
failures included) so the loader's filtering and derivation logic is exercised
too.

They also remain the only input that stresses the state dimension: on the live
`amsterdam` data ~87% of state gas sits in rows excluded as schedule failures, so
the real simulatable trace is almost never state-bound (see `AGENTS.md`).

Shape choices that matter for the simulation are deliberate, not decorative:

* Cohort size and fee level are autocorrelated (AR(1) plus a diurnal cycle), so
  moving-block bootstrap window selection and autocorrelation-based choice of
  `L` have something real to bite on.
* A minority of transactions carry `max_fee_per_gas` below the prevailing base
  fee, so fee-ineligible backlog is non-empty.
* The calldata floor binds on a minority of rows, so
  `schedule_total_gas_spent - schedule_gas_refunded != schedule_gas_used` shows
  up in tests.
* State gas is a large share of total gas for storage-heavy rows, so the state
  dimension can be the binding block constraint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BASE_FEE_MAX_CHANGE_DENOMINATOR,
    ELASTICITY_MULTIPLIER,
    LEGACY_TX_TYPES,
    MIN_BASE_FEE,
)
from schemas import TX_GAS_RESULT_FIELDS, TX_GAS_RESULT_PROVENANCE_FIELDS

DUMMY_ANALYSIS_CONFIG_HASH = "dummy-analysis-cfg-0000000000000000"
DUMMY_SCHEDULE_CONFIG_HASH = "dummy-schedule-cfg-0000000000000000"
DUMMY_GIT_COMMIT = "0000000000000000000000000000000000000000"
DUMMY_SCHEMA_VERSION = 0
DUMMY_REPLAY_SEMANTICS = "dummy-synthetic"

# Type share: legacy, access-list, dynamic-fee, blob, set-code.
_TX_TYPE_SHARES = {0: 0.18, 1: 0.04, 2: 0.70, 3: 0.05, 4: 0.03}

_BLOCKS_PER_DAY = 7200
_SECONDS_PER_BLOCK = 12


def _autocorrelated_series(
    rng: np.random.Generator, n: int, phi: float = 0.9, sigma: float = 0.35
) -> np.ndarray:
    """Standardised AR(1) path, so bootstrap block length has a real ACF to fit."""
    shocks = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = shocks[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + shocks[i]
    return out


def _diurnal(n: int, start_phase: float, amplitude: float) -> np.ndarray:
    steps = np.arange(n)
    return amplitude * np.sin(2 * np.pi * steps / _BLOCKS_PER_DAY + start_phase)


def dummy_tx_gas_results(
    first_block: int = 21_000_000,
    num_blocks: int = 400,
    seed: int = 7,
    mean_txs_per_block: float = 160.0,
    schedule_name: str = "glamsterdam-v1",
) -> pd.DataFrame:
    """Source-table-shaped replay rows for `num_blocks` consecutive blocks."""
    rng = np.random.default_rng(seed)

    intensity = np.exp(
        _autocorrelated_series(rng, num_blocks, phi=0.92, sigma=0.18)
        + _diurnal(num_blocks, start_phase=0.7, amplitude=0.25)
    )
    tx_counts = np.maximum(
        1, rng.poisson(np.clip(mean_txs_per_block * intensity, 5, None))
    )

    fee_level_gwei = np.exp(
        np.log(8.0)
        + _autocorrelated_series(rng, num_blocks, phi=0.95, sigma=0.10)
        + _diurnal(num_blocks, start_phase=0.7, amplitude=0.45)
    )

    block_numbers = np.repeat(np.arange(first_block, first_block + num_blocks), tx_counts)
    total = int(tx_counts.sum())
    tx_index = np.concatenate([np.arange(c) for c in tx_counts])
    block_fee = np.repeat(fee_level_gwei, tx_counts)

    tx_type = rng.choice(
        list(_TX_TYPE_SHARES), size=total, p=list(_TX_TYPE_SHARES.values())
    )

    # Historical (baseline) gas: a heavy tail of contract calls over a floor of
    # plain transfers.
    is_transfer = rng.random(total) < 0.34
    baseline_gas_used = np.where(
        is_transfer,
        21_000,
        np.clip(rng.lognormal(mean=np.log(95_000), sigma=1.15, size=total), 21_000, 8_000_000),
    ).astype(np.int64)

    # Glamsterdam repricing inflates total charged gas; storage-heavy rows most.
    storage_intensity = rng.beta(1.4, 4.0, size=total)
    repricing_factor = 1.15 + 2.6 * storage_intensity + rng.normal(0, 0.08, total)
    schedule_total_gas_spent = np.maximum(
        21_000, (baseline_gas_used * np.clip(repricing_factor, 1.0, None))
    ).astype(np.int64)

    # EIP-8037 state gas is a component of the total, not an extra charge. A
    # slow-moving block-level skew lets whole stretches of the trace be
    # state-bound rather than execution-bound.
    state_skew = np.repeat(
        np.exp(_autocorrelated_series(rng, num_blocks, phi=0.93, sigma=0.22)), tx_counts
    )
    state_share = np.clip(
        storage_intensity * state_skew * rng.uniform(0.6, 2.2, total), 0.0, 0.9
    )
    schedule_state_gas_spent = (schedule_total_gas_spent * state_share).astype(np.int64)

    # Refunds: mostly none, occasionally a storage clear (capped at 1/5).
    has_refund = rng.random(total) < 0.12
    schedule_gas_refunded = np.where(
        has_refund, (schedule_total_gas_spent * rng.uniform(0.02, 0.20, total)), 0
    ).astype(np.int64)

    # Calldata floor: binds on a minority of calldata-heavy, compute-light rows.
    calldata_tokens = np.where(
        is_transfer, 0, rng.lognormal(np.log(220), 1.3, total)
    ).astype(np.int64)
    schedule_intrinsic_gas = (12_000 + 4 * calldata_tokens).astype(np.int64)
    schedule_floor_gas = (12_000 + 16 * calldata_tokens).astype(np.int64)
    floor_binds = rng.random(total) < 0.09
    schedule_floor_gas = np.where(
        floor_binds,
        np.maximum(schedule_floor_gas, schedule_total_gas_spent - schedule_gas_refunded + 1),
        schedule_floor_gas,
    ).astype(np.int64)

    schedule_gas_used = np.maximum(
        schedule_total_gas_spent - schedule_gas_refunded, schedule_floor_gas
    ).astype(np.int64)

    # Signed gas limits: generous headroom over the historical cost, so repricing
    # pushes only a small minority of transactions over their own limit.
    tx_gas_limit = np.maximum(
        baseline_gas_used,
        baseline_gas_used * np.clip(rng.lognormal(np.log(5.0), 0.45, total), 1.02, 40.0),
    ).astype(np.int64)

    baseline_success = (rng.random(total) > 0.004).astype(np.int8)
    schedule_success = (
        (schedule_gas_used <= tx_gas_limit) & (baseline_success == 1)
    ).astype(np.int8)
    min_multiplier_to_succeed = np.where(
        schedule_success == 1,
        1.0,
        np.ceil(schedule_gas_used / np.maximum(tx_gas_limit, 1) * 100) / 100,
    )

    # Fees. `max_fee_per_gas` carries a normalized gas_price for legacy/AL rows.
    priority_gwei = np.clip(rng.lognormal(np.log(0.9), 1.1, total), 0.001, 500.0)
    fee_headroom = rng.lognormal(np.log(1.6), 0.55, total)
    max_fee_gwei = block_fee * fee_headroom
    # ~6% of demand is priced below the prevailing base fee.
    underpriced = rng.random(total) < 0.06
    max_fee_gwei = np.where(underpriced, block_fee * rng.uniform(0.25, 0.95, total), max_fee_gwei)

    max_fee_per_gas = (max_fee_gwei * 1e9).astype(np.int64)
    max_priority_fee_per_gas = np.minimum(
        (priority_gwei * 1e9).astype(np.int64), max_fee_per_gas
    )
    # Legacy/access-list transactions have no priority-fee field, so a producer has
    # nothing to put here; 0 is the likely value. Emitting 0 rather than a copy of
    # the gas price keeps the engine's legacy tip branch load-bearing -- with a
    # copy, `min(prio, max_fee - base_fee)` collapses to the legacy rule and
    # deleting that rule would not change any result on this trace.
    is_legacy = np.isin(tx_type, list(LEGACY_TX_TYPES))
    max_priority_fee_per_gas = np.where(is_legacy, 0, max_priority_fee_per_gas)

    tx_hash = np.array(
        [f"0x{i:064x}" for i in rng.choice(2**48, size=total, replace=False)], dtype=object
    )

    frame = pd.DataFrame(
        {
            "schedule_name": schedule_name,
            "analysis_config_hash": DUMMY_ANALYSIS_CONFIG_HASH,
            "chain_id": 1,
            "schedule_config_hash": DUMMY_SCHEDULE_CONFIG_HASH,
            "block_number": block_numbers,
            "tx_index": tx_index.astype(np.int64),
            "tx_hash": tx_hash,
            "tx_type": tx_type.astype(np.int64),
            "tx_gas_limit": tx_gas_limit,
            "max_fee_per_gas": max_fee_per_gas,
            "max_priority_fee_per_gas": max_priority_fee_per_gas,
            "baseline_success": baseline_success,
            "baseline_gas_used": baseline_gas_used,
            "baseline_total_gas_spent": baseline_gas_used,
            "schedule_success": schedule_success,
            "schedule_gas_used": schedule_gas_used,
            "schedule_total_gas_spent": schedule_total_gas_spent,
            "schedule_gas_refunded": schedule_gas_refunded,
            "schedule_floor_gas": schedule_floor_gas,
            "schedule_state_gas_spent": schedule_state_gas_spent,
            "schedule_intrinsic_gas": schedule_intrinsic_gas,
            "min_multiplier_to_succeed": min_multiplier_to_succeed,
            "block_hash": [f"0x{b:064x}" for b in block_numbers],
            "block_timestamp": pd.to_datetime("2026-06-01", utc=True)
            + pd.to_timedelta((block_numbers - first_block) * _SECONDS_PER_BLOCK, unit="s"),
            "producer_schema_version": DUMMY_SCHEMA_VERSION,
            "producer_git_commit": DUMMY_GIT_COMMIT,
            "replay_semantics": DUMMY_REPLAY_SEMANTICS,
        }
    )
    missing = set(TX_GAS_RESULT_FIELDS + TX_GAS_RESULT_PROVENANCE_FIELDS) - set(frame.columns)
    assert not missing, f"dummy generator missing source fields: {sorted(missing)}"
    return frame.sort_values(["block_number", "tx_index"], ignore_index=True)


def dummy_block_headers(
    tx_gas_results: pd.DataFrame, gas_limit: int = 60_000_000
) -> pd.DataFrame:
    """Headers consistent with the replay rows, base fee stepped by EIP-1559."""
    per_block = (
        tx_gas_results.groupby("block_number", as_index=False)["baseline_gas_used"]
        .sum()
        .rename(columns={"baseline_gas_used": "gas_used"})
    )
    per_block["gas_used"] = per_block["gas_used"].clip(upper=gas_limit)
    per_block["gas_limit"] = gas_limit

    target = gas_limit // ELASTICITY_MULTIPLIER
    base_fees: list[int] = []
    base_fee = 8_000_000_000
    for gas_used in per_block["gas_used"]:
        base_fees.append(int(base_fee))
        base_fee = next_base_fee(int(base_fee), int(gas_used), target)
    per_block["base_fee_per_gas"] = base_fees
    return per_block[["block_number", "gas_used", "gas_limit", "base_fee_per_gas"]]


def next_base_fee(parent_base_fee: int, parent_gas_used: int, parent_gas_target: int) -> int:
    """EIP-1559 base fee update. Duplicated from `sim.metrics`'s canonical form.

    Kept in sync deliberately, including the `MIN_BASE_FEE` floor: these headers
    now supply the demand model's per-cohort anchor prices, and an anchor of 0
    would silently disable the bid rescale for that cohort.
    """
    if parent_gas_target <= 0 or parent_gas_used == parent_gas_target:
        return max(MIN_BASE_FEE, parent_base_fee)
    if parent_gas_used > parent_gas_target:
        delta = parent_base_fee * (parent_gas_used - parent_gas_target)
        return max(
            MIN_BASE_FEE,
            parent_base_fee
            + max(1, delta // parent_gas_target // BASE_FEE_MAX_CHANGE_DENOMINATOR),
        )
    delta = parent_base_fee * (parent_gas_target - parent_gas_used)
    return max(
        MIN_BASE_FEE,
        parent_base_fee - delta // parent_gas_target // BASE_FEE_MAX_CHANGE_DENOMINATOR,
    )
