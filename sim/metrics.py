"""Protocol arithmetic and the per-step output row.

The three protocol rules the simulation depends on -- the EIP-1559 base-fee
update, the 1/1024 gas-limit ramp, and the effective tip by transaction type --
live here so the engine reads as the algorithm and the rules can be tested
against hand-computed cases.
"""

from __future__ import annotations

import numpy as np

from config import (
    BASE_FEE_MAX_CHANGE_DENOMINATOR,
    ELASTICITY_MULTIPLIER,
    GAS_LIMIT_RAMP_DENOMINATOR,
    LEGACY_TX_TYPES,
    MIN_BASE_FEE,
    SimConfig,
)
from schemas import (
    BOTTLENECK_EXECUTION,
    BOTTLENECK_NONE,
    BOTTLENECK_STATE,
    PER_STEP_COLUMNS,
)

_LEGACY_TX_TYPES = np.array(sorted(LEGACY_TX_TYPES), np.int64)

BACKLOG_COLUMNS = (
    "backlog_tx_count",
    "backlog_eligible_tx_count",
    "backlog_fee_ineligible_tx_count",
    "backlog_execution_gas",
    "backlog_state_gas",
    "backlog_eligible_execution_gas",
    "backlog_eligible_state_gas",
    "backlog_fee_ineligible_execution_gas",
    "backlog_fee_ineligible_state_gas",
)


def ramp_gas_limit(current_limit: int, target_limit: int) -> int:
    """One block of the 1/1024 climb toward `target_limit`.

    Only ever raises: a limit already at or above the target is left alone, so a
    scenario configured with a lower Glamsterdam target does not step down.
    """
    if current_limit >= target_limit:
        return current_limit
    return min(current_limit + current_limit // GAS_LIMIT_RAMP_DENOMINATOR, target_limit)


def next_base_fee(parent_base_fee: int, parent_gas_used: int, parent_gas_limit: int) -> int:
    """EIP-1559 base-fee update from the parent header.

    Integer arithmetic in the protocol's order: the increment is floored at 1 wei
    while the decrement is not, which is why a chain at target-plus-one drifts
    upward.

    The result is floored at `MIN_BASE_FEE`. Because the decrement is at most an
    eighth of the base fee itself, an emptying chain sticks at 7 wei or below on
    its own, so the floor binds only from a parent base fee of 0.
    """
    target = parent_gas_limit // ELASTICITY_MULTIPLIER
    if target <= 0 or parent_gas_used == target:
        return max(MIN_BASE_FEE, parent_base_fee)
    if parent_gas_used > target:
        delta = parent_base_fee * (parent_gas_used - target)
        return max(
            MIN_BASE_FEE,
            parent_base_fee + max(1, delta // target // BASE_FEE_MAX_CHANGE_DENOMINATOR),
        )
    delta = parent_base_fee * (target - parent_gas_used)
    return max(
        MIN_BASE_FEE, parent_base_fee - delta // target // BASE_FEE_MAX_CHANGE_DENOMINATOR
    )


def realized_tip_per_gas(
    priority_fees_wei: float, sender_gas_used: int, fallback: float
) -> float:
    """Gas-weighted mean tip actually paid in a block, on the sender-cost basis.

    Same basis as the historical anchor tip in `sim.workload`, so the demand
    model compares like with like. An empty block says nothing about the going
    tip, so it carries the previous estimate forward rather than reporting zero.
    """
    if sender_gas_used <= 0:
        return fallback
    return priority_fees_wei / sender_gas_used


def update_price_signal(previous: float, base_fee: int, tip: float, alpha: float) -> float:
    """One EMA step of the effective gas price users face.

    The demand model reacts to this rather than to the raw base fee, for two
    reasons. The elasticities are estimated on daily data while the per-block
    base fee is a +/-12.5% random walk; and near the base-fee floor the price a
    user actually pays is almost all tip, so base fee alone both diverges and
    wildly overstates how far the price fell. See METHODOLOGY.md section 6.2.
    """
    return alpha * (base_fee + tip) + (1.0 - alpha) * previous


def is_legacy_tx_type(tx_type) -> np.ndarray:
    """Legacy / access-list rows, whose `max_fee_per_gas` is a normalized gas_price."""
    return np.isin(tx_type, _LEGACY_TX_TYPES)


def effective_tip(tx_type, max_fee_per_gas, max_priority_fee_per_gas, base_fee) -> np.ndarray:
    """Tip per gas actually paid to the proposer at this base fee.

    Negative for a transaction priced below the base fee; the caller filters
    those out as fee-ineligible rather than clamping, so the sign carries
    information.
    """
    return tip_given_legacy_mask(
        is_legacy_tx_type(tx_type), max_fee_per_gas, max_priority_fee_per_gas, base_fee
    )


def tip_given_legacy_mask(is_legacy, max_fee_per_gas, max_priority_fee_per_gas, base_fee):
    """`effective_tip` with the type test hoisted out, for the per-block hot loop."""
    headroom = max_fee_per_gas - base_fee
    return np.where(is_legacy, headroom, np.minimum(max_priority_fee_per_gas, headroom))


def bottleneck_dimension(block_execution_gas_used: int, block_state_gas_used: int) -> str:
    """Which dimension bound the block; both share one limit, so compare raw gas."""
    if block_execution_gas_used == 0 and block_state_gas_used == 0:
        return BOTTLENECK_NONE
    if block_state_gas_used > block_execution_gas_used:
        return BOTTLENECK_STATE
    return BOTTLENECK_EXECUTION


def step_record(
    cfg: SimConfig,
    *,
    run_index: int,
    simulation_position: int,
    is_drain_step: bool,
    source_block_number: int | None,
    window_instance: int | None,
    position_in_window: int | None,
    demand_price_signal: float,
    cohort_anchor_price: float | None,
    realized_demand_multiplier: float | None,
    demand_multiplier_clamped: bool,
    base_fee_per_gas: int,
    gas_limit: int,
    block_execution_gas_used: int,
    block_state_gas_used: int,
    included_tx_count: int,
    sender_gas_used: int,
    priority_fees_wei: int,
    arrived_tx_count: int,
    arrived_execution_gas: int,
    arrived_state_gas: int,
    backlog: dict[str, int],
) -> dict:
    """One `schemas.PER_STEP_COLUMNS` row."""
    record = {
        "arrival_mode": cfg.arrival_mode,
        "run_index": run_index,
        "aggregate_elasticity": cfg.aggregate_elasticity,
        "demand_level": cfg.demand_level,
        "bootstrap_window_blocks": cfg.bootstrap_window_blocks,
        "simulation_position": simulation_position,
        "is_drain_step": is_drain_step,
        "source_block_number": source_block_number,
        "window_instance": window_instance,
        "position_in_window": position_in_window,
        "demand_price_signal": demand_price_signal,
        "cohort_anchor_price": cohort_anchor_price,
        "realized_demand_multiplier": realized_demand_multiplier,
        "demand_multiplier_clamped": demand_multiplier_clamped,
        "base_fee_per_gas": base_fee_per_gas,
        "gas_limit": gas_limit,
        # Header-equivalent: the two dimensions share one limit, so the block's
        # gas used is the larger of them, and that is the EIP-1559 parent input.
        "gas_used": max(block_execution_gas_used, block_state_gas_used),
        "block_execution_gas_used": block_execution_gas_used,
        "block_state_gas_used": block_state_gas_used,
        "execution_utilization": block_execution_gas_used / gas_limit,
        "state_utilization": block_state_gas_used / gas_limit,
        "bottleneck_dimension": bottleneck_dimension(
            block_execution_gas_used, block_state_gas_used
        ),
        "included_tx_count": included_tx_count,
        "sender_gas_used": sender_gas_used,
        "priority_fees_wei": priority_fees_wei,
        "arrived_tx_count": arrived_tx_count,
        "arrived_execution_gas": arrived_execution_gas,
        "arrived_state_gas": arrived_state_gas,
        **{name: backlog[name] for name in BACKLOG_COLUMNS},
    }
    assert tuple(record) == PER_STEP_COLUMNS, "step_record drifted from PER_STEP_COLUMNS"
    return record
