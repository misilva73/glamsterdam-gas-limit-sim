"""The per-step simulation loop: fee update, gas-limit ramp, and block fill.

The mempool is parallel numpy arrays kept in inclusion-tiebreak order
(`schemas.INCLUSION_TIEBREAK_COLUMNS`). That invariant is what makes the hot
path affordable: arrivals only ever append (their `arrival_step` is the largest
so far, and `sample_arrivals` emits `(source_block_number, tx_index,
replica_index)` order within a step) and removals preserve relative order, so the
full tie-break chain reduces to a *stable* sort on the tip alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import SimConfig
from schemas import PER_STEP_COLUMNS
from sim.metrics import (
    BACKLOG_COLUMNS,
    next_base_fee,
    ramp_gas_limit,
    realized_tip_per_gas,
    step_record,
    tip_given_legacy_mask,
    update_price_signal,
)
from sim.workload import (
    Cohorts,
    adapt_bids,
    demand_multiplier,
    demand_pool_bounds,
    demand_rng,
    sample_arrivals,
)

MEMPOOL_FIELDS = (
    "max_fee_per_gas",
    "max_priority_fee_per_gas",
    "is_legacy",
    "execution_gas",
    "state_gas",
    "schedule_gas_used",
)

# How many top-of-book candidates to sort at a time. Large enough that a full
# 200M-gas block is normally filled from the first tranche, small enough that a
# million-item backlog is never sorted in full.
TIP_TRANCHE_SIZE = 16_384

# Compact the mempool once this share of it is tombstoned. Lower values spend
# more time copying; higher ones spend more time scanning dead entries.
COMPACTION_DEAD_SHARE = 0.25

_NULLABLE_IDENTITY = ("source_block_number", "window_instance", "position_in_window")


def run_path(
    cohorts: Cohorts,
    path: pd.DataFrame,
    cfg: SimConfig,
    *,
    run_index: int,
    starting_base_fee: int,
) -> pd.DataFrame:
    """Simulate one arrival path plus its drain phase.

    Every path in a scenario starts from the same state -- empty mempool,
    `starting_base_fee`, `cfg.fusaka_gas_limit`, and a price signal seeded at the
    first cohort's anchor -- and sampled cohorts never reset it, so bootstrap
    spread reflects workload variation alone.

    The demand model closes a loop the simulation did not previously have: the
    base fee sets the price signal, the price signal sets how much demand
    arrives, and that demand sets the next base fee. Everything that feeds the
    loop is *parent-derived*, so within a step the multiplier is computed from
    the price signal as it stood before this block was built, and the signal is
    updated from the block's own outcome only afterwards.
    """
    _require_anchors(cohorts, cfg)
    rng = demand_rng(cfg, run_index)
    cohort_index = path["cohort_index"].to_numpy(np.int64)
    source_block_number = path["source_block_number"].to_numpy(np.int64)
    window_instance = path["window_instance"].to_numpy(np.int64)
    position_in_window = path["position_in_window"].to_numpy(np.int64)
    pool_low, pool_high = demand_pool_bounds(
        cohorts, path, cfg.bootstrap_window_blocks, cfg.arrival_mode
    )

    pool = Mempool()
    base_fee = int(starting_base_fee)
    gas_limit = int(cfg.fusaka_gas_limit)
    parent_gas_used: int | None = None
    backlog_execution_gas = backlog_state_gas = 0
    # Seeding the signal at the first cohort's own anchor makes step 0's
    # multiplier exactly `demand_level`, whatever `starting_base_fee` is: the
    # reference path begins at its anchor by construction. The tip estimate is
    # seeded from the same cohort, so an empty first block does not drag the
    # signal toward a tipless price it never actually saw.
    anchor_price = _anchor_prices(cohorts, cohort_index)
    price_signal = float(anchor_price[0]) if anchor_price is not None else float(base_fee)
    prevailing_tip = _seed_tip(cohorts, cohort_index)

    records = []
    for position in range(cohort_index.size + cfg.drain_blocks):
        is_drain_step = position >= cohort_index.size

        # Both persistent-state updates are parent-derived, so position 0 reports
        # the configured initial state verbatim. From then on the ramp applies at
        # the configured step itself, with no activation delay.
        if parent_gas_used is not None:
            base_fee = next_base_fee(base_fee, parent_gas_used, gas_limit)
            if position >= max(1, cfg.first_glamsterdam_simulation_step):
                gas_limit = ramp_gas_limit(gas_limit, cfg.glamsterdam_gas_limit)

        arrived_tx_count = arrived_execution_gas = arrived_state_gas = 0
        cohort_anchor = multiplier = None
        clamped = False
        if not is_drain_step:
            cohort_anchor = (
                float(anchor_price[position]) if anchor_price is not None else float(base_fee)
            )
            multiplier, clamped = demand_multiplier(cfg, price_signal, cohort_anchor)
            arrivals = sample_arrivals(
                cohorts,
                int(cohort_index[position]),
                (int(pool_low[position]), int(pool_high[position])),
                multiplier,
                rng,
            )
            if cfg.adapt_bids:
                arrivals = adapt_bids(arrivals, base_fee)
            admit(pool, arrivals)
            arrived_tx_count = int(arrivals["replica_index"].size)
            arrived_execution_gas = int(arrivals["execution_gas"].sum())
            arrived_state_gas = int(arrivals["state_gas"].sum())
            backlog_execution_gas += arrived_execution_gas
            backlog_state_gas += arrived_state_gas

        chosen, block = fill_and_measure(
            pool, base_fee, gas_limit, backlog_execution_gas, backlog_state_gas
        )
        retire(pool, chosen)
        parent_gas_used = max(block["block_execution_gas_used"], block["block_state_gas_used"])
        backlog_execution_gas = block["backlog"]["backlog_execution_gas"]
        backlog_state_gas = block["backlog"]["backlog_state_gas"]

        records.append(
            step_record(
                cfg,
                run_index=run_index,
                simulation_position=position,
                is_drain_step=is_drain_step,
                source_block_number=None if is_drain_step else int(source_block_number[position]),
                window_instance=None if is_drain_step else int(window_instance[position]),
                position_in_window=None if is_drain_step else int(position_in_window[position]),
                demand_price_signal=price_signal,
                cohort_anchor_price=cohort_anchor,
                realized_demand_multiplier=multiplier,
                demand_multiplier_clamped=clamped,
                base_fee_per_gas=base_fee,
                gas_limit=gas_limit,
                included_tx_count=int(chosen.size),
                arrived_tx_count=arrived_tx_count,
                arrived_execution_gas=arrived_execution_gas,
                arrived_state_gas=arrived_state_gas,
                **block,
            )
        )

        prevailing_tip = realized_tip_per_gas(
            block["priority_fees_wei"], block["sender_gas_used"], prevailing_tip
        )
        price_signal = update_price_signal(
            price_signal, base_fee, prevailing_tip, cfg.price_ema_alpha
        )
    return _per_step_frame(records)


def _require_anchors(cohorts: Cohorts, cfg: SimConfig) -> None:
    """Both price-responsive features need the price the cohort was observed at."""
    if cohorts.has_anchors:
        return
    needs = [
        name
        for name, active in (
            ("aggregate_elasticity", cfg.aggregate_elasticity != 0.0),
            ("adapt_bids", cfg.adapt_bids),
        )
        if active
    ]
    if needs:
        raise ValueError(
            f"{' and '.join(needs)} cannot run without per-cohort anchor prices, which "
            "come from the block headers passed to build_cohorts(tx_frame, headers). "
            "Either supply headers, or set aggregate_elasticity=0 and adapt_bids=False "
            "for a flat demand multiplier with frozen fee caps."
        )


def _anchor_prices(cohorts: Cohorts, cohort_index: np.ndarray) -> np.ndarray | None:
    """Anchor price per simulation position, or None when no headers were given."""
    if not cohorts.has_anchors or cohort_index.size == 0:
        return None
    return cohorts.anchor_price[cohort_index]


def _seed_tip(cohorts: Cohorts, cohort_index: np.ndarray) -> float:
    """The first cohort's realised mean tip, or 0 when there is nothing to go on."""
    if cohorts.anchor_tip is None or cohort_index.size == 0:
        return 0.0
    return float(cohorts.anchor_tip[cohort_index[0]])


# --- Mempool ----------------------------------------------------------------------


@dataclass
class Mempool:
    """Pending workload items as parallel arrays, oldest first.

    Included items are tombstoned rather than removed: at 5x demand the pool
    reaches millions of items, and gathering all of them into fresh arrays every
    block costs more than the rest of the loop put together. Compaction runs once
    a quarter of the pool is dead, so removal is amortised O(1) per item and the
    per-block cost is one scan.
    """

    columns: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            name: np.empty(0, _mempool_dtype(name)) for name in MEMPOOL_FIELDS
        }
    )
    alive: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    size: int = 0
    dead: int = 0

    def live_count(self) -> int:
        return self.size - self.dead

    def view(self, name: str) -> np.ndarray:
        return self.columns[name][: self.size]


def _mempool_dtype(name: str):
    return bool if name == "is_legacy" else np.int64


def admit(pool: Mempool, arrivals: dict[str, np.ndarray]) -> None:
    """Append one expanded cohort to the tail, preserving tie-break order."""
    count = int(arrivals["replica_index"].size)
    if count == 0:
        return
    _reserve(pool, pool.size + count)
    end = pool.size + count
    for name, column in pool.columns.items():
        column[pool.size : end] = arrivals[name]
    pool.alive[pool.size : end] = True
    pool.size = end


def retire(pool: Mempool, chosen: np.ndarray) -> None:
    pool.alive[chosen] = False
    pool.dead += int(chosen.size)
    if pool.dead > COMPACTION_DEAD_SHARE * pool.size:
        _compact(pool)


def _reserve(pool: Mempool, needed: int) -> None:
    capacity = pool.alive.size
    if needed <= capacity:
        return
    capacity = max(2 * capacity, needed, 1024)
    pool.columns = {name: _grown(column, capacity) for name, column in pool.columns.items()}
    pool.alive = _grown(pool.alive, capacity)


def _grown(column: np.ndarray, capacity: int) -> np.ndarray:
    grown = np.empty(capacity, column.dtype)
    grown[: column.size] = column
    return grown


def _compact(pool: Mempool) -> None:
    live = pool.alive[: pool.size]
    kept = pool.live_count()
    for column in pool.columns.values():
        column[:kept] = column[: pool.size][live]
    pool.alive[:kept] = True
    pool.size, pool.dead = kept, 0


# --- One block --------------------------------------------------------------------


def fill_and_measure(
    pool: Mempool,
    base_fee: int,
    gas_limit: int,
    pool_execution_gas: int,
    pool_state_gas: int,
) -> tuple[np.ndarray, dict]:
    """Order, fill, and measure one block. Returns (included rows, metric fields).

    Everything that scales with the whole mempool is a single scan or a gather
    over the fee-eligible subset only -- at high demand multipliers the pool is
    dominated by fee-ineligible items that must not be touched twice. The pool's
    gas totals are carried in by the caller for the same reason.
    """
    execution_gas, state_gas = pool.view("execution_gas"), pool.view("state_gas")
    candidates = np.flatnonzero(
        (pool.view("max_fee_per_gas") >= base_fee) & pool.alive[: pool.size]
    )
    tips = tip_given_legacy_mask(
        pool.view("is_legacy")[candidates],
        pool.view("max_fee_per_gas")[candidates],
        pool.view("max_priority_fee_per_gas")[candidates],
        base_fee,
    )
    taken = select_included(tips, execution_gas, state_gas, gas_limit, candidates)
    chosen = candidates[taken]

    block_execution_gas_used = int(execution_gas[chosen].sum())
    block_state_gas_used = int(state_gas[chosen].sum())
    sender_gas_used = pool.view("schedule_gas_used")[chosen]

    eligible_execution_gas = int(execution_gas[candidates].sum())
    eligible_state_gas = int(state_gas[candidates].sum())
    backlog = {
        "backlog_tx_count": pool.live_count() - chosen.size,
        "backlog_eligible_tx_count": candidates.size - chosen.size,
        "backlog_fee_ineligible_tx_count": pool.live_count() - candidates.size,
        "backlog_execution_gas": pool_execution_gas - block_execution_gas_used,
        "backlog_state_gas": pool_state_gas - block_state_gas_used,
        "backlog_eligible_execution_gas": eligible_execution_gas - block_execution_gas_used,
        "backlog_eligible_state_gas": eligible_state_gas - block_state_gas_used,
        "backlog_fee_ineligible_execution_gas": pool_execution_gas - eligible_execution_gas,
        "backlog_fee_ineligible_state_gas": pool_state_gas - eligible_state_gas,
    }
    metrics = {
        "block_execution_gas_used": block_execution_gas_used,
        "block_state_gas_used": block_state_gas_used,
        # Post-refund, floor-applied sender cost. Never a capacity input.
        "sender_gas_used": int(sender_gas_used.sum()),
        "priority_fees_wei": _priority_fees_wei(tips[taken], sender_gas_used),
        "backlog": {name: int(backlog[name]) for name in BACKLOG_COLUMNS},
    }
    return chosen, metrics


def _priority_fees_wei(tips: np.ndarray, sender_gas_used: np.ndarray) -> float:
    """Proposer tip revenue for the block.

    Kept in float64 end to end, including in the output column: a 200M-gas block
    of high-tip transactions overflows int64 (5e11 wei/gas x 2e8 gas = 1e20).
    Converting to a Python int here would survive the sum but silently make the
    output column object-dtype and break the Parquet write. 15 significant digits
    of wei is far below the resolution any conclusion here rests on.
    """
    return float(np.multiply(tips, sender_gas_used, dtype=np.float64).sum())


def select_included(
    tips: np.ndarray,
    execution_gas: np.ndarray,
    state_gas: np.ndarray,
    gas_limit: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Positions (into `tips`) included, in tip-descending, tie-break order.

    Evaluated in tranches of the top of book so that a huge backlog is never
    sorted in full. A tranche is cut at a tip *value*, never mid-tie, so it is a
    genuine prefix of the full descending order; the stable sort inside it
    reproduces the tie-break chain, which the mempool's ordering already encodes.
    Between tranches the exact stopping test is "does anything left fit at all?".
    """
    if positions is None:
        positions = np.arange(tips.size)
    rest = np.arange(tips.size)
    remaining_execution = remaining_state = int(gas_limit)
    included = []
    while rest.size:
        tranche, rest = _next_tip_tranche(tips, rest)
        rows = positions[tranche]
        taken, remaining_execution, remaining_state = fill_block(
            execution_gas[rows], state_gas[rows], remaining_execution, remaining_state
        )
        included.append(tranche[taken])
        if rest.size:
            left = positions[rest]
            fits = (execution_gas[left] <= remaining_execution) & (
                state_gas[left] <= remaining_state
            )
            if not fits.any():
                break
    return np.concatenate(included) if included else np.empty(0, np.intp)


def _next_tip_tranche(tips: np.ndarray, rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rest.size <= TIP_TRANCHE_SIZE:
        return rest[np.argsort(-tips[rest], kind="stable")], rest[:0]
    values = tips[rest]
    cut = values.size - TIP_TRANCHE_SIZE
    threshold = np.partition(values, cut)[cut]
    head = rest[values >= threshold]
    return head[np.argsort(-tips[head], kind="stable")], rest[values < threshold]


def fill_block(
    execution_gas: np.ndarray,
    state_gas: np.ndarray,
    remaining_execution: int,
    remaining_state: int,
) -> tuple[np.ndarray, int, int]:
    """Greedy fill of candidates already in inclusion order, both dimensions bound.

    Skip-and-continue, not first-fit: an item that does not fit is skipped and
    the walk continues, so a small low-tip transaction can follow a skipped
    large high-tip one. Done vectorised, exactly: headroom only ever shrinks, so
    an item too large for the current headroom can never fit later and is
    dropped, and the longest prefix of the survivors whose cumulative gas fits is
    exactly what the sequential walk would have taken.
    """
    index = np.arange(execution_gas.size)
    taken = []
    while index.size:
        fits = (execution_gas <= remaining_execution) & (state_gas <= remaining_state)
        if not fits.any():
            break
        keep = np.flatnonzero(fits)
        index, execution_gas, state_gas = index[keep], execution_gas[keep], state_gas[keep]

        cumulative_execution = np.cumsum(execution_gas)
        cumulative_state = np.cumsum(state_gas)
        within = (cumulative_execution <= remaining_execution) & (
            cumulative_state <= remaining_state
        )
        # `within[0]` holds because the survivors each fit on their own, so every
        # pass takes at least one item and the loop terminates.
        stop = within.size if within.all() else int(np.argmin(within))
        taken.append(index[:stop])
        remaining_execution -= int(cumulative_execution[stop - 1])
        remaining_state -= int(cumulative_state[stop - 1])
        # Item `stop` overflows the block: skip it, keep walking.
        index = index[stop + 1 :]
        execution_gas, state_gas = execution_gas[stop + 1 :], state_gas[stop + 1 :]
    positions = np.concatenate(taken) if taken else np.empty(0, np.intp)
    return positions, remaining_execution, remaining_state


def _per_step_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=list(PER_STEP_COLUMNS))
    dtypes = {name: "Int64" for name in _NULLABLE_IDENTITY}
    dtypes["is_drain_step"] = bool
    dtypes["demand_multiplier_clamped"] = bool
    dtypes["priority_fees_wei"] = "float64"  # can exceed int64; see _priority_fees_wei
    # NaN on drain steps, where no cohort arrives and so no multiplier is drawn.
    dtypes["cohort_anchor_price"] = "float64"
    dtypes["realized_demand_multiplier"] = "float64"
    dtypes["demand_price_signal"] = "float64"
    return frame.astype(dtypes)
