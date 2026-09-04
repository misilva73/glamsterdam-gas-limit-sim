"""Arrival cohorts, composition pools, and the price-responsive demand model.

The transactions historically included in source block `n` form one cohort.
Cohorts are held as flat numpy arrays plus per-cohort offsets rather than one
frame per block: a full week is ~8M rows and the engine touches a pool of them
every step.

**Quantity and composition are separated.** How much gas arrives at a step is set
entirely by the isoelastic demand model (`demand_multiplier`) against a single
fixed reference point for the whole trace (`DemandReference`). *Which*
transactions make up that gas is drawn from a **composition pool**: a random
contiguous run of `composition_pool_blocks` cohorts, redrawn independently every
step. The pool sets the mix and nothing else -- it does not set the quantity, and
consecutive steps are independent, so this is not a moving-block bootstrap and no
historical autocorrelation survives it. Contiguity earns its place only by
keeping each step's mix inside one stretch of one fee regime rather than smearing
it across the whole trace. See `METHODOLOGY.md` section 6.

A workload item is identified by `(run_index, simulation_position, tx_hash,
replica_index)`. `tx_hash` is deliberately *not* carried through sampling --
`(source_block_number, tx_index)` maps to it one-to-one and costs 16 bytes
instead of a Python string -- so the engine never materialises the multiplied
trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Scenario, SimConfig
from sim.metrics import effective_tip, is_legacy_tx_type

# Columns `build_cohorts` needs; `execution_gas` / `state_gas` are derived by the
# data layer (`data.load_tx_gas_results.derive_gas_dimensions`), not recomputed here.
COHORT_SOURCE_COLUMNS = (
    "block_number",
    "tx_index",
    "tx_type",
    "max_fee_per_gas",
    "max_priority_fee_per_gas",
    "execution_gas",
    "state_gas",
    "schedule_gas_used",
)

PATH_COLUMNS = (
    "simulation_position",
    "pool_start_index",
    "pool_start_block",
)

_MAX_SAMPLING_BATCHES = 64
"""Safety cap on gas-target sampling batches, so a degenerate pool cannot hang."""

_INT64_FLOAT_CAP = float(np.nextafter(2.0**63, 0.0))
"""Largest float64 that survives a cast to int64.

`float64(int64_max)` rounds *up* to 2**63, which is out of range, so clipping a
float to `int64_max` still produces an invalid cast. This is the value below it.
"""


@dataclass(frozen=True)
class DemandReference:
    """The fixed `(price, quantity)` pair the demand curve is anchored at.

    An isoelastic demand curve is only a curve relative to a reference point, so
    both halves are single trace-level constants rather than per-step values:
    "at price `price`, demand is `demand_level * gas` per block". That is what
    makes `demand_level` mean something stable, and it makes
    `realized_demand_multiplier` move with the *simulated* price and nothing
    else. Anchoring on each step's own sampled history instead would put a
    different curve under every step and inject noise that no modelled quantity
    produced.

    `price` is gas-weighted over every transaction in the trace, on the same
    basis as the simulated `priority_fees_wei / sender_gas_used`: historical base
    fee plus realised tip, weighted by sender-facing gas. `tip` is the tip half
    of it alone, which seeds the engine's running tip estimate. `gas` is the mean
    per-cohort `total_gas`, so `demand_level = 1` is one trace-average block's
    worth of demand per step.
    """

    price: float
    gas: float
    tip: float


@dataclass(frozen=True)
class Cohorts:
    """Per-source-block arrays of simulatable transactions.

    `columns` holds one flat array per field, ordered by `(block_number,
    tx_index)`; `offsets[i]:offsets[i + 1]` is cohort `i`.

    `total_gas` is the demand model's quantity measure, `execution_gas +
    state_gas` per transaction -- the report's `S + B`, i.e. the *historical*
    metering basis on which the elasticities were estimated. It is deliberately
    not the simulator's capacity measure, which is `max(execution, state)` per
    block.

    `anchor_price` and `anchor_tip` are per cohort: the effective gas price at
    which that cohort's demand was actually observed, and the realised
    gas-weighted mean tip inside it. The demand model reads neither -- it uses
    the trace-level `reference` -- but they are what `reference` aggregates and
    are kept for diagnostics and for the basis tests.
    """

    block_numbers: np.ndarray
    offsets: np.ndarray
    columns: dict[str, np.ndarray]
    total_gas: np.ndarray
    anchor_price: np.ndarray
    anchor_tip: np.ndarray
    reference: DemandReference

    def __len__(self) -> int:
        return int(self.block_numbers.size)

    def cohort(self, index: int) -> dict[str, np.ndarray]:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return {name: column[start:stop] for name, column in self.columns.items()}

    def cohort_total_gas(self, index: int) -> int:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return int(self.total_gas[start:stop].sum())

    @property
    def tx_counts(self) -> np.ndarray:
        return np.diff(self.offsets)


def build_cohorts(tx_frame: pd.DataFrame, headers: pd.DataFrame) -> Cohorts:
    """Flatten simulatable transactions into cohorts, one per source block.

    `headers` supplies the historical base fee per source block, which the bid
    rescale prices against and the demand reference is aggregated from.
    """
    missing = [c for c in COHORT_SOURCE_COLUMNS if c not in tx_frame.columns]
    if missing:
        raise ValueError(f"tx_frame is missing required columns: {missing}")

    frame = tx_frame
    # Every row is demand, failures included: they still occupy block space and
    # still pay. There is no inclusion filter anywhere in the pipeline, so this
    # takes the frame exactly as given.
    if frame.empty:
        raise ValueError("tx_frame contains no simulatable transactions")

    frame = frame.sort_values(["block_number", "tx_index"], kind="stable")
    source_block_number = frame["block_number"].to_numpy(np.int64)
    block_numbers, counts = np.unique(source_block_number, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    columns = {"source_block_number": source_block_number}
    for name in COHORT_SOURCE_COLUMNS[1:]:
        columns[name] = frame[name].to_numpy(np.int64)
    columns["is_legacy"] = is_legacy_tx_type(columns["tx_type"])
    total_gas = columns["execution_gas"] + columns["state_gas"]

    anchor_base_fee, anchor_tip = _cohort_anchors(columns, block_numbers, counts, headers)
    columns["anchor_base_fee"] = anchor_base_fee
    anchor_price = anchor_base_fee[offsets[:-1]].astype(np.float64) + anchor_tip
    return Cohorts(
        block_numbers=block_numbers,
        offsets=offsets,
        columns=columns,
        total_gas=total_gas,
        anchor_price=anchor_price,
        anchor_tip=anchor_tip,
        reference=_demand_reference(columns, total_gas, block_numbers.size),
    )


def _demand_reference(
    columns: dict[str, np.ndarray], total_gas: np.ndarray, num_cohorts: int
) -> DemandReference:
    """The trace-level reference point of the demand curve. See `DemandReference`.

    The price is gas-weighted over *transactions*, not averaged over cohorts: a
    30M-gas block has to count for six times a 5M-gas one, because that is the
    proportion in which the trace actually supplies gas. Averaging the per-cohort
    anchors instead would weight every block equally and drift off the
    `priority_fees_wei / sender_gas_used` basis the simulated side is measured on.

    A trace with no sender-facing gas at all has no observable price, so the
    reference price and tip fall back to 0 -- which `demand_multiplier` reads as
    "no anchor", returning a flat `demand_level`.
    """
    # float64 throughout: tip x gas reaches ~1e19 for a large high-tip block,
    # which overflows int64.
    weight = columns["schedule_gas_used"].astype(np.float64)
    tip = np.clip(
        effective_tip(
            columns["tx_type"],
            columns["max_fee_per_gas"],
            columns["max_priority_fee_per_gas"],
            columns["anchor_base_fee"],
        ),
        0,
        None,
    ).astype(np.float64)
    total_weight = float(weight.sum())
    if total_weight <= 0:
        return DemandReference(price=0.0, gas=0.0, tip=0.0)
    mean_base_fee = float((columns["anchor_base_fee"].astype(np.float64) * weight).sum())
    mean_tip = float((tip * weight).sum()) / total_weight
    return DemandReference(
        price=mean_base_fee / total_weight + mean_tip,
        gas=float(total_gas.sum()) / max(num_cohorts, 1),
        tip=mean_tip,
    )


def _cohort_anchors(
    columns: dict[str, np.ndarray],
    block_numbers: np.ndarray,
    counts: np.ndarray,
    headers: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-transaction anchor base fee and per-cohort realised mean tip.

    The anchor is the price the cohort's demand was *observed* at, so it must be
    on the same basis as the simulated price signal: base fee plus the tip
    actually realised, gas-weighted by sender-facing gas exactly as
    `priority_fees_wei / sender_gas_used` is on the simulated side.

    Tips are floored at zero. Every row here was historically included, so its
    fee cap covered the base fee of its own block and the tip is non-negative;
    the clip guards against header/replay disagreement rather than modelling
    anything.
    """
    by_block = headers.drop_duplicates("block_number").set_index("block_number")[
        "base_fee_per_gas"
    ]
    aligned = by_block.reindex(block_numbers)
    if aligned.isna().any():
        absent = block_numbers[aligned.isna().to_numpy()]
        raise ValueError(
            f"block headers are missing {absent.size} of {block_numbers.size} cohort "
            f"blocks, so their demand anchor cannot be computed; first missing: "
            f"{absent[:5].tolist()}"
        )
    per_cohort_base_fee = aligned.to_numpy(np.int64)
    anchor_base_fee = np.repeat(per_cohort_base_fee, counts)

    tip = np.clip(
        effective_tip(
            columns["tx_type"],
            columns["max_fee_per_gas"],
            columns["max_priority_fee_per_gas"],
            anchor_base_fee,
        ),
        0,
        None,
    )
    # float64 throughout: tip x gas reaches ~1e19 for a large high-tip block,
    # which overflows int64.
    weight = columns["schedule_gas_used"].astype(np.float64)
    weighted_tip = _segment_sum(tip.astype(np.float64) * weight, counts)
    total_weight = _segment_sum(weight, counts)
    anchor_tip = np.divide(
        weighted_tip, total_weight, out=np.zeros_like(weighted_tip), where=total_weight > 0
    )
    return anchor_base_fee, anchor_tip


def _segment_sum(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Sum `values` within consecutive runs of length `counts`.

    `reduceat` rather than differenced cumulative sums: it accumulates per
    segment, so an 8M-row trace does not have a ~1e21 running total differenced
    to recover a ~1e16 block figure.
    """
    starts = np.concatenate([[0], np.cumsum(counts[:-1])])
    return np.add.reduceat(values, starts)


def composition_pools(
    cohorts: Cohorts, horizon: int, pool_blocks: int, rng: np.random.Generator
) -> pd.DataFrame:
    """One composition pool per step: `pool_blocks` contiguous cohorts, redrawn each step.

    Starts are independent across steps and drawn with replacement, so pools
    repeat and overlap freely. Nothing about a step's arrivals depends on the
    step before it -- the pool supplies the transaction mix, and the demand model
    supplies the quantity.

    Pools never wrap past the end of the source trace, so cohorts in its last
    `pool_blocks - 1` blocks appear in fewer pools than mid-trace ones. That edge
    bias is accepted for the same reason as before: wrapping would build a pool
    out of two unrelated fee regimes spliced end to start.
    """
    if pool_blocks > len(cohorts):
        raise ValueError(
            f"pool_blocks {pool_blocks} exceeds the {len(cohorts)}-cohort source trace"
        )
    starts = rng.integers(0, len(cohorts) - pool_blocks + 1, size=horizon).astype(np.int64)
    return pd.DataFrame(
        {
            "simulation_position": np.arange(starts.size, dtype=np.int64),
            "pool_start_index": starts,
            "pool_start_block": cohorts.block_numbers[starts].astype(np.int64),
        },
        columns=list(PATH_COLUMNS),
    )


# --- Demand model -----------------------------------------------------------------


def demand_pool_bounds(
    cohorts: Cohorts, path: pd.DataFrame, pool_blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per step, the half-open flat-array slice its arrivals are drawn from.

    A pool is contiguous in cohort index, hence a contiguous slice of the flat
    arrays, so sampling is an integer draw with no gather to build the pool.
    """
    starts = path["pool_start_index"].to_numpy(np.int64)
    return cohorts.offsets[starts], cohorts.offsets[starts + pool_blocks]


def demand_multiplier(
    cfg: SimConfig, scenario: Scenario, price_signal: float, anchor_price: float
) -> tuple[float, bool]:
    """Isoelastic demand at `price_signal`, relative to the reference price.

    Returns the multiplier and whether the clamp bound. `anchor_price` is the
    trace-level `DemandReference.price`, fixed for the whole run, so the only
    thing that moves the multiplier is the simulated price signal.

    The clamp is not cosmetic: `p ** -elasticity` is unbounded as the price
    falls, and the elasticity was estimated over base fees spanning roughly
    0.1-10 gwei, so a price far outside that range is an extrapolation the model
    cannot support. Recording when it binds keeps that visible in the output.

    It bounds the **price response only**, not the product. `demand_level` is a
    deliberate choice by the caller about how much latent demand to assume, so a
    level above the bound is not extrapolation and must not be silently capped.
    """
    if scenario.aggregate_elasticity == 0.0 or anchor_price <= 0 or price_signal <= 0:
        return float(scenario.demand_level), False
    low, high = cfg.demand_multiplier_bounds
    response = (price_signal / anchor_price) ** -scenario.aggregate_elasticity
    clamped = min(max(response, low), high)
    return float(scenario.demand_level * clamped), bool(clamped != response)


def sample_arrivals(
    cohorts: Cohorts,
    pool: tuple[int, int],
    gas_target: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """One step's arrivals: `gas_target` gas of transactions drawn from `pool`.

    The whole arrival is sampled -- **uniformly over the pool's transactions,
    with replacement**, until cumulative `total_gas` crosses the target -- so the
    quantity is the demand model's alone and the pool contributes only the mix.
    Uniform draws rather than gas-weighted ones: uniform reproduces the pool's
    own gas distribution, where weighting by gas would over-represent large
    transactions and quietly reshape the mix it is supposed to preserve.

    The crossing draw is kept, so the overshoot is at most one transaction
    against a block-scale target -- far below the resolution of anything measured
    here, and correcting it would buy nothing.

    Arrivals are emitted sorted by flat position, which is `(source_block_number,
    tx_index)` order, with `replica_index` breaking ties among repeats of the
    same source transaction. That is the ordering `sim.engine`'s append-only
    mempool invariant depends on.
    """
    positions = (
        _sample_to_gas_target(cohorts.total_gas, pool, gas_target, rng)
        if gas_target > 0
        else np.empty(0, np.int64)
    )
    positions = np.sort(positions)
    arrivals = {name: column[positions] for name, column in cohorts.columns.items()}
    _, counts = np.unique(positions, return_counts=True)
    arrivals["replica_index"] = _replica_index(counts)
    return arrivals


def _sample_to_gas_target(
    total_gas: np.ndarray, pool: tuple[int, int], target: float, rng: np.random.Generator
) -> np.ndarray:
    """Uniform draws with replacement from `pool` until cumulative gas >= target."""
    low, high = int(pool[0]), int(pool[1])
    mean_gas = float(total_gas[low:high].mean()) if high > low else 0.0
    if mean_gas <= 0:
        return np.empty(0, np.int64)

    drawn = []
    accumulated = 0.0
    for _ in range(_MAX_SAMPLING_BATCHES):
        size = int((target - accumulated) / mean_gas * 1.25) + 8
        picks = rng.integers(low, high, size=size, dtype=np.int64)
        cumulative = accumulated + np.cumsum(total_gas[picks].astype(np.float64))
        crossed = int(np.searchsorted(cumulative, target, side="left"))
        if crossed < size:
            drawn.append(picks[: crossed + 1])
            return np.concatenate(drawn)
        drawn.append(picks)
        accumulated = float(cumulative[-1])
    raise RuntimeError(
        f"gas-target sampling did not reach {target:,.0f} gas in "
        f"{_MAX_SAMPLING_BATCHES} batches from pool [{low}, {high})"
    )


def _replica_index(counts: np.ndarray) -> np.ndarray:
    """0..c-1 within each source transaction, without a Python loop."""
    starts = np.cumsum(counts) - counts
    return np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(starts, counts)


def adapt_bids(arrivals: dict[str, np.ndarray], base_fee: int) -> dict[str, np.ndarray]:
    """Reprice historical fee caps from their own block's base fee to `base_fee`.

    Without this the fee filter, not the elasticity, would set the quantity of
    demand: a cohort observed at 1 gwei placed into a 100 gwei block is entirely
    fee-ineligible, and the same cohort in a 0.01 gwei block has meaningless
    headroom and a flat tip order.

    Two rules, both preserving eligibility exactly (a transaction includable at
    its own block's base fee is includable here), so quantity stays the demand
    model's job and the caps are left to set *ordering*:

    - **Dynamic-fee rows: scale the cap, hold the priority fee.** Scaling the cap
      by `base_fee / anchor_base_fee` is the same thing as scaling its headroom
      over the base fee -- `(f - b0) * bt/b0 + bt == f * bt/b0` -- and leaving
      `max_priority_fee_per_gas` alone keeps tips in absolute wei, which is how
      they actually behave.
    - **Legacy / access-list rows: shift the gas price.** Their
      `max_fee_per_gas` is a normalized gas price whose *whole* headroom is the
      tip (`sim.metrics.tip_given_legacy_mask`), so scaling it would scale the
      tip too. Shifting by `base_fee - anchor_base_fee` holds that tip absolute,
      which is the same intent as holding the priority fee absolute above.

    Rows whose anchor base fee is 0 are left alone: the ratio is undefined, and a
    0 anchor means the source block had no meaningful price to rescale from.
    """
    anchor = arrivals["anchor_base_fee"]
    max_fee = arrivals["max_fee_per_gas"]
    is_legacy = arrivals["is_legacy"]
    scalable = anchor > 0

    # float64 for the product: max_fee x base_fee reaches ~1e23 at plausible
    # values and overflows int64. 15 significant digits of wei is far below the
    # resolution any conclusion here rests on.
    ratio = np.divide(
        float(base_fee), anchor.astype(np.float64), out=np.ones(anchor.shape), where=scalable
    )
    # Clipped before the cast: a near-zero anchor makes the ratio large enough
    # that the product exceeds int64 even though the float64 product is fine.
    scaled = np.clip(np.rint(max_fee.astype(np.float64) * ratio), 0, _INT64_FLOAT_CAP)
    # Same float64-and-clip guard as `scaled`: the sum overflows int64 once the
    # base fee and the cap are both large, which raised `OverflowError` here
    # before `MAX_BASE_FEE` existed. The cap alone is not sufficient -- a fee cap
    # already near int64 would still overflow the addition.
    shifted = np.clip(
        max_fee.astype(np.float64) + (float(base_fee) - anchor.astype(np.float64)),
        0,
        _INT64_FLOAT_CAP,
    )

    repriced = np.where(is_legacy, shifted.astype(np.int64), scaled.astype(np.int64))
    return {
        **arrivals,
        "max_fee_per_gas": np.maximum(np.where(scalable, repriced, max_fee), 0).astype(np.int64),
    }


def bootstrap_rng(cfg: SimConfig, run_index: int) -> np.random.Generator:
    """Pool-selection stream for one path; independent across runs and seeds."""
    return np.random.default_rng([cfg.bootstrap_seed, run_index])


def demand_rng(cfg: SimConfig, run_index: int) -> np.random.Generator:
    """Demand-sampling stream for one path, independent of the pool draw."""
    return np.random.default_rng([cfg.demand_seed, run_index])
