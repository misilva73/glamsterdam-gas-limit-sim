"""Configuration for the Fusaka -> Glamsterdam gas-limit simulation.

One dataclass holds every knob. Seeds for the independent random streams are
derived from `random_seed` so that changing, say, the demand-replication draw
cannot silently shift the bootstrap window selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"

ArrivalMode = Literal["historical", "moving_block_bootstrap"]
TxInclusionPolicy = Literal["all", "gas_rescued", "successful_only"]

DEFAULT_RESCUE_MULTIPLIER = 1.0
"""Assumed gas-limit multiplier when the replay never found one that succeeded.

Those transactions halted for a non-gas reason, so the sender had no reason to
sign anything beyond the limit they actually signed.
"""

GAS_LIMIT_RAMP_DENOMINATOR = 1024
"""EIP-1559 per-block gas-limit adjustment bound: 1/1024 of the parent limit."""

BASE_FEE_MAX_CHANGE_DENOMINATOR = 8
ELASTICITY_MULTIPLIER = 2

MIN_BASE_FEE = 1
"""Wei floor on the base fee.

Mostly a guard rather than a live constraint: the 1559 decrement is
`parent_base_fee * (target - gas_used) // target // 8`, which floors to zero once
the base fee drops below 8, so an emptying chain sticks at 7 wei or below on its
own rather than reaching zero. It binds only from a base fee of exactly 0, which
is reachable only via an explicit `starting_base_fee=0`.
"""

# Transaction types whose effective tip is capped by max_priority_fee_per_gas.
# 0 = legacy, 1 = access list (both carry a normalized gas_price in
# max_fee_per_gas); 2 = dynamic fee, 3 = blob, 4 = set code.
LEGACY_TX_TYPES = frozenset({0, 1})


@dataclass(frozen=True)
class SimConfig:
    # --- Gas limit trajectory -------------------------------------------------
    fusaka_gas_limit: int = 60_000_000
    glamsterdam_gas_limit: int = 200_000_000
    first_glamsterdam_simulation_step: int = 0

    # --- Source trace --------------------------------------------------------
    source_block_range: tuple[int, int] | None = None
    reference_start_block: int | None = None

    # --- Simulation ----------------------------------------------------------
    simulation_horizon_blocks: int | None = None  # None -> source trace length
    arrival_mode: ArrivalMode = "moving_block_bootstrap"
    bootstrap_window_blocks: int = 32
    starting_base_fee: int | None = None  # None -> derived from parent header
    drain_blocks: int = 0
    num_bootstrap_runs: int = 20
    random_seed: int = 20260831

    # --- Demand model --------------------------------------------------------
    # Isoelastic aggregate demand against a smoothed effective gas price:
    #   m = demand_level * (price_signal / cohort_anchor_price) ** -aggregate_elasticity
    # See METHODOLOGY.md section 6. `aggregate_elasticity = 0` reduces the model to
    # a flat `demand_level` multiplier -- the pre-demand-model behaviour, kept as
    # the sweep's baseline arm and as the mode that needs no anchor prices.
    aggregate_elasticity: float = 0.175
    demand_level: float = 1.0
    price_ema_blocks: int = 300
    demand_multiplier_bounds: tuple[float, float] = (0.05, 20.0)
    # Reprice historical fee caps from their own block's base fee to the
    # simulated one. Off means frozen caps, where the fee filter rather than the
    # elasticity sets how much demand is eligible. See `sim.workload.adapt_bids`.
    adapt_bids: bool = True

    # --- Data source ---------------------------------------------------------
    # Which replay rows count as demand. See `split_simulatable`.
    #   all              - every row, failures included (default)
    #   gas_rescued      - successes plus rows a higher gas limit would rescue
    #   successful_only  - the strict double-success rule
    tx_inclusion_policy: TxInclusionPolicy = "all"
    max_rescue_multiplier: float | None = None
    # Which reth replay config to read. `None` is only for tests and other callers
    # that never reach the table: `run_simulation` requires it on the command line,
    # and `clickhouse_tx_gas_results_sql` refuses to build a query without it, so a
    # run can never silently average two replay configurations.
    analysis_config_hash: str | None = None
    # The live table names the composite schedule 'amsterdam' (the execution-layer
    # fork name in execution-specs), not 'glamsterdam-v1'. A wrong name here
    # returns zero rows and the loader lists what is actually available.
    schedule_name: str = "amsterdam"
    schedule_config_hash: str | None = None
    chain_id: int = 1

    # --- Xatu ----------------------------------------------------------------
    meta_network_name: str = "mainnet"
    secrets_path: Path = REPO_ROOT / "secrets.json"

    # --- IO ------------------------------------------------------------------
    cache_dir: Path = DEFAULT_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        if self.demand_level <= 0:
            raise ValueError("demand_level must be positive")
        if self.aggregate_elasticity < 0:
            raise ValueError(
                "aggregate_elasticity is the absolute value of a negative slope, "
                "so it must be >= 0; a negative value would make demand rise with price"
            )
        if self.price_ema_blocks < 1:
            raise ValueError("price_ema_blocks must be >= 1")
        low, high = self.demand_multiplier_bounds
        if not 0 < low <= high:
            raise ValueError(f"demand_multiplier_bounds must satisfy 0 < low <= high, got {(low, high)}")
        if self.bootstrap_window_blocks < 1:
            raise ValueError("bootstrap_window_blocks must be >= 1")
        if self.drain_blocks < 0:
            raise ValueError("drain_blocks must be >= 0")

    # --- Derived seeds -------------------------------------------------------
    def derived_seed(self, stream: str) -> int:
        """Stable per-stream seed so independent draws never share a sequence."""
        digest = hashlib.blake2b(
            f"{self.random_seed}:{stream}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big")

    @property
    def bootstrap_seed(self) -> int:
        return self.derived_seed("bootstrap_window")

    @property
    def demand_seed(self) -> int:
        return self.derived_seed("demand_replication")

    @property
    def price_ema_alpha(self) -> float:
        """EMA smoothing factor, on pandas `ewm(span=...)` semantics."""
        return 2.0 / (self.price_ema_blocks + 1.0)

    def with_(self, **overrides) -> "SimConfig":
        return replace(self, **overrides)


@dataclass(frozen=True)
class SimulationGrid:
    """Axes swept by a simulation; each combination gets `num_bootstrap_runs` paths.

    The demand axes are the two halves of the model: `aggregate_elasticities`
    sets the *shape* of the response to price, `demand_levels` its *level*. The
    elasticity defaults span the report's event-based range (0.10-0.28) around
    its central 0.175, plus 0 for the flat-multiplier baseline.
    """

    aggregate_elasticities: tuple[float, ...] = (0.0, 0.10, 0.175, 0.28)
    demand_levels: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    bootstrap_window_blocks: tuple[int, ...] = (16, 32, 64)
    include_historical_reference: bool = True


DEFAULT_CONFIG = SimConfig()
"""Base every CLI override is layered onto, so unset flags mean the field default."""
