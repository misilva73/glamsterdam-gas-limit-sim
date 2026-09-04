#!/usr/bin/env python
"""Run a Fusaka -> Glamsterdam gas-limit simulation end to end.

Sweeps the elasticity x demand-level x bootstrap-window grid, runs
`num_bootstrap_runs` independent moving-block-bootstrap paths plus one matching
historical reference path per cell, and writes the per-step frame, the scenario
summaries, and a run manifest.

It does no analysis and draws no figures. The job here is to produce data;
reading it belongs in `notebooks/`.

Every path in a scenario starts from the identical initial state: empty mempool,
the base fee of the actual parent of `reference_start_block`, and
`fusaka_gas_limit`. Per-run randomness is derived from `random_seed`, so runs
are independent and the whole simulation is reproducible.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, SimConfig, SimulationGrid
from data.fetch_blocks import fetch_block_headers, starting_base_fee
from data.load_tx_gas_results import (
    derive_gas_dimensions,
    load_tx_gas_results,
    replay_outcome_summary,
)
from sim.engine import run_path
from sim.workload import bootstrap_path, bootstrap_rng, build_cohorts, historical_path

BOOTSTRAP_MODE = "moving_block_bootstrap"
HISTORICAL_MODE = "historical"

SATURATION_THRESHOLD = 0.99
"""Utilization at or above which a block counts as saturated in a dimension."""

CAVEAT = (
    "Frozen-trace simulation: repriced gas and success are fixed, and demand is an "
    "isoelastic model fitted to daily aggregates, extrapolated well outside its "
    "estimation range. Demand level 1x is an included-demand LOWER BOUND, and every "
    "number here is a sensitivity estimate under the observed transaction mix -- not "
    "an equilibrium forecast for a 200M limit."
)

REPORTED_LIBRARIES = (
    "pandas",
    "numpy",
    "pyarrow",
)


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_simulation",
        description=__doc__.splitlines()[0],
        epilog=CAVEAT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_argument_group("source data")
    source.add_argument(
        "--analysis-config-hash",
        required=True,
        help="reth replay config to read; mandatory, so a run can never silently "
        "mix datasets",
    )
    source.add_argument(
        "--schedule-name",
        help="repricing schedule whose gas is simulated (default: amsterdam)",
    )
    source.add_argument(
        "--schedule-config-hash",
        help="pin one revision of --schedule-name (default: whatever the table holds)",
    )
    source.add_argument("--chain-id", type=int, help="chain of the replay rows (default: 1)")
    source.add_argument(
        "--block-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="inclusive source block range",
    )
    source.add_argument(
        "--reference-start-block",
        type=int,
        help="first block of the historical path; its parent sets the starting base "
        "fee (default: first block of the range)",
    )
    source.add_argument(
        "--cache-dir", type=Path, help="parquet cache for fetched data (default: data/cache)"
    )

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--horizon", type=int, help="arrival steps (default: trace length)")
    sim.add_argument(
        "--arrival-mode",
        choices=(HISTORICAL_MODE, BOOTSTRAP_MODE),
        help="historical replays the trace once, with no bands; "
        "moving_block_bootstrap resamples cohort windows (default)",
    )
    sim.add_argument(
        "--num-runs", type=int, help="bootstrap paths per grid cell (default: 20)"
    )
    sim.add_argument(
        "--seed",
        type=int,
        help="master seed; all streams derive from it (default: 20260831)",
    )
    sim.add_argument("--starting-base-fee", type=int, help="wei; default: parent header")
    sim.add_argument(
        "--fusaka-gas-limit", type=int, help="gas limit at step 0 (default: 60,000,000)"
    )
    sim.add_argument(
        "--glamsterdam-gas-limit",
        type=int,
        help="ceiling the 1/1024 ramp climbs toward (default: 200,000,000)",
    )

    demand = parser.add_argument_group("demand model")
    demand.add_argument(
        "--price-ema-blocks",
        type=int,
        help="span of the EMA that smooths the effective gas price the demand model "
        "responds to; the elasticities are daily, so this should be hours not blocks "
        "(default: 300)",
    )
    demand.add_argument(
        "--multiplier-bounds",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="clamp on the demand multiplier, since p**-e is unbounded as the price "
        "falls and the elasticity was estimated over a narrow price range "
        "(default: 0.05 20)",
    )
    demand.add_argument(
        "--no-bid-adaptation",
        action="store_true",
        help="freeze historical fee caps instead of repricing them from their own "
        "block's base fee to the simulated one; makes the fee filter, not the "
        "elasticity, set how much demand is eligible",
    )

    grid = parser.add_argument_group("simulation grid")
    grid.add_argument(
        "--elasticities",
        nargs="+",
        type=float,
        metavar="E",
        help="demand-shape axis: aggregate price elasticity of demand "
        "(default: 0.10 0.175 0.28). 0 is a flat multiplier with no price response "
        "and is not swept: it cannot shed demand, so above demand level 1 the base "
        "fee runs to the MAX_BASE_FEE ceiling",
    )
    grid.add_argument(
        "--demand-levels",
        nargs="+",
        type=float,
        metavar="A",
        help="demand-level axis: latent-demand multiplier at the anchor price, "
        "standing in for never-included and secular-growth demand "
        "(default: 1 1.5 2)",
    )
    grid.add_argument(
        "--window-blocks",
        nargs="+",
        type=int,
        metavar="L",
        help="bootstrap axis: length in cohorts of each resampled window "
        "(default: 16 32 64)",
    )
    grid.add_argument(
        "--no-historical",
        action="store_true",
        help="skip the historical reference path (bands only)",
    )

    parser.add_argument("--output-dir", type=Path, help="destination (default: output/)")
    return parser


def build_config(args: argparse.Namespace, base: SimConfig = DEFAULT_CONFIG) -> SimConfig:
    """Resolve CLI overrides into a `SimConfig`. Pure, so it is directly testable."""
    windows = args.window_blocks or None
    elasticities = args.elasticities or None
    levels = args.demand_levels or None
    overrides = {
        "analysis_config_hash": args.analysis_config_hash,
        "schedule_name": args.schedule_name,
        "schedule_config_hash": args.schedule_config_hash,
        "chain_id": args.chain_id,
        "source_block_range": tuple(args.block_range) if args.block_range else None,
        "reference_start_block": args.reference_start_block,
        "cache_dir": args.cache_dir,
        "simulation_horizon_blocks": args.horizon,
        "arrival_mode": args.arrival_mode,
        "num_bootstrap_runs": args.num_runs,
        "random_seed": args.seed,
        "starting_base_fee": args.starting_base_fee,
        "fusaka_gas_limit": args.fusaka_gas_limit,
        "glamsterdam_gas_limit": args.glamsterdam_gas_limit,
        "output_dir": args.output_dir,
        "price_ema_blocks": args.price_ema_blocks,
        "demand_multiplier_bounds": (
            tuple(args.multiplier_bounds) if args.multiplier_bounds else None
        ),
        "adapt_bids": False if args.no_bid_adaptation else None,
        # The grid drives the sweep; the config carries its first cell so that a
        # single-scenario config is always self-consistent.
        "bootstrap_window_blocks": windows[0] if windows else None,
        "aggregate_elasticity": elasticities[0] if elasticities else None,
        "demand_level": levels[0] if levels else None,
    }
    return base.with_(**{k: v for k, v in overrides.items() if v is not None})


def build_grid(args: argparse.Namespace, base: SimulationGrid = SimulationGrid()) -> SimulationGrid:
    overrides = {
        "aggregate_elasticities": tuple(args.elasticities) if args.elasticities else None,
        "demand_levels": tuple(args.demand_levels) if args.demand_levels else None,
        "bootstrap_window_blocks": tuple(args.window_blocks) if args.window_blocks else None,
        "include_historical_reference": False if args.no_historical else None,
    }
    return SimulationGrid(
        **{**asdict(base), **{k: v for k, v in overrides.items() if v is not None}}
    )


# --- Simulation -------------------------------------------------------------


class SimulationResult(NamedTuple):
    per_step: pd.DataFrame
    outcomes: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict
    outputs: list[Path]


def run_simulation(cfg: SimConfig, grid: SimulationGrid) -> SimulationResult:
    timings: dict[str, float] = {}
    started = time.perf_counter()

    with _timed(timings, "load_seconds"):
        # Every replay row is demand; the outcome breakdown is a report on what
        # the trace contains, not a filter applied to it.
        simulatable = derive_gas_dimensions(load_tx_gas_results(cfg))
        outcomes = replay_outcome_summary(simulatable)
        # Headers first: the demand model anchors every cohort on the base fee of
        # its own source block, so cohorts cannot be built without them.
        headers = fetch_headers(cfg, simulatable)
        cohorts = build_cohorts(simulatable, headers)
        base_fee = resolve_starting_base_fee(cfg, headers, simulatable)

    horizon = cfg.simulation_horizon_blocks or len(cohorts)

    with _timed(timings, "simulate_seconds"):
        per_step = pd.concat(
            [
                frame
                for elasticity in grid.aggregate_elasticities
                for level in grid.demand_levels
                for window in grid.bootstrap_window_blocks
                for frame in run_scenario(
                    cohorts,
                    cfg.with_(
                        aggregate_elasticity=elasticity,
                        demand_level=level,
                        bootstrap_window_blocks=window,
                    ),
                    horizon=horizon,
                    base_fee=base_fee,
                    include_historical=grid.include_historical_reference,
                )
            ],
            ignore_index=True,
        )

    summary = scenario_summary(per_step)
    with _timed(timings, "write_seconds"):
        outputs = write_outputs(cfg, per_step, outcomes, summary)

    timings["total_seconds"] = time.perf_counter() - started
    manifest = run_manifest(cfg, grid, horizon, base_fee, timings, outputs)
    manifest_path = Path(cfg.output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return SimulationResult(
        per_step, outcomes, summary, manifest, outputs + [manifest_path]
    )


def run_scenario(
    cohorts,
    cfg: SimConfig,
    *,
    horizon: int,
    base_fee: int,
    include_historical: bool = True,
) -> list[pd.DataFrame]:
    """One grid cell: `num_bootstrap_runs` bootstrap paths plus the reference.

    `cfg.arrival_mode == "historical"` runs the reference path alone, for a
    single-trace replay with no Monte Carlo bands.
    """
    historical_only = cfg.arrival_mode == HISTORICAL_MODE
    bootstrap_cfg = cfg.with_(arrival_mode=BOOTSTRAP_MODE)
    num_runs = 0 if historical_only else cfg.num_bootstrap_runs
    frames = [
        _tag(
            run_path(
                cohorts,
                # Common random numbers: the same window draw serves every
                # demand scenario, so scenarios differ by demand alone.
                bootstrap_path(
                    cohorts,
                    horizon,
                    cfg.bootstrap_window_blocks,
                    bootstrap_rng(cfg, run_index),
                ),
                bootstrap_cfg,
                run_index=run_index,
                starting_base_fee=base_fee,
            ),
            bootstrap_cfg,
            run_index,
        )
        for run_index in range(num_runs)
    ]
    if include_historical or historical_only:
        historical_cfg = cfg.with_(arrival_mode=HISTORICAL_MODE)
        frames.append(
            _tag(
                run_path(
                    cohorts,
                    historical_path(cohorts, horizon),
                    historical_cfg,
                    run_index=0,
                    starting_base_fee=base_fee,
                ),
                historical_cfg,
                run_index=0,
            )
        )
    return frames


def fetch_headers(cfg: SimConfig, simulatable: pd.DataFrame) -> pd.DataFrame:
    """Headers covering every source block, plus the reference path's parent.

    Every cohort needs its own block's base fee for the demand anchor, and the
    reference path additionally needs the block *before* its first, whose base
    fee is where the simulation starts.
    """
    blocks = simulatable["block_number"].to_numpy(np.int64)
    first = min(int(cfg.reference_start_block or blocks.min()), int(blocks.min()))
    return fetch_block_headers(cfg, first - 1, int(blocks.max()))


def resolve_starting_base_fee(
    cfg: SimConfig, headers: pd.DataFrame, simulatable: pd.DataFrame
) -> int:
    """Base fee of the parent of `reference_start_block`, or the configured override."""
    if cfg.starting_base_fee is not None:
        return int(cfg.starting_base_fee)
    reference = int(
        cfg.reference_start_block or simulatable["block_number"].to_numpy(np.int64).min()
    )
    return starting_base_fee(headers, reference)


def scenario_summary(per_step: pd.DataFrame) -> pd.DataFrame:
    """Per scenario: saturation shares, terminal backlog, and final base fee.

    "Terminal" is the last arrival step, which is where a path now ends.
    """
    keys = ["arrival_mode", "aggregate_elasticity", "demand_level", "bootstrap_window_blocks"]
    flagged = per_step.assign(
        execution_saturated=per_step["execution_utilization"] >= SATURATION_THRESHOLD,
        state_saturated=per_step["state_utilization"] >= SATURATION_THRESHOLD,
    )
    saturation = flagged.groupby(keys, observed=True).agg(
        execution_saturated_share=("execution_saturated", "mean"),
        state_saturated_share=("state_saturated", "mean"),
        # How far the demand model actually travelled, and whether the clamp had
        # to hold it: a high clamped share means the price left the range the
        # elasticity was estimated over, so that cell is extrapolation.
        median_demand_multiplier=("realized_demand_multiplier", "median"),
        max_demand_multiplier=("realized_demand_multiplier", "max"),
        multiplier_clamped_share=("demand_multiplier_clamped", "mean"),
        # Non-zero means the base fee ran into `config.MAX_BASE_FEE`, so this
        # scenario could not shed demand and its fees are not interpretable.
        base_fee_clamped_share=("base_fee_clamped", "mean"),
    )
    terminal = (
        flagged.sort_values("simulation_position")
        .groupby(keys + ["run_index"], observed=True)
        .tail(1)
        .assign(ended_empty=lambda f: f["backlog_tx_count"] == 0)
    )
    outcome = terminal.groupby(keys, observed=True).agg(
        runs=("run_index", "nunique"),
        # Share of runs whose *last arrival block* left nothing queued. Not a
        # drain result: the run stops with the last arrival, so this says demand
        # fitted inside capacity at the end, not how fast a queue would clear.
        ended_empty_share=("ended_empty", "mean"),
        median_terminal_backlog_txs=("backlog_tx_count", "median"),
        max_terminal_backlog_txs=("backlog_tx_count", "max"),
        final_base_fee_wei=("base_fee_per_gas", "median"),
    )
    summary = saturation.join(outcome).reset_index()
    summary["median_final_base_fee_gwei"] = summary.pop("final_base_fee_wei") * 1e-9
    return summary


def write_outputs(
    cfg: SimConfig,
    per_step: pd.DataFrame,
    outcomes: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[Path]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet only. A CSV copy of the same frame is ~3.5x the bytes, ~11x slower
    # to write, and lossy on reload (bools become strings, the Int64/int64
    # distinction goes), so it cost real wall clock while being the worse copy.
    # The small summaries below stay CSV because they are meant to be read.
    paths = [out_dir / "per_step.parquet"]
    per_step.to_parquet(paths[0], index=False)

    outcomes_path = out_dir / "replay_outcome_summary.csv"
    outcomes.to_csv(outcomes_path)
    summary_path = out_dir / "scenario_summary.csv"
    summary.to_csv(summary_path, index=False)
    return paths + [outcomes_path, summary_path]


def run_manifest(
    cfg: SimConfig,
    grid: SimulationGrid,
    horizon: int,
    base_fee: int,
    timings: dict[str, float],
    outputs: list[Path],
) -> dict:
    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "grid": asdict(grid),
        "resolved": {
            "simulation_horizon_blocks": horizon,
            "starting_base_fee_wei": int(base_fee),
            "bootstrap_seed": cfg.bootstrap_seed,
            "demand_seed": cfg.demand_seed,
        },
        "library_versions": library_versions(),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "outputs": [str(p) for p in outputs],
        "caveat": CAVEAT,
    }


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version()} | {
        name: _installed_version(name) for name in REPORTED_LIBRARIES
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = build_config(args)
    grid = build_grid(args)
    result = run_simulation(cfg, grid)

    print(_banner("REPLAY OUTCOME MIX (every row is simulated; nothing is excluded)"))
    print(
        result.outcomes.to_string()
        if not result.outcomes.empty
        else "empty trace: no replay rows in range"
    )
    print(_banner("SCENARIO SUMMARY"))
    print(result.summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\n{CAVEAT}\n")
    print(f"wrote {len(result.outputs)} files to {cfg.output_dir}")
    return 0


# --- internals --------------------------------------------------------------


def _tag(per_step: pd.DataFrame, cfg: SimConfig, run_index: int) -> pd.DataFrame:
    """Stamp the grid identity, so a concatenated simulation is always groupable."""
    return per_step.assign(
        arrival_mode=cfg.arrival_mode,
        run_index=run_index,
        aggregate_elasticity=cfg.aggregate_elasticity,
        demand_level=cfg.demand_level,
        bootstrap_window_blocks=cfg.bootstrap_window_blocks,
    )


@contextmanager
def _timed(timings: dict[str, float], key: str):
    """Record wall-clock seconds for one simulation phase."""
    started = time.perf_counter()
    yield
    timings[key] = time.perf_counter() - started


def _installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _banner(text: str) -> str:
    return f"\n{'=' * len(text)}\n{text}\n{'=' * len(text)}"


if __name__ == "__main__":
    raise SystemExit(main())
