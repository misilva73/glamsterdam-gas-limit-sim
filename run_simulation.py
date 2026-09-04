#!/usr/bin/env python
"""Run a Fusaka -> Glamsterdam gas-limit simulation end to end.

Sweeps the elasticity x demand-level x bootstrap-window grid, runs
`num_bootstrap_runs` independent moving-block-bootstrap paths per cell, and
writes the per-step data, the scenario summaries, and a run manifest.

Output is **checkpointed per grid cell**: when a cell finishes, its paths go to
`per_step/<cell>.parquet` and its summary row is appended to
`scenario_summary.csv` before the next cell starts. Nothing but the cell in
flight is held in memory, so a whole sweep fits in one process and a crash at
hour nine costs one cell rather than everything. Every invocation writes into a
fresh timestamped directory under `output_dir`, so runs never mix -- unless
`--resume STAMP` names an interrupted one, in which case the cells it already
checkpointed are skipped and the rest are simulated into the same directory.

It does no analysis and draws no figures. The job here is to produce data for a
separate downstream analysis.

Every path in a scenario starts from the identical initial state: empty mempool,
the base fee of the actual parent of the first source cohort, and
`fusaka_gas_limit`. Per-run randomness is derived from `random_seed`, so runs are
independent and the whole simulation is reproducible.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, Scenario, SimConfig, SimulationGrid
from data.fetch_blocks import fetch_block_headers, starting_base_fee
from data.load_tx_gas_results import (
    derive_gas_dimensions,
    load_tx_gas_results,
    replay_outcome_summary,
)
from schemas import PER_STEP_COLUMNS
from sim.engine import run_path
from sim.workload import bootstrap_path, bootstrap_rng, build_cohorts

PER_STEP_DIR = "per_step"
SUMMARY_FILE = "scenario_summary.csv"
OUTCOMES_FILE = "replay_outcome_summary.csv"
MANIFEST_FILE = "manifest.json"

RUN_DIR_FORMAT = "%Y%m%dT%H%M%SZ"
"""UTC stamp naming a run's own output directory, so runs sort chronologically."""

PATH_CONFIG_FIELDS = frozenset({"output_dir", "cache_dir", "secrets_path"})
"""`SimConfig` fields the manifest stores as strings and must read back as `Path`."""

RESUME_EXEMPT_CONFIG_FIELDS = PATH_CONFIG_FIELDS
"""Config fields `--resume` tolerates differing.

Exactly the path fields: where the cache, the secrets, and the output tree live
says nothing about the numbers, and a resume may happen on another machine.
"""

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
        help="reth replay config to read; mandatory unless --resume supplies it, so "
        "a run can never silently mix datasets",
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
        "--cache-dir", type=Path, help="parquet cache for fetched data (default: data/cache)"
    )

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--horizon", type=int, help="arrival steps (default: trace length)")
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
        "(default: 0.1 0.2 0.3). 0 is a flat multiplier with no price response "
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
        "(default: 32; pass multiple values for a robustness sweep)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="parent of the timestamped run directory (default: output/)",
    )
    parser.add_argument(
        "--resume",
        metavar="STAMP",
        help="continue an interrupted run: the name of its directory under "
        "--output-dir (e.g. 20260904T083556Z). Cells already checkpointed there "
        "are skipped. The config and grid come from that run's manifest, so no "
        "other flag is needed; any flag given anyway must agree with what the run "
        "recorded, or nothing is written",
    )
    return parser


def build_config(args: argparse.Namespace, base: SimConfig = DEFAULT_CONFIG) -> SimConfig:
    """Resolve CLI overrides into a `SimConfig`. Pure, so it is directly testable."""
    overrides = {
        "analysis_config_hash": args.analysis_config_hash,
        "schedule_name": args.schedule_name,
        "schedule_config_hash": args.schedule_config_hash,
        "chain_id": args.chain_id,
        "source_block_range": tuple(args.block_range) if args.block_range else None,
        "cache_dir": args.cache_dir,
        "simulation_horizon_blocks": args.horizon,
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
    }
    return base.with_(**{k: v for k, v in overrides.items() if v is not None})


def build_grid(args: argparse.Namespace, base: SimulationGrid = SimulationGrid()) -> SimulationGrid:
    overrides = {
        "aggregate_elasticities": tuple(args.elasticities) if args.elasticities else None,
        "demand_levels": tuple(args.demand_levels) if args.demand_levels else None,
        "bootstrap_window_blocks": tuple(args.window_blocks) if args.window_blocks else None,
    }
    return SimulationGrid(
        **{**asdict(base), **{k: v for k, v in overrides.items() if v is not None}}
    )


# --- Simulation -------------------------------------------------------------


class SimulationResult(NamedTuple):
    """What a finished run leaves behind.

    The per-step frame is deliberately absent: it is streamed to
    `run_dir / PER_STEP_DIR` a cell at a time and never assembled in memory. Read
    the whole sweep back with `pd.read_parquet(result.run_dir / "per_step")`.
    """

    run_dir: Path
    outcomes: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict
    outputs: list[Path]


def run_simulation(
    cfg: SimConfig, grid: SimulationGrid, *, resume: str | None = None
) -> SimulationResult:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    cells = grid_cells(grid)
    # Checked before the load, which is the expensive part: an incompatible
    # --resume should fail in milliseconds, not after a 500-second fetch.
    resumed = open_resumed_run(cfg, grid, resume, cells) if resume else None

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
    resolved = resolved_facts(cfg, horizon, base_fee)

    if resumed is None:
        run_dir, done, summaries = new_run_dir(cfg.output_dir), 0, []
    else:
        # The trace itself has to match too: the same block range can load
        # different rows as the upstream table grows, and cells simulated against
        # two different traces do not belong in one output directory.
        require_resumable_trace(resumed, resolved)
        run_dir, done, summaries = resumed.run_dir, resumed.cells_done, [resumed.summary]

    per_step_dir = run_dir / PER_STEP_DIR
    per_step_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / SUMMARY_FILE
    with _timed(timings, "write_seconds"):
        # The outcome mix describes the loaded trace, so it is already final.
        write_replay_outcomes(run_dir, outcomes)
    # Written before the first cell, so an interrupted run can be resumed at all:
    # `--resume` reads this to check that it is continuing the same simulation.
    manifest_path = write_manifest(
        run_dir,
        run_manifest(
            cfg, grid, resolved, timings, run_dir_files(run_dir), run_dir, completed=False
        ),
    )

    for scenario in cells[done:]:
        with _timed(timings, "simulate_seconds"):
            frames = run_scenario(
                cohorts,
                cfg,
                scenario,
                horizon=horizon,
                base_fee=base_fee,
            )
        with _timed(timings, "write_seconds"):
            summaries.append(
                checkpoint_cell(
                    per_step_dir,
                    summary_path,
                    cell_slug(
                        scenario.aggregate_elasticity,
                        scenario.demand_level,
                        scenario.bootstrap_window_blocks,
                    ),
                    frames,
                )
            )
        # Drop the cell before the next one is simulated: keeping the binding
        # alive would hold two cells at the peak instead of one.
        del frames

    summary = pd.concat([f for f in summaries if not f.empty], ignore_index=True)
    timings["total_seconds"] = time.perf_counter() - started
    outputs = run_dir_files(run_dir)
    manifest = run_manifest(
        cfg, grid, resolved, timings, outputs, run_dir, completed=True
    )
    write_manifest(run_dir, manifest)
    return SimulationResult(run_dir, outcomes, summary, manifest, outputs + [manifest_path])


def grid_cells(grid: SimulationGrid) -> list[Scenario]:
    """Every cell of the sweep, in the order it is simulated and written.

    `--resume` depends on this order being stable: cells are checkpointed
    sequentially, so what is on disk is always a prefix of this list.
    """
    cells = [
        Scenario(elasticity, level, window)
        for elasticity in grid.aggregate_elasticities
        for level in grid.demand_levels
        for window in grid.bootstrap_window_blocks
    ]
    if not cells:
        raise ValueError("empty simulation grid: every axis needs at least one value")
    return cells


def run_scenario(
    cohorts,
    cfg: SimConfig,
    scenario: Scenario,
    *,
    horizon: int,
    base_fee: int,
) -> list[pd.DataFrame]:
    """Run the bootstrap samples for one grid cell."""
    return [
        run_path(
            cohorts,
            # Common random numbers: the same window draw serves every
            # demand scenario, so scenarios differ by demand alone.
            bootstrap_path(
                cohorts,
                horizon,
                scenario.bootstrap_window_blocks,
                bootstrap_rng(cfg, run_index),
            ),
            cfg,
            scenario,
            run_index=run_index,
            starting_base_fee=base_fee,
        )
        for run_index in range(cfg.num_bootstrap_runs)
    ]


def fetch_headers(cfg: SimConfig, simulatable: pd.DataFrame) -> pd.DataFrame:
    """Headers covering every source block, plus the first cohort's parent.

    Every cohort needs its own block's base fee for the demand anchor, and the
    simulation additionally needs the block *before* its first cohort, whose
    base fee is where every path starts.
    """
    blocks = simulatable["block_number"].to_numpy(np.int64)
    first = int(blocks.min())
    return fetch_block_headers(cfg, first - 1, int(blocks.max()))


def resolve_starting_base_fee(
    cfg: SimConfig, headers: pd.DataFrame, simulatable: pd.DataFrame
) -> int:
    """Base fee of the first source cohort's parent, or the configured override."""
    if cfg.starting_base_fee is not None:
        return int(cfg.starting_base_fee)
    first_cohort = int(simulatable["block_number"].to_numpy(np.int64).min())
    return starting_base_fee(headers, first_cohort)


def scenario_summary(per_step: pd.DataFrame) -> pd.DataFrame:
    """Per scenario: saturation shares, terminal backlog, and final base fee.

    "Terminal" is the last arrival step, which is where a path now ends.
    """
    keys = ["aggregate_elasticity", "demand_level", "bootstrap_window_blocks"]
    flagged = per_step.assign(
        execution_saturated=(
            per_step["block_execution_gas_used"] / per_step["gas_limit"]
            >= SATURATION_THRESHOLD
        ),
        state_saturated=(
            per_step["block_state_gas_used"] / per_step["gas_limit"]
            >= SATURATION_THRESHOLD
        ),
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


def new_run_dir(output_dir: Path) -> Path:
    """A fresh timestamped directory under `output_dir`, holding this run's files.

    Every invocation gets its own unless it passes `--resume`, so an incremental
    sweep can never append into another run's part files -- which is what makes a
    directory of parts safe to read as one dataset.
    """
    stamp = time.strftime(RUN_DIR_FORMAT, time.gmtime())
    for attempt in range(100):
        run_dir = Path(output_dir) / (stamp if attempt == 0 else f"{stamp}-{attempt}")
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            continue  # same second as a previous run, or a directory left behind
        return run_dir
    raise RuntimeError(f"no unique run directory available under {output_dir}")


def checkpoint_cell(
    per_step_dir: Path,
    summary_path: Path,
    slug: str,
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Persist one finished grid cell: its per-step part, then its summary row.

    One parquet holds every bootstrap run of the cell, with one appended
    `scenario_summary.csv` row. Data is written before the
    summary that describes it, so an interrupted sweep never leaves a summary row
    without the per-step data behind it -- and `--resume` can therefore trust that
    a summary row means a complete cell.

    Parquet only for the per-step data. A CSV copy of the same frame is ~3.5x the
    bytes, ~11x slower to write, and lossy on reload (bools become strings, the
    Int64/int64 distinction goes), so it cost real wall clock while being the
    worse copy. The small summaries stay CSV because they are meant to be read.
    """
    frame = pd.concat(frames, ignore_index=True)[list(PER_STEP_COLUMNS)]
    frame.to_parquet(per_step_dir / part_file(slug), index=False)
    summary = scenario_summary(frame)
    first_cell = not summary_path.exists()
    summary.to_csv(
        summary_path, mode="w" if first_cell else "a", header=first_cell, index=False
    )
    return summary


def cell_slug(elasticity: float, level: float, window: int) -> str:
    """Grid identity as a filename stem, e.g. `e0.175_d1.5_w32`.

    `repr` of a float is its shortest round-tripping form, so distinct axis values
    can never collide into one part file.
    """
    return f"e{float(elasticity)!r}_d{float(level)!r}_w{int(window)}"


def part_file(slug: str) -> str:
    return f"{slug}.parquet"


def write_replay_outcomes(run_dir: Path, outcomes: pd.DataFrame) -> Path:
    path = run_dir / OUTCOMES_FILE
    outcomes.to_csv(path)
    return path


def run_dir_files(run_dir: Path) -> list[Path]:
    """Every data file in a run directory, in a stable order.

    Scanned rather than accumulated so that a resumed run reports the cells its
    earlier invocations wrote too. `manifest.json` is excluded: it is the file
    that lists these.
    """
    parts = sorted((run_dir / PER_STEP_DIR).glob("*.parquet"))
    named = [run_dir / OUTCOMES_FILE, run_dir / SUMMARY_FILE]
    return parts + [path for path in named if path.exists()]


# --- Resume -----------------------------------------------------------------


def read_prior_manifest(output_dir: Path, resume: str) -> tuple[Path, dict]:
    """Find the `--resume STAMP` directory under `output_dir` and read its manifest."""
    run_dir = Path(output_dir) / resume
    manifest_path = run_dir / MANIFEST_FILE
    if not run_dir.is_dir():
        available = sorted(p.name for p in Path(output_dir).glob("*") if p.is_dir())
        raise ValueError(
            f"nothing to resume: {run_dir} does not exist. "
            f"Available runs: {available or 'none'}"
        )
    if not manifest_path.exists():
        raise ValueError(
            f"cannot resume {run_dir}: no {MANIFEST_FILE}, so that run died during "
            "the load and simulated nothing. Start a fresh run instead"
        )
    return run_dir, json.loads(manifest_path.read_text())


def stored_inputs(manifest: dict, output_dir: Path) -> tuple[SimConfig, SimulationGrid]:
    """The config and grid a run recorded, so resuming need not repeat its flags.

    `output_dir` comes from where the run was actually found rather than from the
    manifest, since that is the one field the tree itself can contradict -- the
    output may have been moved, or mounted somewhere else on this machine.

    A manifest holding fields `SimConfig` no longer has is refused rather than
    quietly dropped: the missing knob had a value in the finished cells, and
    silently substituting today's default would make the two halves incomparable.
    """
    stored = manifest.get("config", {})
    unknown = set(stored) - {field.name for field in fields(SimConfig)}
    if unknown:
        raise ValueError(
            f"cannot rebuild the config from {MANIFEST_FILE}: it records "
            f"{', '.join(sorted(unknown))}, which this version of SimConfig does "
            "not have. That run predates a config change; start a fresh run"
        )
    cfg = SimConfig(**{name: _config_value(name, v) for name, v in stored.items()})
    grid = SimulationGrid(
        **{name: _tupled(v) for name, v in manifest.get("grid", {}).items()}
    )
    return cfg.with_(output_dir=Path(output_dir)), grid


class ResumedRun(NamedTuple):
    run_dir: Path
    cells_done: int
    """Leading cells of `grid_cells` already checkpointed; the sweep restarts here."""
    summary: pd.DataFrame
    """Their `scenario_summary.csv` rows, kept so the result covers the whole grid."""
    manifest: dict


def open_resumed_run(
    cfg: SimConfig,
    grid: SimulationGrid,
    resume: str,
    cells: list[Scenario],
) -> ResumedRun:
    """Validate `--resume STAMP` and work out where the sweep stopped.

    Refuses anything but a directory holding a matching, unfinished run: resuming
    into a different config, grid, or seed would leave one output directory
    describing two simulations, which no reader could untangle. The CLI rebuilds
    both from the same manifest, so this normally has nothing to reject -- it is
    the guard for callers that pass their own config in.
    """
    run_dir, prior = read_prior_manifest(cfg.output_dir, resume)
    if prior.get("completed"):
        raise ValueError(f"cannot resume {run_dir}: that run completed")
    mismatched = resume_mismatches(prior, cfg, grid)
    if mismatched:
        raise ValueError(
            f"cannot resume {run_dir}: it was run with a different "
            f"{', '.join(mismatched)}. Resuming would mix two simulations in one "
            "output directory"
        )

    cells_done, summary = completed_cells(run_dir, cells)
    return ResumedRun(run_dir, cells_done, summary, prior)


def completed_cells(run_dir: Path, cells: list[Scenario]) -> tuple[int, pd.DataFrame]:
    """How many leading cells are fully checkpointed, and their summary rows.

    Cells are simulated in `grid_cells` order, so what is on disk is a prefix and
    the count is the first cell whose parts are not all there. A cell is only
    complete once its summary rows landed as well, so the count is also bounded by
    the row count; `scenario_summary.csv` is truncated to match, and any cell
    caught mid-checkpoint is simulated again over its own part files.
    """
    per_step_dir = run_dir / PER_STEP_DIR
    written = 0
    for scenario in cells:
        slug = cell_slug(
            scenario.aggregate_elasticity,
            scenario.demand_level,
            scenario.bootstrap_window_blocks,
        )
        if not (per_step_dir / part_file(slug)).exists():
            break
        written += 1

    summary_path = run_dir / SUMMARY_FILE
    rows = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    cells_done = min(written, len(rows))
    kept = rows.iloc[:cells_done]
    if len(kept) != len(rows):
        if kept.empty:
            # Leave no header behind: the next checkpoint writes the file fresh.
            summary_path.unlink(missing_ok=True)
        else:
            kept.to_csv(summary_path, index=False)
    return cells_done, kept


def resume_mismatches(prior: dict, cfg: SimConfig, grid: SimulationGrid) -> list[str]:
    """Config and grid fields that differ from the run being resumed.

    Paths are exempt: where the cache and the output tree live says nothing about
    the numbers, and a resume may well happen on another machine.
    """
    stored = prior.get("config", {})
    current = _jsonable(manifest_config(cfg))
    mismatched = [
        name
        for name, value in current.items()
        if name not in RESUME_EXEMPT_CONFIG_FIELDS and stored.get(name) != value
    ]
    extra_stored = set(stored) - set(current) - RESUME_EXEMPT_CONFIG_FIELDS
    if extra_stored:
        mismatched.append("config schema")
    if prior.get("grid") != _jsonable(asdict(grid)):
        mismatched.append("grid")
    if prior.get("per_step_columns") != list(PER_STEP_COLUMNS):
        mismatched.append("per-step schema")
    return mismatched


def require_resumable_trace(resumed: ResumedRun, resolved: dict) -> None:
    """Refuse to resume if the loaded trace or the derived seeds moved.

    The upstream replay table is still being written, so the same block range can
    load more rows than it did yesterday. That changes the horizon, and every
    already-written cell was simulated against the shorter one.
    """
    stored = resumed.manifest.get("resolved", {})
    mismatched = [
        name for name, value in resolved.items() if stored.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            f"cannot resume {resumed.run_dir}: {', '.join(mismatched)} differs from "
            "the completed cells, so the trace or the seeds moved under the sweep. "
            "Start a fresh run instead"
        )


# --- Manifest ---------------------------------------------------------------


def resolved_facts(cfg: SimConfig, horizon: int, base_fee: int) -> dict:
    """What the run resolved from its inputs; also what `--resume` must match."""
    return {
        "simulation_horizon_blocks": int(horizon),
        "starting_base_fee_wei": int(base_fee),
        "bootstrap_seed": cfg.bootstrap_seed,
        "demand_seed": cfg.demand_seed,
    }


def manifest_config(cfg: SimConfig) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}


def run_manifest(
    cfg: SimConfig,
    grid: SimulationGrid,
    resolved: dict,
    timings: dict[str, float],
    outputs: list[Path],
    run_dir: Path,
    *,
    completed: bool,
) -> dict:
    return {
        # False in the copy written before the first cell: a manifest with
        # `completed: false` marks a run that was interrupted and can be resumed.
        # Timings and outputs in that copy cover only what had been written.
        "completed": completed,
        "config": manifest_config(cfg),
        "grid": asdict(grid),
        "per_step_columns": list(PER_STEP_COLUMNS),
        # `config.output_dir` is the parent; `run_dir` is where the run wrote.
        "resolved": {"run_dir": str(run_dir)} | resolved,
        "library_versions": library_versions(),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "outputs": [str(p.relative_to(run_dir)) for p in outputs],
        "caveat": CAVEAT,
    }


def write_manifest(run_dir: Path, manifest: dict) -> Path:
    path = run_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version()} | {
        name: _installed_version(name) for name in REPORTED_LIBRARIES
    }


def resolve_inputs(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[SimConfig, SimulationGrid]:
    """The config and grid to run, from the command line or from a resumed run.

    Resuming layers whatever flags were given onto the stored ones rather than
    ignoring them, so `--cache-dir` can move and a typo still gets caught: the
    result goes through `open_resumed_run`, which refuses any difference that
    would change the numbers.
    """
    if args.resume:
        run_dir, prior = read_prior_manifest(
            args.output_dir or DEFAULT_CONFIG.output_dir, args.resume
        )
        base_cfg, base_grid = stored_inputs(prior, run_dir.parent)
        return build_config(args, base_cfg), build_grid(args, base_grid)
    if not args.analysis_config_hash:
        parser.error(
            "--analysis-config-hash is required, so a run can never silently mix "
            "datasets; --resume takes it from the run it continues"
        )
    return build_config(args), build_grid(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, grid = resolve_inputs(parser, args)
    result = run_simulation(cfg, grid, resume=args.resume)

    print(_banner("REPLAY OUTCOME MIX (every row is simulated; nothing is excluded)"))
    print(
        result.outcomes.to_string()
        if not result.outcomes.empty
        else "empty trace: no replay rows in range"
    )
    print(_banner("SCENARIO SUMMARY"))
    print(result.summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\n{CAVEAT}\n")
    print(f"wrote {len(result.outputs)} files to {result.run_dir}")
    return 0


# --- internals --------------------------------------------------------------


def _jsonable(value: dict) -> dict:
    """Round-trip through JSON so tuples compare equal to the lists on disk."""
    return json.loads(json.dumps(value, default=str))


def _tupled(value):
    """JSON has no tuples; the dataclasses want the ones they wrote back."""
    return tuple(value) if isinstance(value, list) else value


def _config_value(name: str, value):
    """One manifest field back in the type `SimConfig` declares for it."""
    return Path(value) if name in PATH_CONFIG_FIELDS else _tupled(value)


@contextmanager
def _timed(timings: dict[str, float], key: str):
    """Accumulate wall-clock seconds for one simulation phase.

    Phases interleave now that each cell is written as it finishes, so a phase is
    entered once per cell and its total is the sum.
    """
    started = time.perf_counter()
    yield
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - started


def _installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _banner(text: str) -> str:
    return f"\n{'=' * len(text)}\n{text}\n{'=' * len(text)}"


if __name__ == "__main__":
    raise SystemExit(main())
