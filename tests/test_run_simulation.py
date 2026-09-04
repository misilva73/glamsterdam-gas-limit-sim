"""Tests for the simulation entry point: CLI, summaries, and the output contract.

The engine and the workload model are exercised elsewhere; here everything is
driven from a synthetic per-step frame with exactly `schemas.PER_STEP_COLUMNS`,
so the CLI, the summaries, and the output contract are testable on their own.
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import run_simulation
import schemas
from config import DEFAULT_CONFIG, SimulationGrid
from sim.workload import bootstrap_path, bootstrap_rng, build_cohorts
from tests.dummy import (
    DUMMY_ANALYSIS_CONFIG_HASH,
    dummy_block_headers,
    dummy_tx_gas_results,
)

BOOTSTRAP = "moving_block_bootstrap"
HISTORICAL = "historical"

SCENARIO_KEYS = [
    "aggregate_elasticity",
    "demand_level",
    "bootstrap_window_blocks",
]

# --- synthetic per-step frames ----------------------------------------------


def synthetic_per_step(
    *,
    elasticities=(0.175,),
    levels=(1.0,),
    windows=(32,),
    num_runs: int = 4,
    positions: int = 24,
    arrival_mode: str = BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """A per-step frame shaped exactly like the engine's output."""
    frame = pd.MultiIndex.from_product(
        [list(elasticities), list(levels), list(windows), range(num_runs), range(positions)],
        names=[
            "aggregate_elasticity",
            "demand_level",
            "bootstrap_window_blocks",
            "run_index",
            "simulation_position",
        ],
    ).to_frame(index=False)

    rng = np.random.default_rng(seed)
    position = frame["simulation_position"].to_numpy()
    window = frame["bootstrap_window_blocks"].to_numpy()
    elasticity = frame["aggregate_elasticity"].to_numpy()
    scale = frame["demand_level"].to_numpy()
    size = len(frame)

    gas_limit = np.minimum(60_000_000 * (1 + 1 / 1024) ** position, 200_000_000)
    execution_utilization = np.clip(0.35 * scale + rng.normal(0, 0.05, size), 0, 1)
    state_utilization = np.clip(0.25 * scale + rng.normal(0, 0.05, size), 0, 1)
    execution_gas_used = execution_utilization * gas_limit
    state_gas_used = state_utilization * gas_limit
    backlog_tx_count = rng.poisson(120 * scale, size)
    ineligible_tx_count = rng.poisson(8 * scale, size)

    base_fee = 8e9 * (1 + 0.01 * position) * scale
    anchor_price = np.full(size, 9e9)
    # The model's own arithmetic, so the synthetic frame stays internally
    # consistent with what the engine would have recorded.
    multiplier = scale * (base_fee / anchor_price) ** -elasticity

    frame["arrival_mode"] = arrival_mode
    frame["source_block_number"] = 21_000_000 + position
    frame["window_instance"] = position // window
    frame["position_in_window"] = position % window
    frame["demand_price_signal"] = base_fee
    frame["cohort_anchor_price"] = anchor_price
    frame["realized_demand_multiplier"] = multiplier
    frame["demand_multiplier_clamped"] = False
    frame["base_fee_per_gas"] = base_fee.astype(np.int64)
    frame["base_fee_clamped"] = False
    frame["gas_limit"] = gas_limit.astype(np.int64)
    frame["gas_used"] = np.maximum(execution_gas_used, state_gas_used).astype(np.int64)
    frame["block_execution_gas_used"] = execution_gas_used.astype(np.int64)
    frame["block_state_gas_used"] = state_gas_used.astype(np.int64)
    frame["execution_utilization"] = execution_utilization
    frame["state_utilization"] = state_utilization
    frame["bottleneck_dimension"] = np.where(
        execution_gas_used >= state_gas_used,
        schemas.BOTTLENECK_EXECUTION,
        schemas.BOTTLENECK_STATE,
    )
    frame["included_tx_count"] = rng.poisson(150, size)
    frame["sender_gas_used"] = (0.9 * execution_gas_used).astype(np.int64)
    frame["priority_fees_wei"] = (frame["included_tx_count"] * 1e15).astype(np.int64)
    frame["arrived_tx_count"] = rng.poisson(160 * scale)
    frame["arrived_execution_gas"] = (execution_gas_used * 1.1).astype(np.int64)
    frame["arrived_state_gas"] = (state_gas_used * 1.1).astype(np.int64)
    frame["backlog_tx_count"] = backlog_tx_count + ineligible_tx_count
    frame["backlog_eligible_tx_count"] = backlog_tx_count
    frame["backlog_fee_ineligible_tx_count"] = ineligible_tx_count
    frame["backlog_eligible_execution_gas"] = backlog_tx_count * 120_000
    frame["backlog_eligible_state_gas"] = backlog_tx_count * 45_000
    frame["backlog_fee_ineligible_execution_gas"] = ineligible_tx_count * 120_000
    frame["backlog_fee_ineligible_state_gas"] = ineligible_tx_count * 45_000
    frame["backlog_execution_gas"] = (
        frame["backlog_eligible_execution_gas"] + frame["backlog_fee_ineligible_execution_gas"]
    )
    frame["backlog_state_gas"] = (
        frame["backlog_eligible_state_gas"] + frame["backlog_fee_ineligible_state_gas"]
    )
    return frame[list(schemas.PER_STEP_COLUMNS)]


def test_synthetic_frame_matches_the_contract():
    frame = synthetic_per_step()
    assert list(frame.columns) == list(schemas.PER_STEP_COLUMNS)


# --- cohort fixtures --------------------------------------------------------


def dummy_cohorts(num_blocks: int = 200) -> pd.DataFrame:
    """Dummy replay rows with the gas dimensions derived here, not imported.

    Every row is kept: there is no inclusion policy, so failures are demand too.
    """
    frame = dummy_tx_gas_results(num_blocks=num_blocks)
    return frame.assign(
        state_gas=frame["schedule_state_gas_spent"],
        execution_gas=np.maximum(
            frame["schedule_total_gas_spent"] - frame["schedule_state_gas_spent"],
            frame["schedule_floor_gas"],
        ),
    )


def anchored_cohorts(num_blocks: int = 200):
    """Cohorts with demand anchors, as `run_simulation` builds them."""
    frame = dummy_cohorts(num_blocks)
    return build_cohorts(frame, dummy_block_headers(frame))


# --- CLI and simulation plumbing --------------------------------------------


def parse(argv: list[str]):
    """`--analysis-config-hash` is mandatory, so every parse carries one."""
    return run_simulation.build_parser().parse_args(
        ["--analysis-config-hash", DUMMY_ANALYSIS_CONFIG_HASH, *argv]
    )


def test_build_config_applies_every_override(tmp_path):
    args = parse(
        [
            "--block-range",
            "21000000",
            "21000399",
            "--reference-start-block",
            "21000000",
            "--horizon",
            "120",
            "--arrival-mode",
            "historical",
            "--num-runs",
            "5",
            "--seed",
            "99",
            "--starting-base-fee",
            "7000000000",
            "--elasticities",
            "0.175",
            "0.28",
            "--demand-levels",
            "1",
            "2.5",
            "--price-ema-blocks",
            "600",
            "--multiplier-bounds",
            "0.2",
            "8",
            "--no-bid-adaptation",
            "--window-blocks",
            "16",
            "64",
            "--output-dir",
            str(tmp_path),
        ]
    )
    cfg = run_simulation.build_config(args)

    assert cfg.analysis_config_hash == DUMMY_ANALYSIS_CONFIG_HASH
    assert cfg.source_block_range == (21_000_000, 21_000_399)
    assert cfg.reference_start_block == 21_000_000
    assert cfg.simulation_horizon_blocks == 120
    assert cfg.arrival_mode == "historical"
    assert cfg.num_bootstrap_runs == 5
    assert cfg.random_seed == 99
    assert cfg.starting_base_fee == 7_000_000_000
    assert cfg.output_dir == tmp_path
    assert cfg.price_ema_blocks == 600
    assert cfg.demand_multiplier_bounds == (0.2, 8.0)
    assert cfg.adapt_bids is False
    # The config carries the grid's first cell so a lone config is consistent.
    assert (cfg.aggregate_elasticity, cfg.demand_level) == (0.175, 1.0)
    assert cfg.bootstrap_window_blocks == 16


def test_build_config_leaves_unspecified_fields_at_their_defaults():
    cfg = run_simulation.build_config(parse([]))
    assert cfg == DEFAULT_CONFIG.with_(analysis_config_hash=DUMMY_ANALYSIS_CONFIG_HASH)


def test_build_config_derives_independent_reproducible_seeds():
    cfg = run_simulation.build_config(parse(["--seed", "7"]))
    same = run_simulation.build_config(parse(["--seed", "7"]))
    other = run_simulation.build_config(parse(["--seed", "8"]))

    assert cfg.bootstrap_seed == same.bootstrap_seed
    assert cfg.bootstrap_seed != cfg.demand_seed
    assert cfg.bootstrap_seed != other.bootstrap_seed


def test_build_grid_overrides_the_sweep():
    grid = run_simulation.build_grid(
        parse(
            [
                "--elasticities", "0", "0.3",
                "--demand-levels", "1", "3",
                "--window-blocks", "32",
                "--no-historical",
            ]
        )
    )
    assert grid.aggregate_elasticities == (0.0, 0.3)
    assert grid.demand_levels == (1.0, 3.0)
    assert grid.bootstrap_window_blocks == (32,)
    assert grid.include_historical_reference is False


def test_build_grid_defaults_to_the_plan_grid():
    grid = run_simulation.build_grid(parse([]))
    assert grid.aggregate_elasticities == (0.10, 0.175, 0.28)
    assert grid.demand_levels == (1.0, 1.5, 2.0)
    assert grid.bootstrap_window_blocks == (16, 32, 64)
    assert grid.include_historical_reference is True


def test_the_analysis_config_hash_is_mandatory_on_the_command_line():
    """A run can never silently average two replay configurations."""
    with pytest.raises(SystemExit):
        run_simulation.build_parser().parse_args(["--block-range", "1", "2"])


def test_bootstrap_paths_are_reproducible_and_independent_across_runs():
    cohorts = build_cohorts(dummy_cohorts(num_blocks=80))
    cfg = run_simulation.build_config(parse(["--seed", "5"]))
    path = lambda c, run: bootstrap_path(
        cohorts, 40, c.bootstrap_window_blocks, bootstrap_rng(c, run)
    )["cohort_index"].to_numpy()

    assert (path(cfg, 0) == path(cfg, 0)).all()
    assert (path(cfg, 0) != path(cfg, 1)).any()
    assert (path(cfg, 0) != path(cfg.with_(random_seed=6), 0)).any()


def test_scenario_summary_reports_one_row_per_scenario():
    axes = dict(elasticities=(0.0, 0.175), levels=(1.0, 3.0), windows=(16, 32))
    per_step = pd.concat(
        [
            synthetic_per_step(**axes),
            synthetic_per_step(**axes, num_runs=1, arrival_mode=HISTORICAL),
        ],
        ignore_index=True,
    )
    summary = run_simulation.scenario_summary(per_step)

    # arrival modes x elasticities x levels x windows
    assert len(summary) == 2 * 2 * 2 * 2
    assert summary.loc[summary["arrival_mode"] == BOOTSTRAP, "runs"].eq(4).all()
    assert summary["median_final_base_fee_gwei"].gt(0).all()
    assert summary["execution_saturated_share"].between(0, 1).all()
    assert summary["median_demand_multiplier"].gt(0).all()
    assert (summary["multiplier_clamped_share"] == 0.0).all()


def end_to_end_args(tmp_path, *overrides: str):
    """A small two-cell sweep on dummy data, writing under `tmp_path`."""
    return parse(
        [
            "--block-range",
            "21000000",
            "21000119",
            "--horizon",
            "40",
            "--num-runs",
            "3",
            "--elasticities",
            "0",
            "0.175",
            "--demand-levels",
            "2",
            "--window-blocks",
            "16",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
            *overrides,
        ]
    )


def run_end_to_end(tmp_path, *overrides: str, resume: str | None = None):
    args = end_to_end_args(tmp_path, *overrides)
    return run_simulation.run_simulation(
        run_simulation.build_config(args),
        run_simulation.build_grid(args),
        resume=resume,
    )


def test_run_simulation_end_to_end_on_dummy_data(tmp_path, offline_data):
    result = run_end_to_end(tmp_path)

    # The per-step data is never assembled in memory; the directory of parts is
    # readable as one dataset, which is the contract downstream depends on.
    per_step = pd.read_parquet(result.run_dir / "per_step")
    assert list(per_step.columns) == list(schemas.PER_STEP_COLUMNS)
    assert per_step["simulation_position"].max() == 40 - 1
    assert set(per_step["arrival_mode"]) == {BOOTSTRAP, HISTORICAL}
    assert per_step.groupby("arrival_mode")["run_index"].nunique().to_dict() == {
        BOOTSTRAP: 3,
        HISTORICAL: 1,
    }
    assert len(result.summary) == 4  # arrival modes x elasticities

    # Data only: per-step parts are parquet, the summaries are small CSVs, and a
    # run writes no figures and no CSV copy of the per-step data.
    written = {path.relative_to(result.run_dir).as_posix() for path in result.outputs}
    assert written == {
        "per_step/e0.0_d2.0_w16_bootstrap.parquet",
        "per_step/e0.0_d2.0_w16_historical.parquet",
        "per_step/e0.175_d2.0_w16_bootstrap.parquet",
        "per_step/e0.175_d2.0_w16_historical.parquet",
        "replay_outcome_summary.csv",
        "scenario_summary.csv",
        "manifest.json",
    }
    assert not any(p.suffix == ".png" for p in result.outputs)
    for path in result.outputs:
        assert path.stat().st_size > 0

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["resolved"]["run_dir"] == str(result.run_dir)
    assert manifest["resolved"]["simulation_horizon_blocks"] == 40
    assert manifest["grid"]["aggregate_elasticities"] == [0.0, 0.175]
    assert manifest["grid"]["demand_levels"] == [2.0]
    assert manifest["timings"]["total_seconds"] > 0
    assert manifest["timings"]["write_seconds"] > 0  # accumulated across cells
    assert "lower bound" in manifest["caveat"].lower()


def test_each_cell_is_checkpointed_as_it_finishes(tmp_path, offline_data):
    """The summary on disk matches the returned one, cell by cell and in order."""
    result = run_end_to_end(tmp_path)

    on_disk = pd.read_csv(result.run_dir / "scenario_summary.csv")
    pd.testing.assert_frame_equal(on_disk, result.summary, check_dtype=False)
    # One row per part file, appended cell by cell: the first cell's rows lead.
    assert on_disk["aggregate_elasticity"].tolist() == [0.0, 0.0, 0.175, 0.175]
    assert on_disk["arrival_mode"].tolist() == [BOOTSTRAP, HISTORICAL] * 2

    for part in sorted((result.run_dir / "per_step").glob("*.parquet")):
        frame = pd.read_parquet(part)
        # A part holds every run of exactly one cell and one arrival mode.
        assert frame["arrival_mode"].nunique() == 1
        assert len(frame.groupby(SCENARIO_KEYS, observed=True)) == 1
        assert frame["run_index"].nunique() == (
            3 if frame["arrival_mode"].iloc[0] == BOOTSTRAP else 1
        )


def test_every_run_writes_into_its_own_directory(tmp_path, offline_data):
    first = run_end_to_end(tmp_path)
    second = run_end_to_end(tmp_path)

    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == second.run_dir.parent == tmp_path / "out"
    # Neither run appended into the other, so each is a complete sweep on its own.
    for run_dir in (first.run_dir, second.run_dir):
        assert len(pd.read_csv(run_dir / "scenario_summary.csv")) == 4
        assert len(list((run_dir / "per_step").glob("*.parquet"))) == 4


def test_cell_slug_round_trips_the_axis_values():
    assert run_simulation.cell_slug(0.175, 1.0, 32) == "e0.175_d1.0_w32"
    # Neighbouring values must not collide into one part file.
    assert run_simulation.cell_slug(0.1750001, 1.0, 32) != run_simulation.cell_slug(
        0.175, 1.0, 32
    )


def test_grid_cells_refuses_an_empty_axis():
    with pytest.raises(ValueError, match="empty simulation grid"):
        run_simulation.grid_cells(SimulationGrid(demand_levels=()))


def test_cell_arrival_modes_tracks_what_a_cell_actually_writes():
    cfg, grid = DEFAULT_CONFIG, SimulationGrid()
    assert run_simulation.cell_arrival_modes(cfg, grid) == (BOOTSTRAP, HISTORICAL)
    assert run_simulation.cell_arrival_modes(
        cfg, SimulationGrid(include_historical_reference=False)
    ) == (BOOTSTRAP,)
    assert run_simulation.cell_arrival_modes(
        cfg.with_(arrival_mode=HISTORICAL), grid
    ) == (HISTORICAL,)
    with pytest.raises(ValueError, match="nothing to simulate"):
        run_simulation.cell_arrival_modes(
            cfg.with_(num_bootstrap_runs=0),
            SimulationGrid(include_historical_reference=False),
        )


# --- resume -----------------------------------------------------------------

FOUR_CELLS = ("--demand-levels", "1", "2")


class Killed(Exception):
    """Stands in for the OOM or the kill that ends a long sweep."""


def kill_after(cells: int):
    """A `run_scenario` that dies once `cells` cells have been simulated."""
    real = run_simulation.run_scenario
    simulated = itertools.count()

    def guard(*args, **kwargs):
        if next(simulated) >= cells:
            raise Killed("killed mid-sweep")
        return real(*args, **kwargs)

    return guard


def sole_run_dir(tmp_path) -> pathlib.Path:
    (run_dir,) = sorted((tmp_path / "out").iterdir())
    return run_dir


def interrupted_run(tmp_path, monkeypatch, *, after: int) -> pathlib.Path:
    """Run the four-cell sweep and kill it after `after` cells; return its directory."""
    monkeypatch.setattr(run_simulation, "run_scenario", kill_after(after))
    with pytest.raises(Killed):
        run_end_to_end(tmp_path, *FOUR_CELLS)
    monkeypatch.undo()
    return sole_run_dir(tmp_path)


def test_resuming_an_interrupted_sweep_reproduces_the_uninterrupted_one(
    tmp_path, offline_data, monkeypatch
):
    """The whole point: resume must be indistinguishable from never stopping."""
    reference = run_end_to_end(tmp_path / "whole", *FOUR_CELLS)

    partial = tmp_path / "killed"
    run_dir = interrupted_run(partial, monkeypatch, after=2)
    # Killed mid-sweep: two cells checkpointed, and the manifest says unfinished.
    assert len(list((run_dir / "per_step").glob("*.parquet"))) == 2 * 2
    assert json.loads((run_dir / "manifest.json").read_text())["completed"] is False

    result = run_end_to_end(partial, *FOUR_CELLS, resume=run_dir.name)

    assert result.run_dir == run_dir  # same directory, no second timestamp
    assert result.manifest["completed"] is True
    assert len(list((partial / "out").iterdir())) == 1
    pd.testing.assert_frame_equal(
        result.summary, reference.summary, check_dtype=False
    )
    resumed_steps, whole_steps = (
        pd.read_parquet(directory / "per_step").sort_values(
            list(schemas.PER_STEP_IDENTITY_COLUMNS), ignore_index=True
        )
        for directory in (result.run_dir, reference.run_dir)
    )
    pd.testing.assert_frame_equal(resumed_steps, whole_steps)


def test_resume_redoes_a_cell_caught_between_its_parts_and_its_summary_row(
    tmp_path, offline_data, monkeypatch
):
    """A crash inside a checkpoint must not leave a summary hole or a duplicate."""
    run_dir = interrupted_run(tmp_path, monkeypatch, after=3)
    summary_path = run_dir / "scenario_summary.csv"
    # Rewind the third cell's rows, leaving its part files behind: exactly the
    # state a kill between the parquet write and the CSV append would leave.
    kept = summary_path.read_text().splitlines()[: 1 + 2 * 2]
    summary_path.write_text("\n".join(kept) + "\n")

    result = run_end_to_end(tmp_path, *FOUR_CELLS, resume=run_dir.name)

    on_disk = pd.read_csv(summary_path)
    assert len(on_disk) == 4 * 2  # four cells x two arrival modes, no duplicates
    assert not on_disk.duplicated(SCENARIO_KEYS + ["arrival_mode"]).any()
    pd.testing.assert_frame_equal(on_disk, result.summary, check_dtype=False)


def test_resume_refuses_a_run_it_would_contradict(tmp_path, offline_data, monkeypatch):
    run_dir = interrupted_run(tmp_path, monkeypatch, after=2)

    with pytest.raises(ValueError, match="different grid"):
        run_end_to_end(tmp_path, "--demand-levels", "1", "3", resume=run_dir.name)
    with pytest.raises(ValueError, match="different random_seed"):
        run_end_to_end(tmp_path, *FOUR_CELLS, "--seed", "1234", resume=run_dir.name)
    with pytest.raises(ValueError, match="does not exist"):
        run_end_to_end(tmp_path, *FOUR_CELLS, resume="20200101T000000Z")

    # Nothing was written by any of the refusals.
    assert len(list((tmp_path / "out").iterdir())) == 1
    assert len(list((run_dir / "per_step").glob("*.parquet"))) == 2 * 2


def test_resume_refuses_a_trace_that_moved_under_the_sweep(
    tmp_path, offline_data, monkeypatch
):
    """Same config, different resolved horizon: the upstream table grew."""
    run_dir = interrupted_run(tmp_path, monkeypatch, after=2)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resolved"]["simulation_horizon_blocks"] = 39
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="simulation_horizon_blocks differs"):
        run_end_to_end(tmp_path, *FOUR_CELLS, resume=run_dir.name)


def test_resume_refuses_a_completed_run(tmp_path, offline_data):
    result = run_end_to_end(tmp_path)

    with pytest.raises(ValueError, match="completed"):
        run_end_to_end(tmp_path, resume=result.run_dir.name)


def test_run_scenario_honours_historical_only_mode():
    cohorts = anchored_cohorts(num_blocks=40)
    cfg = run_simulation.build_config(parse(["--num-runs", "2"]))

    frames = run_simulation.run_scenario(cohorts, cfg, horizon=10, base_fee=8_000_000_000)
    assert [f["arrival_mode"].iloc[0] for f in frames] == [BOOTSTRAP] * 2 + [HISTORICAL]

    reference_only = run_simulation.run_scenario(
        cohorts, cfg.with_(arrival_mode=HISTORICAL), horizon=10, base_fee=8_000_000_000
    )
    assert [f["arrival_mode"].iloc[0] for f in reference_only] == [HISTORICAL]


def test_library_versions_reports_the_analysis_stack():
    versions = run_simulation.library_versions()
    assert versions["pandas"].startswith("3.")
    assert set(run_simulation.REPORTED_LIBRARIES) <= set(versions)
