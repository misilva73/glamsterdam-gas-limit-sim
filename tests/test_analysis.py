"""Tests for the analysis and simulation layer.

The engine and the workload model are exercised elsewhere; here everything is
driven from a synthetic per-step frame with exactly `schemas.PER_STEP_COLUMNS`,
so the aggregation and plotting code is testable on its own.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import run_simulation
import schemas
from analysis.plots import (
    aggregate_bands,
    plot_backlog,
    plot_base_fee,
    plot_demand_response,
    plot_gas_limit_ramp,
    plot_simulation,
    plot_utilization,
)
from analysis.window_length import (
    SUMMARY_SERIES,
    autocorrelation,
    cohort_summary,
    plot_autocorrelation,
    suggest_window_blocks,
)
from config import DEFAULT_CONFIG
from sim.workload import bootstrap_path, bootstrap_rng, build_cohorts
from tests.dummy import (
    DUMMY_ANALYSIS_CONFIG_HASH,
    dummy_block_headers,
    dummy_tx_gas_results,
)

BOOTSTRAP = "moving_block_bootstrap"
HISTORICAL = "historical"

PLOT_FUNCTIONS = (
    plot_base_fee,
    plot_utilization,
    plot_gas_limit_ramp,
    plot_backlog,
    plot_demand_response,
)


# --- synthetic per-step frames ----------------------------------------------


def synthetic_per_step(
    *,
    elasticities=(0.175,),
    levels=(1.0,),
    windows=(32,),
    num_runs: int = 4,
    positions: int = 24,
    arrival_mode: str = BOOTSTRAP,
    drain_from: int | None = None,
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
    is_drain = np.zeros(size, bool) if drain_from is None else position >= drain_from
    # The model's own arithmetic, so the synthetic frame stays internally
    # consistent with what the engine would have recorded.
    multiplier = np.where(is_drain, np.nan, scale * (base_fee / anchor_price) ** -elasticity)

    frame["arrival_mode"] = arrival_mode
    frame["is_drain_step"] = is_drain
    frame["source_block_number"] = 21_000_000 + position
    frame["window_instance"] = position // window
    frame["position_in_window"] = position % window
    frame["demand_price_signal"] = base_fee
    frame["cohort_anchor_price"] = np.where(is_drain, np.nan, anchor_price)
    frame["realized_demand_multiplier"] = multiplier
    frame["demand_multiplier_clamped"] = False
    frame["base_fee_per_gas"] = base_fee.astype(np.int64)
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
    frame["arrived_tx_count"] = np.where(is_drain, 0, rng.poisson(160 * scale))
    frame["arrived_execution_gas"] = np.where(is_drain, 0, execution_gas_used * 1.1).astype(
        np.int64
    )
    frame["arrived_state_gas"] = np.where(is_drain, 0, state_gas_used * 1.1).astype(np.int64)
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


# --- aggregate_bands --------------------------------------------------------


def hand_built_per_step(values_by_level: dict[float, list[float]]) -> pd.DataFrame:
    """One position, one elasticity, one window, one run per listed value."""
    rows = [
        {
            "aggregate_elasticity": 0.175,
            "demand_level": level,
            "bootstrap_window_blocks": 32,
            "simulation_position": 0,
            "run_index": run_index,
            "base_fee_per_gas": value,
        }
        for level, values in values_by_level.items()
        for run_index, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_aggregate_bands_computes_known_quantiles():
    per_step = hand_built_per_step({1.0: [0.0, 1.0, 2.0, 3.0, 4.0]})
    bands = aggregate_bands(per_step, ["base_fee_per_gas"]).set_index("statistic")["value"]

    assert bands["min"] == 0.0
    assert bands["max"] == 4.0
    assert bands["p50"] == 2.0
    assert bands["p10"] == pytest.approx(0.4)
    assert bands["p90"] == pytest.approx(3.6)


def test_aggregate_bands_groups_by_every_scenario_axis():
    per_step = pd.concat(
        [
            hand_built_per_step({1.0: [1.0, 2.0, 3.0], 2.0: [10.0, 20.0, 30.0]}),
            hand_built_per_step({1.0: [5.0, 5.0, 5.0]}).assign(bootstrap_window_blocks=64),
        ],
        ignore_index=True,
    )
    medians = (
        aggregate_bands(per_step, ["base_fee_per_gas"])
        .query("statistic == 'p50'")
        .set_index(["demand_level", "bootstrap_window_blocks"])["value"]
    )
    assert medians[(1.0, 32)] == 2.0
    assert medians[(2.0, 32)] == 20.0
    assert medians[(1.0, 64)] == 5.0


def test_aggregate_bands_is_tidy_and_covers_every_position():
    per_step = synthetic_per_step(levels=(1.0, 2.0), windows=(16, 32), positions=10)
    bands = aggregate_bands(per_step, ["base_fee_per_gas", "gas_limit"])

    assert list(bands.columns) == [
        "aggregate_elasticity",
        "demand_level",
        "bootstrap_window_blocks",
        "simulation_position",
        "metric",
        "statistic",
        "value",
    ]
    assert len(bands) == 2 * 2 * 10 * 2 * 5  # groups x positions x metrics x statistics
    assert bands["value"].notna().all()


def test_aggregate_bands_without_min_max():
    per_step = synthetic_per_step(positions=3)
    bands = aggregate_bands(per_step, ["gas_used"], quantiles=(0.25, 0.75), also_min_max=False)
    assert set(bands["statistic"]) == {"p25", "p75"}


# --- autocorrelation and window length --------------------------------------


def ar1(phi: float, n: int = 4000, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n)
    values = np.empty(n)
    values[0] = shocks[0]
    for i in range(1, n):
        values[i] = phi * values[i - 1] + shocks[i]
    return pd.Series(values, name="tx_count")


def test_autocorrelation_tracks_the_ar1_decay():
    correlations = autocorrelation(ar1(0.9), nlags=60)

    assert correlations.index.name == "lag"
    assert correlations.iloc[0] == pytest.approx(1.0)
    assert correlations[1] == pytest.approx(0.9, abs=0.05)
    assert correlations[5] == pytest.approx(0.9**5, abs=0.08)


def test_autocorrelation_clips_nlags_to_the_series_length():
    assert len(autocorrelation(pd.Series([1.0, 2.0, 3.0, 4.0]), nlags=100)) == 4


def test_suggest_window_blocks_recovers_the_ar1_timescale():
    # rho_k = 0.9^k crosses 1/e near lag 10; integral timescale is ~19 blocks.
    suggestion = suggest_window_blocks(pd.DataFrame({"tx_count": ar1(0.9)})).iloc[0]

    assert 6 <= suggestion["lag_below_1_over_e"] <= 16
    assert 10 <= suggestion["integral_timescale_blocks"] <= 40
    assert suggestion["supported_window_blocks"] in (32.0, 64.0)


def test_suggest_window_blocks_on_white_noise_supports_the_shortest_candidate():
    noise = pd.DataFrame({"tx_count": np.random.default_rng(1).normal(size=3000)})
    suggestion = suggest_window_blocks(noise).iloc[0]

    assert suggestion["lag_below_1_over_e"] == 1.0
    assert suggestion["lag_inside_white_noise_band"] == 1.0
    assert suggestion["integral_timescale_blocks"] == pytest.approx(1.0, abs=0.3)
    assert suggestion["supported_window_blocks"] == 16.0


def test_suggest_window_blocks_reports_when_no_candidate_is_long_enough():
    slow = pd.DataFrame({"tx_count": ar1(0.995, n=6000)})
    suggestion = suggest_window_blocks(slow, candidates=(2, 4)).iloc[0]
    assert np.isnan(suggestion["supported_window_blocks"])


def test_suggest_window_blocks_covers_every_summary_series():
    summary = cohort_summary(dummy_cohorts())
    suggestion = suggest_window_blocks(summary, nlags=80)
    assert list(suggestion["series"]) == list(SUMMARY_SERIES)


# --- cohort_summary ---------------------------------------------------------


def dummy_cohorts(num_blocks: int = 200) -> pd.DataFrame:
    """Dummy replay rows with the gas dimensions derived here, not imported."""
    frame = dummy_tx_gas_results(num_blocks=num_blocks)
    return frame.assign(
        state_gas=frame["schedule_state_gas_spent"],
        execution_gas=np.maximum(
            frame["schedule_total_gas_spent"] - frame["schedule_state_gas_spent"],
            frame["schedule_floor_gas"],
        ),
    )


def simulatable_cohorts(num_blocks: int = 200) -> pd.DataFrame:
    """Row selection is the loader's job; `build_cohorts` rejects an unsplit frame."""
    frame = dummy_cohorts(num_blocks)
    return frame[(frame["baseline_success"] == 1) & (frame["schedule_success"] == 1)]


def anchored_cohorts(num_blocks: int = 200):
    """Cohorts with demand anchors, as `run_simulation` builds them."""
    frame = dummy_cohorts(num_blocks)
    return build_cohorts(simulatable_cohorts(num_blocks), dummy_block_headers(frame))


def test_cohort_summary_is_one_row_per_source_block():
    tx_frame = dummy_cohorts(num_blocks=50)
    summary = cohort_summary(tx_frame)

    assert list(summary.columns) == list(SUMMARY_SERIES)
    assert summary.index.name == "block_number"
    assert len(summary) == tx_frame["block_number"].nunique()
    assert summary.index.is_monotonic_increasing
    assert summary["tx_count"].sum() == len(tx_frame)


def test_cohort_summary_totals_match_a_manual_groupby():
    tx_frame = dummy_cohorts(num_blocks=20)
    summary = cohort_summary(tx_frame)
    block = tx_frame["block_number"].iloc[0]
    cohort = tx_frame[tx_frame["block_number"] == block]

    assert summary.loc[block, "execution_gas"] == cohort["execution_gas"].sum()
    assert summary.loc[block, "state_gas"] == cohort["state_gas"].sum()
    assert summary.loc[block, "median_max_fee_per_gas"] == cohort["max_fee_per_gas"].median()


def test_dummy_cohorts_are_autocorrelated_enough_to_need_a_window():
    summary = cohort_summary(dummy_cohorts(num_blocks=400))
    assert autocorrelation(summary["tx_count"], nlags=10)[1] > 0.5


# --- figures ----------------------------------------------------------------


def assert_png(path):
    assert path.exists(), path
    assert path.stat().st_size > 5_000, f"{path} looks empty ({path.stat().st_size} bytes)"


@pytest.mark.parametrize("plot", PLOT_FUNCTIONS, ids=lambda f: f.__name__)
def test_plot_writes_a_png_for_a_single_scenario(plot, tmp_path):
    bootstrap = synthetic_per_step()
    historical = synthetic_per_step(num_runs=1, arrival_mode=HISTORICAL, seed=9)
    assert_png(plot(bootstrap, historical, tmp_path / f"{plot.__name__}.png"))


@pytest.mark.parametrize("plot", PLOT_FUNCTIONS, ids=lambda f: f.__name__)
def test_plot_writes_a_png_for_the_full_grid(plot, tmp_path):
    grid = dict(elasticities=(0.0, 0.175), levels=(1.0, 2.0), windows=(16, 64), positions=30)
    bootstrap = synthetic_per_step(drain_from=24, **grid)
    historical = synthetic_per_step(
        num_runs=1, arrival_mode=HISTORICAL, drain_from=24, seed=9, **grid
    )
    assert_png(plot(bootstrap, historical, tmp_path / f"{plot.__name__}_grid.png"))


@pytest.mark.parametrize("plot", PLOT_FUNCTIONS, ids=lambda f: f.__name__)
def test_plot_survives_a_missing_historical_reference(plot, tmp_path):
    bootstrap = synthetic_per_step()
    empty = bootstrap.iloc[:0]
    assert_png(plot(bootstrap, empty, tmp_path / f"{plot.__name__}_bands_only.png"))


def test_plot_simulation_writes_every_figure(tmp_path):
    bootstrap = synthetic_per_step(levels=(1.0, 2.0), drain_from=20)
    historical = synthetic_per_step(
        levels=(1.0, 2.0), num_runs=1, arrival_mode=HISTORICAL, drain_from=20
    )
    paths = plot_simulation(bootstrap, historical, tmp_path / "figures")

    assert len(paths) == len(PLOT_FUNCTIONS)
    assert {p.name for p in paths} == {
        "base_fee.png",
        "utilization.png",
        "gas_limit_ramp.png",
        "backlog.png",
        "demand_response.png",
    }
    for path in paths:
        assert_png(path)


def test_plot_autocorrelation_writes_a_png(tmp_path):
    summary = cohort_summary(dummy_cohorts(num_blocks=300))
    assert_png(plot_autocorrelation(summary, 120, tmp_path / "acf.png"))


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
            "--drain-blocks",
            "30",
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
    assert cfg.drain_blocks == 30
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
    assert grid.aggregate_elasticities == (0.0, 0.10, 0.175, 0.28)
    assert grid.demand_levels == (1.0, 1.5, 2.0, 3.0)
    assert grid.bootstrap_window_blocks == (16, 32, 64)
    assert grid.include_historical_reference is True


def test_the_analysis_config_hash_is_mandatory_on_the_command_line():
    """A run can never silently average two replay configurations."""
    with pytest.raises(SystemExit):
        run_simulation.build_parser().parse_args(["--block-range", "1", "2"])


def test_bootstrap_paths_are_reproducible_and_independent_across_runs():
    cohorts = build_cohorts(simulatable_cohorts(num_blocks=80))
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


def test_run_simulation_end_to_end_on_dummy_data(tmp_path, offline_data):
    args = parse(
        [
            "--block-range",
            "21000000",
            "21000119",
            "--horizon",
            "40",
            "--num-runs",
            "3",
            "--drain-blocks",
            "10",
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
        ]
    )
    result = run_simulation.run_simulation(
        run_simulation.build_config(args), run_simulation.build_grid(args)
    )

    assert set(schemas.PER_STEP_COLUMNS) <= set(result.per_step.columns)
    assert result.per_step["simulation_position"].max() == 40 + 10 - 1
    assert set(result.per_step["arrival_mode"]) == {BOOTSTRAP, HISTORICAL}
    assert result.per_step.groupby("arrival_mode")["run_index"].nunique().to_dict() == {
        BOOTSTRAP: 3,
        HISTORICAL: 1,
    }
    assert result.per_step["is_drain_step"].sum() == 10 * (3 + 1) * 2
    assert len(result.summary) == 4  # arrival modes x elasticities

    written = {path.name for path in result.outputs}
    assert {
        "per_step.csv",
        "per_step.parquet",
        "excluded_summary.csv",
        "scenario_summary.csv",
        "manifest.json",
        "base_fee.png",
        "utilization.png",
        "gas_limit_ramp.png",
        "backlog.png",
    } <= written
    for path in result.outputs:
        assert path.stat().st_size > 0

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["resolved"]["simulation_horizon_blocks"] == 40
    assert manifest["grid"]["aggregate_elasticities"] == [0.0, 0.175]
    assert manifest["grid"]["demand_levels"] == [2.0]
    assert manifest["timings"]["total_seconds"] > 0
    assert "lower bound" in manifest["caveat"].lower()


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
