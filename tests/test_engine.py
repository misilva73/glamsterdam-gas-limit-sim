"""Protocol arithmetic, the two-dimensional block fill, and end-to-end runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_CONFIG, MIN_BASE_FEE, SimConfig
from tests.dummy import dummy_tx_gas_results
from schemas import (
    BOTTLENECK_EXECUTION,
    BOTTLENECK_NONE,
    BOTTLENECK_STATE,
    PER_STEP_COLUMNS,
    PER_STEP_DEMAND_COLUMNS,
)
from sim.engine import fill_block, run_path, select_included
from sim.metrics import (
    effective_tip,
    next_base_fee,
    ramp_gas_limit,
    realized_tip_per_gas,
    update_price_signal,
)
from sim.workload import (
    bootstrap_path,
    build_cohorts,
    demand_multiplier,
    historical_path,
)
from tests.test_workload import anchored_cohorts, simulatable

GWEI = 1_000_000_000

DEFAULT_ROW = dict(
    block_number=100,
    tx_type=2,
    tx_gas_limit=30_000_000,
    max_fee_per_gas=100 * GWEI,
    max_priority_fee_per_gas=GWEI,
    execution_gas=21_000,
    state_gas=0,
    schedule_gas_used=21_000,
)


def tx_frame(*rows: dict) -> pd.DataFrame:
    """Hand-built simulatable rows; `tx_index` follows insertion order per block."""
    frame = pd.DataFrame([DEFAULT_ROW | row for row in rows])
    frame["tx_index"] = frame.groupby("block_number").cumcount()
    return frame


def mechanism_config(**overrides) -> SimConfig:
    """A config with the demand model switched off.

    `aggregate_elasticity=0` plus `adapt_bids=False` is exactly the behaviour
    before the demand model existed: a flat multiplier and frozen fee caps. The
    block-fill, protocol-arithmetic, and mempool tests below are about those
    mechanisms, so they hold the demand side still and need no anchor prices.
    The demand model has its own section at the end of this file.
    """
    return DEFAULT_CONFIG.with_(aggregate_elasticity=0.0, adapt_bids=False, **overrides)


def fixed_limit_config(gas_limit: int, **overrides) -> SimConfig:
    """A config whose gas limit never moves, so a test can hand-pick capacity."""
    return mechanism_config(
        arrival_mode="historical",
        fusaka_gas_limit=gas_limit,
        glamsterdam_gas_limit=gas_limit,
        **overrides,
    )


def run(frame: pd.DataFrame, cfg: SimConfig, base_fee: int = 0) -> pd.DataFrame:
    cohorts = build_cohorts(frame)
    path = historical_path(cohorts, len(cohorts))
    return run_path(cohorts, path, cfg, run_index=0, starting_base_fee=base_fee)


# --- Protocol arithmetic ----------------------------------------------------------


def test_gas_limit_ramp_climbs_to_the_target_without_overshooting():
    limit, target = 60_000_000, 200_000_000
    trajectory = [limit]
    while trajectory[-1] < target:
        trajectory.append(ramp_gas_limit(trajectory[-1], target))
        assert trajectory[-1] <= target
        assert trajectory[-1] > trajectory[-2]
        assert len(trajectory) < 5_000

    assert trajectory[-1] == target
    assert trajectory[1] == 60_000_000 + 60_000_000 // 1024
    # 1/1024 compound growth from 60M to 200M: ~1233 blocks, about four hours.
    assert 1_200 < len(trajectory) < 1_300
    assert ramp_gas_limit(target, target) == target


def test_base_fee_at_target_is_unchanged():
    assert next_base_fee(1_000 * GWEI, 30_000_000, 60_000_000) == 1_000 * GWEI


def test_base_fee_rises_by_an_eighth_of_the_relative_overshoot():
    # Full block: used is 2x target, so the rise is the maximum 12.5%.
    assert next_base_fee(8 * GWEI, 60_000_000, 60_000_000) == 9 * GWEI
    # Half-way over target: 6.25%.
    assert next_base_fee(8 * GWEI, 45_000_000, 60_000_000) == 8_500_000_000


def test_base_fee_falls_by_an_eighth_of_the_relative_undershoot():
    assert next_base_fee(8 * GWEI, 0, 60_000_000) == 7 * GWEI
    assert next_base_fee(8 * GWEI, 15_000_000, 60_000_000) == 7_500_000_000


def test_base_fee_increment_is_floored_at_one_wei_but_the_decrement_is_not():
    # A tiny base fee one gas over target would round its increase to zero, so
    # the protocol forces a 1-wei step; the decrease has no such floor.
    assert next_base_fee(1, 30_000_001, 60_000_000) == 2
    assert next_base_fee(0, 60_000_000, 60_000_000) == 1
    assert next_base_fee(8, 0, 60_000_000) == 7


def test_the_base_fee_never_falls_below_one_wei():
    """MIN_BASE_FEE, and why it is a guard rather than a live constraint."""
    # The decrement is at most an eighth of the base fee itself, so it floors to
    # nothing below 8: an emptying chain sticks in 1..7 on its own arithmetic.
    assert next_base_fee(1, 0, 60_000_000) == 1
    assert next_base_fee(7, 0, 60_000_000) == 7
    # The floor therefore binds only from a base fee of exactly 0, reachable
    # only via an explicit --starting-base-fee 0.
    assert next_base_fee(0, 0, 60_000_000) == MIN_BASE_FEE == 1
    assert next_base_fee(0, 30_000_000, 60_000_000) == 1  # the at-target branch


def test_effective_tip_by_transaction_type():
    base_fee = 10 * GWEI
    tx_type = np.array([2, 3, 4, 0, 1, 2])
    max_fee = np.array([12, 100, 11, 15, 30, 4]) * GWEI
    max_priority = np.array([1, 2, 5, 15, 30, 1]) * GWEI

    tip = effective_tip(tx_type, max_fee, max_priority, base_fee)

    assert tip.tolist() == [
        1 * GWEI,  # dynamic fee: priority fee binds
        2 * GWEI,  # blob: priority fee binds
        1 * GWEI,  # set code: fee cap binds
        5 * GWEI,  # legacy: normalized gas price minus base fee
        20 * GWEI,  # access list: same rule
        -6 * GWEI,  # priced below the base fee, left negative for the caller
    ]


# --- Block fill ------------------------------------------------------------------


def test_fill_block_skips_a_non_fitting_item_and_keeps_walking():
    execution_gas = np.array([900_000, 200_000, 50_000, 60_000, 40_000])
    state_gas = np.zeros(5, np.int64)

    taken, remaining_execution, remaining_state = fill_block(
        execution_gas, state_gas, 1_000_000, 1_000_000
    )

    # Items 1 and 3 do not fit once item 0 is in; the walk continues past both.
    assert taken.tolist() == [0, 2, 4]
    assert remaining_execution == 1_000_000 - 990_000
    assert remaining_state == 1_000_000


def test_fill_block_is_not_first_fit_across_tranches():
    # One transaction larger than the whole block is skipped, not a stop signal.
    execution_gas = np.array([2_000_000, 10_000])
    state_gas = np.zeros(2, np.int64)
    taken, _, _ = fill_block(execution_gas, state_gas, 1_000_000, 1_000_000)
    assert taken.tolist() == [1]


def test_select_included_walks_the_tip_order_and_skips_what_does_not_fit():
    tips = np.array([100, 50, 10])
    execution_gas = np.array([900_000, 200_000, 50_000])
    state_gas = np.zeros(3, np.int64)

    chosen = select_included(tips, execution_gas, state_gas, 1_000_000)
    assert chosen.tolist() == [0, 2]


def naive_fill(execution_gas, state_gas, gas_limit):
    """The sequential skip-and-continue walk, written out literally."""
    used_execution = used_state = 0
    taken = []
    for i, (execution, state) in enumerate(zip(execution_gas, state_gas)):
        if used_execution + execution <= gas_limit and used_state + state <= gas_limit:
            taken.append(i)
            used_execution += execution
            used_state += state
    return taken


def test_fill_block_matches_the_sequential_walk_on_random_blocks():
    rng = np.random.default_rng(0)
    for _ in range(200):
        size = int(rng.integers(1, 60))
        execution_gas = rng.integers(0, 400_000, size)
        state_gas = rng.integers(0, 400_000, size)
        taken, _, _ = fill_block(execution_gas, state_gas, 1_000_000, 1_000_000)
        assert taken.tolist() == naive_fill(execution_gas, state_gas, 1_000_000)


def test_tip_tranching_reproduces_a_single_full_sort(monkeypatch):
    """Tranche boundaries must never reorder the book, ties included."""
    import sim.engine as engine

    rng = np.random.default_rng(1)
    for _ in range(50):
        size = int(rng.integers(1, 200))
        tips = rng.integers(0, 5, size)  # heavy ties, so cuts land mid-value
        execution_gas = rng.integers(1, 200_000, size)
        state_gas = rng.integers(0, 200_000, size)

        order = np.argsort(-tips, kind="stable")
        expected = order[naive_fill(execution_gas[order], state_gas[order], 1_000_000)]

        monkeypatch.setattr(engine, "TIP_TRANCHE_SIZE", int(rng.integers(1, 8)))
        chosen = engine.select_included(tips, execution_gas, state_gas, 1_000_000)
        assert chosen.tolist() == expected.tolist()


def test_high_tip_transaction_is_skipped_and_a_small_low_tip_one_included():
    frame = tx_frame(
        dict(max_priority_fee_per_gas=100 * GWEI, execution_gas=900_000, schedule_gas_used=900_000),
        dict(max_priority_fee_per_gas=50 * GWEI, execution_gas=200_000, schedule_gas_used=200_000),
        dict(max_priority_fee_per_gas=10 * GWEI, execution_gas=50_000, schedule_gas_used=50_000),
    )
    step = run(frame, fixed_limit_config(1_000_000)).iloc[0]

    assert step["included_tx_count"] == 2
    assert step["block_execution_gas_used"] == 950_000
    assert step["backlog_tx_count"] == 1
    assert step["backlog_eligible_execution_gas"] == 200_000


def test_state_bound_block_leaves_execution_gas_unused():
    rows = [dict(execution_gas=10_000, state_gas=100_000) for _ in range(20)]
    step = run(tx_frame(*rows), fixed_limit_config(1_000_000)).iloc[0]

    assert step["block_state_gas_used"] == 1_000_000
    assert step["block_execution_gas_used"] == 100_000
    assert step["gas_used"] == 1_000_000
    assert step["bottleneck_dimension"] == BOTTLENECK_STATE
    assert step["state_utilization"] == 1.0
    assert step["execution_utilization"] == pytest.approx(0.1)
    assert step["included_tx_count"] == 10


def test_execution_bound_block_leaves_state_gas_unused():
    rows = [dict(execution_gas=100_000, state_gas=1_000) for _ in range(20)]
    step = run(tx_frame(*rows), fixed_limit_config(1_000_000)).iloc[0]

    assert step["block_execution_gas_used"] == 1_000_000
    assert step["block_state_gas_used"] == 10_000
    assert step["gas_used"] == 1_000_000
    assert step["bottleneck_dimension"] == BOTTLENECK_EXECUTION
    assert step["included_tx_count"] == 10


def test_empty_block_has_no_bottleneck():
    # Priced below the starting base fee, so nothing is includable.
    frame = tx_frame(dict(max_fee_per_gas=GWEI))
    step = run(frame, fixed_limit_config(1_000_000), base_fee=10 * GWEI).iloc[0]

    assert step["included_tx_count"] == 0
    assert step["gas_used"] == 0
    assert step["bottleneck_dimension"] == BOTTLENECK_NONE


def test_sender_gas_and_priority_fees_are_post_refund_amounts():
    frame = tx_frame(
        dict(
            max_fee_per_gas=50 * GWEI,
            max_priority_fee_per_gas=2 * GWEI,
            execution_gas=100_000,
            state_gas=40_000,
            schedule_gas_used=90_000,  # below execution_gas: refunded, floor not binding
        )
    )
    step = run(frame, fixed_limit_config(1_000_000), base_fee=10 * GWEI).iloc[0]

    assert step["block_execution_gas_used"] == 100_000
    assert step["sender_gas_used"] == 90_000
    assert step["priority_fees_wei"] == 2 * GWEI * 90_000


# --- Fee eligibility --------------------------------------------------------------


def test_fee_ineligible_demand_waits_for_the_base_fee_to_fall():
    frame = tx_frame(
        dict(max_fee_per_gas=50 * GWEI, execution_gas=21_000, state_gas=5_000)
    )
    cfg = fixed_limit_config(60_000_000, drain_blocks=10)
    steps = run(frame, cfg, base_fee=100 * GWEI)

    ineligible = steps["backlog_fee_ineligible_tx_count"].to_numpy()
    assert ineligible[0] == 1
    assert steps.loc[0, "backlog_fee_ineligible_execution_gas"] == 21_000
    assert steps.loc[0, "backlog_fee_ineligible_state_gas"] == 5_000
    assert steps.loc[0, "backlog_eligible_tx_count"] == 0

    # Empty blocks decay the base fee 12.5% per block until the cap clears.
    included_at = steps.index[steps["included_tx_count"] == 1]
    assert len(included_at) == 1
    first = int(included_at[0])
    assert steps.loc[first, "base_fee_per_gas"] <= 50 * GWEI
    assert steps.loc[first - 1, "base_fee_per_gas"] > 50 * GWEI
    assert (ineligible[first:] == 0).all()
    assert steps.loc[first, "gas_used"] == 21_000


# --- Gas-limit trajectory ---------------------------------------------------------


def test_gas_limit_is_flat_before_the_glamsterdam_step_then_ramps():
    frame = tx_frame(*[dict(block_number=100 + i) for i in range(8)])
    cfg = mechanism_config(arrival_mode="historical", first_glamsterdam_simulation_step=5)
    limits = run(frame, cfg, base_fee=GWEI)["gas_limit"].tolist()

    # No activation delay: the bump lands on the configured step itself.
    assert limits[:5] == [60_000_000] * 5
    assert limits[5] == 60_000_000 + 60_000_000 // 1024
    assert limits[6] > limits[5]
    assert max(limits) < 200_000_000


def test_first_simulated_block_reports_the_configured_initial_state():
    frame = tx_frame(*[dict(block_number=100 + i) for i in range(4)])
    cfg = mechanism_config(arrival_mode="historical")
    steps = run(frame, cfg, base_fee=7 * GWEI)

    assert steps.loc[0, "gas_limit"] == cfg.fusaka_gas_limit
    assert steps.loc[0, "base_fee_per_gas"] == 7 * GWEI
    assert steps.loc[1, "gas_limit"] == 60_000_000 + 60_000_000 // 1024


# --- Output dtypes ----------------------------------------------------------------


def test_priority_fees_survive_a_block_that_overflows_int64(tmp_path):
    """A 200M-gas block of high-tip transactions exceeds int64 (~1e20 wei).

    Regression: returning a Python int here made the column object-dtype, which
    only failed later, at the Parquet write, after a whole simulation had run.
    """
    rows = [
        dict(
            max_fee_per_gas=500 * GWEI,
            max_priority_fee_per_gas=500 * GWEI,
            execution_gas=1_000_000,
            schedule_gas_used=1_000_000,
            tx_gas_limit=1_000_000,
        )
        for _ in range(200)
    ]
    steps = run(tx_frame(*rows), fixed_limit_config(200_000_000), base_fee=0)

    assert steps["priority_fees_wei"].dtype == np.float64
    assert steps.loc[0, "priority_fees_wei"] > np.iinfo(np.int64).max
    steps.to_parquet(tmp_path / "per_step.parquet")  # must not raise


def test_legacy_rows_are_priced_by_gas_price_not_a_zero_priority_fee():
    """Legacy/access-list rows carry a normalized gas_price in max_fee_per_gas.

    The producer has no priority-fee field to fill for them, so a 0 there must not
    be read as a 0 tip -- that would starve every legacy transaction.
    """
    legacy = dict(tx_type=0, max_fee_per_gas=50 * GWEI, max_priority_fee_per_gas=0)
    dynamic = dict(tx_type=2, max_fee_per_gas=50 * GWEI, max_priority_fee_per_gas=GWEI)
    cfg = fixed_limit_config(21_000)  # room for exactly one transaction
    steps = run(tx_frame(legacy, dynamic), cfg, base_fee=10 * GWEI)

    # Legacy tip is 50 - 10 = 40 gwei, beating the dynamic row's 1 gwei.
    assert steps.loc[0, "included_tx_count"] == 1
    assert steps.loc[0, "priority_fees_wei"] == pytest.approx(40 * GWEI * 21_000)


# --- Drain phase and determinism --------------------------------------------------


def test_drain_phase_adds_no_arrivals_and_clears_the_backlog():
    rows = [dict(execution_gas=400_000, schedule_gas_used=400_000) for _ in range(6)]
    cfg = fixed_limit_config(1_000_000, drain_blocks=3)
    steps = run(tx_frame(*rows), cfg)

    assert steps["is_drain_step"].tolist() == [False, True, True, True]
    assert steps.loc[0, "included_tx_count"] == 2
    assert steps.loc[1:, "arrived_tx_count"].eq(0).all()
    assert steps.loc[1:, ["source_block_number", "window_instance", "position_in_window"]].isna().all().all()
    assert steps["backlog_tx_count"].tolist() == [4, 2, 0, 0]
    # Backlog gas is carried incrementally across steps: check it against the
    # arithmetic that arrivals minus inclusions implies.
    assert steps["backlog_execution_gas"].tolist() == [1_600_000, 800_000, 0, 0]
    assert steps["backlog_eligible_execution_gas"].tolist() == [1_600_000, 800_000, 0, 0]
    assert (steps["backlog_state_gas"] == 0).all()
    assert steps.loc[3, "bottleneck_dimension"] == BOTTLENECK_NONE


@pytest.fixture(scope="module")
def dummy_cohorts():
    return build_cohorts(simulatable(dummy_tx_gas_results(num_blocks=80, seed=13)))


def test_same_seed_and_config_reproduce_an_identical_frame(dummy_cohorts):
    cfg = mechanism_config(demand_level=1.5, bootstrap_window_blocks=16, drain_blocks=3)
    path = bootstrap_path(dummy_cohorts, 40, cfg.bootstrap_window_blocks, np.random.default_rng(4))

    first = run_path(dummy_cohorts, path, cfg, run_index=0, starting_base_fee=8 * GWEI)
    second = run_path(dummy_cohorts, path, cfg, run_index=0, starting_base_fee=8 * GWEI)
    other_run = run_path(dummy_cohorts, path, cfg, run_index=1, starting_base_fee=8 * GWEI)

    pd.testing.assert_frame_equal(first, second)
    assert first.to_csv(index=False) == second.to_csv(index=False)
    # A different run index draws its own replication stream.
    assert not first["arrived_tx_count"].equals(other_run["arrived_tx_count"])


def test_backlog_gas_conserves_arrivals_minus_inclusions(dummy_cohorts):
    """Every gas unit that arrives is either included or still in the backlog."""
    cfg = mechanism_config(arrival_mode="historical", drain_blocks=4)
    path = historical_path(dummy_cohorts, 40)
    steps = run_path(dummy_cohorts, path, cfg, run_index=0, starting_base_fee=8 * GWEI)

    for dimension in ("execution_gas", "state_gas"):
        arrived = np.array(
            [dummy_cohorts.cohort(i)[dimension].sum() for i in path["cohort_index"]] + [0] * 4
        )
        used = steps[f"block_{dimension.replace('_gas', '')}_gas_used"].to_numpy()
        assert (steps[f"backlog_{dimension}"].to_numpy() == np.cumsum(arrived - used)).all()

    assert steps["backlog_tx_count"].iloc[-1] < steps["backlog_tx_count"].max()


def test_mempool_compaction_cadence_does_not_change_results(dummy_cohorts, monkeypatch):
    """Tombstoning is an optimisation: when it is collected must not matter."""
    import sim.engine as engine

    cfg = mechanism_config(demand_level=2.5, bootstrap_window_blocks=16, drain_blocks=10)
    path = bootstrap_path(dummy_cohorts, 60, 16, np.random.default_rng(5))
    run = lambda: run_path(dummy_cohorts, path, cfg, run_index=0, starting_base_fee=8 * GWEI)

    reference = run()
    for share in (0.0, 0.99):  # compact every block; never compact
        monkeypatch.setattr(engine, "COMPACTION_DEAD_SHARE", share)
        pd.testing.assert_frame_equal(reference, run())


def test_end_to_end_run_on_dummy_data(dummy_cohorts):
    cfg = mechanism_config(demand_level=2.0, bootstrap_window_blocks=16, drain_blocks=5)
    path = bootstrap_path(dummy_cohorts, 60, cfg.bootstrap_window_blocks, np.random.default_rng(0))
    steps = run_path(dummy_cohorts, path, cfg, run_index=3, starting_base_fee=8 * GWEI)

    assert tuple(steps.columns) == PER_STEP_COLUMNS
    assert len(steps) == 65
    assert steps["simulation_position"].tolist() == list(range(65))
    assert (steps["run_index"] == 3).all()
    assert (steps["arrival_mode"] == cfg.arrival_mode).all()
    assert (steps["gas_used"] <= steps["gas_limit"]).all()
    assert (
        steps["gas_used"]
        == np.maximum(steps["block_execution_gas_used"], steps["block_state_gas_used"])
    ).all()
    assert steps["included_tx_count"].sum() > 0
    assert steps["priority_fees_wei"].min() >= 0
    assert (
        steps["backlog_tx_count"]
        == steps["backlog_eligible_tx_count"] + steps["backlog_fee_ineligible_tx_count"]
    ).all()
    assert (steps["gas_limit"].diff().dropna() >= 0).all()


# --- The demand model -------------------------------------------------------------


def headers_for(frame: pd.DataFrame, base_fee: int) -> pd.DataFrame:
    """Headers pinning every source block to one historical base fee."""
    return pd.DataFrame(
        {
            "block_number": np.sort(frame["block_number"].unique()),
            "base_fee_per_gas": int(base_fee),
        }
    )


def run_priced(
    frame: pd.DataFrame, cfg: SimConfig, *, anchor_base_fee: int, base_fee: int
) -> pd.DataFrame:
    cohorts = build_cohorts(frame, headers_for(frame, anchor_base_fee))
    return run_path(
        cohorts,
        historical_path(cohorts, len(cohorts)),
        cfg,
        run_index=0,
        starting_base_fee=base_fee,
    )


def priced_config(**overrides) -> SimConfig:
    """A config with the demand model live, historical arrivals unless overridden."""
    return DEFAULT_CONFIG.with_(**{"arrival_mode": "historical", **overrides})


def test_a_price_responsive_config_refuses_cohorts_without_anchors(dummy_cohorts):
    """Both features need the price the cohort was observed at, and say so."""
    path = historical_path(dummy_cohorts, 5)
    with pytest.raises(ValueError, match="aggregate_elasticity and adapt_bids cannot run without"):
        run_path(
            dummy_cohorts,
            path,
            priced_config(aggregate_elasticity=0.175, adapt_bids=True),
            run_index=0,
            starting_base_fee=GWEI,
        )
    with pytest.raises(ValueError, match="^adapt_bids cannot run without"):
        run_path(
            dummy_cohorts,
            path,
            priced_config(aggregate_elasticity=0.0, adapt_bids=True),
            run_index=0,
            starting_base_fee=GWEI,
        )
    # Neither feature on: the flat-multiplier mode needs no anchors.
    run_path(
        dummy_cohorts,
        path,
        priced_config(aggregate_elasticity=0.0, adapt_bids=False),
        run_index=0,
        starting_base_fee=GWEI,
    )


def test_step_zero_sits_exactly_on_the_anchor_whatever_the_starting_base_fee():
    """The anchor fixed point: the price signal is seeded at the first anchor."""
    frame = tx_frame(*[dict(block_number=100 + i) for i in range(4)])
    cfg = priced_config(aggregate_elasticity=0.28, demand_level=1.5)

    for base_fee in (1, GWEI, 500 * GWEI):
        steps = run_priced(frame, cfg, anchor_base_fee=10 * GWEI, base_fee=base_fee)
        assert steps.loc[0, "realized_demand_multiplier"] == pytest.approx(1.5)
        assert not steps.loc[0, "demand_multiplier_clamped"]
        # tip is min(1 gwei, 100 - 10) = 1 gwei, so the anchor price is 11 gwei.
        assert steps.loc[0, "cohort_anchor_price"] == pytest.approx(11 * GWEI)
        assert steps.loc[0, "demand_price_signal"] == pytest.approx(11 * GWEI)


def test_the_engine_applies_the_multiplier_the_demand_model_specifies(dummy_cohorts):
    """Exact wiring check: every step's multiplier is the model's own answer."""
    priced = anchored_cohorts(num_blocks=40, seed=21)
    cfg = priced_config(aggregate_elasticity=0.175, demand_level=1.2, price_ema_blocks=8)
    steps = run_priced_cohorts(priced, cfg, base_fee=4 * GWEI, drain=3)

    arrivals = steps[~steps["is_drain_step"]]
    expected = [
        demand_multiplier(cfg, row.demand_price_signal, row.cohort_anchor_price)[0]
        for row in arrivals.itertuples()
    ]
    assert arrivals["realized_demand_multiplier"].tolist() == pytest.approx(expected)


def run_priced_cohorts(cohorts, cfg: SimConfig, *, base_fee: int, drain: int = 0):
    return run_path(
        cohorts,
        historical_path(cohorts, len(cohorts)),
        cfg.with_(drain_blocks=drain),
        run_index=0,
        starting_base_fee=base_fee,
    )


def test_the_price_signal_is_an_ema_of_the_effective_price_not_the_base_fee():
    frame = tx_frame(*[dict(block_number=100 + i) for i in range(12)])
    cfg = priced_config(aggregate_elasticity=0.175, price_ema_blocks=4)
    steps = run_priced(frame, cfg, anchor_base_fee=10 * GWEI, base_fee=200 * GWEI)

    signal = steps["demand_price_signal"].to_numpy()
    # Seeded at the anchor, then dragged a fraction of the way toward the much
    # higher simulated price -- never jumping straight to it.
    assert signal[0] == pytest.approx(11 * GWEI)
    assert 11 * GWEI < signal[1] < 200 * GWEI

    # Reproduce the EMA by hand from the recorded blocks.
    expected, tip = signal[0], 0.0
    for row in steps.itertuples():
        assert row.demand_price_signal == pytest.approx(expected)
        tip = realized_tip_per_gas(row.priority_fees_wei, row.sender_gas_used, tip)
        expected = update_price_signal(
            expected, row.base_fee_per_gas, tip, cfg.price_ema_alpha
        )


def test_demand_shrinks_above_the_anchor_price_and_grows_below_it():
    """The sign of the response, and the loop that produces both halves of it.

    These blocks hold one 21k-gas transaction against a 60M limit, so they are
    far under target and the base fee collapses whatever it starts at. Starting
    it well above the anchor therefore walks the price signal through both
    regimes in one run: demand contracts while the signal is above the anchor,
    then expands once the collapsing base fee drags it below.
    """
    frame = tx_frame(*[dict(block_number=100 + i) for i in range(60)])
    cfg = priced_config(aggregate_elasticity=0.28, price_ema_blocks=3)
    steps = run_priced(frame, cfg, anchor_base_fee=GWEI, base_fee=1_000 * GWEI)

    multiplier = steps["realized_demand_multiplier"].to_numpy()
    signal = steps["demand_price_signal"].to_numpy()
    anchor = steps["cohort_anchor_price"].to_numpy()

    assert (multiplier < 1.0).any() and (multiplier > 1.0).any()
    # Exactly, at every step: demand is below its level iff the price the model
    # sees is above the price the cohort was observed at.
    assert ((multiplier < 1.0) == (signal > anchor)).all()
    assert ((multiplier > 1.0) == (signal < anchor)).all()


def test_bid_adaptation_makes_a_cohort_includable_at_a_far_higher_base_fee():
    """Without it the fee filter, not the elasticity, would set the quantity."""
    frame = tx_frame(
        dict(max_fee_per_gas=20 * GWEI, max_priority_fee_per_gas=GWEI, execution_gas=21_000)
    )
    cfg = priced_config(aggregate_elasticity=0.0)
    kwargs = dict(anchor_base_fee=10 * GWEI, base_fee=100 * GWEI)

    frozen = run_priced(frame, cfg.with_(adapt_bids=False), **kwargs)
    adapted = run_priced(frame, cfg.with_(adapt_bids=True), **kwargs)

    # A 20 gwei cap is worthless at a 100 gwei base fee...
    assert frozen.loc[0, "included_tx_count"] == 0
    assert frozen.loc[0, "backlog_fee_ineligible_tx_count"] == 1
    # ...but the same 2x headroom over its own block's base fee is not.
    assert adapted.loc[0, "included_tx_count"] == 1
    assert adapted.loc[0, "priority_fees_wei"] == pytest.approx(GWEI * 21_000)


def test_drain_steps_record_no_demand_because_no_cohort_arrives():
    priced = anchored_cohorts(num_blocks=20, seed=31)
    steps = run_priced_cohorts(
        priced, priced_config(aggregate_elasticity=0.175), base_fee=8 * GWEI, drain=4
    )

    drain = steps[steps["is_drain_step"]]
    assert len(drain) == 4
    assert drain["realized_demand_multiplier"].isna().all()
    assert drain["cohort_anchor_price"].isna().all()
    assert not drain["demand_multiplier_clamped"].any()
    assert (drain[["arrived_tx_count", "arrived_execution_gas", "arrived_state_gas"]] == 0).all().all()
    # The price signal keeps updating: the drain phase still has a base fee.
    assert drain["demand_price_signal"].notna().all()


def test_the_clamp_is_recorded_when_the_price_leaves_the_estimation_range():
    frame = tx_frame(*[dict(block_number=100 + i) for i in range(30)])
    cfg = priced_config(
        aggregate_elasticity=0.28, price_ema_blocks=2, demand_multiplier_bounds=(0.9, 1.1)
    )
    steps = run_priced(frame, cfg, anchor_base_fee=GWEI, base_fee=1_000 * GWEI)

    assert steps["demand_multiplier_clamped"].any()
    clamped = steps.loc[steps["demand_multiplier_clamped"], "realized_demand_multiplier"]
    assert clamped.between(0.9, 1.1).all()


def test_arrived_gas_is_recorded_and_conserved_against_the_backlog():
    """Sampling makes arrivals unrecoverable from the path, so they are output."""
    priced = anchored_cohorts(num_blocks=40, seed=17)
    steps = run_priced_cohorts(
        priced, priced_config(aggregate_elasticity=0.175), base_fee=8 * GWEI, drain=6
    )

    for dimension in ("execution", "state"):
        arrived = steps[f"arrived_{dimension}_gas"].to_numpy()
        used = steps[f"block_{dimension}_gas_used"].to_numpy()
        assert (steps[f"backlog_{dimension}_gas"].to_numpy() == np.cumsum(arrived - used)).all()


def test_the_demand_model_run_is_reproducible_and_run_dependent():
    priced = anchored_cohorts(num_blocks=60, seed=41)
    cfg = priced_config(
        arrival_mode="moving_block_bootstrap",
        aggregate_elasticity=0.175,
        demand_level=1.5,
        bootstrap_window_blocks=16,
    )
    path = bootstrap_path(priced, 40, 16, np.random.default_rng(4))
    run = lambda index: run_path(
        priced, path, cfg, run_index=index, starting_base_fee=8 * GWEI
    )

    pd.testing.assert_frame_equal(run(0), run(0))
    # The sampling stream is per run, so paths differ even on one window draw.
    assert not run(0)["arrived_execution_gas"].equals(run(1)["arrived_execution_gas"])


def test_end_to_end_with_the_demand_model_on_dummy_data():
    priced = anchored_cohorts(num_blocks=80, seed=13)
    cfg = priced_config(
        arrival_mode="moving_block_bootstrap",
        aggregate_elasticity=0.175,
        demand_level=2.0,
        bootstrap_window_blocks=16,
    )
    path = bootstrap_path(priced, 60, 16, np.random.default_rng(0))
    steps = run_path(priced, path, cfg, run_index=3, starting_base_fee=8 * GWEI)

    assert tuple(steps.columns) == PER_STEP_COLUMNS
    for name in PER_STEP_DEMAND_COLUMNS:
        assert name in steps.columns
    assert (steps["aggregate_elasticity"] == 0.175).all()
    assert (steps["demand_level"] == 2.0).all()
    assert steps["realized_demand_multiplier"].notna().all()
    assert (steps["realized_demand_multiplier"] > 0).all()
    assert (steps["base_fee_per_gas"] >= MIN_BASE_FEE).all()
    assert (steps["gas_used"] <= steps["gas_limit"]).all()
    assert steps["arrived_execution_gas"].sum() > 0
    assert steps["included_tx_count"].sum() > 0
