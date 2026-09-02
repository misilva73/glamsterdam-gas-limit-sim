"""Arrival-path construction, demand anchors, sampling, and bid adaptation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_CONFIG
from tests.dummy import dummy_block_headers, dummy_tx_gas_results
from sim.metrics import effective_tip
from sim.workload import (
    PATH_COLUMNS,
    adapt_bids,
    bootstrap_path,
    build_cohorts,
    demand_multiplier,
    demand_pool_bounds,
    historical_path,
    sample_arrivals,
)

GWEI = 1_000_000_000


def simulatable(raw: pd.DataFrame) -> pd.DataFrame:
    """The data layer's gas derivation, so tests share its contract."""
    frame = raw[(raw["baseline_success"] == 1) & (raw["schedule_success"] == 1)].copy()
    frame["state_gas"] = frame["schedule_state_gas_spent"]
    frame["execution_gas"] = np.maximum(
        frame["schedule_total_gas_spent"] - frame["schedule_state_gas_spent"],
        frame["schedule_floor_gas"],
    )
    return frame


def anchored_cohorts(num_blocks: int = 120, seed: int = 3):
    """Cohorts with demand anchors, as the CLI builds them."""
    raw = dummy_tx_gas_results(num_blocks=num_blocks, seed=seed)
    return build_cohorts(simulatable(raw), dummy_block_headers(raw))


@pytest.fixture(scope="module")
def cohorts():
    return build_cohorts(simulatable(dummy_tx_gas_results(num_blocks=120, seed=3)))


@pytest.fixture(scope="module")
def priced_cohorts():
    return anchored_cohorts()


def test_cohorts_are_one_per_source_block_in_tx_index_order(cohorts):
    assert len(cohorts) == 120
    assert cohorts.tx_counts.sum() == cohorts.columns["tx_index"].size

    first = cohorts.cohort(0)
    assert (first["source_block_number"] == cohorts.block_numbers[0]).all()
    assert (np.diff(first["tx_index"]) > 0).all()


def test_build_cohorts_drops_unsuccessful_rows_and_demands_derived_columns():
    raw = dummy_tx_gas_results(num_blocks=20, seed=5)
    with pytest.raises(ValueError, match="execution_gas"):
        build_cohorts(raw)

    kept = build_cohorts(simulatable(raw)).tx_counts.sum()
    assert kept < len(raw)


def test_total_gas_is_the_historical_metering_basis(cohorts):
    """The demand model's quantity is S + B, not the simulator's max(S, B)."""
    assert (
        cohorts.total_gas == cohorts.columns["execution_gas"] + cohorts.columns["state_gas"]
    ).all()
    assert cohorts.cohort_total_gas(0) == int(
        cohorts.cohort(0)["execution_gas"].sum() + cohorts.cohort(0)["state_gas"].sum()
    )


# --- Demand anchors ---------------------------------------------------------------


def test_cohort_anchor_is_base_fee_plus_the_gas_weighted_realized_tip():
    frame = pd.DataFrame(
        {
            "block_number": [100, 100],
            "tx_index": [0, 1],
            "tx_type": [2, 2],
            "max_fee_per_gas": [100 * GWEI, 100 * GWEI],
            "max_priority_fee_per_gas": [2 * GWEI, 8 * GWEI],
            "execution_gas": [100_000, 300_000],
            "state_gas": [0, 0],
            "schedule_gas_used": [100_000, 300_000],
        }
    )
    headers = pd.DataFrame({"block_number": [100], "base_fee_per_gas": [10 * GWEI]})

    built = build_cohorts(frame, headers)

    # Tips of 2 and 8 gwei, weighted 100k:300k, give 6.5 gwei.
    assert built.anchor_tip.tolist() == [pytest.approx(6.5 * GWEI)]
    assert built.anchor_price.tolist() == [pytest.approx(16.5 * GWEI)]
    assert built.columns["anchor_base_fee"].tolist() == [10 * GWEI] * 2
    assert built.has_anchors


def test_anchor_matches_the_realized_tip_definition_used_on_the_simulated_side(
    priced_cohorts,
):
    """`anchor_price` must be on the same basis as `priority_fees_wei / sender_gas_used`."""
    columns = priced_cohorts.columns
    tip = np.clip(
        effective_tip(
            columns["tx_type"],
            columns["max_fee_per_gas"],
            columns["max_priority_fee_per_gas"],
            columns["anchor_base_fee"],
        ),
        0,
        None,
    )
    for index in (0, 7, len(priced_cohorts) - 1):
        start, stop = priced_cohorts.offsets[index], priced_cohorts.offsets[index + 1]
        gas = columns["schedule_gas_used"][start:stop]
        expected = columns["anchor_base_fee"][start] + (tip[start:stop] * gas).sum() / gas.sum()
        assert priced_cohorts.anchor_price[index] == pytest.approx(expected)


def test_cohorts_without_headers_carry_no_anchors(cohorts):
    assert cohorts.anchor_price is None
    assert cohorts.anchor_tip is None
    assert not cohorts.has_anchors


def test_missing_headers_for_a_cohort_block_are_refused():
    raw = dummy_tx_gas_results(num_blocks=10, seed=5)
    headers = dummy_block_headers(raw).iloc[2:]
    with pytest.raises(ValueError, match="missing 2 of 10 cohort blocks"):
        build_cohorts(simulatable(raw), headers)


# --- The demand multiplier --------------------------------------------------------


def test_multiplier_is_the_demand_level_at_the_anchor_price():
    cfg = DEFAULT_CONFIG.with_(aggregate_elasticity=0.175, demand_level=2.0)
    multiplier, clamped = demand_multiplier(cfg, price_signal=8 * GWEI, anchor_price=8 * GWEI)
    assert multiplier == pytest.approx(2.0)
    assert not clamped


def test_demand_rises_as_the_price_falls_and_falls_as_it_rises():
    cfg = DEFAULT_CONFIG.with_(aggregate_elasticity=0.175, demand_level=1.0)
    anchor = 10 * GWEI

    cheaper, _ = demand_multiplier(cfg, anchor / 4, anchor)
    dearer, _ = demand_multiplier(cfg, anchor * 4, anchor)

    assert cheaper == pytest.approx(4.0**0.175)
    assert dearer == pytest.approx(0.25**0.175)
    assert dearer < 1.0 < cheaper


def test_zero_elasticity_is_a_flat_multiplier_whatever_the_price():
    cfg = DEFAULT_CONFIG.with_(aggregate_elasticity=0.0, demand_level=3.0)
    for price in (1, GWEI, 1_000 * GWEI):
        assert demand_multiplier(cfg, price, 8 * GWEI) == (3.0, False)


def test_a_more_elastic_demand_responds_more_to_the_same_price_fall():
    anchor = 10 * GWEI
    responses = [
        demand_multiplier(
            DEFAULT_CONFIG.with_(aggregate_elasticity=e, demand_multiplier_bounds=(0.001, 1e9)),
            anchor / 10,
            anchor,
        )[0]
        for e in (0.0, 0.10, 0.175, 0.28)
    ]
    assert responses == sorted(responses)
    assert responses[0] == 1.0


def test_the_clamp_bounds_an_extrapolated_multiplier_and_says_so():
    cfg = DEFAULT_CONFIG.with_(
        aggregate_elasticity=0.175, demand_multiplier_bounds=(0.5, 2.0)
    )
    anchor = GWEI

    # A base fee at the 1-wei floor is a 1e9 price fall: (1e9 ** 0.175) ~ 38x.
    high, clamped_high = demand_multiplier(cfg, 1, anchor)
    assert (high, clamped_high) == (2.0, True)

    low, clamped_low = demand_multiplier(cfg, anchor * 10**9, anchor)
    assert (low, clamped_low) == (0.5, True)

    inside, clamped = demand_multiplier(cfg, anchor, anchor)
    assert (inside, clamped) == (1.0, False)


def test_the_clamp_bounds_the_price_response_not_the_chosen_demand_level():
    """A level above the bound is a deliberate assumption, not an extrapolation."""
    cfg = DEFAULT_CONFIG.with_(
        aggregate_elasticity=0.175, demand_level=30.0, demand_multiplier_bounds=(0.5, 2.0)
    )
    anchor = GWEI

    at_anchor, clamped = demand_multiplier(cfg, anchor, anchor)
    assert (at_anchor, clamped) == (30.0, False)

    # The response saturates at 2x, so the product is 60x, not capped at 2x.
    extrapolated, clamped = demand_multiplier(cfg, 1, anchor)
    assert (extrapolated, clamped) == (60.0, True)


def test_a_non_positive_price_or_anchor_falls_back_to_the_level():
    cfg = DEFAULT_CONFIG.with_(aggregate_elasticity=0.175, demand_level=1.5)
    assert demand_multiplier(cfg, 0, 8 * GWEI) == (1.5, False)
    assert demand_multiplier(cfg, 8 * GWEI, 0) == (1.5, False)


# --- Arrival paths ----------------------------------------------------------------


def test_historical_path_is_the_continuous_trace(cohorts):
    path = historical_path(cohorts, 50)
    assert tuple(path.columns) == PATH_COLUMNS
    assert path["cohort_index"].tolist() == list(range(50))
    assert (path["window_instance"] == 0).all()
    assert path["position_in_window"].tolist() == list(range(50))
    assert (path["source_block_number"].to_numpy() == cohorts.block_numbers[:50]).all()

    with pytest.raises(ValueError, match="exceeds"):
        historical_path(cohorts, len(cohorts) + 1)


def test_bootstrap_path_never_wraps_and_preserves_within_window_order(cohorts):
    horizon, window = 100, 16
    path = bootstrap_path(cohorts, horizon, window, np.random.default_rng(1))

    assert tuple(path.columns) == PATH_COLUMNS
    assert len(path) == horizon
    assert path["simulation_position"].tolist() == list(range(horizon))
    assert path["cohort_index"].max() < len(cohorts)

    windows = path.groupby("window_instance")["cohort_index"]
    assert windows.ngroups == -(-horizon // window)
    for _, indices in windows:
        assert (np.diff(indices.to_numpy()) == 1).all()
        assert indices.max() < len(cohorts)  # a window that wrapped would restart at 0
    assert (windows.size() <= window).all()
    # A window instance is unique per sampled occurrence, so a repeated start
    # index still gets its own instance id.
    assert path["window_instance"].nunique() == windows.ngroups


def test_bootstrap_path_is_reproducible_from_its_seed(cohorts):
    kwargs = dict(cohorts=cohorts, horizon=64, window_blocks=32)
    same = bootstrap_path(**kwargs, rng=np.random.default_rng(7))
    again = bootstrap_path(**kwargs, rng=np.random.default_rng(7))
    other = bootstrap_path(**kwargs, rng=np.random.default_rng(8))

    pd.testing.assert_frame_equal(same, again)
    assert not same["cohort_index"].equals(other["cohort_index"])


def test_bootstrap_window_cannot_exceed_the_source_trace(cohorts):
    with pytest.raises(ValueError, match="exceeds"):
        bootstrap_path(cohorts, 10, len(cohorts) + 1, np.random.default_rng(0))


# --- Sampling pools ---------------------------------------------------------------


def test_bootstrap_pool_is_exactly_the_step_s_own_window(cohorts):
    window = 16
    path = bootstrap_path(cohorts, 64, window, np.random.default_rng(2))
    low, high = demand_pool_bounds(cohorts, path, window, "moving_block_bootstrap")

    for position in range(len(path)):
        instance = path.loc[position, "window_instance"]
        members = path.loc[path["window_instance"] == instance, "cohort_index"]
        assert low[position] == cohorts.offsets[members.min()]
        assert high[position] == cohorts.offsets[members.max() + 1]
        # Every step in a window shares one pool, and the arriving cohort is in it.
        assert low[position] <= cohorts.offsets[path.loc[position, "cohort_index"]]


def test_historical_pool_is_a_trailing_window(cohorts):
    window = 8
    path = historical_path(cohorts, 30)
    low, high = demand_pool_bounds(cohorts, path, window, "historical")

    assert low[0] == cohorts.offsets[0]  # clamped at the start of the trace
    assert high[0] == cohorts.offsets[1]
    assert low[20] == cohorts.offsets[13]  # 20 - 8 + 1
    assert high[20] == cohorts.offsets[21]


# --- Arrival sampling -------------------------------------------------------------


def test_unit_multiplier_is_exactly_the_source_cohort(cohorts):
    cohort = cohorts.cohort(4)
    arrivals = sample_arrivals(cohorts, 4, (0, len(cohorts)), 1.0, np.random.default_rng(0))

    assert (arrivals["replica_index"] == 0).all()
    assert (arrivals["execution_gas"] == cohort["execution_gas"]).all()
    assert (arrivals["tx_index"] == cohort["tx_index"]).all()


def test_integer_multiplier_replicates_every_transaction_exactly(cohorts):
    cohort = cohorts.cohort(0)
    size = cohort["tx_index"].size
    arrivals = sample_arrivals(cohorts, 0, (0, 10), 3.0, np.random.default_rng(0))

    assert arrivals["tx_index"].size == 3 * size
    assert arrivals["replica_index"].tolist() == np.tile([0, 1, 2], size).tolist()
    assert (arrivals["max_fee_per_gas"] == np.repeat(cohort["max_fee_per_gas"], 3)).all()
    assert (arrivals["tx_index"] == np.repeat(cohort["tx_index"], 3)).all()


def arrived_gas(cohorts, cohort_index, pool, multiplier, rng, draws):
    """Total S + B gas arriving over repeated draws of one step."""
    totals = []
    for _ in range(draws):
        arrivals = sample_arrivals(cohorts, cohort_index, pool, multiplier, rng)
        totals.append(int(arrivals["execution_gas"].sum() + arrivals["state_gas"].sum()))
    return totals


def test_fractional_multiplier_hits_its_gas_target_in_expectation(cohorts):
    pool = (0, int(cohorts.offsets[20]))
    target = 1.5 * cohorts.cohort_total_gas(0)
    rng = np.random.default_rng(11)

    sampled = arrived_gas(cohorts, 0, pool, 1.5, rng, draws=200)

    # Gas-based, so it is the *gas* that lands on target; the transaction count
    # is not fixed and varies draw to draw.
    assert np.mean(sampled) == pytest.approx(target, rel=0.02)
    # The crossing draw is always kept, so no sample undershoots by more than
    # the single largest transaction the pool could have contributed.
    assert min(sampled) >= target - int(cohorts.total_gas[pool[0] : pool[1]].max())


def test_sub_unit_multiplier_thins_the_cohort_by_gas(cohorts):
    pool = (0, int(cohorts.offsets[20]))
    target = 0.5 * cohorts.cohort_total_gas(0)
    rng = np.random.default_rng(2)

    sampled = arrived_gas(cohorts, 0, pool, 0.5, rng, draws=200)

    assert np.mean(sampled) == pytest.approx(target, rel=0.03)
    assert min(sampled) > 0


def test_the_sampled_remainder_is_drawn_from_the_whole_window_not_just_the_cohort(cohorts):
    pool = (int(cohorts.offsets[0]), int(cohorts.offsets[12]))
    rng = np.random.default_rng(4)

    arrivals = sample_arrivals(cohorts, 3, pool, 1.5, rng)
    blocks = set(arrivals["source_block_number"].tolist())

    assert len(blocks) > 1
    assert blocks <= set(cohorts.block_numbers[:12].tolist())
    # The whole copy still comes from the arriving cohort alone.
    assert cohorts.block_numbers[3] in blocks


def test_arrivals_are_emitted_in_the_inclusion_tiebreak_order(cohorts):
    """`sim.engine`'s append-only mempool depends on this ordering."""
    rng = np.random.default_rng(6)
    for multiplier in (0.4, 1.0, 2.7):
        arrivals = sample_arrivals(
            cohorts, 5, (0, int(cohorts.offsets[15])), multiplier, rng
        )
        key = np.stack(
            [
                arrivals["source_block_number"],
                arrivals["tx_index"],
                arrivals["replica_index"],
            ]
        )
        assert (np.diff(np.lexsort(key[::-1])) == 1).all()
        # replica_index restarts at 0 for each distinct source transaction.
        _, first = np.unique(
            np.stack([arrivals["source_block_number"], arrivals["tx_index"]]),
            axis=1,
            return_index=True,
        )
        assert (arrivals["replica_index"][np.sort(first)] == 0).all()


def test_sampling_is_reproducible_from_its_generator(cohorts):
    pool = (0, int(cohorts.offsets[10]))
    same = sample_arrivals(cohorts, 1, pool, 2.3, np.random.default_rng(9))
    again = sample_arrivals(cohorts, 1, pool, 2.3, np.random.default_rng(9))
    other = sample_arrivals(cohorts, 1, pool, 2.3, np.random.default_rng(10))

    assert (same["tx_index"] == again["tx_index"]).all()
    assert same["tx_index"].size != other["tx_index"].size or not (
        same["tx_index"] == other["tx_index"]
    ).all()


# --- Bid adaptation ---------------------------------------------------------------


def bids(**overrides) -> dict[str, np.ndarray]:
    row = {
        "max_fee_per_gas": [100 * GWEI],
        "max_priority_fee_per_gas": [2 * GWEI],
        "anchor_base_fee": [10 * GWEI],
        "is_legacy": [False],
    }
    return {name: np.array(overrides.get(name, value)) for name, value in row.items()}


def test_dynamic_fee_cap_scales_with_the_base_fee_and_the_tip_does_not():
    adapted = adapt_bids(bids(), base_fee=30 * GWEI)

    assert adapted["max_fee_per_gas"].tolist() == [300 * GWEI]
    assert adapted["max_priority_fee_per_gas"].tolist() == [2 * GWEI]


def test_scaling_the_cap_is_the_same_as_scaling_its_headroom():
    """The design note writes this as headroom scaling; the two are identical."""
    max_fee, anchor, base_fee = 137 * GWEI, 11 * GWEI, 400 * GWEI
    adapted = adapt_bids(
        bids(max_fee_per_gas=[max_fee], anchor_base_fee=[anchor]), base_fee=base_fee
    )
    headroom_form = (max_fee - anchor) * (base_fee / anchor) + base_fee
    assert adapted["max_fee_per_gas"][0] == pytest.approx(headroom_form, rel=1e-12)


def test_legacy_gas_price_shifts_so_its_tip_stays_absolute():
    """A legacy row's whole headroom *is* its tip, so scaling would scale the tip."""
    adapted = adapt_bids(
        bids(is_legacy=[True], max_fee_per_gas=[12 * GWEI]), base_fee=30 * GWEI
    )

    # Historical tip was 12 - 10 = 2 gwei; it must still be 2 gwei at 30 gwei.
    assert adapted["max_fee_per_gas"].tolist() == [32 * GWEI]


@pytest.mark.parametrize("is_legacy", [False, True])
@pytest.mark.parametrize("base_fee", [1, GWEI // 100, GWEI, 500 * GWEI])
def test_adaptation_preserves_fee_eligibility_in_both_directions(is_legacy, base_fee):
    """Quantity is the demand model's job, so the fee filter must not change it."""
    adapted = adapt_bids(
        bids(is_legacy=[is_legacy], max_fee_per_gas=[10 * GWEI]), base_fee=base_fee
    )
    assert adapted["max_fee_per_gas"][0] >= base_fee


def test_a_zero_anchor_base_fee_leaves_the_bid_untouched():
    adapted = adapt_bids(bids(anchor_base_fee=[0]), base_fee=30 * GWEI)
    assert adapted["max_fee_per_gas"].tolist() == [100 * GWEI]


def test_adaptation_saturates_instead_of_overflowing_int64():
    """A 1-wei anchor against a 500-gwei base fee makes the product ~1e23.

    Regression: clipping the float64 product at `int64_max` is not enough, since
    `float64(int64_max)` rounds up to 2**63 and the cast is still invalid --
    which numpy reports as a warning and a garbage (negative) value, not an
    error. Hence `simplefilter("error")` here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapted = adapt_bids(
            bids(anchor_base_fee=[1], max_fee_per_gas=[500 * GWEI]), base_fee=500 * GWEI
        )

    assert adapted["max_fee_per_gas"].dtype == np.int64
    capped = int(adapted["max_fee_per_gas"][0])
    assert 2**62 < capped <= np.iinfo(np.int64).max


def test_adaptation_leaves_every_other_field_alone(cohorts):
    priced = anchored_cohorts(num_blocks=20, seed=8)
    arrivals = sample_arrivals(priced, 2, (0, int(priced.offsets[5])), 1.0, np.random.default_rng(0))
    adapted = adapt_bids(arrivals, base_fee=42 * GWEI)

    assert set(adapted) == set(arrivals)
    for name in ("execution_gas", "state_gas", "tx_index", "schedule_gas_used", "is_legacy"):
        assert (adapted[name] == arrivals[name]).all()
