"""Arrival-path construction, demand anchors, sampling, and bid adaptation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_CONFIG, Scenario
from tests.dummy import dummy_block_headers, dummy_tx_gas_results
from sim.metrics import effective_tip
from sim.workload import (
    PATH_COLUMNS,
    adapt_bids,
    build_cohorts,
    composition_pools,
    demand_multiplier,
    demand_pool_bounds,
    sample_arrivals,
)

GWEI = 1_000_000_000


def scenario(**overrides) -> Scenario:
    return Scenario(
        **{
            "aggregate_elasticity": 0.175,
            "demand_level": 1.0,
            "composition_pool_blocks": 16,
            **overrides,
        }
    )


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
    return anchored_cohorts()


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
        build_cohorts(raw, dummy_block_headers(raw))

    kept = build_cohorts(simulatable(raw), dummy_block_headers(raw)).tx_counts.sum()
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


def test_the_demand_reference_is_gas_weighted_over_transactions_not_cohorts():
    """Two blocks of very different size: the big one must dominate the price."""
    frame = pd.DataFrame(
        {
            "block_number": [100, 200],
            "tx_index": [0, 0],
            "tx_type": [2, 2],
            "max_fee_per_gas": [100 * GWEI, 100 * GWEI],
            "max_priority_fee_per_gas": [GWEI, GWEI],
            "execution_gas": [1_000_000, 9_000_000],
            "state_gas": [0, 0],
            "schedule_gas_used": [1_000_000, 9_000_000],
        }
    )
    headers = pd.DataFrame(
        {"block_number": [100, 200], "base_fee_per_gas": [10 * GWEI, 20 * GWEI]}
    )

    reference = build_cohorts(frame, headers).reference

    # Base fees of 10 and 20 gwei weighted 1M:9M give 19 gwei, plus a flat 1 gwei
    # tip. An unweighted cohort mean would have said 15 + 1 = 16 gwei.
    assert reference.price == pytest.approx(20 * GWEI)
    assert reference.tip == pytest.approx(GWEI)
    # Mean cohort S + B gas across the two blocks.
    assert reference.gas == pytest.approx(5_000_000)


def test_the_reference_price_is_the_gas_weighted_mean_of_the_cohort_anchors(
    priced_cohorts,
):
    """`DemandReference.price` must stay on the `priority_fees_wei / gas` basis."""
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
    weight = columns["schedule_gas_used"].astype(np.float64)
    expected = ((columns["anchor_base_fee"] + tip) * weight).sum() / weight.sum()

    assert priced_cohorts.reference.price == pytest.approx(expected)


def test_a_trace_with_no_sender_gas_has_no_observable_reference_price():
    frame = pd.DataFrame(
        {
            "block_number": [100],
            "tx_index": [0],
            "tx_type": [2],
            "max_fee_per_gas": [100 * GWEI],
            "max_priority_fee_per_gas": [GWEI],
            "execution_gas": [0],
            "state_gas": [0],
            "schedule_gas_used": [0],
        }
    )
    headers = pd.DataFrame({"block_number": [100], "base_fee_per_gas": [10 * GWEI]})

    reference = build_cohorts(frame, headers).reference

    # No weight to average over, so there is no price to anchor on;
    # `demand_multiplier` reads a 0 anchor as "flat at the demand level".
    assert (reference.price, reference.tip, reference.gas) == (0.0, 0.0, 0.0)
    assert demand_multiplier(DEFAULT_CONFIG, scenario(demand_level=2.0), GWEI, 0.0) == (
        2.0,
        False,
    )


def test_missing_headers_for_a_cohort_block_are_refused():
    raw = dummy_tx_gas_results(num_blocks=10, seed=5)
    headers = dummy_block_headers(raw).iloc[2:]
    with pytest.raises(ValueError, match="missing 2 of 10 cohort blocks"):
        build_cohorts(simulatable(raw), headers)


# --- The demand multiplier --------------------------------------------------------


def test_multiplier_is_the_demand_level_at_the_anchor_price():
    multiplier, clamped = demand_multiplier(
        DEFAULT_CONFIG, scenario(demand_level=2.0), 8 * GWEI, 8 * GWEI
    )
    assert multiplier == pytest.approx(2.0)
    assert not clamped


def test_demand_rises_as_the_price_falls_and_falls_as_it_rises():
    cfg, cell = DEFAULT_CONFIG, scenario()
    anchor = 10 * GWEI

    cheaper, _ = demand_multiplier(cfg, cell, anchor / 4, anchor)
    dearer, _ = demand_multiplier(cfg, cell, anchor * 4, anchor)

    assert cheaper == pytest.approx(4.0**0.175)
    assert dearer == pytest.approx(0.25**0.175)
    assert dearer < 1.0 < cheaper


def test_zero_elasticity_is_a_flat_multiplier_whatever_the_price():
    cell = scenario(aggregate_elasticity=0.0, demand_level=3.0)
    for price in (1, GWEI, 1_000 * GWEI):
        assert demand_multiplier(DEFAULT_CONFIG, cell, price, 8 * GWEI) == (3.0, False)


def test_a_more_elastic_demand_responds_more_to_the_same_price_fall():
    anchor = 10 * GWEI
    responses = [
        demand_multiplier(
            DEFAULT_CONFIG.with_(demand_multiplier_bounds=(0.001, 1e9)),
            scenario(aggregate_elasticity=e),
            anchor / 10,
            anchor,
        )[0]
        for e in (0.0, 0.10, 0.175, 0.28)
    ]
    assert responses == sorted(responses)
    assert responses[0] == 1.0


def test_the_clamp_bounds_an_extrapolated_multiplier_and_says_so():
    cfg = DEFAULT_CONFIG.with_(demand_multiplier_bounds=(0.5, 2.0))
    cell = scenario()
    anchor = GWEI

    # A base fee at the 1-wei floor is a 1e9 price fall: (1e9 ** 0.175) ~ 38x.
    high, clamped_high = demand_multiplier(cfg, cell, 1, anchor)
    assert (high, clamped_high) == (2.0, True)

    low, clamped_low = demand_multiplier(cfg, cell, anchor * 10**9, anchor)
    assert (low, clamped_low) == (0.5, True)

    inside, clamped = demand_multiplier(cfg, cell, anchor, anchor)
    assert (inside, clamped) == (1.0, False)


def test_the_clamp_bounds_the_price_response_not_the_chosen_demand_level():
    """A level above the bound is a deliberate assumption, not an extrapolation."""
    cfg = DEFAULT_CONFIG.with_(demand_multiplier_bounds=(0.5, 2.0))
    cell = scenario(demand_level=30.0)
    anchor = GWEI

    at_anchor, clamped = demand_multiplier(cfg, cell, anchor, anchor)
    assert (at_anchor, clamped) == (30.0, False)

    # The response saturates at 2x, so the product is 60x, not capped at 2x.
    extrapolated, clamped = demand_multiplier(cfg, cell, 1, anchor)
    assert (extrapolated, clamped) == (60.0, True)


def test_a_non_positive_price_or_anchor_falls_back_to_the_level():
    cell = scenario(demand_level=1.5)
    assert demand_multiplier(DEFAULT_CONFIG, cell, 0, 8 * GWEI) == (1.5, False)
    assert demand_multiplier(DEFAULT_CONFIG, cell, 8 * GWEI, 0) == (1.5, False)


# --- Arrival paths ----------------------------------------------------------------


def test_every_step_draws_its_own_pool_and_none_of_them_wrap(cohorts):
    horizon, pool_blocks = 100, 16
    path = composition_pools(cohorts, horizon, pool_blocks, np.random.default_rng(1))

    assert tuple(path.columns) == PATH_COLUMNS
    assert len(path) == horizon
    assert path["simulation_position"].tolist() == list(range(horizon))
    # One start per step, and every pool fits inside the trace without wrapping.
    starts = path["pool_start_index"].to_numpy()
    assert starts.min() >= 0
    assert starts.max() + pool_blocks <= len(cohorts)
    assert (path["pool_start_block"].to_numpy() == cohorts.block_numbers[starts]).all()


def test_pool_starts_are_independent_across_steps(cohorts):
    """Not a moving-block bootstrap: no step's pool follows on from the last."""
    path = composition_pools(cohorts, 400, 16, np.random.default_rng(5))
    starts = path["pool_start_index"].to_numpy()

    # Consecutive draws are unrelated, so steps almost never continue each other
    # and the draws spread over the whole legal range.
    assert (np.diff(starts) == 16).mean() < 0.05
    assert starts.max() - starts.min() > 0.5 * (len(cohorts) - 16)
    assert np.unique(starts).size > 20


def test_composition_pools_are_reproducible_from_the_seed(cohorts):
    kwargs = dict(cohorts=cohorts, horizon=64, pool_blocks=32)
    same = composition_pools(**kwargs, rng=np.random.default_rng(7))
    again = composition_pools(**kwargs, rng=np.random.default_rng(7))
    other = composition_pools(**kwargs, rng=np.random.default_rng(8))

    pd.testing.assert_frame_equal(same, again)
    assert not same["pool_start_index"].equals(other["pool_start_index"])


def test_a_shorter_horizon_is_a_prefix_of_a_longer_one(cohorts):
    """One draw per step, so halving the horizon just stops the same run early."""
    long = composition_pools(cohorts, 80, 16, np.random.default_rng(3))
    short = composition_pools(cohorts, 40, 16, np.random.default_rng(3))

    assert short["pool_start_index"].tolist() == long["pool_start_index"].tolist()[:40]


def test_a_pool_cannot_exceed_the_source_trace(cohorts):
    with pytest.raises(ValueError, match="exceeds"):
        composition_pools(cohorts, 10, len(cohorts) + 1, np.random.default_rng(0))


# --- Sampling pools ---------------------------------------------------------------


def test_pool_bounds_are_the_contiguous_cohort_run_from_the_step_s_start(cohorts):
    pool_blocks = 16
    path = composition_pools(cohorts, 64, pool_blocks, np.random.default_rng(2))
    low, high = demand_pool_bounds(cohorts, path, pool_blocks)

    for position in range(len(path)):
        start = int(path.loc[position, "pool_start_index"])
        assert low[position] == cohorts.offsets[start]
        assert high[position] == cohorts.offsets[start + pool_blocks]


# --- Arrival sampling -------------------------------------------------------------


def arrived_gas(cohorts, pool, gas_target, rng, draws):
    """Total S + B gas arriving over repeated draws of one step."""
    totals = []
    for _ in range(draws):
        arrivals = sample_arrivals(cohorts, pool, gas_target, rng)
        totals.append(int(arrivals["execution_gas"].sum() + arrivals["state_gas"].sum()))
    return totals


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.5])
def test_arrivals_hit_their_gas_target_in_expectation(cohorts, scale):
    """The demand model sets the quantity, so the *gas* is what lands on target."""
    pool = (0, int(cohorts.offsets[20]))
    target = scale * cohorts.reference.gas
    rng = np.random.default_rng(11)

    sampled = arrived_gas(cohorts, pool, target, rng, draws=200)
    largest = int(cohorts.total_gas[pool[0] : pool[1]].max())

    # The transaction count is not fixed and varies draw to draw.
    assert np.mean(sampled) == pytest.approx(target, rel=0.05)
    # Keeping the crossing draw biases arrivals *up* by part of one transaction,
    # so the mean sits just above target -- proportionally more so for a small
    # target, which is why the tolerance above is not tighter.
    assert 0 <= np.mean(sampled) - target < largest
    # And no individual sample undershoots by more than that same one draw.
    assert min(sampled) >= target - largest


def test_a_non_positive_gas_target_arrives_nothing(cohorts):
    arrivals = sample_arrivals(cohorts, (0, 100), 0.0, np.random.default_rng(1))

    assert arrivals["tx_index"].size == 0
    assert arrivals["replica_index"].size == 0


def test_arrivals_are_drawn_from_the_whole_pool(cohorts):
    """Composition comes from the pool, never from one privileged cohort."""
    pool = (int(cohorts.offsets[0]), int(cohorts.offsets[12]))
    rng = np.random.default_rng(4)

    arrivals = sample_arrivals(cohorts, pool, 3 * cohorts.reference.gas, rng)
    blocks = set(arrivals["source_block_number"].tolist())

    assert len(blocks) > 1
    assert blocks <= set(cohorts.block_numbers[:12].tolist())


def test_arrivals_never_come_from_outside_the_pool(cohorts):
    pool = (int(cohorts.offsets[30]), int(cohorts.offsets[34]))
    arrivals = sample_arrivals(
        cohorts, pool, 5 * cohorts.reference.gas, np.random.default_rng(12)
    )

    assert set(arrivals["source_block_number"].tolist()) <= set(
        cohorts.block_numbers[30:34].tolist()
    )


def test_arrivals_are_emitted_in_the_inclusion_tiebreak_order(cohorts):
    """`sim.engine`'s append-only mempool depends on this ordering."""
    rng = np.random.default_rng(6)
    for scale in (0.4, 1.0, 2.7):
        arrivals = sample_arrivals(
            cohorts, (0, int(cohorts.offsets[15])), scale * cohorts.reference.gas, rng
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
    target = 2.3 * cohorts.reference.gas
    same = sample_arrivals(cohorts, pool, target, np.random.default_rng(9))
    again = sample_arrivals(cohorts, pool, target, np.random.default_rng(9))
    other = sample_arrivals(cohorts, pool, target, np.random.default_rng(10))

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
    arrivals = sample_arrivals(
        priced, (0, int(priced.offsets[5])), priced.reference.gas, np.random.default_rng(0)
    )
    adapted = adapt_bids(arrivals, base_fee=42 * GWEI)

    assert set(adapted) == set(arrivals)
    for name in ("execution_gas", "state_gas", "tx_index", "schedule_gas_used", "is_legacy"):
        assert (adapted[name] == arrivals[name]).all()
