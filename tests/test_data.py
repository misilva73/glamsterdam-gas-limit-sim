"""Data-layer tests, driven entirely by `tests.dummy`.

The query paths are covered by asserting on the SQL text produced by the pure
builders; everything behavioural runs against synthetic frames, injected at the
two network seams by the `offline_data` fixture, so the suite needs no
credentials and no network. The live table's real column types are recorded in
`AGENTS.md`.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from config import SimConfig
from data import cache, fetch_blocks
from data import load_tx_gas_results as loader
from schemas import (
    BLOCK_HEADER_COLUMNS,
    TX_GAS_RESULT_COLUMNS,
    TX_GAS_RESULT_FIELDS,
    TX_GAS_RESULT_PROVENANCE_FIELDS,
)
from tests import dummy
from tests.conftest import FIRST_BLOCK, LAST_BLOCK, NUM_BLOCKS


@pytest.fixture
def clickhouse_cfg(tmp_path) -> SimConfig:
    return SimConfig(
        analysis_config_hash="analysis-cfg-abc",
        schedule_config_hash="schedule-cfg-def",
        source_block_range=(FIRST_BLOCK, FIRST_BLOCK + 50_399),
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    return dummy.dummy_tx_gas_results(first_block=FIRST_BLOCK, num_blocks=NUM_BLOCKS)


# --- Gas dimension derivation -----------------------------------------------------


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_derive_gas_dimensions_splits_total_into_state_and_execution():
    frame = _frame(
        schedule_total_gas_spent=[100_000, 50_000],
        schedule_state_gas_spent=[40_000, 0],
        schedule_floor_gas=[1_000, 1_000],
    )
    derived = loader.derive_gas_dimensions(frame)
    assert derived["state_gas"].tolist() == [40_000, 0]
    assert derived["execution_gas"].tolist() == [60_000, 50_000]
    assert derived["execution_gas"].dtype == np.dtype("int64")


def test_calldata_floor_can_push_execution_gas_above_the_remainder():
    # The floor is a lower bound on charged gas, so it can exceed
    # total - state and then binds the execution dimension on its own.
    frame = _frame(
        schedule_total_gas_spent=[100_000],
        schedule_state_gas_spent=[90_000],
        schedule_floor_gas=[75_000],
    )
    derived = loader.derive_gas_dimensions(frame)
    remainder = 100_000 - 90_000
    assert derived["execution_gas"].iloc[0] == 75_000 > remainder


def test_derived_dimensions_hold_on_the_dummy_trace(raw_rows):
    derived = loader.derive_gas_dimensions(loader.normalize_dtypes(raw_rows))
    remainder = derived["schedule_total_gas_spent"] - derived["schedule_state_gas_spent"]
    assert (derived["execution_gas"] >= remainder).all()
    assert (derived["execution_gas"] >= derived["schedule_floor_gas"]).all()
    assert (derived["state_gas"] == derived["schedule_state_gas_spent"]).all()
    assert (derived["execution_gas"] > remainder).any(), "floor never binds in dummy trace"


def test_block_gas_used_is_the_binding_dimension():
    selected = _frame(execution_gas=[10, 20], state_gas=[100, 5])
    assert loader.block_gas_used(selected) == 105
    assert loader.block_gas_used(selected.iloc[:0]) == 0


# --- Simulatable / excluded split -------------------------------------------------


def test_split_simulatable_keeps_only_doubly_successful_rows():
    frame = _frame(
        baseline_success=[1, 1, 0, 0],
        schedule_success=[1, 0, 1, 0],
        execution_gas=[10, 20, 30, 40],
        state_gas=[1, 2, 3, 4],
        schedule_gas_used=[5, 6, 7, 8],
    )
    selected, excluded = loader.split_simulatable(frame, policy="successful_only")
    assert selected["execution_gas"].tolist() == [10]
    # With no `min_multiplier_to_succeed` column there is no rescue evidence, so a
    # schedule failure cannot be attributed to a too-small gas limit.
    assert excluded["exclusion_reason"].tolist() == [
        "schedule_halted_regardless",
        "baseline_only_failure",
        "both_failed",
    ]


def test_excluded_summary_reports_counts_and_gas_shares():
    frame = _frame(
        baseline_success=[1, 1, 0, 0],
        schedule_success=[1, 0, 1, 0],
        execution_gas=[10, 20, 30, 50],
        state_gas=[1, 2, 3, 5],
        schedule_gas_used=[5, 6, 7, 9],
    )
    _, excluded = loader.split_simulatable(frame, policy="successful_only")
    summary = loader.excluded_summary(excluded)

    assert list(summary.index) == list(loader.EXCLUSION_REASONS)
    assert summary["tx_count"].sum() == 3
    assert summary.loc["both_failed", "execution_gas"] == 50
    assert summary["execution_gas_share"].sum() == pytest.approx(1.0)
    assert summary.loc["both_failed", "execution_gas_share"] == pytest.approx(50 / 100)


def test_excluded_summary_on_an_empty_frame_is_empty_not_an_error():
    summary = loader.excluded_summary(pd.DataFrame(columns=["exclusion_reason"]))
    assert summary.empty
    assert "execution_gas_share" in summary.columns


def test_dummy_trace_has_both_simulatable_and_excluded_rows(raw_rows):
    selected, excluded = loader.split_simulatable(
        loader.derive_gas_dimensions(loader.normalize_dtypes(raw_rows)),
        policy="successful_only",
    )
    assert len(selected) > 0 and len(excluded) > 0
    assert len(selected) + len(excluded) == len(raw_rows)
    assert loader.excluded_summary(excluded)["tx_count"].sum() == len(excluded)


# --- Dataset identity guard -------------------------------------------------------


def test_mixed_analysis_config_hash_is_rejected(raw_rows):
    mixed = raw_rows.copy()
    mixed.loc[mixed.index[:5], "analysis_config_hash"] = "other-analysis-cfg"
    with pytest.raises(ValueError, match="analysis_config_hash must be pinned"):
        loader.assert_single_dataset(mixed)


def test_mixed_schedule_config_hash_names_both_values(raw_rows):
    mixed = raw_rows.copy()
    mixed.loc[mixed.index[-3:], "schedule_config_hash"] = "other-schedule-cfg"
    with pytest.raises(ValueError) as excinfo:
        loader.assert_single_dataset(mixed)
    assert "other-schedule-cfg" in str(excinfo.value)
    assert dummy.DUMMY_SCHEDULE_CONFIG_HASH in str(excinfo.value)


def test_mixed_chain_id_is_rejected(raw_rows):
    mixed = raw_rows.copy()
    mixed.loc[mixed.index[:1], "chain_id"] = 11155111
    with pytest.raises(ValueError, match="chain_id must be pinned"):
        loader.assert_single_dataset(mixed)


def test_config_pinned_hash_must_match_the_data(raw_rows, offline_cfg):
    cfg = offline_cfg.with_(schedule_config_hash="not-the-dummy-hash")
    with pytest.raises(ValueError, match="schedule_config_hash mismatch"):
        loader.assert_single_dataset(raw_rows, cfg)


def test_matching_pinned_hashes_pass(raw_rows, offline_cfg):
    cfg = offline_cfg.with_(
        analysis_config_hash=dummy.DUMMY_ANALYSIS_CONFIG_HASH,
        schedule_config_hash=dummy.DUMMY_SCHEDULE_CONFIG_HASH,
    )
    assert loader.assert_single_dataset(raw_rows, cfg) is raw_rows


# --- Fee / integer normalization --------------------------------------------------


def test_decimal_strings_normalize_to_int64_wei():
    normalized = loader.as_int64(["1000000000", " 25 ", "0"], "max_fee_per_gas")
    assert normalized.tolist() == [1_000_000_000, 25, 0]
    assert normalized.dtype == np.dtype("int64")


def test_python_int_objects_from_uint256_normalize():
    values = pd.Series([2**62, 7], dtype=object)
    assert loader.as_int64(values, "max_fee_per_gas").tolist() == [2**62, 7]


def test_uint256_overflow_is_rejected_not_wrapped():
    with pytest.raises(OverflowError, match="max_fee_per_gas"):
        loader.as_int64(pd.Series([1, 2**200], dtype=object), "max_fee_per_gas")


def test_uint64_overflow_is_rejected():
    with pytest.raises(OverflowError, match="exceeds int64"):
        loader.as_int64(pd.Series([2**63], dtype="uint64"), "max_fee_per_gas")


def test_null_fee_is_rejected():
    with pytest.raises(ValueError, match="contains nulls"):
        loader.as_int64(pd.Series([1, None], dtype=object), "max_priority_fee_per_gas")


def test_non_integral_value_is_rejected():
    with pytest.raises(ValueError, match="non-integral"):
        loader.as_int64(pd.Series([1.5], dtype=object), "max_fee_per_gas")


def test_normalize_dtypes_coerces_string_fees_and_adds_absent_provenance(raw_rows):
    stringly = raw_rows.drop(columns=["producer_git_commit", "replay_semantics"]).assign(
        max_fee_per_gas=lambda f: f["max_fee_per_gas"].astype(str),
        max_priority_fee_per_gas=lambda f: f["max_priority_fee_per_gas"].astype(str),
    )
    normalized = loader.normalize_dtypes(stringly)
    assert normalized["max_fee_per_gas"].dtype == np.dtype("int64")
    assert normalized["max_fee_per_gas"].tolist() == raw_rows["max_fee_per_gas"].tolist()
    assert normalized["producer_git_commit"].isna().all()


# --- End-to-end offline load + cache ----------------------------------------------


def test_load_tx_gas_results_returns_the_declared_contract(offline_cfg, offline_data):
    frame = loader.load_tx_gas_results(offline_cfg)
    assert list(frame.columns)[: len(TX_GAS_RESULT_COLUMNS)] == list(TX_GAS_RESULT_COLUMNS)
    assert frame["block_number"].between(FIRST_BLOCK, LAST_BLOCK).all()
    assert frame[["execution_gas", "state_gas", "tx_gas_limit"]].dtypes.eq("int64").all()
    assert frame[["block_number", "tx_index"]].apply(tuple, axis=1).is_monotonic_increasing


def test_cache_round_trip_writes_a_sidecar_and_skips_the_second_fetch(
    offline_cfg, offline_data, monkeypatch
):
    calls = []
    real_fetch = loader.fetch_tx_gas_results

    def counting_fetch(cfg):
        calls.append(cfg)
        return real_fetch(cfg)

    monkeypatch.setattr(loader, "fetch_tx_gas_results", counting_fetch)

    first = loader.load_tx_gas_results(offline_cfg)
    path = loader.tx_gas_results_cache_path(offline_cfg)
    assert path.exists()

    second = loader.load_tx_gas_results(offline_cfg)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)

    provenance = json.loads(cache.sidecar_path(path).read_text())
    assert provenance["analysis_config_hash"] == dummy.DUMMY_ANALYSIS_CONFIG_HASH
    assert provenance["schedule_config_hash"] == dummy.DUMMY_SCHEDULE_CONFIG_HASH
    assert provenance["chain_id"] == 1
    assert provenance["observed_block_range"] == [FIRST_BLOCK, LAST_BLOCK]
    assert provenance["row_count"] == len(first)
    assert provenance["producer_git_commit"] == [dummy.DUMMY_GIT_COMMIT]
    assert provenance["replay_semantics"] == [dummy.DUMMY_REPLAY_SEMANTICS]
    assert provenance["block_hash_coverage"]["present"] is True
    assert provenance["block_timestamp_coverage"]["min"] < (
        provenance["block_timestamp_coverage"]["max"]
    )
    assert dt.datetime.fromisoformat(provenance["fetched_at"]).tzinfo is not None


def test_cache_key_separates_block_ranges_and_hashes(offline_cfg):
    other_range = loader.tx_gas_results_cache_path(
        offline_cfg.with_(source_block_range=(FIRST_BLOCK, LAST_BLOCK + 1))
    )
    other_hash = loader.tx_gas_results_cache_path(
        offline_cfg.with_(analysis_config_hash="something-else")
    )
    base = loader.tx_gas_results_cache_path(offline_cfg)
    assert len({base, other_range, other_hash}) == 3


def test_provenance_tolerates_a_source_without_provenance_columns(offline_cfg, raw_rows):
    bare = raw_rows.drop(columns=list(TX_GAS_RESULT_PROVENANCE_FIELDS))
    frame = loader.prepare_tx_gas_results(bare, offline_cfg)
    provenance = loader.tx_gas_results_provenance(bare, frame, offline_cfg)
    assert provenance["block_hash_coverage"] == {"present": False}
    assert provenance["producer_git_commit"] is None


def test_a_fetch_without_a_block_range_is_refused(tmp_path):
    cfg = SimConfig(analysis_config_hash="abc", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="source_block_range is required"):
        loader.fetch_tx_gas_results(cfg)


# --- ClickHouse SQL construction --------------------------------------------------


def test_clickhouse_sql_pins_dataset_network_and_block_range(clickhouse_cfg):
    sql = loader.clickhouse_tx_gas_results_sql(clickhouse_cfg, 100, 199)
    # Database-qualified: the client connects to `default`, the table lives in
    # `gas_analysis`. Verified against the live cluster.
    assert "FROM gas_analysis.gas_analysis_tx_gas_result FINAL" in sql
    assert "analysis_config_hash = 'analysis-cfg-abc'" in sql
    assert "schedule_config_hash = 'schedule-cfg-def'" in sql
    assert "chain_id = 1" in sql
    assert f"schedule_name = '{clickhouse_cfg.schedule_name}'" in sql
    assert "block_number >= 100" in sql and "block_number <= 199" in sql
    assert sql.rstrip().endswith("ORDER BY block_number, tx_index")


def test_clickhouse_sql_selects_every_contract_field(clickhouse_cfg):
    sql = loader.clickhouse_tx_gas_results_sql(clickhouse_cfg, 100, 199)
    for field in TX_GAS_RESULT_FIELDS + TX_GAS_RESULT_PROVENANCE_FIELDS:
        assert field in sql


def test_clickhouse_sql_requires_a_pinned_analysis_hash():
    # `run_simulation` makes --analysis-config-hash mandatory; the builder is the
    # backstop for a config assembled in code, which can still leave it None.
    unpinned = SimConfig(analysis_config_hash=None)
    with pytest.raises(ValueError, match="analysis_config_hash must be pinned"):
        loader.clickhouse_tx_gas_results_sql(unpinned, 100, 199)


def test_clickhouse_sql_escapes_quotes_in_pinned_values(clickhouse_cfg):
    sql = loader.clickhouse_tx_gas_results_sql(
        clickhouse_cfg.with_(schedule_name="o'clock"), 1, 2
    )
    assert "schedule_name = 'o\\'clock'" in sql


def test_an_empty_query_result_still_yields_the_declared_contract(offline_cfg):
    frame = loader.prepare_tx_gas_results(loader._concat_chunks([]), offline_cfg)
    assert frame.empty
    assert list(frame.columns) == list(TX_GAS_RESULT_COLUMNS)
    assert frame["execution_gas"].dtype == np.dtype("int64")
    assert loader.block_gas_used(frame) == 0


def test_a_week_of_blocks_is_chunked(clickhouse_cfg):
    first, last = clickhouse_cfg.source_block_range
    chunks = list(loader.block_chunks(first, last))
    assert len(chunks) > 1
    assert chunks[0][0] == first and chunks[-1][1] == last
    assert all(hi - lo + 1 <= loader.BLOCK_CHUNK for lo, hi in chunks)
    # contiguous and non-overlapping
    assert all(nxt[0] == cur[1] + 1 for cur, nxt in zip(chunks, chunks[1:]))


# --- Xatu SQL construction --------------------------------------------------------


def test_beacon_range_sql_pins_network_and_a_bounded_slot_window(offline_cfg):
    sql = fetch_blocks.beacon_block_range_sql(
        offline_cfg,
        dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 8, tzinfo=dt.timezone.utc),
    )
    assert "FROM canonical_beacon_block FINAL" in sql
    assert "meta_network_name = 'mainnet'" in sql
    assert "slot_start_date_time >= toDateTime('2026-06-01 00:00:00', 'UTC')" in sql
    assert "slot_start_date_time < toDateTime('2026-06-08 00:00:00', 'UTC')" in sql
    assert "execution_payload_block_number" in sql


def test_header_sql_pins_network_and_block_range(offline_cfg):
    sql = fetch_blocks.execution_block_headers_sql(offline_cfg, 100, 199)
    assert "FROM canonical_execution_block FINAL" in sql
    assert "meta_network_name = 'mainnet'" in sql
    assert "block_number >= 100" in sql and "block_number <= 199" in sql
    for column in BLOCK_HEADER_COLUMNS:
        assert column in sql
    assert sql.rstrip().endswith("ORDER BY block_number")


def test_resolve_block_range_rejects_an_empty_window(offline_cfg):
    moment = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    with pytest.raises(ValueError, match="empty slot-time window"):
        fetch_blocks.resolve_block_range(offline_cfg, moment, moment)


# --- Headers ----------------------------------------------------------------------


def test_fetched_headers_are_normalized_to_the_declared_contract(
    offline_cfg, offline_data
):
    headers = fetch_blocks.fetch_block_headers(offline_cfg, FIRST_BLOCK, LAST_BLOCK)
    assert list(headers.columns) == list(BLOCK_HEADER_COLUMNS)
    assert headers["block_number"].is_monotonic_increasing
    assert headers.dtypes.eq("int64").all()
    assert (headers["gas_limit"] == offline_cfg.fusaka_gas_limit).all()


def test_header_cache_skips_the_second_fetch(offline_cfg, offline_data, monkeypatch):
    calls = []
    real_fetch = fetch_blocks.fetch_block_headers_uncached

    def counting_fetch(cfg, first_block, last_block):
        calls.append((first_block, last_block))
        return real_fetch(cfg, first_block, last_block)

    monkeypatch.setattr(fetch_blocks, "fetch_block_headers_uncached", counting_fetch)

    first = fetch_blocks.fetch_block_headers(offline_cfg, FIRST_BLOCK, LAST_BLOCK)
    second = fetch_blocks.fetch_block_headers(offline_cfg, FIRST_BLOCK, LAST_BLOCK)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)

    path = fetch_blocks.block_headers_cache_path(offline_cfg, FIRST_BLOCK, LAST_BLOCK)
    provenance = json.loads(cache.sidecar_path(path).read_text())
    assert provenance["network"] == "mainnet"
    assert provenance["requested_block_range"] == [FIRST_BLOCK, LAST_BLOCK]
    assert provenance["missing_block_count"] == 0


def test_starting_base_fee_reads_the_actual_parent():
    headers = pd.DataFrame(
        {
            "block_number": [99, 100, 101],
            "gas_used": [0, 0, 0],
            "gas_limit": [60_000_000] * 3,
            "base_fee_per_gas": [7_000_000_000, 8_000_000_000, 9_000_000_000],
        }
    )
    assert fetch_blocks.starting_base_fee(headers, 100) == 7_000_000_000


def test_starting_base_fee_raises_when_the_parent_is_absent():
    headers = pd.DataFrame(
        {
            "block_number": [100, 101],
            "gas_used": [0, 0],
            "gas_limit": [60_000_000] * 2,
            "base_fee_per_gas": [8_000_000_000, 9_000_000_000],
        }
    )
    with pytest.raises(ValueError, match="parent block 99"):
        fetch_blocks.starting_base_fee(headers, 100)


def test_starting_base_fee_on_the_dummy_header_frame(offline_cfg, offline_data):
    headers = fetch_blocks.fetch_block_headers(offline_cfg, FIRST_BLOCK, LAST_BLOCK)
    base_fee = fetch_blocks.starting_base_fee(headers, offline_cfg.reference_start_block)
    assert base_fee == int(
        headers.loc[headers["block_number"] == FIRST_BLOCK, "base_fee_per_gas"].iloc[0]
    )


# --- Gas-rescued inclusion policy -------------------------------------------------


def _rescue_frame() -> pd.DataFrame:
    return _frame(
        baseline_success=[1, 1, 1, 1, 0],
        schedule_success=[1, 0, 0, 0, 0],
        min_multiplier_to_succeed=[1.0, 2.5, 9.7429, None, 3.0],
        execution_gas=[10, 20, 30, 40, 50],
        state_gas=[1, 200, 300, 4, 5],
        schedule_gas_used=[5, 6, 7, 8, 9],
    )


def test_strict_policy_drops_every_schedule_failure():
    selected, excluded = loader.split_simulatable(
        _rescue_frame(), policy="successful_only"
    )
    assert selected["execution_gas"].tolist() == [10]
    assert excluded["exclusion_reason"].tolist() == [
        "schedule_gas_rescuable",
        "schedule_gas_rescuable",
        "schedule_halted_regardless",
        "both_failed",
    ]


def test_gas_rescued_policy_keeps_limit_censored_rows_but_not_real_halts():
    selected, excluded = loader.split_simulatable(_rescue_frame(), policy="gas_rescued")
    assert selected["execution_gas"].tolist() == [10, 20, 30]
    # The rescued rows are where the state gas lives; this is the whole point.
    assert selected["state_gas"].sum() == 501
    assert excluded["exclusion_reason"].tolist() == [
        "schedule_halted_regardless",
        "both_failed",
    ]


def test_max_rescue_multiplier_bounds_the_assumed_limit_increase():
    selected, _ = loader.split_simulatable(
        _rescue_frame(), policy="gas_rescued", max_rescue_multiplier=5.0
    )
    assert selected["execution_gas"].tolist() == [10, 20]


def test_a_baseline_failure_is_never_rescued():
    """Rescue only forgives a too-small gas limit, not a broken baseline."""
    selected, _ = loader.split_simulatable(_rescue_frame(), policy="gas_rescued")
    assert 50 not in selected["execution_gas"].tolist()


def test_all_policy_keeps_every_row_including_failures():
    selected, excluded = loader.split_simulatable(_rescue_frame(), policy="all")
    assert len(selected) == 5 and excluded.empty


def test_capacity_stays_on_the_pre_refund_basis_not_schedule_gas_used():
    """EIP-7778: block accounting ignores refunds, so gas_used is not the basis."""
    frame = _frame(
        schedule_total_gas_spent=[100_000],
        schedule_gas_used=[60_000],  # post-refund, much smaller
        schedule_state_gas_spent=[10_000],
        schedule_floor_gas=[0],
    )
    derived = loader.derive_gas_dimensions(frame)
    assert derived["execution_gas"].iloc[0] == 90_000  # from total, not gas_used
