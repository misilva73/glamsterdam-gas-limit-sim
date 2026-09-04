"""Offline injection of the synthetic datasets.

The simulation reads ClickHouse and nothing else, so there is no offline mode to
switch on. Instead the two functions that touch the network --
`load_tx_gas_results.fetch_tx_gas_result_chunks` and
`fetch_blocks.fetch_block_headers_uncached` -- are replaced with `tests.dummy`
generators. Both are called through their module globals by the cached wrappers
above them, so patching the module attribute leaves caching, dtype normalisation,
the dataset guard and provenance on the real code path.

The replay stand-in yields the trace in `block_chunks` pieces exactly as the real
fetch does, so the streamed cache write -- one Parquet part per chunk, provenance
accumulated across them -- is what the suite actually exercises.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import SimConfig
from data import fetch_blocks
from data import load_tx_gas_results as loader
from tests.dummy import (
    DUMMY_ANALYSIS_CONFIG_HASH,
    DUMMY_SCHEDULE_CONFIG_HASH,
    dummy_block_headers,
    dummy_tx_gas_results,
)

FIRST_BLOCK = 21_000_000
NUM_BLOCKS = 60
LAST_BLOCK = FIRST_BLOCK + NUM_BLOCKS - 1


OFFLINE_BLOCK_CHUNK = 25


def offline_tx_gas_results(cfg: SimConfig) -> pd.DataFrame:
    """Stand-in for `fetch_tx_gas_results`: source-shaped rows for the pinned range."""
    first_block, last_block = loader.require_block_range(cfg)
    return dummy_tx_gas_results(
        first_block=first_block,
        num_blocks=last_block - first_block + 1,
        schedule_name=cfg.schedule_name,
    )


def offline_tx_gas_result_chunks(cfg: SimConfig):
    """Stand-in for `fetch_tx_gas_result_chunks`, cut the way the real fetch cuts.

    The chunk is deliberately far smaller than `BLOCK_CHUNK` so the 60-block
    synthetic range still arrives as several chunks. A single-chunk stand-in would
    leave part ordering and provenance merging -- the parts of the streamed write
    that can actually be wrong -- uncovered.
    """
    rows = offline_tx_gas_results(cfg)
    first_block, last_block = loader.require_block_range(cfg)
    for lo, hi in loader.block_chunks(first_block, last_block, chunk=OFFLINE_BLOCK_CHUNK):
        yield rows[rows["block_number"].between(lo, hi)].reset_index(drop=True)


def offline_block_headers(
    cfg: SimConfig, first_block: int, last_block: int
) -> pd.DataFrame:
    """Stand-in for `fetch_block_headers_uncached`, from the same synthetic trace.

    Generation is anchored at the configured source range rather than at
    `first_block`: the generator's autocorrelated draw depends on its start index,
    so regenerating from a shifted start would describe a *different* trace than
    the one being simulated -- and the derived starting base fee would not match
    the demand it is supposed to summarise. Blocks before the anchor (the parent
    of the first source cohort) inherit the anchor block's header.
    """
    anchor = cfg.source_block_range[0] if cfg.source_block_range else first_block
    replay_rows = dummy_tx_gas_results(
        first_block=anchor, num_blocks=max(last_block, anchor) - anchor + 1
    )
    headers = dummy_block_headers(replay_rows, gas_limit=cfg.fusaka_gas_limit)

    if first_block < anchor:
        parent = headers.iloc[[0]].drop(columns="block_number")
        pad = pd.DataFrame({"block_number": range(first_block, anchor)}).join(
            pd.concat([parent] * (anchor - first_block), ignore_index=True)
        )
        headers = pd.concat([pad, headers], ignore_index=True)
    return headers[headers["block_number"].between(first_block, last_block)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make any unstubbed fetch fail loudly instead of reaching a real cluster.

    Both clients build lazily via `clickhouse_connect.get_client`, so blocking it
    proves the suite is offline even on a machine that has `secrets.json`.
    """
    import clickhouse_connect

    def refuse(*args, **kwargs):
        raise AssertionError(
            "the test suite must not open a ClickHouse connection; request the "
            "`offline_data` fixture to serve `tests.dummy` frames instead"
        )

    monkeypatch.setattr(clickhouse_connect, "get_client", refuse)


@pytest.fixture
def offline_data(monkeypatch):
    """Serve both datasets from `tests.dummy` instead of ClickHouse."""
    monkeypatch.setattr(loader, "fetch_tx_gas_result_chunks", offline_tx_gas_result_chunks)
    monkeypatch.setattr(
        fetch_blocks, "fetch_block_headers_uncached", offline_block_headers
    )


@pytest.fixture
def offline_cfg(tmp_path) -> SimConfig:
    """Config pinned to the synthetic dataset. Pair with `offline_data` to fetch."""
    return SimConfig(
        analysis_config_hash=DUMMY_ANALYSIS_CONFIG_HASH,
        schedule_config_hash=DUMMY_SCHEDULE_CONFIG_HASH,
        source_block_range=(FIRST_BLOCK, LAST_BLOCK),
        cache_dir=tmp_path / "cache",
    )
