# AGENTS.md — Fusaka → Glamsterdam gas-limit simulation

Orientation for anyone changing this code. Read this before changing behaviour,
and update it in the same change.

**`METHODOLOGY.md` is the reference for what the simulation does and why** — data
sources, the workload and demand models, the engine step by step, every output
column, the assumptions, and how to read a result. `README.md` is the quickstart
and flag reference. This file is the *how to work on it* half: ownership, the
traps, dataset status, performance. Where they overlap, `METHODOLOGY.md` is
canonical for methodology and this file for code behaviour.

In one paragraph: real mainnet transactions re-executed under the Glamsterdam gas
schedule (upstream, by the `reth-research` replay in the `CarlBeek/reth` fork) are
replayed as per-source-block arrival cohorts against a gas limit ramping 60M →
200M at 1/1024 per block. Blocks have **two** capacity dimensions, execution and
state, and `gas_used = max(execution, state)`. How much demand arrives comes from
an isoelastic model driven by a smoothed effective gas price, so base fee → demand
→ utilisation → base fee is a closed loop. It is a **frozen-trace simulation, not
counterfactual EVM replay**.

## Layout and ownership

```text
config.py                     SimConfig (every knob), SimulationGrid, protocol constants
schemas.py                    column contracts shared by all three layers
conftest.py                   makes the repo root importable for bare `pytest`

data/load_tx_gas_results.py   ClickHouse loading, filtering, gas derivation
data/fetch_blocks.py          Xatu block headers, block-range resolution, starting base fee
data/cache.py                 Parquet + JSON-sidecar cache used by both loaders

sim/workload.py               cohorts + demand anchors, arrival paths (historical /
                              moving-block bootstrap), the demand model (multiplier,
                              sampling pools, gas-target sampling, bid adaptation),
                              RNG streams
sim/engine.py                 the per-block loop: fee update, gas-limit ramp, demand
                              draw, block fill, price-signal update
sim/metrics.py                pure protocol functions + per-step record assembly

analysis/window_length.py     cohort summaries, autocorrelation, choice of bootstrap L
analysis/plots.py             quantile bands + historical overlay figures
run_simulation.py             CLI: load → simulate the grid → write output → plot
tests/dummy.py                synthetic replay rows + headers (test fixtures only)
tests/conftest.py             `offline_data`, which injects them at the fetch seams
tests/                        pytest suite (166 tests)
```

`schemas.py` is the contract between layers. Changing an existing name there is a
breaking change across all three; appending is safe.

Run it with `README.md`'s quickstart. One gotcha: `pip install` may fail inside a
sandboxed shell because TLS is intercepted — run that one command with the sandbox
disabled.

## Live dataset status

`gas_analysis.gas_analysis_tx_gas_result` exists, is being populated, and the full
pipeline has been run end to end against it. As measured 2026-09-01: ~12.9M rows,
41,380 blocks spanning `25278577`–`25319985`, ~5.75 days, one
`(analysis_config_hash, schedule_config_hash, schedule_name)`. It grows fast —
**re-measure the range before planning a simulation** rather than trusting those
numbers.

Operational facts, confirmed live:

- The table is **database-qualified**: `gas_analysis.gas_analysis_tx_gas_result`.
  The client connects to `default`, so an unqualified name raises `UNKNOWN_TABLE`.
- **`schedule_name` is `amsterdam`** (the execution-layer fork name in
  execution-specs), not `glamsterdam-v1`. `SimConfig.schedule_name` defaults to it,
  and a wrong value raises an error listing what the table holds.
- **`force_primary_key` is on.** The key is
  `(chain_id, analysis_config_hash, schedule_name, block_hash, tx_index, tx_hash)`
  — note `block_hash`, not `block_number`. Our query pins the first three, which
  satisfies the prefix, so the block-number range is a scan within that partition
  and `BLOCK_CHUNK` is about query size, not index use. Ad-hoc inventory queries
  must pin at least `chain_id`: a bare `GROUP BY` raises `INDEX_NOT_USED` (code
  277) rather than returning slowly.
- **Coverage has 25 gaps totalling 29 blocks.** All but one are single
  zero-transaction mainnet blocks, which a per-transaction table cannot represent;
  the one cluster is `25278577`–`25278594`, a ragged replay start. **Start a range
  at 25278600 or later.** Positional cohorts tolerate skipped block numbers, so the
  single-block gaps need no handling.
- Column names and types are confirmed; the loader normalization rules are
  summarized in `METHODOLOGY.md` §3.1.

ClickHouse is the only input, so **every real run needs credentials** — gitignored
`secrets.json` (`METHODOLOGY.md` §3.2). Both loaders cache to `cfg.cache_dir` with
a provenance sidecar (§3.4).

The suite stays offline by injecting `tests/dummy.py` frames at the two functions
that talk to a cluster (`fetch_tx_gas_results`, `fetch_block_headers_uncached`) —
the `offline_data` fixture. Never add an offline branch back into `data/`: an
autouse fixture blocks `clickhouse_connect.get_client`, so a test that needs data
and forgets `offline_data` fails with a message telling it what to request.

## Capacity traps — all guarded by tests

Formulas in `METHODOLOGY.md` §4.2 and §7.5.

- `schedule_state_gas_spent` is **already part of** `schedule_total_gas_spent`. It
  is not an extra charge, so execution gas is the remainder, floored by
  `schedule_floor_gas`.
- `schedule_gas_used` is post-refund and floor-applied: **sender-cost metrics
  only, never capacity.** Capacity stays on the pre-refund basis whatever the
  inclusion policy, per EIP-7778, which has block accounting ignore refunds.
- Do **not** assume `schedule_total_gas_spent - schedule_gas_refunded ==
  schedule_gas_used`; the calldata floor may bind (~9% of dummy rows).
- Block fill is **skip-and-continue, not first-fit-stop**: an item that doesn't fit
  is skipped and the walk continues, so a cheap low-tip transaction can land after
  an expensive high-tip one is skipped.
- The **legacy tip branch is load-bearing**, not redundant:
  `max_priority_fee_per_gas` is null (filled 0) for every type 0/1 row, so without
  `tx_type` to select the branch every legacy and access-list transaction would
  price at a zero tip and starve. Any future source must supply `tx_type`.

## Demand-model traps — all guarded by tests

Model in `METHODOLOGY.md` §6.

- **The quantity measure is `execution_gas + state_gas`** (`Cohorts.total_gas`),
  not the simulator's `max(execution, state)`. It is the *historical* metering
  basis the elasticities were estimated on; mixing the two silently miscalibrates.
- **The price signal is the smoothed effective price, not the base fee.** Base fee
  alone is unbounded below and overstates the price fall whenever tips dominate —
  exactly the 200M regime. The EMA is also the damping term on a loop whose gain is
  large at `e ~ 0.175`.
- **`cohort_anchor_price` must stay on the same basis as the simulated side**:
  header base fee plus the *realised gas-weighted mean tip*, matching
  `priority_fees_wei / sender_gas_used`. Anchor on the bare header base fee and the
  ratio is biased at every step, so `m != demand_level` even at the anchor.
- **The price signal is seeded at the first cohort's anchor**, not at
  `starting_base_fee`, which makes step 0's multiplier exactly `demand_level`
  whatever the starting base fee. A test pins that fixed point.
- **Everything feeding the loop is parent-derived.** The multiplier uses the signal
  as it stood *before* this block was built; the signal updates from the block's
  outcome afterwards.
- **Arrivals must be emitted in `(source_block_number, tx_index, replica_index)`
  order.** `sample_arrivals` draws from a whole bootstrap window, so its picks are
  unordered; it sorts by flat position, which *is* that order. Skip that sort and
  the engine's append-only mempool invariant breaks silently.
- **Legacy rows shift, dynamic-fee rows scale.** A legacy row's whole headroom is
  its tip, so scaling its gas price would scale the tip; shifting holds it
  absolute, matching what leaving `max_priority_fee_per_gas` alone does for
  dynamic-fee rows. Both preserve eligibility exactly.
- **Anchors are required for either price-responsive feature.** `build_cohorts`
  needs `headers`; `run_path` refuses `aggregate_elasticity != 0` or `adapt_bids`
  without them, naming which. `aggregate_elasticity=0` plus `adapt_bids=False` is
  the pre-demand-model behaviour and needs no headers — that is what the mechanism
  tests run under.
- **Arrivals are recorded, not derived.** Sampling means `arrived_*_gas` cannot be
  recomputed from the path plus a scalar, so anything reconciling arrivals against
  the backlog must read those columns.
- **`float64(int64_max)` rounds up to 2\*\*63.** The bid rescale clips to
  `_INT64_FLOAT_CAP`, the largest float64 that survives the cast; clipping to
  `int64_max` still produces an invalid cast, which numpy reports as a warning and
  a negative value rather than an error.

## Behavioural decisions worth knowing

- **Both persistent-state updates are parent-derived.** Neither the base fee nor
  the ramp is applied at position 0, so the first recorded block reports the
  configured initial state verbatim. From then on the ramp applies at
  `first_glamsterdam_simulation_step` itself, with no activation delay, via
  `position >= max(1, first_glamsterdam_simulation_step)`.
- **`tx_hash` is not carried through cohorts or expansion.** A week is ~8M rows and
  object-dtype hashes cost ~1 GB. Identity is `(run_index, window_instance,
  source_block_number, tx_index, replica_index)`, 1:1 with a `tx_hash`-based one.
  `schemas.WORKLOAD_COLUMNS` is consequently never materialised as a frame.
- **`priority_fees_wei` is float64 end to end, including the output column**: a
  200M-gas block of 500-gwei-tip transactions overflows int64 (~1e20). Converting
  to a Python int would survive the sum but make the column object-dtype and break
  the Parquet write — the bug the regression test in `tests/test_engine.py` pins.
- **`MIN_BASE_FEE` is 1 wei and is a guard, not a live constraint.** The 1559
  decrement floors to zero once the base fee drops below 8, so an emptying chain
  sticks at 7 wei or below on its own arithmetic; the floor binds only from exactly
  0, reachable only via `--starting-base-fee 0`. It is therefore *not* what bounds
  the demand multiplier — `--multiplier-bounds` does that job.
- **`build_cohorts` does not re-filter rows.** Which rows count as demand is
  entirely `split_simulatable`'s decision; re-applying a success filter here would
  silently undo `--tx-inclusion-policy`. It only refuses an empty frame.
- **Cohorts are positional and may skip block numbers** (a source block whose every
  row was filtered out has no cohort), so bootstrap windows are contiguous in
  *cohort index*, not block number. The induced sampling remainder is drawn from
  the step's whole window rather than the arriving cohort, which would make it a
  near copy of one block; under `historical` the pool is a trailing window instead.
- **Gas-target sampling accepts its overshoot**, keeping the draw that crosses the
  target rather than Bernoulli-correcting it: the error is one transaction against
  a cohort-scale target. `_MAX_SAMPLING_BATCHES` raises rather than looping forever
  on a degenerate pool.
- **`tests/dummy.py` writes `max_priority_fee_per_gas = 0` for legacy/access-list
  rows**, the likely real-producer value. Deliberately not a copy of the gas price:
  with a copy, `min(prio, max_fee - base_fee)` collapses into the legacy rule and
  deleting that branch would not change a single dummy result, whereas with 0 it
  would starve every legacy transaction — so the branch stays testable. **Check
  what the real producer writes**; the gas price would also work, but a null would
  need the loader's dtype handling revisited.
- `historical_path` raises rather than truncating when `horizon > len(cohorts)`; a
  short reference path would misalign against bands aggregated by
  `simulation_position`.
- `aggregate_bands` requires a single `arrival_mode` and raises otherwise, rather
  than folding the historical reference into the bands it is compared against.
- Backlog is measured **post-inclusion**, with the eligible/fee-ineligible split
  taken at the current block's base fee.
- Bottleneck ties go to `execution`; `none` only when both dimensions are zero.
- `ramp_gas_limit` never steps down.

### No per-transaction gas limit is modelled

An `effective_gas_limit` derivation (`tx_gas_limit × min_multiplier_to_succeed`)
was removed as dead weight — nothing reads a per-transaction gas limit.
`tx_gas_limit` stays in the loaded frame for analysis but is not carried into
cohort arrays. Two facts before reintroducing one:

- Selection cannot use it today. Ordering is by effective tip, a price *per gas*
  from the fee fields alone, so transaction size never enters the sort. The
  candidate use is block-space *reservation*, as a conservative builder must.
- `min_multiplier_to_succeed` is **≤ 1 for already-successful transactions**
  (median 0.73, min 0.003 live): it reports the *minimum* limit they needed, so it
  measures headroom, not a shortfall. Multiplying by it shrinks a successful
  transaction's limit ~61% below what the sender signed. Any derivation almost
  certainly wants `max(1, multiplier)`.

## Reporting results

The assumptions and limitations are `METHODOLOGY.md` §9, with §10 on what each one
costs a reader. State them whenever results are reported.
`run_simulation.CAVEAT` is the short form printed by every run and stored in every
`manifest.json`; keep it in sync with §9 if either changes.

## Performance envelope

Measured on the dummy trace (~190 tx/block). The per-block scan is Θ(backlog) and
at high demand the backlog grows monotonically, so throughput falls with horizon:

| case | time | steps/s | peak RSS | max backlog |
|---|---|---|---|---|
| 7,200 steps @1x | 0.3 s | ~22,900 | 655 MB | 2,057 |
| 50,400 steps @1x | 2.5 s | ~19,800 | 668 MB | 2,057 |
| 7,200 steps @5x | 8.6 s | ~840 | 834 MB | 2.8M |
| 50,400 steps @5x | 503 s | ~100 | 1,749 MB | 16.7M |

A full-week 5x cell is ~8.4 min/path, so 20 bootstrap runs ≈ 2.8 h at ~1.8 GB per
worker; low-demand cells are essentially free. **Plan the grid accordingly** — cost
is dominated entirely by the cells whose realized multiplier runs high, and the two
demand axes multiply (the defaults are 48 cells).

The mempool is tombstoned parallel numpy arrays with amortised compaction
(`COMPACTION_DEAD_SHARE`), and the tip sort is tranched (`TIP_TRANCHE_SIZE`),
cutting only at tip *values*, never mid-tie. Nothing is capped or approximated:
tranching and the vectorised fill are tested against literal sequential reference
implementations, and compaction cadence is asserted not to change output.

## Known gaps

1. **Block fill is driven purely by actual gas.** For a conservative-builder model,
   see "No per-transaction gas limit is modelled" above first.
2. **Per-step output is concatenated in memory.** `run_simulation` builds the whole
   per-step frame before writing; a full week × 48 cells × 20 runs is ~48M rows,
   which needs partitioned writes.
3. **Re-derive the bootstrap window `L` on real data.** The dummy trace's
   autocorrelation is set by its own AR(1) parameters, so the current answer is
   circular. `tests/dummy.py` in particular draws dynamic-fee priority fees i.i.d.
   per transaction, so `median_max_priority_fee_per_gas` decorrelates in ~1 block —
   an artifact, not a finding.
4. **`baseline_only_failure` is always empty on dummy data** by construction, so
   that branch of `excluded_summary` is unit-tested but never exercised end to end.
5. **The dataset is still being written**, so a cached extract and a fresh query can
   disagree. Only one `analysis_config_hash` and one schedule exist so far, so the
   hash-mixing guard has never fired on real data.
6. **Chunked reads are unproven at scale.** `BLOCK_CHUNK = 5_000` is validated for
   Xatu but the replay table has never been read in a range needing more than one
   chunk, so whether `FINAL` dedup can straddle a chunk boundary is untested — it
   should not, since chunks are disjoint block ranges and the primary key leads with
   `(chain_id, analysis_config_hash, schedule_name, block_hash, ...)`. The largest
   range actually simulated is 4,000 blocks, so gap #2's memory ceiling is likewise
   untested at week scale.

## Conventions

- Python 3.12, pandas 3.x (copy-on-write default — no chained assignment, no
  `inplace=`), numpy, scipy, statsmodels, matplotlib/seaborn.
- Flat layout, plain functions, dataclasses for config. No class hierarchies,
  registries, or dependency injection.
- Comments explain protocol subtleties, dataset caveats, and *why* — not what the
  code already says.
- Prefer library primitives (pandas groupby/quantile, statsmodels `acf`) over
  hand-rolled equivalents. Custom code is for the project-specific parts: the
  block-fill rule, the gas-limit ramp, and two-dimensional capacity accounting.
