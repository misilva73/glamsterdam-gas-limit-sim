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
config.py                     SimConfig (fixed run controls), Scenario (one cell),
                              SimulationGrid (swept axes), protocol constants
schemas.py                    column contracts shared by all three layers
conftest.py                   makes the repo root importable for bare `pytest`

data/load_tx_gas_results.py   ClickHouse loading, filtering, gas derivation
data/fetch_blocks.py          Xatu block headers, block-range resolution, starting base fee
data/cache.py                 Parquet + JSON-sidecar cache used by both loaders;
                              an entry is one file, or a directory of parts when
                              streamed

sim/workload.py               cohorts + the trace-level demand reference, per-step
                              composition pools, the demand model (multiplier,
                              gas-target sampling, bid adaptation), RNG streams
sim/engine.py                 the per-block loop: fee update, gas-limit ramp, demand
                              draw, block fill, price-signal update
sim/metrics.py                pure protocol functions + per-step record assembly

run_simulation.py             CLI: load → simulate the grid, checkpointing each
                              cell as it finishes → write the manifest
tests/dummy.py                synthetic replay rows + headers (test fixtures only)
tests/conftest.py             `offline_data`, which injects them at the fetch seams
tests/                        pytest suite
```

`schemas.py` is the contract between layers. Changing an existing name there is a
breaking change across all three; appending is safe.

Run it with `README.md`'s quickstart. One gotcha: `pip install` may fail inside a
sandboxed shell because TLS is intercepted — run that one command with the sandbox
disabled.

## Live dataset status

`gas_analysis.gas_analysis_tx_gas_result` exists, is being populated, and the full
pipeline has been run end to end against it. As measured 2026-09-04: ~44.0M rows,
151,144 blocks spanning `25168786`–`25319985`, ~21 days, one
`(analysis_config_hash, schedule_config_hash, schedule_name)`. It grows fast, and
it is **backfilled at the head as well as the tail** — the 2026-09-01 measurement
had it starting at `25278577`, 110k blocks later — so **re-measure the range before
planning a simulation** rather than trusting those numbers.

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
- **Coverage gaps are all single blocks** as of the 2026-09-04 backfill: every gap
  in `25168786`–`25319985` is one zero-transaction mainnet block, which a
  per-transaction table cannot represent. Positional cohorts tolerate skipped block
  numbers, so they need no handling. The ragged `25278577`–`25278594` start that
  once forced a `25278600` floor has been filled in; the current first block is
  clean and needs no offset.
- Column names and types are confirmed; the loader normalization rules are
  summarized in `METHODOLOGY.md` §3.1.

ClickHouse is the only input, so **every real run needs credentials** — gitignored
`secrets.json` (`METHODOLOGY.md` §3.2). Both loaders cache to `cfg.cache_dir` with
a provenance sidecar (§3.4).

The suite stays offline by injecting `tests/dummy.py` frames at the two functions
that talk to a cluster (`fetch_tx_gas_result_chunks`, `fetch_block_headers_uncached`) —
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
- **`DemandReference.price` must stay on the same basis as the simulated side**:
  historical base fee plus the *realised gas-weighted mean tip*, matching
  `priority_fees_wei / sender_gas_used`. Anchor on bare base fees and the ratio is
  biased at every step, so `m != demand_level` even at the reference.
- **The demand reference is one constant for the whole trace, gas-weighted over
  transactions.** An isoelastic curve is only a curve relative to a fixed
  reference pair, so both `price` and `gas` are trace-level; that is what gives
  `demand_level` a stable meaning and keeps `realized_demand_multiplier` a
  function of the simulated price alone. Averaging per-cohort anchors instead
  would weight a 5M-gas block like a 30M-gas one and drift off basis; anchoring
  per step on its own pool would put a different curve under every step.
- **The price signal is seeded at the reference price**, not at
  `starting_base_fee`, which makes step 0's multiplier exactly `demand_level`
  whatever the starting base fee — and keeps it there until the simulated price
  leaves the historical average. A test pins that fixed point.
- **Everything feeding the loop is parent-derived.** The multiplier uses the signal
  as it stood *before* this block was built; the signal updates from the block's
  outcome afterwards.
- **Arrivals must be emitted in `(source_block_number, tx_index, replica_index)`
  order.** `sample_arrivals` draws from a whole composition pool, so its picks are
  unordered; it sorts by flat position, which *is* that order. Skip that sort and
  the engine's append-only mempool invariant breaks silently.
- **Legacy rows shift, dynamic-fee rows scale.** A legacy row's whole headroom is
  its tip, so scaling its gas price would scale the tip; shifting holds it
  absolute, matching what leaving `max_priority_fee_per_gas` alone does for
  dynamic-fee rows. Both preserve eligibility exactly.
- **Every cohort is anchored.** `build_cohorts` always requires `headers`; there is
  no headerless engine mode. This keeps bid adaptation and the demand fixed point
  on one invariant even when a scenario uses zero elasticity.
- **Arrivals are recorded, not derived.** The *whole* arrival is sampled, so
  `arrived_*_gas` cannot be recomputed from the path at all — anything reconciling
  arrivals against the backlog must read those columns. It also means a single
  path's per-step mix is noisy; aggregates across runs carry the signal.
- **Uniform draws, gas-based stopping.** `_sample_to_gas_target` picks positions
  uniformly and only *stops* on cumulative gas. Weighting the draw by gas would
  over-represent large transactions and reshape the mix the pool exists to supply.
- **`float64(int64_max)` rounds up to 2\*\*63.** The bid rescale clips to
  `_INT64_FLOAT_CAP`, the largest float64 that survives the cast; clipping to
  `int64_max` still produces an invalid cast, which numpy reports as a warning and
  a negative value rather than an error.

## Behavioural decisions worth knowing

- **Composition pools are not a moving-block bootstrap.** Every step draws its own
  start independently, so consecutive steps are unrelated and none of the trace's
  autocorrelation reaches the arrival process; all persistence comes from mempool,
  base fee, and the EMA. Contiguity buys *local coherence* — one fee regime per
  mix — not preserved dependence. There is no historical-only mode, historical
  reference path, or `arrival_mode` output dimension.
- **`pool_start_block` identifies the draw, not the arrivals.** A step's
  transactions come from anywhere in its pool; they are reproducible from
  `run_index` plus the recorded seeds, never from that column alone.
- **A checkpoint writes its one cell part before the summary row describing it.**
  `--resume` treats complete cells as the leading run whose parts are present,
  bounded by the summary row count, and truncates the summary to that prefix.
  Reordering those writes silently breaks resume.
- **The manifest is written twice**, before the first cell and at the end, and
  `completed` tells them apart. It is also the *input* to a resumed run:
  `resolve_inputs` rebuilds `SimConfig` and `SimulationGrid` from it
  (`stored_inputs`) and layers whatever flags were given on top, so `--resume
  STAMP` alone is a complete command line — which is why
  `--analysis-config-hash` is enforced in `resolve_inputs` rather than by
  `required=True`. `open_resumed_run` then compares the stored `config`, `grid`,
  per-step schema, and `resolved` against the current ones and refuses on any
  difference except the path fields in `RESUME_EXEMPT_CONFIG_FIELDS`. Via the CLI
  that check has nothing left to catch; it is the guard for callers passing their
  own config to `run_simulation`.
- **Reading a config back is type-lossy and deliberately strict.** JSON has no
  `Path` and no tuple, so `stored_inputs` restores `PATH_CONFIG_FIELDS` as paths
  and every list as a tuple. A manifest naming a field `SimConfig` no longer has
  is refused outright: the retired knob had a value in the finished cells, and
  substituting today's default would make the two halves of the sweep
  incomparable.
- **Both persistent-state updates are parent-derived.** Neither the base fee nor
  the ramp is applied at position 0, so the first recorded block reports the
  configured initial state verbatim and the ramp first applies at position 1.
  Glamsterdam is always live from position 0; there is no activation step.
- **`tx_hash` is not carried through cohorts or expansion.** A week is ~8M rows and
  object-dtype hashes cost ~1 GB. Identity is `(run_index, simulation_position,
  source_block_number, tx_index, replica_index)`, 1:1 with a `tx_hash`-based one.
  Expanded workload is kept only in the engine's parallel arrays, never a frame.
- **`priority_fees_wei` is float64 end to end, including the output column**: a
  200M-gas block of 500-gwei-tip transactions overflows int64 (~1e20). Converting
  to a Python int would survive the sum but make the column object-dtype and break
  the Parquet write — the bug the regression test in `tests/test_engine.py` pins.
- **`MIN_BASE_FEE` is 1 wei and is a guard, not a live constraint.** The 1559
  decrement floors to zero once the base fee drops below 8, so an emptying chain
  sticks at 7 wei or below on its own arithmetic; the floor binds only from exactly
  0, reachable only via `--starting-base-fee 0`. It is therefore *not* what bounds
  the demand multiplier — `--multiplier-bounds` does that job.
- **`build_cohorts` does not filter rows.** Every replay row is demand, failures
  included; there is no inclusion policy anywhere in the pipeline, so applying a
  success filter here would invent one. It only refuses an empty frame.
- **Cohorts are positional and may skip block numbers** (a source block with no
  replay rows has no cohort), so composition pools are contiguous in *cohort
  index*, not block number. A pool never wraps the trace end, so its last `L - 1`
  cohorts appear in fewer pools — the accepted edge bias; wrapping would build a
  mix out of two unrelated fee regimes.
- **Gas-target sampling accepts its overshoot**, keeping the draw that crosses the
  target rather than Bernoulli-correcting it: the error is one transaction against
  a block-scale target, biasing arrived gas slightly up. `_MAX_SAMPLING_BATCHES`
  raises rather than looping forever on a degenerate pool.
- **`tests/dummy.py` writes `max_priority_fee_per_gas = 0` for legacy/access-list
  rows**, the likely real-producer value. Deliberately not a copy of the gas price:
  with a copy, `min(prio, max_fee - base_fee)` collapses into the legacy rule and
  deleting that branch would not change a single dummy result, whereas with 0 it
  would starve every legacy transaction — so the branch stays testable. **Check
  what the real producer writes**; the gas price would also work, but a null would
  need the loader's dtype handling revisited.
- Backlog is measured **post-inclusion**, with the eligible/fee-ineligible split
  taken at the current block's base fee. Per-step output stores total and eligible
  values; fee-ineligible values are their exact difference.
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

`analysis.ipynb` uses blocks 3,000–7,199 for settled-period medians and
distributions in the one-day run `20260907T122124Z`, sampling 20,000 individual
blocks per scenario across all paths. This excludes both the gas-limit ramp and
the subsequent adjustment. The sample cache is named for the window
(`steady_3000_7200.parquet`) and retains `simulation_position`; do not reuse the
older post-ramp `steady.parquet`. The convergence diagnostic compares the two
halves of the settled window using 24-block averages, with the midpoint rounded
up to a bin boundary. Recalculate the notebook outputs and figures before
updating the report's statistics when changing this window.

The notebook caches are scoped to `output/analysis/20260907T122124Z/` and their
filenames identify the selected windows. Delete that run's analysis-cache directory
when source results or metric definitions change. Full-run guard, dimension and
backlog diagnostics read the original parts, not the settled sample. Classify
near-target scenarios from utilisation (within one percentage point of 50%), not
from the largest demand multiplier. Report medians as medians: zero median eligible
backlog does not mean zero maximum, and a ratio of scenario state/execution medians
does not establish which dimension is larger per block. In this run state is larger
in about 19–21% of blocks across scenarios.

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

A full-week 5x cell is ~8.4 min/path, so 20 runs ≈ 2.8 h at ~1.8 GB per worker;
low-demand cells are essentially free. **Plan the grid accordingly** — cost is
dominated entirely by the cells whose realized multiplier runs high, and the two
demand axes multiply (the defaults are 9 cells at the single default `L = 16`).

These figures predate moving *all* arrivals into `_sample_to_gas_target`. Whole
cohort copies used to cover the integer part of the multiplier with a cheap
`np.repeat` and no RNG; now every arriving transaction is drawn and the full
arrival is sorted, so expect the arrival stage to cost several times what it did
at high multipliers. Re-measure before planning a full-week sweep.

Memory no longer scales with the grid: each cell is written to `per_step/` and
freed before the next one starts, so the retained source frame and cohorts set the
floor and one cell's paths sit on top of it. A whole sweep therefore belongs in one
process — sharding it per cell across invocations is no longer necessary, and each
shard would pay the ~16 GB source-frame floor again. Peak is briefly two copies of
one cell, from the concat that assembles its part file.

**Nor does it scale with the fetch.** The loader used to hold every raw block-chunk
and then concatenate them, which cost roughly `10 GB + 1.24 GB per million rows`
and was OOM-killed at 64.7 GB fetching the 44M-row range on the 62 GB rig. Chunks
are now normalised straight into their own Parquet part under the cache entry
(`load_tx_gas_results._stream_to_cache`), so fetch peak is set by `BLOCK_CHUNK`
rather than by range length, and the combined frame is materialised exactly once
when the parts are read back. A cache entry is consequently a **directory of
`part-NNNNN.parquet`**, not a single file; `cache.read_cached` accepts either, and
parts are read in lexical order because that is block order — `load_tx_gas_results`
re-checks that the reassembled frame is block-sorted rather than trusting it, since
cohorts are cut positionally and a shuffled trace would fail nowhere else.

The mempool is tombstoned parallel numpy arrays with amortised compaction
(`COMPACTION_DEAD_SHARE`), and the tip sort is tranched (`TIP_TRANCHE_SIZE`),
cutting only at tip *values*, never mid-tie. Nothing is capped or approximated:
tranching and the vectorised fill are tested against literal sequential reference
implementations, and compaction cadence is asserted not to change output.

## Known gaps

1. **Block fill is driven purely by actual gas.** For a conservative-builder model,
   see "No per-transaction gas limit is modelled" above first.
2. **Resume is per cell and prefix-based, not per path.** A cell killed halfway
   through its 20 paths is simulated again from scratch, which on a full-week
   high-demand cell is ~2.8 h of rework. `--resume` also trusts that the
   checkpointed cells are a *prefix* of `grid_cells` — true because cells run
   sequentially, but it means hand-deleting a part file from the middle silently
   rewinds the sweep to that point rather than filling the hole.
3. **Re-derive the composition-pool width `L` on real data.** What `L` has to be
   wide enough for is now *local coherence* — enough transaction variety to look
   like real demand, without spanning two fee regimes — so the dummy trace's own
   AR(1) parameters say nothing about it. `tests/dummy.py` in particular draws
   dynamic-fee priority fees i.i.d. per transaction, so its fee mix decorrelates
   in ~1 block, an artifact rather than a finding.
4. **`baseline_only_failure` is always empty on dummy data** by construction, so
   that row of `replay_outcome_summary` is unit-tested but never exercised end to
   end.
5. **The dataset is still being written**, so a cached extract and a fresh query can
   disagree. Only one `analysis_config_hash` and one schedule exist so far, so the
   hash-mixing guard has never fired on real data.
6. **`FINAL` dedup across chunk boundaries is still untested.** Multi-chunk reads
   are now routine, but whether `FINAL` can straddle a `BLOCK_CHUNK` boundary has
   never been checked directly — it should not, since chunks are disjoint block
   ranges and the primary key leads with
   `(chain_id, analysis_config_hash, schedule_name, block_hash, ...)`.

## Conventions

- Python 3.12, pandas 3.x (copy-on-write default — no chained assignment, no
  `inplace=`), numpy. No plotting dependencies at all: a simulation run writes
  data and draws nothing.
- Flat layout, plain functions, dataclasses for config. No class hierarchies,
  registries, or dependency injection.
- Comments explain protocol subtleties, dataset caveats, and *why* — not what the
  code already says.
- Prefer library primitives (pandas groupby/quantile) over
  hand-rolled equivalents. Custom code is for the project-specific parts: the
  block-fill rule, the gas-limit ramp, and two-dimensional capacity accounting.
