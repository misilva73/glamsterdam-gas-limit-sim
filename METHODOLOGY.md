# Methodology

This is the source of truth for what the Fusaka → Glamsterdam gas-limit
simulation does, why it does it, and how to interpret its outputs. `README.md` is
the quickstart and CLI reference. `AGENTS.md` covers implementation and operating
details.

## 1. Overview

### The model in one minute

The simulation asks what happens to Ethereum's base fee, block utilisation, and
mempool while the gas limit rises from 60M toward 200M under the Glamsterdam gas
schedule, including [EIP-8037](https://eips.ethereum.org/EIPS/eip-8037) state gas
and [EIP-7778](https://eips.ethereum.org/EIPS/eip-7778) block accounting.

It combines two historical inputs:

1. Real mainnet transactions re-executed upstream under the `amsterdam` schedule.
   Their measured gas and success are frozen inputs to this model.
2. Historical block headers, used for base-fee initialization and to anchor each
   source block's observed demand to its historical price.

Transactions arrive in source-block cohorts. A run either preserves their
historical order or resamples contiguous windows with a moving-block bootstrap.
An isoelastic demand curve changes the quantity arriving as the simulated price
changes. Pending transactions are ordered by effective tip and greedily included
only when they fit both execution-gas and state-gas capacity.

For each simulated block, the engine does this:

```text
parent block
    ↓
update base fee and gas limit
    ↓
price signal ÷ cohort anchor → demand multiplier
    ↓
sample arrivals and adapt their fee caps
    ↓
add to mempool → filter by base fee → order by effective tip
    ↓
fill under execution AND state limits
    ↓
record block, arrivals, and remaining backlog
    ↓
update the price signal for the next block
```

The central feedback loop is:

```text
base fee → effective price → demand → utilisation → next base fee
```

### The three facts to remember

- This is a **frozen-trace simulation, not a counterfactual EVM replay**.
  Reordering or sampling a transaction does not re-execute it.
- Capacity is two-dimensional. A transaction must fit both dimensions, and the
  header-equivalent gas used is `max(block execution gas, block state gas)`.
- Results are sensitivity estimates under assumed demand parameters and the
  observed transaction mix. They are not equilibrium forecasts for a 200M limit.

Before quoting a result, read [§9 Assumptions](#9-assumptions) and check the three
diagnostics in [§10 Interpretation](#10-how-to-interpret-results).

## 2. Question and pipeline

The model studies two changes together:

- the block gas limit ramps from the Fusaka value toward a Glamsterdam target;
- Glamsterdam introduces state gas alongside execution gas.

Which resource binds is an output. Demand is also endogenous: lower simulated
prices can cause more arrivals, which can raise utilisation and the next base fee.

```text
ClickHouse replay rows + Xatu block headers
                  ↓
load, validate, derive gas dimensions, cache
                  ↓
group transactions into source-block cohorts
                  ↓
construct a historical or bootstrap arrival path
                  ↓
simulate demand, mempool, block fill, gas limit, and base fee
                  ↓
write per-step data, scenario summaries, and manifest
```

The upstream replay determines per-transaction gas and success under the candidate
schedule. This repository models arrival quantity, fee eligibility, ordering,
capacity, backlog, and fee feedback; it does not run the EVM.

## 3. Data sources

### 3.1 Replay rows — the demand trace

The transaction trace comes from
`gas_analysis.gas_analysis_tx_gas_result FINAL` on ClickHouse, produced by the
`reth-research` replay in the `CarlBeek/reth` fork. It contains one row per real
historical mainnet transaction re-executed under a candidate gas schedule.

Every query is bounded by an inclusive block range and pins `chain_id`,
`analysis_config_hash`, and `schedule_name` (default `amsterdam`).
`schedule_config_hash` is optional. The loader rejects mixed datasets both after a
fresh fetch and after a cache read.

The engine ultimately needs:

| Input | Purpose |
| --- | --- |
| `block_number`, `tx_index` | cohort membership and deterministic order |
| `tx_type`, fee-cap fields | eligibility and effective tip |
| schedule total, state, floor, and used gas | capacity dimensions and sender cost |
| baseline/schedule success, rescue multiplier | replay-outcome classification (§4.1) |

The loader also preserves transaction and producer provenance for analysis and
cache auditing. `schemas.TX_GAS_RESULT_FIELDS` and
`TX_GAS_RESULT_PROVENANCE_FIELDS` are the exact contracts.

Important normalization rules:

- Fee caps arrive as decimal strings and are parsed exactly into `int64`; overflow
  raises instead of wrapping.
- `max_priority_fee_per_gas` is nullable for legacy and access-list transactions.
  It is filled with zero because their tip rule does not read it. A null on any
  other transaction type is an error.
- Success flags arrive as booleans; nullable intrinsic gas and rescue multipliers
  retain nulls; fixed-size hash fields are decoded from bytes.

These are loader requirements, not modelling choices.

### 3.2 Block headers — prices and initial state

`canonical_execution_block FINAL` on Xatu supplies `block_number`, `gas_used`,
`gas_limit`, and `base_fee_per_gas` for mainnet.

The normal pipeline uses headers for:

- each cohort's historical base fee, needed for its demand anchor and bid
  adaptation;
- the base fee of the actual parent of `reference_start_block`, used as the
  simulation's initial base fee unless explicitly overridden.

The fetch therefore starts one block before the first reference block.
`canonical_beacon_block FINAL` is used only to translate a slot-time interval into
an execution-block range.

Credentials live in gitignored `secrets.json`; see `README.md` for setup.

### 3.3 Synthetic data — tests only

There is no production offline-data branch. Tests inject synthetic replay rows and
headers at the two network seams. The fixtures exercise autocorrelation, fee
ineligibility, calldata-floor binding, and state-gas saturation, but they are not
evidence about mainnet. In particular, any bootstrap window inferred from the
synthetic trace must be re-derived on real data.

### 3.4 Caching and provenance

Both loaders cache Parquet data plus a JSON provenance sidecar in `cache_dir`.
Cache identity includes the source table or network, pinned dataset fields, and
requested block range.

The replay sidecar records the requested and observed range, row and block counts,
hash and timestamp coverage, producer versions, replay semantics, and fetch time.
The header sidecar records its requested range and missing-block count.

The upstream replay table is still growing. For a reproducible result, pin an
explicit block range and retain both the cache and its sidecar.

## 4. Inputs and preprocessing

### 4.1 Which replay rows count as demand

**Every replay row does.** There is no inclusion policy and no filter: a failed
transaction still occupies block space and still pays, so it is real demand. Its
gas figures come from whichever replay run the producer recorded — the rescued run
where one succeeded, the original otherwise. Nothing is dropped for failing, and
nothing is dropped for how much extra gas limit it would have needed.

Rows are still *classified* by how they fared in the replay, and the breakdown is
reported in `replay_outcome_summary.csv`:

| Outcome | Meaning |
| --- | --- |
| `simulatable` | succeeded in both baseline and schedule |
| `schedule_gas_rescuable` | failed under the schedule, but succeeded once the replay raised the gas limit — censored by the sender's signed limit, not broken |
| `schedule_halted_regardless` | failed under the schedule at every swept multiplier |
| `baseline_only_failure` | failed in baseline only |
| `both_failed` | failed in both |

This breakdown is a description of the trace, not a choice about it, and it matters
because `schedule_gas_rescuable` rows can hold a disproportionate share of state
gas — gas that is only realisable if senders raise their signed limits.
`min_multiplier_to_succeed` is the evidence for that classification, not a gas
multiplier: values below one on already-successful transactions measure headroom,
and values at `RESCUE_SWEEP_CEILING` are censored, meaning *at least* that much.

### 4.2 The two capacity dimensions

For each transaction:

```text
state_gas = schedule_state_gas_spent

execution_gas = max(
    schedule_total_gas_spent - schedule_state_gas_spent,
    schedule_floor_gas,
)
```

For an included set:

```text
block_execution_gas = Σ execution_gas
block_state_gas     = Σ state_gas
gas_used            = max(block_execution_gas, block_state_gas)
```

`schedule_state_gas_spent` is already part of `schedule_total_gas_spent`; it is
not added again. The calldata floor can make execution gas exceed the simple
remainder.

Capacity is always pre-refund. `schedule_gas_used` is post-refund and
floor-applied, and is used only for sender-cost and tip-revenue metrics. It is
never a capacity input.

### 4.3 Configuration and grid

All single-run parameters live in `config.SimConfig`; swept axes live in
`config.SimulationGrid`. `README.md` is the complete flag reference, and every
resolved value is written to `manifest.json`.

Key defaults are:

| Parameter | Default |
| --- | ---: |
| initial / target gas limit | 60,000,000 / 200,000,000 |
| ramp rate | current limit ÷ 1024 per block |
| elasticities | 0.10, 0.175, 0.28 |
| demand levels | 1, 1.5, 2 |
| bootstrap windows | 16, 32, 64 cohorts |
| bootstrap runs per cell | 20 |
| price EMA span | 300 blocks |
| price-response bounds | 0.05, 20 |

The defaults form 27 grid cells before repeated runs and historical references.
High-demand cells dominate runtime because each block scans a growing mempool.

## 5. Workload model

### 5.1 Cohorts

All selected transactions from source block `n` form one arrival cohort. They are
ordered by `(block_number, tx_index)` and stored in flat arrays with cohort
offsets.

Cohorts are positional and may skip block numbers: if filtering removes every
transaction in a block, no empty cohort is created. Transaction identity after
sampling is `(run_index, window_instance, source_block_number, tx_index,
replica_index)`; hashes are omitted from the hot path to save memory.

### 5.2 Arrival paths

`historical` uses cohorts once in their original order. The horizon must not exceed
the available cohorts.

`moving_block_bootstrap` draws window starts uniformly with replacement, copies
`L` contiguous cohorts from each start, and truncates the concatenated path to the
horizon. Windows do not wrap around the trace. Each occurrence has a unique
`window_instance`.

Run `k` uses the same bootstrap windows in every demand scenario. These common
random numbers make scenario differences reflect parameters rather than different
window draws. A matching historical path is included by default as a reference,
not as part of the bootstrap bands.

### 5.3 Choosing the bootstrap length `L`

`analysis/window_length.py` summarizes each cohort by transaction count, both gas
dimensions, and fee statistics. For every series it reports:

- first autocorrelation lag below `1/e`;
- first lag inside the approximate 95% white-noise band;
- integral timescale through the first non-positive autocorrelation.

The most conservative estimate is `decorrelation_blocks`.
`supported_window_blocks` is the smallest candidate `L` that covers it, or null if
none does. `window_length.csv` records the evidence.
Treat `L` as a robustness axis and re-estimate it for each real trace.

## 6. Demand model

### 6.1 Quantity response

For an arriving cohort:

```text
price_response = clamp(
    (price_signal / cohort_anchor_price) ** -aggregate_elasticity,
    lower_bound,
    upper_bound,
)

realized_demand_multiplier = demand_level * price_response
gas_target = realized_demand_multiplier * cohort_total_gas
```

where:

```text
cohort_total_gas = Σ(execution_gas + state_gas)
```

The sum is deliberate: it is the historical metering basis on which the
elasticity estimates were calibrated. It is not the block-capacity measure
`max(execution, state)`.

`demand_level` sets latent demand at the anchor price. It represents secular
growth and demand missing from an included-only trace. `aggregate_elasticity`
sets how quantity responds to price. With elasticity zero, the multiplier is
always `demand_level` and needs no price anchor.

The default elasticity grid follows the central estimate 0.175 and rounded
event-based range 0.10–0.28 in the
[EIP-8037 empirical analysis](https://ethresear.ch/t/empirical-analysis-of-price-elasticities-for-ethereum-state-and-burst-resources/24166).
This model uses one aggregate elasticity, so it cannot model substitution between
execution and state demand.

### 6.2 Price signal and anchor

Demand reacts to an exponentially smoothed effective price:

```text
effective_price = base_fee + prevailing_tip
alpha           = 2 / (price_ema_blocks + 1)
next_signal     = alpha * effective_price + (1 - alpha) * current_signal
```

`prevailing_tip` is `priority_fees_wei / sender_gas_used` for the latest non-empty
block and is carried forward across empty blocks. Effective price is used instead
of base fee alone because tips dominate near the base-fee floor. The EMA dampens
per-block EIP-1559 noise and a high-gain feedback loop.

Each cohort's anchor price is:

```text
source block base fee + historical gas-weighted mean realized tip
```

The historical and simulated tips use the same `schedule_gas_used` weighting.
The price signal is seeded at the first arriving cohort's anchor, so **step 0's**
multiplier is exactly `demand_level`, regardless of the configured starting base
fee. Later steps are not fixed at their anchors; the simulated feedback loop
determines their multipliers.

### 6.3 Sampling arrivals

For multiplier `m`, arrivals contain:

1. `floor(m)` complete copies of the arriving cohort;
2. a fractional remainder sampled uniformly with replacement until its accumulated
   `execution_gas + state_gas` crosses the remaining gas target.

The crossing transaction is retained, so the target may be exceeded by at most
one sampled transaction. Remainders are sampled from the current bootstrap window;
for a historical path they use a trailing window of length `L`.

Arrivals are sorted by source position and replica index before admission. This
preserves the engine's deterministic append-only mempool order.

### 6.4 Bid adaptation

By default, historical fee caps are adapted from source base fee `b0` to simulated
base fee `bt`:

```text
dynamic fee (types 2/3/4):  max_fee' = max_fee * bt / b0
legacy/access list (0/1):   max_fee' = max_fee + (bt - b0)
```

Dynamic priority fees remain unchanged. The legacy shift also holds the absolute
tip unchanged. For source transactions that were eligible historically, both
rules preserve eligibility while allowing fee caps to continue determining order.
Rows with `b0 = 0` are left unchanged.

Without bid adaptation, historical fee caps are frozen. In that mode the fee
filter, rather than only the demand curve, can determine eligible quantity.

### 6.5 Bounds and diagnostics

The isoelastic response is unbounded as price approaches zero and is extrapolated
beyond the estimation range. `demand_multiplier_bounds` therefore clamps the
**price response before multiplying by `demand_level`**.

`demand_multiplier_clamped` marks each saturated response;
`multiplier_clamped_share` summarizes it. A high share means the result is being
set by the chosen bound and should be treated as unsupported extrapolation.

## 7. Block-production engine

### 7.1 State and step order

A path starts with an empty mempool, `starting_base_fee`, `fusaka_gas_limit`, and
the first cohort's anchor as its price signal. It runs the arrival horizon and then
**stops**: whatever is still queued at the last arrival is discarded.

There is no drain phase. How long a leftover queue would take to clear is
`backlog / gas_limit` blocks of arithmetic rather than a simulation result, and
whether a backlog is transient or structural is already visible in its trajectory
across the arrival phase.

At each position:

1. From position 1 onward, update base fee from the previous block and ramp the gas
   limit. Glamsterdam is always active from position 0.
2. Compute the multiplier from the pre-block price signal and current cohort anchor.
3. Sample arrivals, adapt bids, and append them to the mempool.
4. Filter fee-eligible transactions, order them, and fill the block.
5. Record block outcome, arrivals, and post-inclusion backlog.
6. Update prevailing tip and the price signal for the next position.

The first recorded block therefore uses the configured initial base fee and gas
limit exactly.

### 7.2 Base fee

The update uses [EIP-1559](https://eips.ethereum.org/EIPS/eip-1559) integer
arithmetic with target `parent_gas_limit / 2` and maximum change denominator 8.
An upward non-zero change is at least 1 wei; the downward change is not.
`MIN_BASE_FEE = 1` is a guard, but ordinary integer arithmetic can leave an empty
chain stuck at 7 wei or below before the guard binds.

`MAX_BASE_FEE = 1e17` wei (0.1 ETH per gas) caps the update. The protocol has no
such ceiling; this one exists because the 1559 increment is multiplicative. A
scenario that cannot shed demand — no price response (§6.1) combined with
eligibility-preserving bid adaptation (§6.4) — saturates every block and raises
the fee ~12.5% per block without limit, crossing int64 in about 213 blocks and
crashing in `adapt_bids`.

The ceiling is set where the fee is unambiguously absurd but the arithmetic is
still safe: ~6 orders of magnitude above any base fee mainnet has seen, and a
200M-gas block at this fee would burn about 20 million ETH.

Hitting it is a statement about the scenario, never about the chain.
`base_fee_clamped` marks each step at the ceiling and `base_fee_clamped_share`
summarizes it — read exactly like `demand_multiplier_clamped` (§6.5): a non-zero
share means the bound, not the mechanism, set the fees, and they are not
interpretable.

### 7.3 Gas-limit ramp

Each non-initial block applies:

```text
next_limit = min(current_limit + floor(current_limit / 1024), target_limit)
```

The ramp never lowers a gas limit. Position 0 never ramps because both persistent
updates are parent-derived, so the first increase appears at position 1. There is
no activation step: Glamsterdam is live from position 0 in every run.

### 7.4 Eligibility and ordering

A transaction is eligible when `max_fee_per_gas >= base_fee`. Ineligible items
remain in the mempool.

```text
legacy/access list tip = max_fee_per_gas - base_fee

dynamic-fee tip = min(
    max_priority_fee_per_gas,
    max_fee_per_gas - base_fee,
)
```

Eligible items sort by effective tip descending, then by ascending
`(arrival_step, window_instance, source_block_number, tx_index, replica_index)`.
The implementation uses stable tip sorting and tip-value tranches, but these are
exact performance optimizations, not approximations.

### 7.5 Two-dimensional fill

Walk the ordered candidates and include a transaction only if both totals remain
within the same gas limit:

```text
used_execution + tx_execution <= gas_limit
AND
used_state + tx_state <= gas_limit
```

The rule is **skip and continue**. A transaction that does not fit stays pending,
and a later, smaller transaction may still be included.

`bottleneck_dimension` is whichever included-gas total is larger. Ties are
reported as `execution`; `none` is used only when both totals are zero.

### 7.6 Backlog

Backlog is measured after inclusion at the current block's base fee. Counts,
execution gas, and state gas are each reported as total, eligible, and
fee-ineligible.

Actual sampled arrivals are also recorded per step. They cannot be reconstructed
exactly from the cohort and multiplier because fractional sampling accepts an
overshoot.

## 8. Outputs

All files are written under `output_dir`.

### 8.1 Per-step data

`per_step.parquet` contains one row per simulated block and uses the exact order
in `schemas.PER_STEP_COLUMNS`. Parquet only: a CSV copy of the same frame is
~3.5x the bytes, ~11x slower to write, and lossy on reload, so it cost wall clock
while being the worse copy. The small summaries stay CSV because they are meant
to be read directly.

| Group | Columns |
| --- | --- |
| identity | `arrival_mode`, `run_index`, `aggregate_elasticity`, `demand_level`, `bootstrap_window_blocks`, `simulation_position`, `source_block_number`, `window_instance`, `position_in_window` |
| demand | `demand_price_signal`, `cohort_anchor_price`, `realized_demand_multiplier`, `demand_multiplier_clamped` |
| capacity | `base_fee_per_gas`, `base_fee_clamped`, `gas_limit`, `gas_used`, `block_execution_gas_used`, `block_state_gas_used`, `execution_utilization`, `state_utilization`, `bottleneck_dimension` |
| included | `included_tx_count`, `sender_gas_used`, `priority_fees_wei` |
| arrivals | `arrived_tx_count`, `arrived_execution_gas`, `arrived_state_gas` |
| backlog | `backlog_tx_count`, `backlog_eligible_tx_count`, `backlog_fee_ineligible_tx_count`, plus execution/state gas with the same total/eligible/fee-ineligible split |

Units are wei and gas unless a name states otherwise. `priority_fees_wei` is
`float64` because plausible block totals can exceed signed `int64`.

### 8.2 Scenario summary

`scenario_summary.csv` groups by `arrival_mode`, `aggregate_elasticity`,
`demand_level`, and `bootstrap_window_blocks`. Its metrics are
`execution_saturated_share`, `state_saturated_share`,
`median_demand_multiplier`, `max_demand_multiplier`,
`multiplier_clamped_share`, `base_fee_clamped_share`, `runs`, `ended_empty_share`,
`median_terminal_backlog_txs`, `max_terminal_backlog_txs`, and
`median_final_base_fee_gwei`. Saturation means utilisation at or above 99%.

### 8.3 Other artifacts

- `replay_outcome_summary.csv`: the whole trace broken down by replay outcome (§4.1). Nothing is excluded; this reports what is being simulated.
- `window_length.csv`: autocorrelation evidence for bootstrap `L`.
- `manifest.json`: resolved config and grid, seeds, source range, initial base fee,
  library versions, timings, output paths, and the standing caveat.

A run writes data and nothing else -- no figures, no analysis. Reading the results
belongs in `notebooks/`, where `analysis.bands.aggregate_bands` collapses the
bootstrap runs into per-position p10-p90 bands. The historical path is a separate
reference and is never folded into those bands.

## 9. Assumptions

State these whenever reporting results from this repository.

1. **Frozen execution.** Gas, success, and effects come from the upstream replay
   and do not change after sampling or reordering. Nonces, balances, state
   dependencies, and bundles are not enforced.
2. **Cohort timing.** Every transaction historically included in source block `n`
   becomes available at one simulated step because submission times are unknown.
3. **Included-demand trace.** The data omits never-included, replaced, and dropped
   transactions. Demand level 1 is therefore a lower-bound scenario, not total
   latent demand.
4. **Aggregate demand.** One elasticity controls total `execution + state` gas. It
   does not model substitution between the two resources.
5. **Extrapolated demand curve.** A constant elasticity estimated from daily data
   is extended into a much lower-price regime. Clamp flags reveal when the chosen
   bounds determine the response.
6. **Demand sees price, not queue delay.** A growing backlog does not directly
   deter new arrivals. Terminal backlog is therefore not an equilibrium claim.
7. **Adapted bids, not sender rebidding.** Fee-cap transforms preserve historical
   eligibility and absolute tips; they do not model new willingness to pay.
8. **Different demand and capacity measures.** Demand is calibrated on
   `execution + state`; inclusion is constrained separately in both dimensions
   and summarized by their maximum.
9. **Observed transaction mix.** Results are sensitivity estimates under that mix,
   not forecasts of how applications or users adapt to 200M blocks.

## 10. How to interpret results

### Check these first

1. `multiplier_clamped_share`: a material value means the bound, not evidence,
   controls part of the demand response.
2. `median_demand_multiplier` and `max_demand_multiplier`: these show how far the
   loop moved from the configured demand level.
3. `ended_empty_share` and terminal backlog: these say whether **price alone**,
   under this model, restrained demand enough that nothing was left queued at the
   last arrival. They are not a drain result -- a run stops with its last
   arrival, so they describe the state reached, not how fast a queue would clear.

### Read comparisons, not isolated cells

- `demand_level` changes latent quantity at the anchor price.
- `aggregate_elasticity` changes sensitivity to price. Zero is not swept: with no
  feedback and bid adaptation on, demand cannot be shed at `demand_level > 1` and
  the base fee runs to `MAX_BASE_FEE` (§7.2). It remains available as a
  single-scenario setting, where `demand_level <= 1` keeps it meaningful.
- `L` changes which dependence structure the bootstrap preserves. If conclusions
  change across `L`, consult `window_length.csv`.
- The historical line shows the original cohort ordering under simulated rules.
  It is a reference path, not observed historical outcomes and not a confidence
  interval.

Execution and state utilisation share a denominator and do not add to 100%.
Compare `bottleneck_dimension` or saturation shares to learn which resource binds.
Check `replay_outcome_summary.csv`, because the share of state gas sitting in
`schedule_gas_rescuable` rows says how much of the simulated demand depends on
senders raising their signed gas limits.

### Supported questions

- Under a stated demand level and elasticity, how do base fee and backlog evolve?
- Does eligible backlog clear while the gas limit ramps?
- Under the observed transaction mix, how often does state or execution bind?
- Which conclusions are robust across demand assumptions and bootstrap windows?

### Unsupported questions

- What will the actual base fee be at a 200M gas limit?
- How will demand substitute between execution and state gas?
- Would reordered transactions still execute successfully on the resulting state?
- How will users rebid or react to confirmation delay?

## 11. Reproducibility

- One master seed deterministically derives independent bootstrap and demand
  streams. Bootstrap run `k` is shared across scenarios.
- Inclusion order is fully specified, and performance optimizations are tested
  against literal sequential implementations.
- Dataset fields and block range are pinned and stored in cache provenance. Keep
  the cache sidecars because the upstream table changes over time.
- `manifest.json` records resolved parameters, derived seeds, input range, initial
  base fee, library versions, timings, outputs, and the caveat shown by the CLI.

A run is reproducible given the same cached inputs, configuration, seed, and
library versions.
