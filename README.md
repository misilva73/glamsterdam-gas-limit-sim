# Glamsterdam gas-limit simulation

What happens to base fee, block utilisation, and the mempool when the Ethereum
gas limit ramps from Fusaka's 60M toward 200M under Glamsterdam gas repricing?

This simulator takes real historical mainnet transactions re-executed under
Glamsterdam rules (via the `reth-research` replay), replays them as arrival
cohorts against a gas limit climbing at 1/1024 per block, and reports per-block
outcomes across a grid of demand scenarios with Monte Carlo bootstrap bands.

Two things distinguish it from a one-dimensional gas model:

- Glamsterdam charges **state gas** (EIP-8037) alongside execution gas, so every
  block has two simultaneous capacity constraints and
  `gas_used = max(execution, state)`. Which dimension binds is one of the outputs.
- Demand **responds to price**. How much demand arrives at a step comes from an
  isoelastic demand model calibrated on empirical elasticities and anchored per
  cohort on the price that cohort was historically observed at. Base fee → demand
  → utilisation → base fee is a closed loop.

**`METHODOLOGY.md` is the deep dive**: data sources, inputs, the demand model, the
engine, every output column, the assumptions, and how to read a result. Read it
before quoting a number.

## Quickstart

The simulator reads its replay rows from ClickHouse, so it needs Xatu credentials
in a gitignored `secrets.json` at the repo root:

```json
{"xatu_username": "...", "xatu_password": "..."}
```

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_simulation.py \
    --analysis-config-hash <hash> --block-range <first> <last> \
    --reference-start-block <block> --num-runs 20

.venv/bin/python -m pytest -q     # offline, no credentials needed
```

`--analysis-config-hash` is mandatory: it pins one reth replay configuration, so a
run can never silently average two of them. The loader lists the datasets the
table actually holds if the pinned one returns no rows.

The test suite needs neither credentials nor network — it generates synthetic
source-shaped frames in `tests/dummy.py` and injects them at the two fetch seams
(`tests/conftest.py`), and an autouse fixture fails any test that tries to open a
real connection.

## Flags

Same list as `run_simulation.py --help`. Defaults come from `SimConfig` and
`SimulationGrid` in `config.py`; what a given run actually resolved to is recorded
in its `manifest.json`.

**Source data** — which replay rows to simulate.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--analysis-config-hash HASH` | Which reth replay config to read. **Required**, so a run can never silently mix datasets. | — |
| `--schedule-name NAME` | Repricing schedule whose gas is simulated. | `amsterdam` |
| `--schedule-config-hash HASH` | Pin one revision of `--schedule-name`. | whatever the table holds |
| `--chain-id N` | Chain of the replay rows. | `1` |
| `--block-range FIRST LAST` | Inclusive source block range. Required — unbounded scans of the replay table are refused. | — |
| `--reference-start-block N` | First block of the historical path; its **parent** supplies the starting base fee. | first block of the range |
| `--cache-dir PATH` | Parquet cache for fetched data. | `data/cache` |

**Simulation** — how one path is run.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--horizon N` | Arrival steps, i.e. how many cohorts are fed in. | trace length |
| `--arrival-mode {historical,moving_block_bootstrap}` | `historical` replays the trace once, with no bands; the bootstrap resamples cohort windows. | `moving_block_bootstrap` |
| `--num-runs N` | Bootstrap paths per grid cell — the width of the p10–p90 band. | `20` |
| `--seed N` | Master seed; every stream derives from it, so a simulation is reproducible. | `20260831` |
| `--starting-base-fee WEI` | Override the base fee at step 0. | parent header |
| `--fusaka-gas-limit N` | Gas limit at step 0. | `60,000,000` |
| `--glamsterdam-gas-limit N` | Ceiling the 1/1024 ramp climbs toward. | `200,000,000` |

**Demand model** — how much demand arrives, and what it is willing to pay.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--price-ema-blocks N` | Span of the EMA smoothing the effective gas price the model reacts to. The elasticities are daily, so this is hours, not blocks. | `300` |
| `--multiplier-bounds LOW HIGH` | Clamp on the **price response** `(p/p0)**-e`, not on the product with the demand level. `p**-e` is unbounded as the price falls and the elasticity was estimated over a narrow price range, so a saturating response is capped and flagged per step; a deliberately high `--demand-levels` is not capped. | `0.05 20` |
| `--no-bid-adaptation` | Freeze historical fee caps instead of repricing them from their own block's base fee to the simulated one. Makes the fee filter, not the elasticity, set how much demand is eligible. | off |

**Simulation grid** — every combination of the three axes is a scenario, each run
`--num-runs` times. The two demand axes multiply, so the defaults are 27 cells;
narrow them explicitly on a long trace.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--elasticities E [E ...]` | Demand-*shape* axis: aggregate price elasticity. `0` is a flat multiplier with no price response, and is no longer swept by default: with bid adaptation on it cannot shed demand above `--demand-levels 1`, so the base fee runs to the `MAX_BASE_FEE` ceiling and `base_fee_clamped_share` goes non-zero. | `0.10 0.175 0.28` |
| `--demand-levels A [A ...]` | Demand-*level* axis: latent-demand multiplier at the anchor price, standing in for never-included and secular-growth demand. | `1 1.5 2` |
| `--window-blocks L [L ...]` | Bootstrap axis: length in cohorts of each resampled window. | `16 32 64` |
| `--no-historical` | Skip the historical reference path (bands only). | off |
| `--output-dir PATH` | Parent of the timestamped directory this run writes. | `output/` |
| `--resume STAMP` | Continue an interrupted run: the name of its directory under `--output-dir`. Cells already checkpointed there are skipped and the rest are written into the same directory. Refused unless the config, grid, seed, and resolved trace all match the run being resumed. | off |

## Output

Every run creates its own UTC-timestamped directory under `--output-dir`
(`output/20260904T083556Z/`), so a new sweep never writes into an older one.
Every column is documented in `METHODOLOGY.md` §8.

```text
output/20260904T083556Z/
├── per_step/
│   ├── e0.1_d1.0_w16_bootstrap.parquet     every bootstrap run of that cell
│   ├── e0.1_d1.0_w16_historical.parquet    that cell's reference path
│   └── ...                                 one pair per grid cell
├── scenario_summary.csv
├── replay_outcome_summary.csv
└── manifest.json
```

- `per_step/` — one row per simulated block: base fee, gas limit,
  header-equivalent gas used, execution/state gas and utilisation, bottleneck
  dimension, included transactions, sender-facing gas, priority fees, arrivals by
  dimension, eligible / fee-ineligible backlog by count and both gas dimensions,
  and the demand model's own state. Written **one grid cell at a time, as each
  cell finishes**: a sweep never holds more than the cell in flight, and an
  interrupted run keeps every cell that completed. Read the whole sweep back as
  one frame with `pd.read_parquet("output/<stamp>/per_step")` — every part shares
  the `schemas.PER_STEP_COLUMNS` schema and carries its own grid identity
  columns, so no partition decoding is needed.
- `scenario_summary.csv` — per demand scenario: saturation share by dimension,
  median and max realized multiplier, multiplier and base-fee clamped shares,
  whether anything was left queued at the last arrival, terminal backlog, final
  base fee. Rows are appended as each cell is checkpointed, in grid order.
- `replay_outcome_summary.csv` — the whole trace broken down by how each row fared
  in the replay. Nothing is excluded; this reports what is being simulated, and in
  particular how much state gas sits in gas-rescuable rows.
- `manifest.json` — resolved config, grid, seeds, library versions, timings, and
  `completed`. Written twice: once before the first cell, so an interrupted run
  can be resumed, and again at the end with `completed: true`.

### Resuming an interrupted sweep

A long sweep that dies keeps every cell it checkpointed. Restart it with the
directory name and it picks up where it stopped:

```bash
.venv/bin/python run_simulation.py <same flags as before> \
    --resume 20260904T083556Z
```

Cells are simulated in a fixed order, so what is on disk is a prefix of the grid;
`--resume` counts the complete cells and simulates the rest into the same
directory. A cell caught between its parquet parts and its summary row is
simulated again rather than trusted. Resuming is refused — before the load, so it
fails in milliseconds — if the config, grid, or seed differs from the run being
resumed, or if the trace itself moved (the replay table grows, so the same block
range can resolve to a longer horizon than the finished cells were run against).
Either way the answer is a fresh run, not a mixed directory.

## Read this before quoting any number

This is a **frozen-trace simulation, not counterfactual EVM replay**:

- Repriced gas and success are fixed after reordering. Nonce chains, balances,
  state dependencies, and bundles are not enforced.
- Bid adaptation is a **rescale of the observed fee distribution**, not a model of
  how senders would actually rebid.
- The demand model is **aggregate**: one elasticity on total gas, so it cannot say
  how repricing shifts demand *between* the two dimensions.
- The elasticities come from **daily** aggregates over base fees spanning roughly
  0.1–10 gwei, and a 200M gas limit pushes the price far below that. Every cell is
  an extrapolation; `demand_multiplier_clamped` and `multiplier_clamped_share`
  record when it left the supportable range.
- Demand responds to **price, not queue depth**, so terminal backlog is not an
  equilibrium statement.
- The trace holds only historically *included* transactions, so **demand level 1x
  is an included-demand lower bound**. That is what `--demand-levels` above 1 is
  for.

Results are sensitivity estimates under the observed transaction mix, **not
equilibrium forecasts** for a 200M gas limit. `METHODOLOGY.md` §9–10 has the full
list and what each caveat costs you.

## Documentation

- **`METHODOLOGY.md`** — how the simulation works: data sources, inputs, the
  workload and demand models, the engine, outputs, assumptions, interpretation.
- **`AGENTS.md`** — the working reference for changing the code: module ownership,
  the domain rules and their traps, live dataset status, measured performance
  envelope, and known gaps.
