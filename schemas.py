"""Column contracts shared by the loaders, the engine, and downstream analysis.

Keeping these in one place lets the data layer, the simulator, and the analysis
layer be developed and tested against the same frame shapes -- including against
synthetic data, since the real `gas_analysis_tx_gas_result` table does not exist
yet.
"""

from __future__ import annotations

# --- Replay rows: gas_analysis_tx_gas_result -------------------------------------

TX_GAS_RESULT_FIELDS = (
    "schedule_config_hash",
    "block_number",
    "tx_index",
    "tx_hash",
    "tx_type",
    "tx_gas_limit",
    "max_fee_per_gas",
    "max_priority_fee_per_gas",
    "baseline_success",
    "baseline_gas_used",
    "baseline_total_gas_spent",
    "schedule_success",
    "schedule_gas_used",
    "schedule_total_gas_spent",
    "schedule_gas_refunded",
    "schedule_floor_gas",
    "schedule_state_gas_spent",
    "schedule_intrinsic_gas",
    "min_multiplier_to_succeed",
)

TX_GAS_RESULT_PROVENANCE_FIELDS = (
    "block_hash",
    "block_timestamp",
    "producer_schema_version",
    "producer_git_commit",
    "replay_semantics",
)

# Added by the loader, not present in the source table.
TX_GAS_RESULT_DERIVED_FIELDS = ("state_gas", "execution_gas")

TX_GAS_RESULT_COLUMNS = (
    TX_GAS_RESULT_FIELDS + TX_GAS_RESULT_PROVENANCE_FIELDS + TX_GAS_RESULT_DERIVED_FIELDS
)

# --- Block headers: canonical_execution_block ------------------------------------

BLOCK_HEADER_COLUMNS = (
    "block_number",
    "gas_used",
    "gas_limit",
    "base_fee_per_gas",
)

# --- Per-step simulation output ---------------------------------------------------

PER_STEP_IDENTITY_COLUMNS = (
    "run_index",
    "aggregate_elasticity",
    "demand_level",
    "composition_pool_blocks",
    "simulation_position",
    # First source block of the step's composition pool. A step has no single
    # source block any more -- its arrivals are sampled across the whole pool --
    # so this identifies the *draw*, not the arrivals. The arrivals themselves
    # are reproducible from `run_index` plus the seeds.
    "pool_start_block",
)

# The demand axes are scenario-level (identity); the multiplier they produce is a
# per-step outcome, hence `realized_` -- it moves with the price signal within a
# single run and is not a scenario key.
#
# `demand_anchor_price` is constant within a run (the trace-level reference the
# curve is anchored at) and is carried per row anyway, so a per-step frame can be
# read back without its manifest and the multiplier recomputed from it alone.
PER_STEP_DEMAND_COLUMNS = (
    "demand_price_signal",
    "demand_anchor_price",
    "realized_demand_multiplier",
    "demand_multiplier_clamped",
)

PER_STEP_METRIC_COLUMNS = (
    "base_fee_per_gas",
    # Marks a step sitting at `config.MAX_BASE_FEE`. Like
    # `demand_multiplier_clamped`, it says the bound rather than the mechanism set
    # this value, so any scenario with a non-zero share has uninterpretable fees.
    "base_fee_clamped",
    "gas_limit",
    "block_execution_gas_used",
    "block_state_gas_used",
    "included_tx_count",
    "sender_gas_used",
    "priority_fees_wei",
    "arrived_tx_count",
    # Arrivals are wholly sampled, so they are not recoverable from the path plus
    # a scalar multiplier -- they have to be recorded.
    "arrived_execution_gas",
    "arrived_state_gas",
    "backlog_tx_count",
    "backlog_eligible_tx_count",
    "backlog_execution_gas",
    "backlog_state_gas",
    "backlog_eligible_execution_gas",
    "backlog_eligible_state_gas",
)

PER_STEP_COLUMNS = (
    PER_STEP_IDENTITY_COLUMNS + PER_STEP_DEMAND_COLUMNS + PER_STEP_METRIC_COLUMNS
)
