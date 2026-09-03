"""Column contracts shared by the loaders, the engine, and the plots.

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

# --- Workload items fed to the engine --------------------------------------------
# One row per (source transaction x demand replica), tagged with its arrival step.

WORKLOAD_COLUMNS = (
    "arrival_step",
    "window_instance",
    "source_block_number",
    "tx_index",
    "tx_hash",
    "replica_index",
    "tx_type",
    "max_fee_per_gas",
    "max_priority_fee_per_gas",
    "execution_gas",
    "state_gas",
    "schedule_gas_used",
    # Base fee of the item's own source block: what the bid rescale prices from.
    "anchor_base_fee",
)

# Deterministic inclusion order after the tip-descending sort. Documented here
# because reproducibility of a run depends on this exact tie-break chain.
INCLUSION_TIEBREAK_COLUMNS = (
    "arrival_step",
    "window_instance",
    "source_block_number",
    "tx_index",
    "replica_index",
)

# --- Per-step simulation output ---------------------------------------------------

PER_STEP_IDENTITY_COLUMNS = (
    "arrival_mode",
    "run_index",
    "aggregate_elasticity",
    "demand_level",
    "bootstrap_window_blocks",
    "simulation_position",
    "source_block_number",
    "window_instance",
    "position_in_window",
)

# The demand axes are scenario-level (identity); the multiplier they produce is a
# per-step outcome, hence `realized_` -- it moves with the price signal within a
# single run and is not a scenario key.
PER_STEP_DEMAND_COLUMNS = (
    "demand_price_signal",
    "cohort_anchor_price",
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
    "gas_used",
    "block_execution_gas_used",
    "block_state_gas_used",
    "execution_utilization",
    "state_utilization",
    "bottleneck_dimension",
    "included_tx_count",
    "sender_gas_used",
    "priority_fees_wei",
    "arrived_tx_count",
    # Arrivals are sampled, so they are no longer recoverable from the path plus
    # a scalar multiplier -- they have to be recorded.
    "arrived_execution_gas",
    "arrived_state_gas",
    "backlog_tx_count",
    "backlog_eligible_tx_count",
    "backlog_fee_ineligible_tx_count",
    "backlog_execution_gas",
    "backlog_state_gas",
    "backlog_eligible_execution_gas",
    "backlog_eligible_state_gas",
    "backlog_fee_ineligible_execution_gas",
    "backlog_fee_ineligible_state_gas",
)

PER_STEP_COLUMNS = (
    PER_STEP_IDENTITY_COLUMNS + PER_STEP_DEMAND_COLUMNS + PER_STEP_METRIC_COLUMNS
)

BOTTLENECK_EXECUTION = "execution"
BOTTLENECK_STATE = "state"
BOTTLENECK_NONE = "none"
