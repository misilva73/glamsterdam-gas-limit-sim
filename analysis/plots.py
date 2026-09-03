"""Simulation charts: bootstrap bands with the historical trajectory overlaid.

Every figure is a grid of the simulation axes -- one column per `demand_level`,
one row block per `(aggregate_elasticity, bootstrap_window_blocks)` pair -- so a
scenario sweep can be read side by side. Bootstrap runs collapse into per
`simulation_position` quantile bands; the matching historical reference path is
drawn on top as a dashed line.

The two demand axes multiply, so sweeping all three axes at their defaults makes
a very tall figure. Narrow the grid on the command line when reading results
rather than plotting every cell at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: figures are written, never shown

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from schemas import BOTTLENECK_EXECUTION, BOTTLENECK_NONE, BOTTLENECK_STATE

SCENARIO_KEYS = ("aggregate_elasticity", "demand_level", "bootstrap_window_blocks")
BAND_GROUP_KEYS = (*SCENARIO_KEYS, "simulation_position")

WEI_PER_GWEI = 1e-9
PER_MILLION = 1e-6

BOTTLENECK_ORDER = (BOTTLENECK_EXECUTION, BOTTLENECK_STATE, BOTTLENECK_NONE)
# The bottleneck dimension is categorical per block, so its per-position share is
# jagged at small run counts; both the bands and the single historical path get the
# same rolling mean, which leaves the stacked shares summing to 1.
MIX_SMOOTHING_BLOCKS = 32

FIGURE_DPI = 150


def use_style() -> None:
    """Shared seaborn look; idempotent, so any entry point may call it."""
    sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")


def aggregate_bands(
    per_step: pd.DataFrame,
    value_columns: Sequence[str],
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    also_min_max: bool = True,
) -> pd.DataFrame:
    """Collapse bootstrap runs into per-position bands, tidy long.

    Returns one row per (elasticity, demand level, window length, position,
    metric, statistic), where statistic is `p10`/`p50`/`p90` (per `quantiles`)
    plus `min`/`max`.

    Expects a single `arrival_mode`: folding the historical reference into the
    bootstrap bands would bias them by exactly the path they are compared against.
    """
    modes = set(per_step.get("arrival_mode", pd.Series(dtype=object)).unique())
    if len(modes) > 1:
        raise ValueError(
            f"aggregate_bands expects one arrival_mode, got {sorted(modes)}; "
            "split the frame before banding"
        )
    value_columns = list(value_columns)
    grouped = per_step.groupby(list(BAND_GROUP_KEYS), observed=True)[value_columns]

    statistics = {_quantile_label(q): grouped.quantile(q) for q in quantiles}
    if also_min_max:
        statistics["min"] = grouped.min()
        statistics["max"] = grouped.max()

    tidy = pd.concat(
        {
            statistic: wide.rename_axis(columns="metric").stack().rename("value")
            for statistic, wide in statistics.items()
        },
        names=["statistic"],
    ).reset_index()
    return (
        tidy[[*BAND_GROUP_KEYS, "metric", "statistic", "value"]]
        .sort_values([*BAND_GROUP_KEYS, "metric", "statistic"], ignore_index=True)
    )


class Line(NamedTuple):
    """One metric drawn as a band plus its historical overlay."""

    metric: str
    label: str


class Panel(NamedTuple):
    """One row of the figure grid: either metric bands or the bottleneck mix."""

    ylabel: str
    series: tuple[Line, ...] = ()
    scale: float = 1.0
    mix: bool = False


def plot_base_fee(
    bootstrap_per_step: pd.DataFrame, historical_per_step: pd.DataFrame, out_path: Path
) -> Path:
    return _band_grid(
        bootstrap_per_step,
        historical_per_step,
        [Panel("base fee (gwei)", (Line("base_fee_per_gas", "base fee"),), WEI_PER_GWEI)],
        out_path,
        suptitle="Base fee under Glamsterdam gas-limit ramp",
    )


def plot_utilization(
    bootstrap_per_step: pd.DataFrame, historical_per_step: pd.DataFrame, out_path: Path
) -> Path:
    return _band_grid(
        bootstrap_per_step,
        historical_per_step,
        [
            Panel(
                "utilization (% of gas limit)",
                (
                    Line("execution_utilization", "execution"),
                    Line("state_utilization", "state"),
                ),
                scale=100.0,
            ),
            Panel(
                f"bottleneck share of blocks\n({MIX_SMOOTHING_BLOCKS}-block mean)", mix=True
            ),
        ],
        out_path,
        suptitle="Utilization by gas dimension and binding constraint",
    )


def plot_gas_limit_ramp(
    bootstrap_per_step: pd.DataFrame, historical_per_step: pd.DataFrame, out_path: Path
) -> Path:
    return _band_grid(
        bootstrap_per_step,
        historical_per_step,
        [
            Panel(
                "gas (million)",
                (Line("gas_limit", "gas limit"), Line("gas_used", "gas used")),
                PER_MILLION,
            )
        ],
        out_path,
        suptitle="Gas-limit ramp (1/1024 per block) against realized gas used",
    )


def plot_backlog(
    bootstrap_per_step: pd.DataFrame, historical_per_step: pd.DataFrame, out_path: Path
) -> Path:
    eligible_vs_ineligible = [
        ("backlog (transactions)", "tx_count", 1.0),
        ("backlog execution gas (million)", "execution_gas", PER_MILLION),
        ("backlog state gas (million)", "state_gas", PER_MILLION),
    ]
    panels = [
        Panel(
            ylabel,
            (
                Line(f"backlog_eligible_{suffix}", "eligible"),
                Line(f"backlog_fee_ineligible_{suffix}", "fee-ineligible"),
            ),
            scale,
        )
        for ylabel, suffix, scale in eligible_vs_ineligible
    ]
    return _band_grid(
        bootstrap_per_step,
        historical_per_step,
        panels,
        out_path,
        suptitle="Mempool backlog, split by fee eligibility at the prevailing base fee",
    )


def plot_demand_response(
    bootstrap_per_step: pd.DataFrame, historical_per_step: pd.DataFrame, out_path: Path
) -> Path:
    """The demand model's own state: what price it saw and what it did about it."""
    return _band_grid(
        bootstrap_per_step,
        historical_per_step,
        [
            Panel(
                "price signal (gwei)",
                (
                    Line("demand_price_signal", "EMA effective price"),
                    Line("cohort_anchor_price", "cohort anchor"),
                ),
                WEI_PER_GWEI,
            ),
            Panel(
                "realized demand multiplier",
                (Line("realized_demand_multiplier", "multiplier"),),
            ),
            Panel(
                "arrived gas (million)",
                (
                    Line("arrived_execution_gas", "execution"),
                    Line("arrived_state_gas", "state"),
                ),
                PER_MILLION,
            ),
        ],
        out_path,
        suptitle="Demand model: price signal, multiplier, and induced arrivals",
    )


SIMULATION_FIGURES = {
    "base_fee": plot_base_fee,
    "utilization": plot_utilization,
    "gas_limit_ramp": plot_gas_limit_ramp,
    "backlog": plot_backlog,
    "demand_response": plot_demand_response,
}


def plot_simulation(
    bootstrap_per_step: pd.DataFrame, historical_per_step: pd.DataFrame, out_dir: Path
) -> list[Path]:
    """Write every simulation figure into `out_dir` and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot(bootstrap_per_step, historical_per_step, out_dir / f"{name}.png")
        for name, plot in SIMULATION_FIGURES.items()
    ]


# --- internals -------------------------------------------------------------


def _quantile_label(quantile: float) -> str:
    return f"p{quantile * 100:g}"


def _band_grid(
    bootstrap: pd.DataFrame,
    historical: pd.DataFrame,
    panels: Sequence[Panel],
    out_path: Path,
    *,
    suptitle: str,
) -> Path:
    if bootstrap.empty:
        raise ValueError("nothing to plot: the per-step frame is empty")
    use_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    levels = sorted(bootstrap["demand_level"].unique())
    elasticities = sorted(bootstrap["aggregate_elasticity"].unique())
    windows = sorted(bootstrap["bootstrap_window_blocks"].unique())
    # The demand axes multiply: levels are columns, and each (elasticity, window)
    # pair gets its own block of panel rows.
    rows = [
        (elasticity, window, panel)
        for elasticity in elasticities
        for window in windows
        for panel in panels
    ]

    metrics = sorted({line.metric for panel in panels for line in panel.series})
    bands = aggregate_bands(bootstrap, metrics) if metrics else None

    fig, axes = plt.subplots(
        len(rows),
        len(levels),
        figsize=(1.0 + 4.0 * len(levels), 1.0 + 2.7 * len(rows)),
        squeeze=False,
        sharex="col",
        sharey="row",
    )
    colors = sns.color_palette("colorblind")

    for row, (elasticity, window, panel) in enumerate(rows):
        for column, level in enumerate(levels):
            ax = axes[row][column]
            scenario = _scenario_slice(bootstrap, elasticity, level, window)
            reference = _historical_slice(historical, elasticity, level, window)

            if panel.mix:
                _draw_bottleneck_mix(ax, scenario, reference)
            else:
                scenario_bands = _scenario_slice(bands, elasticity, level, window)
                for line, color in zip(panel.series, colors):
                    _draw_band(ax, scenario_bands, line, color=color, scale=panel.scale)
                    _draw_historical(ax, reference, line, color=color, scale=panel.scale)

            if row == 0:
                ax.set_title(f"demand level {level:g}x")
            if column == 0:
                prefix = _row_label(elasticity, window, len(elasticities), len(windows))
                ax.set_ylabel(f"{prefix}{panel.ylabel}")
                ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
            if row == len(rows) - 1:
                ax.set_xlabel("simulation position (blocks)")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


def _row_label(
    elasticity: float, window: int, num_elasticities: int, num_windows: int
) -> str:
    """Name only the axes that actually discriminate between rows."""
    parts = []
    if num_elasticities > 1:
        parts.append(f"e={elasticity:g}")
    if num_windows > 1:
        parts.append(f"L={window}")
    return f"{', '.join(parts)}\n" if parts else ""


def _scenario_slice(
    frame: pd.DataFrame, elasticity: float, level: float, window: int
) -> pd.DataFrame:
    return frame[
        (frame["aggregate_elasticity"] == elasticity)
        & (frame["demand_level"] == level)
        & (frame["bootstrap_window_blocks"] == window)
    ]


def _historical_slice(
    historical: pd.DataFrame, elasticity: float, level: float, window: int
) -> pd.DataFrame:
    """The reference path for this scenario.

    A historical run has no sampled windows, so the engine may tag it with any
    `bootstrap_window_blocks`; match on it only when it actually discriminates.
    """
    if historical is None or historical.empty:
        return pd.DataFrame()
    matched = historical[
        (historical["aggregate_elasticity"] == elasticity)
        & (historical["demand_level"] == level)
    ]
    by_window = matched[matched["bootstrap_window_blocks"] == window]
    return by_window if not by_window.empty else matched


def _draw_band(
    ax, bands: pd.DataFrame, line: Line, *, color, scale: float
) -> None:
    band = bands[bands["metric"] == line.metric]
    if band.empty:
        return
    wide = (
        band.pivot(index="simulation_position", columns="statistic", values="value")
        .sort_index()
        * scale
    )
    if {"p10", "p90"} <= set(wide.columns):
        ax.fill_between(
            wide.index,
            wide["p10"],
            wide["p90"],
            color=color,
            alpha=0.25,
            linewidth=0,
            label=f"{line.label} p10-p90",
        )
    if {"min", "max"} <= set(wide.columns):
        ax.plot(wide.index, wide["min"], color=color, lw=0.5, alpha=0.6, ls=":")
        ax.plot(
            wide.index,
            wide["max"],
            color=color,
            lw=0.5,
            alpha=0.6,
            ls=":",
            label=f"{line.label} min/max",
        )
    if "p50" in wide.columns:
        ax.plot(wide.index, wide["p50"], color=color, lw=1.6, label=f"{line.label} median")


def _draw_historical(
    ax, reference: pd.DataFrame, line: Line, *, color, scale: float
) -> None:
    if reference.empty or line.metric not in reference.columns:
        return
    path = reference.sort_values("simulation_position")
    ax.plot(
        path["simulation_position"],
        path[line.metric] * scale,
        color=color,
        lw=1.8,
        ls="--",
        label=f"{line.label} historical",
    )


def _draw_bottleneck_mix(ax, scenario: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Share of bootstrap runs bound by each gas dimension at every position."""
    if scenario.empty:
        return
    mix = (
        pd.crosstab(
            scenario["simulation_position"], scenario["bottleneck_dimension"], normalize="index"
        )
        .reindex(columns=list(BOTTLENECK_ORDER), fill_value=0.0)
        .rolling(MIX_SMOOTHING_BLOCKS, min_periods=1)
        .mean()
    )
    ax.stackplot(
        mix.index,
        mix.to_numpy().T,
        labels=[f"{name} bound" for name in mix.columns],
        alpha=0.7,
    )
    ax.set_ylim(0, 1)

    if not reference.empty:
        path = reference.sort_values("simulation_position")
        state_bound = (
            path["bottleneck_dimension"]
            .eq(BOTTLENECK_STATE)
            .rolling(MIX_SMOOTHING_BLOCKS, min_periods=1)
            .mean()
        )
        ax.plot(
            path["simulation_position"],
            state_bound,
            color="black",
            lw=1.4,
            ls="--",
            label=f"historical state-bound ({MIX_SMOOTHING_BLOCKS}-block mean)",
        )

