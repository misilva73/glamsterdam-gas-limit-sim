"""Collapse bootstrap runs into per-position quantile bands.

Pure aggregation, no plotting: a simulation run writes raw per-step data and
nothing else, so banding happens downstream in whatever reads it (see
`notebooks/`).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

SCENARIO_KEYS = ("aggregate_elasticity", "demand_level", "bootstrap_window_blocks")
BAND_GROUP_KEYS = (*SCENARIO_KEYS, "simulation_position")


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
    return tidy[[*BAND_GROUP_KEYS, "metric", "statistic", "value"]].sort_values(
        [*BAND_GROUP_KEYS, "metric", "statistic"], ignore_index=True
    )


def _quantile_label(quantile: float) -> str:
    return f"p{round(quantile * 100)}"
