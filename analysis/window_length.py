"""Choosing the moving-block bootstrap block length `L`.

`L` must be long enough to carry the dependence that matters -- congestion
persists across blocks, so cohort size, gas mix, and fee level are all
autocorrelated -- and short enough that resampling still produces genuinely new
paths. The decision is argued from the empirical autocorrelation of the cohort
summary series rather than picked: for each series we report the lag at which
the ACF first decays past 1/e, the lag at which it first falls inside the 95%
white-noise band, and the integral timescale, then say which of the candidate
`L` values covers them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

DECAY_THRESHOLD = 1.0 / np.e
WHITE_NOISE_Z = 1.96

SUMMARY_SERIES = (
    "tx_count",
    "execution_gas",
    "state_gas",
    "median_max_fee_per_gas",
    "mean_max_fee_per_gas",
    "median_max_priority_fee_per_gas",
)


def cohort_summary(tx_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per source block: cohort size, both gas dimensions, fee level.

    Indexed by `block_number` in trace order, so every column is a time series
    ready for `autocorrelation`.
    """
    summary = tx_frame.groupby("block_number").agg(
        tx_count=("tx_hash", "size"),
        execution_gas=("execution_gas", "sum"),
        state_gas=("state_gas", "sum"),
        median_max_fee_per_gas=("max_fee_per_gas", "median"),
        mean_max_fee_per_gas=("max_fee_per_gas", "mean"),
        median_max_priority_fee_per_gas=("max_priority_fee_per_gas", "median"),
    )
    return summary.sort_index()[list(SUMMARY_SERIES)]


def autocorrelation(series: pd.Series, nlags: int) -> pd.Series:
    """ACF of `series` for lags 0..nlags, indexed by lag."""
    values = pd.Series(series).astype(float).to_numpy()
    nlags = min(nlags, len(values) - 1)
    estimates = acf(values, nlags=nlags, fft=True, missing="drop")
    return pd.Series(estimates, index=pd.RangeIndex(len(estimates), name="lag"), name="acf")


def suggest_window_blocks(
    summary: pd.DataFrame,
    candidates: Sequence[int] = (16, 32, 64),
    nlags: int = 200,
) -> pd.DataFrame:
    """Per-series decorrelation diagnostics and the `L` each one supports.

    `decorrelation_blocks` is the most conservative of the three estimates; the
    recommendation is the smallest candidate that covers it (NaN when every
    candidate is too short, which is itself the finding).
    """
    rows = []
    for name in summary.columns:
        correlations = autocorrelation(summary[name], nlags)
        band = WHITE_NOISE_Z / np.sqrt(len(summary[name].dropna()))
        below_threshold = _first_lag_below(correlations, DECAY_THRESHOLD)
        inside_band = _first_lag_below(correlations.abs(), band)
        timescale = _integral_timescale(correlations)
        decorrelation = np.nanmax([below_threshold, inside_band, timescale])
        rows.append(
            {
                "series": name,
                "lag_below_1_over_e": below_threshold,
                "lag_inside_white_noise_band": inside_band,
                "white_noise_band": band,
                "integral_timescale_blocks": timescale,
                "decorrelation_blocks": decorrelation,
                "supported_window_blocks": _smallest_covering(candidates, decorrelation),
            }
        )
    return pd.DataFrame(rows)


def _first_lag_below(correlations: pd.Series, threshold: float) -> float:
    """Smallest positive lag whose value is below `threshold`; NaN if never."""
    positive_lags = correlations.iloc[1:]
    crossings = positive_lags.index[positive_lags < threshold]
    return float(crossings[0]) if len(crossings) else np.nan


def _integral_timescale(correlations: pd.Series) -> float:
    """1 + 2 * sum of the ACF up to its first non-positive lag, in blocks."""
    positive_lags = correlations.iloc[1:].to_numpy()
    non_positive = np.flatnonzero(positive_lags <= 0)
    cutoff = non_positive[0] if len(non_positive) else len(positive_lags)
    return float(1.0 + 2.0 * positive_lags[:cutoff].sum())


def _smallest_covering(candidates: Sequence[int], target: float) -> float:
    covering = [c for c in sorted(candidates) if c >= target]
    return float(covering[0]) if covering else np.nan
