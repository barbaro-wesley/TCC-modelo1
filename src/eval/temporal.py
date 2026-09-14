"""Release shifted labels only at their target date (post-publication convention).

Dates here identify observation periods, not verified publication timestamps.
Historical revisions/publication delays require a separate point-in-time dataset.
"""

import numpy as np
import pandas as pd

TEMPORAL_PROTOCOL = "calendar-mature-labels-v1"


def training_ends(n, horizon=1, origin_dates=None, target_dates=None):
    """Exclusive training ends, one per origin; equality means already observed."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if (origin_dates is None) != (target_dates is None):
        raise ValueError("Pass both origin_dates and target_dates")
    if origin_dates is None:
        return np.maximum(0, np.arange(n) - horizon + 1)
    origins = pd.DatetimeIndex(origin_dates).as_unit("ns")
    targets = pd.DatetimeIndex(target_dates).as_unit("ns")
    if len(origins) != n or len(targets) != n:
        raise ValueError("Dates and predictions must have equal lengths")
    if (origins.hasnans or targets.hasnans or not origins.is_monotonic_increasing
            or not targets.is_monotonic_increasing or origins.has_duplicates
            or np.any(targets <= origins)):
        raise ValueError("Dates must be ordered, unique origins and future targets")
    return np.searchsorted(targets.asi8, origins.asi8, side="right")


def weekly_history_starts(n, dates=None):
    """Start of each uninterrupted weekly block; never compress calendar gaps."""
    if dates is None:
        return np.zeros(n, dtype=int)
    dates = pd.DatetimeIndex(dates).as_unit("ns")
    if len(dates) != n or dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Expected ordered, unique weekly dates")
    breaks = np.r_[True, np.diff(dates.asi8) != pd.Timedelta(weeks=1).value]
    return np.maximum.accumulate(np.where(breaks, np.arange(n), 0))
