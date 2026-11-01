"""Turn period labels (YTD, 1Y, 2026-Q3, ...) into a pair of business dates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

PERIOD_HELP = (
    "MTD, QTD, YTD, 1M, 3M, 6M, 1Y, 3Y, ITD (since inception), a calendar year (2025), "
    "a quarter (2026-Q3) or a month (2026-09)"
)
_TRAILING = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36}


@dataclass(frozen=True)
class Period:
    """Performance is measured from the close of `start` to the close of `end`."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    truncated: bool = False  # the requested start was before the first available date

    @property
    def days(self) -> int:
        return int((self.end - self.start).days)


def _last_before(dates: pd.DatetimeIndex, ts: pd.Timestamp, *, inclusive: bool) -> pd.Timestamp:
    mask = dates <= ts if inclusive else dates < ts
    candidates = dates[mask]
    return pd.Timestamp(candidates[-1]) if len(candidates) else pd.Timestamp(dates[0])


def _calendar_bounds(label: str) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    """Canonical label and first/last calendar day of a year, quarter or month."""
    if m := re.fullmatch(r"(\d{4})", label):
        year = int(m.group(1))
        return label, pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    if m := re.fullmatch(r"(\d{4})[- ]?Q([1-4])|Q([1-4])[- ]?(\d{4})", label):
        year = int(m.group(1) or m.group(4))
        quarter = int(m.group(2) or m.group(3))
        start = pd.Timestamp(year, 3 * quarter - 2, 1)
        return f"{year}-Q{quarter}", start, start + pd.offsets.QuarterEnd(0)
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})", label):
        year, month = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year, month, 1)
        return f"{year}-{month:02d}", start, start + pd.offsets.MonthEnd(0)
    return None


def resolve_period(
    dates: pd.DatetimeIndex,
    period: str | None = "YTD",
    start_date: date | None = None,
    end_date: date | None = None,
) -> Period:
    """Resolve a period against the available business dates.

    Explicit dates win over the label. A period starting on day D is measured from the close
    of the last business day before D, so that "2026-Q3" covers the whole of July to September.
    """
    first, last = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    end = _last_before(dates, pd.Timestamp(end_date), inclusive=True) if end_date else last

    if start_date is not None:
        requested = pd.Timestamp(start_date)
        label = f"{start_date} to {end.date()}"
        start = _last_before(dates, requested, inclusive=False)
        return Period(label, start, end, truncated=requested <= first)

    label = (period or "YTD").strip().upper()
    if label in ("ITD", "SI", "INCEPTION", "SINCE_INCEPTION", "MAX"):
        return Period("ITD", first, end)
    if label in ("MTD", "QTD", "YTD"):
        anchor = {
            "MTD": pd.Timestamp(end.year, end.month, 1),
            "QTD": pd.Timestamp(end.year, 3 * ((end.month - 1) // 3) + 1, 1),
            "YTD": pd.Timestamp(end.year, 1, 1),
        }[label]
        start = _last_before(dates, anchor, inclusive=False)
        return Period(label, start, end, truncated=anchor <= first)
    if label in _TRAILING:
        anchor = end - pd.DateOffset(months=_TRAILING[label])
        return Period(label, _last_before(dates, anchor, inclusive=True), end, anchor < first)
    if bounds := _calendar_bounds(label):
        label, cal_start, cal_end = bounds
        if cal_start > last:
            raise ValueError(f"Period {label} is after the last available date {last.date()}.")
        end = _last_before(dates, min(cal_end, end), inclusive=True)
        start = _last_before(dates, cal_start, inclusive=False)
        return Period(label, start, end, truncated=cal_start <= first)
    raise ValueError(f"Unknown period '{period}'. Use {PERIOD_HELP}.")
