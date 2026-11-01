"""Daily FX rates, expressed as CHF per unit of currency (CHF itself is always 1)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pi.data.universe import CURRENCIES

log = logging.getLogger(__name__)

# (start rate in CHF, annual drift, annual volatility)
_SYNTHETIC = {"EUR": (0.96, -0.01, 0.05), "USD": (0.90, -0.02, 0.07)}
_YF = {"EUR": "EURCHF=X", "USD": "USDCHF=X"}


def synthetic_fx(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rates: dict[str, np.ndarray] = {"CHF": np.ones(len(dates))}
    for ccy, (start, mu, sigma) in _SYNTHETIC.items():
        steps = (mu - 0.5 * sigma**2) / 252 + sigma / np.sqrt(252) * rng.standard_normal(len(dates))
        steps[0] = 0.0
        rates[ccy] = np.round(start * np.exp(np.cumsum(steps)), 6)
    return pd.DataFrame(rates, index=dates)[list(CURRENCIES)]


def load_fx(
    dates: pd.DatetimeIndex, rng: np.random.Generator, *, offline: bool, cache_dir: Path
) -> pd.DataFrame:
    fx = synthetic_fx(dates, rng)
    if offline:
        return fx
    cache = cache_dir / f"fx_{dates[0].date()}_{dates[-1].date()}.csv"
    try:
        if cache.exists():
            raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            import yfinance as yf

            data = yf.download(
                list(_YF.values()),
                start=str(dates[0].date()),
                end=str((dates[-1] + pd.Timedelta(days=1)).date()),
                progress=False,
            )
            raw = data["Close"] if data is not None and not data.empty else pd.DataFrame()
            if raw.empty:
                raise ValueError("empty FX download")
            cache_dir.mkdir(parents=True, exist_ok=True)
            raw.to_csv(cache)
        raw = raw.rename(columns={v: k for k, v in _YF.items()})
        fx.update(raw.reindex(dates).ffill().bfill().round(6))
    except Exception as exc:
        log.warning("FX download failed (%s); using synthetic rates.", exc)
    return fx


def conversion(fx: pd.DataFrame, from_ccy: str, to_ccy: str) -> pd.Series:
    """Daily rate to convert an amount in `from_ccy` into `to_ccy`."""
    return fx[from_ccy] / fx[to_ccy]
