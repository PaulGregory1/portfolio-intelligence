"""Daily close prices: Yahoo Finance with a local cache, or a correlated synthetic generator.

The synthetic generator is a geometric Brownian motion driven by a small factor model
(global equity, regional equity, interest rates per currency, credit, gold, real estate),
so that correlations between instruments look plausible. Tests always use it.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pi.data.universe import Instrument

log = logging.getLogger(__name__)

TRADING_DAYS = 252
HISTORY_YEARS = 3

FACTORS = (
    "equity_global",
    "equity_ch",
    "equity_eu",
    "equity_us",
    "equity_em",
    "equity_jp",
    "rates_chf",
    "rates_eur",
    "rates_usd",
    "credit",
    "gold",
    "real_estate",
)
_REGION_FACTOR = {
    "Switzerland": "equity_ch",
    "Europe": "equity_eu",
    "North America": "equity_us",
    "Emerging Markets": "equity_em",
    "Japan": "equity_jp",
}


def business_days(end: date, years: int = HISTORY_YEARS) -> pd.DatetimeIndex:
    start = pd.Timestamp(end) - pd.DateOffset(years=years)
    return pd.bdate_range(start=start, end=pd.Timestamp(end), name="date")


def factor_loadings(inst: Instrument) -> dict[str, float]:
    """Loadings on unit-variance factors; the remaining variance is idiosyncratic."""
    rates = f"rates_{inst.currency.lower()}"
    sub = inst.sub_asset_class
    if inst.asset_class == "equity":
        single = sub == "single_stock"
        loadings = {"equity_global": 0.55 if single else 0.75}
        if inst.region in _REGION_FACTOR:
            loadings[_REGION_FACTOR[inst.region]] = 0.3 if single else 0.45
        return loadings
    if inst.asset_class == "fixed_income":
        if sub in ("high_yield", "emerging_debt"):
            return {"credit": 0.6, "equity_global": 0.35, rates: 0.35}
        if sub in ("corporate", "aggregate"):
            return {rates: 0.8, "credit": 0.4}
        return {rates: 0.95}
    if inst.asset_class == "alternatives":
        if sub == "gold":
            return {"gold": 0.97}
        return {"real_estate": 0.6, "equity_global": 0.45, rates: 0.3}
    return {rates: 0.5}


def synthetic_prices(
    instruments: list[Instrument], dates: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Correlated GBM paths, one column per ticker."""
    n = len(dates)
    factors = rng.standard_normal((n, len(FACTORS)))
    columns: dict[str, np.ndarray] = {}
    for inst in instruments:
        loadings = factor_loadings(inst)
        load = np.array([loadings.get(f, 0.0) for f in FACTORS])
        idio_weight = np.sqrt(max(0.0, 1.0 - float(load @ load)))
        shocks = factors @ load + idio_weight * rng.standard_normal(n)
        daily_sigma = inst.sigma / np.sqrt(TRADING_DAYS)
        drift = (inst.mu - 0.5 * inst.sigma**2) / TRADING_DAYS
        log_returns = drift + daily_sigma * shocks
        log_returns[0] = 0.0
        start_price = float(np.round(rng.uniform(20.0, 400.0), 2))
        columns[inst.ticker] = np.round(start_price * np.exp(np.cumsum(log_returns)), 4)
    return pd.DataFrame(columns, index=dates)


def download_prices(
    instruments: list[Instrument], dates: pd.DatetimeIndex, cache_dir: Path
) -> pd.DataFrame:
    """Download close prices from Yahoo Finance (cached). Missing tickers are left out."""
    symbols = {i.yf_ticker: i.ticker for i in instruments if i.yf_ticker}
    cache = cache_dir / f"prices_{dates[0].date()}_{dates[-1].date()}.csv"
    if cache.exists():
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        data = yf.download(
            list(symbols),
            start=str(dates[0].date()),
            end=str((dates[-1] + pd.Timedelta(days=1)).date()),
            auto_adjust=False,
            progress=False,
        )
        raw = data["Close"] if data is not None and not data.empty else pd.DataFrame()
        if not raw.empty:
            cache_dir.mkdir(parents=True, exist_ok=True)
            raw.to_csv(cache)
    raw = raw.rename(columns=symbols)
    aligned = raw.reindex(dates).ffill().bfill()
    # Keep only tickers with an almost complete history.
    usable = [c for c in aligned.columns if raw[c].notna().mean() > 0.9]
    return aligned[usable].round(4)


def load_prices(
    instruments: list[Instrument],
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    *,
    offline: bool,
    cache_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Return prices for every instrument and the list of tickers that are synthetic.

    The synthetic paths are always drawn (for every ticker) so that the random stream, and
    therefore the rest of the generated data, does not depend on network availability.
    """
    prices = synthetic_prices(instruments, dates, rng)
    if offline:
        return prices, list(prices.columns)
    try:
        real = download_prices(instruments, dates, cache_dir)
    except Exception as exc:  # network errors, API changes: fall back to synthetic data
        log.warning("Price download failed (%s); using synthetic prices.", exc)
        return prices, list(prices.columns)
    prices.update(real)
    synthetic = [t for t in prices.columns if t not in real.columns]
    return prices, synthetic
