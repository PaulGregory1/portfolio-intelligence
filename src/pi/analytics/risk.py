"""Ex-post and ex-ante risk measures. Losses (VaR, CVaR, drawdown) are positive numbers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pi.analytics.benchmark import benchmark_returns
from pi.analytics.history import CASH, history
from pi.analytics.performance import period_slice
from pi.analytics.periods import Period, resolve_period
from pi.analytics.valuation import exposures, valuation
from pi.data.store import PortfolioData

TRADING_DAYS = 252


def volatility(returns: np.ndarray | pd.Series) -> float:
    """Annualised standard deviation of daily returns (sample, ddof=1)."""
    return float(np.std(np.asarray(returns, dtype=float), ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Largest peak-to-trough loss of the cumulative index, with the peak and trough dates."""
    if returns.empty:
        return 0.0, None, None
    index = (1.0 + returns).cumprod()
    index = pd.concat([pd.Series([1.0], index=[index.index[0] - pd.Timedelta(days=1)]), index])
    drawdown = 1.0 - index / index.cummax()
    trough = pd.Timestamp(drawdown.idxmax())
    peak = pd.Timestamp(index.loc[:trough].idxmax())
    return float(drawdown.max()), peak, trough


def tracking_error(r_p: np.ndarray | pd.Series, r_b: np.ndarray | pd.Series) -> float:
    return volatility(np.asarray(r_p, dtype=float) - np.asarray(r_b, dtype=float))


def information_ratio(r_p: np.ndarray | pd.Series, r_b: np.ndarray | pd.Series) -> float:
    active = np.asarray(r_p, dtype=float) - np.asarray(r_b, dtype=float)
    te = tracking_error(r_p, r_b)
    return float(np.mean(active) * TRADING_DAYS / te) if te > 0 else 0.0


def historical_var(returns: np.ndarray | pd.Series, confidence: float) -> float:
    """One-day historical Value at Risk: the loss not exceeded with the given confidence."""
    return float(-np.quantile(np.asarray(returns, dtype=float), 1.0 - confidence))


def historical_cvar(returns: np.ndarray | pd.Series, confidence: float) -> float:
    """Expected shortfall: average loss on the days beyond the VaR."""
    arr = np.asarray(returns, dtype=float)
    cutoff = np.quantile(arr, 1.0 - confidence)
    return float(-arr[arr <= cutoff].mean())


def ex_ante_volatility(weights: pd.Series, asset_returns: pd.DataFrame) -> float:
    """sqrt(w' S w), S the annualised sample covariance of the holdings' daily returns."""
    cov = asset_returns[weights.index].cov().to_numpy() * TRADING_DAYS
    w = weights.to_numpy()
    return float(np.sqrt(w @ cov @ w))


def hhi(weights: np.ndarray | pd.Series) -> float:
    """Herfindahl-Hirschman index of line weights (1 = a single line)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w**2))


@dataclass(frozen=True)
class RiskReport:
    client_id: str
    currency: str
    period: Period
    observations: int
    volatility: float
    benchmark_volatility: float
    max_drawdown: float
    drawdown_peak: pd.Timestamp | None
    drawdown_trough: pd.Timestamp | None
    tracking_error: float
    information_ratio: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    ex_ante_volatility: float
    hhi: float
    top5_weight: float
    top10_weight: float
    largest_position: str
    largest_position_weight: float
    currency_exposure: dict[str, float]


def risk_report(
    data: PortfolioData,
    client_id: str,
    period: str | None = "1Y",
    start_date: date | None = None,
    end_date: date | None = None,
) -> RiskReport:
    hist = history(data, client_id)
    per = resolve_period(data.dates, period, start_date, end_date)
    r_p = period_slice(hist.returns, per)
    r_b = period_slice(benchmark_returns(data, client_id), per)
    mdd, peak, trough = max_drawdown(r_p)

    val = valuation(data, client_id, per.end)
    lines = val.lines[val.lines["ticker"] != CASH]
    weights = lines.set_index("ticker")["weight"]
    # Ex-ante risk always uses one year of daily returns up to the valuation date.
    window = data.prices_in(hist.currency).loc[: per.end].iloc[-(TRADING_DAYS + 1) :]
    asset_returns = window[list(weights.index)].pct_change().iloc[1:]
    top = weights.sort_values(ascending=False)
    return RiskReport(
        client_id=client_id,
        currency=hist.currency,
        period=per,
        observations=len(r_p),
        volatility=volatility(r_p),
        benchmark_volatility=volatility(r_b),
        max_drawdown=mdd,
        drawdown_peak=peak,
        drawdown_trough=trough,
        tracking_error=tracking_error(r_p, r_b),
        information_ratio=information_ratio(r_p, r_b),
        var_95=historical_var(r_p, 0.95),
        var_99=historical_var(r_p, 0.99),
        cvar_95=historical_cvar(r_p, 0.95),
        cvar_99=historical_cvar(r_p, 0.99),
        ex_ante_volatility=ex_ante_volatility(weights, asset_returns),
        hhi=hhi(weights),
        top5_weight=float(top.iloc[:5].sum()),
        top10_weight=float(top.iloc[:10].sum()),
        largest_position=str(top.index[0]),
        largest_position_weight=float(top.iloc[0]),
        currency_exposure={str(k): float(v) for k, v in exposures(val, "currency").items()},
    )
