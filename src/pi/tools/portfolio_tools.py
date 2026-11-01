"""The tools themselves: thin adapters from the analytics engine to rounded, typed outputs."""

from __future__ import annotations

from functools import cache

import pandas as pd

from pi.analytics.attribution import attribution
from pi.analytics.benchmark import compare_to_benchmark
from pi.analytics.performance import performance
from pi.analytics.periods import Period, resolve_period
from pi.analytics.risk import risk_report
from pi.analytics.valuation import GROUPINGS, exposures, valuation
from pi.data.store import PortfolioData
from pi.screening.filters import ScreenCriteria, instrument_stats, screen
from pi.screening.search import BM25Searcher
from pi.tools import schemas as s
from pi.tools.access import ToolContext, resolve_client
from pi.tools.registry import ToolRegistry

UNITS = (
    " Fields ending in _pct are percentages rounded to 2 decimals (4.23 means 4.23%); amounts "
    "are in the client's reference currency."
)


def pct(x: float | None) -> float | None:
    return None if x is None else round(100.0 * x, 2)


def pct0(x: float) -> float:
    return round(100.0 * x, 2)


def money(x: float) -> float:
    return round(float(x), 2)


def period_out(p: Period) -> s.PeriodOut:
    return s.PeriodOut(
        label=p.label, start_date=p.start.date(), end_date=p.end.date(), truncated=p.truncated
    )


registry = ToolRegistry()


@registry.register(
    "list_clients",
    "List the clients of the current banker with their mandate, reference currency and "
    "current total value. Call this first to find client ids.",
    s.ListClientsInput,
    s.ClientList,
)
def list_clients(ctx: ToolContext, _: s.ListClientsInput) -> s.ClientList:
    clients = []
    for c in ctx.data.clients_of(ctx.banker_id):
        clients.append(
            s.ClientSummary(
                client_id=c.client_id,
                name=c.name,
                reference_currency=c.currency,
                mandate=c.mandate,
                profile=c.profile,
                total_value=money(valuation(ctx.data, c.client_id).total_value),
            )
        )
    return s.ClientList(
        banker_id=ctx.banker_id,
        banker_name=ctx.data.bankers.get(ctx.banker_id, "unknown"),
        as_of_date=ctx.data.end_date.date(),
        clients=clients,
    )


@registry.register(
    "get_portfolio",
    "Positions of a client portfolio: quantity, price, market value and weight of every line "
    "(including cash), and the total value." + UNITS,
    s.GetPortfolioInput,
    s.Portfolio,
)
def get_portfolio(ctx: ToolContext, p: s.GetPortfolioInput) -> s.Portfolio:
    client = resolve_client(ctx, p.client_id)
    as_of = pd.Timestamp(p.as_of_date) if p.as_of_date else None
    val = valuation(ctx.data, client.client_id, as_of)
    positions = [
        s.Position(
            ticker=str(r["ticker"]),
            name=str(r["name"]),
            asset_class=str(r["asset_class"]),
            region=str(r["region"]),
            currency=str(r["currency"]),
            quantity=round(float(r["quantity"]), 2),
            price=round(float(r["price"]), 2),
            market_value=money(float(r["market_value"])),
            weight_pct=pct0(float(r["weight"])),
        )
        for r in val.lines.to_dict("records")
    ]
    return s.Portfolio(
        client_id=client.client_id,
        client_name=client.name,
        reference_currency=client.currency,
        mandate=client.mandate,
        as_of_date=val.as_of.date(),
        total_value=money(val.total_value),
        positions=positions,
    )


@registry.register(
    "get_transactions",
    "Transactions of a client over a period (default YTD), optionally filtered by type or "
    "instrument, most recent first, with totals by type." + UNITS,
    s.GetTransactionsInput,
    s.TransactionList,
)
def get_transactions(ctx: ToolContext, p: s.GetTransactionsInput) -> s.TransactionList:
    client = resolve_client(ctx, p.client_id)
    per = resolve_period(ctx.data.dates, p.period, p.start_date, p.end_date)
    tx = ctx.data.client_transactions(client.client_id)
    tx = tx[(tx["date"] > per.start) & (tx["date"] <= per.end)]
    if p.types:
        tx = tx[tx["type"].isin(p.types)]
    if p.ticker:
        tx = tx[tx["ticker"].fillna("").str.upper() == p.ticker.strip().upper()]
    totals = tx.groupby("type")["amount"].sum()
    shown = tx.sort_values(["date", "txn_id"], ascending=False).head(p.limit)
    return s.TransactionList(
        client_id=client.client_id,
        reference_currency=client.currency,
        period=period_out(per),
        total_count=len(tx),
        shown=len(shown),
        totals_by_type={str(k): money(float(v)) for k, v in totals.items()},
        transactions=[
            s.Transaction(
                date=pd.Timestamp(r["date"]).date(),
                type=str(r["type"]),
                ticker=None if pd.isna(r["ticker"]) else str(r["ticker"]),
                quantity=float(r["quantity"]),
                price=None if pd.isna(r["price"]) else round(float(r["price"]), 2),
                amount=money(float(r["amount"])),
            )
            for r in shown.to_dict("records")
        ],
    )


@registry.register(
    "get_performance",
    "Performance of a client portfolio over a period: time-weighted return (net of fees), "
    "money-weighted return, composite benchmark return, active return, start and end values, "
    "net flows, income and fees." + UNITS,
    s.GetPerformanceInput,
    s.Performance,
)
def get_performance(ctx: ToolContext, p: s.GetPerformanceInput) -> s.Performance:
    client = resolve_client(ctx, p.client_id)
    r = performance(ctx.data, client.client_id, p.period, p.start_date, p.end_date)
    return s.Performance(
        client_id=client.client_id,
        reference_currency=r.currency,
        period=period_out(r.period),
        twr_pct=pct0(r.twr),
        twr_annualised_pct=pct(r.twr_annualised),
        mwr_pct=pct0(r.mwr_period),
        mwr_annualised_pct=pct(r.mwr_annualised),
        benchmark_return_pct=pct0(r.benchmark_return),
        active_return_pct=pct0(r.active_return),
        start_value=money(r.start_value),
        end_value=money(r.end_value),
        net_flows=money(r.net_flows),
        income=money(r.income),
        fees=money(r.fees),
        investment_gain=money(r.investment_gain),
    )


@registry.register(
    "get_attribution",
    "Brinson-Fachler attribution of the active return (portfolio minus benchmark) over a "
    "period, by asset class or by region: allocation, selection and interaction effects per "
    "segment, plus the fee drag. Use it to explain out- or under-performance." + UNITS,
    s.GetAttributionInput,
    s.Attribution,
)
def get_attribution(ctx: ToolContext, p: s.GetAttributionInput) -> s.Attribution:
    client = resolve_client(ctx, p.client_id)
    r = attribution(ctx.data, client.client_id, p.period, p.by, p.start_date, p.end_date)
    segments = [
        s.AttributionSegment(
            segment=str(row["segment"]),
            avg_portfolio_weight_pct=pct0(row["portfolio_weight"]),
            benchmark_weight_pct=pct0(row["benchmark_weight"]),
            portfolio_return_pct=pct0(row["portfolio_return"]),
            benchmark_return_pct=pct0(row["benchmark_return"]),
            allocation_pct=pct0(row["allocation"]),
            selection_pct=pct0(row["selection"]),
            interaction_pct=pct0(row["interaction"]),
            total_effect_pct=pct0(row["total"]),
        )
        for row in r.segments.to_dict("records")
    ]
    return s.Attribution(
        client_id=client.client_id,
        reference_currency=r.currency,
        period=period_out(r.period),
        method="Brinson-Fachler, daily, linked with Carino factors",
        by=r.by,
        portfolio_return_pct=pct0(r.portfolio_return),
        benchmark_return_pct=pct0(r.benchmark_return),
        active_return_pct=pct0(r.active_return),
        allocation_pct=pct0(r.allocation),
        selection_pct=pct0(r.selection),
        interaction_pct=pct0(r.interaction),
        fees_pct=pct0(r.fees),
        segments=segments,
    )


@registry.register(
    "get_risk_report",
    "Risk metrics of a client portfolio over a period (default 1Y): volatility, max drawdown, "
    "tracking error, information ratio, 1-day historical VaR and CVaR at 95% and 99%, ex-ante "
    "volatility, concentration (HHI, top 5 / top 10 weights) and currency exposure." + UNITS,
    s.GetRiskReportInput,
    s.RiskReport,
)
def get_risk_report(ctx: ToolContext, p: s.GetRiskReportInput) -> s.RiskReport:
    client = resolve_client(ctx, p.client_id)
    r = risk_report(ctx.data, client.client_id, p.period, p.start_date, p.end_date)
    return s.RiskReport(
        client_id=client.client_id,
        reference_currency=r.currency,
        period=period_out(r.period),
        observations=r.observations,
        volatility_pct=pct0(r.volatility),
        benchmark_volatility_pct=pct0(r.benchmark_volatility),
        max_drawdown_pct=pct0(r.max_drawdown),
        drawdown_peak_date=r.drawdown_peak.date() if r.drawdown_peak is not None else None,
        drawdown_trough_date=r.drawdown_trough.date() if r.drawdown_trough is not None else None,
        tracking_error_pct=pct0(r.tracking_error),
        information_ratio=round(r.information_ratio, 4),
        var_95_1d_pct=pct0(r.var_95),
        var_99_1d_pct=pct0(r.var_99),
        cvar_95_1d_pct=pct0(r.cvar_95),
        cvar_99_1d_pct=pct0(r.cvar_99),
        ex_ante_volatility_pct=pct0(r.ex_ante_volatility),
        hhi=round(r.hhi, 4),
        top5_weight_pct=pct0(r.top5_weight),
        top10_weight_pct=pct0(r.top10_weight),
        largest_position=r.largest_position,
        largest_position_weight_pct=pct0(r.largest_position_weight),
        currency_exposure_pct={k: pct0(v) for k, v in r.currency_exposure.items()},
    )


@registry.register(
    "compare_to_benchmark",
    "Compare a portfolio with its mandate's composite benchmark, by asset class or region: "
    "current weights versus benchmark weights (active weights), and segment returns and "
    "contributions over a period." + UNITS,
    s.CompareToBenchmarkInput,
    s.BenchmarkComparison,
)
def compare_benchmark(ctx: ToolContext, p: s.CompareToBenchmarkInput) -> s.BenchmarkComparison:
    client = resolve_client(ctx, p.client_id)
    r = compare_to_benchmark(ctx.data, client.client_id, p.period, p.by, p.start_date, p.end_date)
    rows = [
        s.BenchmarkRow(
            segment=str(row["segment"]),
            portfolio_weight_pct=pct0(row["portfolio_weight"]),
            benchmark_weight_pct=pct0(row["benchmark_weight"]),
            active_weight_pct=pct0(row["active_weight"]),
            portfolio_return_pct=pct0(row["portfolio_return"]),
            benchmark_return_pct=pct(row["benchmark_return"]),
            portfolio_contribution_pct=pct0(row["portfolio_contribution"]),
            benchmark_contribution_pct=pct0(row["benchmark_contribution"]),
        )
        for row in r.rows.to_dict("records")
    ]
    return s.BenchmarkComparison(
        client_id=client.client_id,
        reference_currency=r.currency,
        mandate=r.mandate,
        period=period_out(r.period),
        by=r.by,
        portfolio_return_pct=pct0(r.portfolio_return),
        benchmark_return_pct=pct0(r.benchmark_return),
        fees_pct=pct0(r.fees),
        rows=rows,
    )


@registry.register(
    "get_exposures",
    "Current exposures of a portfolio by asset class, sub-asset class, region and currency "
    "(weights in percent of the total value, cash included)." + UNITS,
    s.GetExposuresInput,
    s.Exposures,
)
def get_exposures(ctx: ToolContext, p: s.GetExposuresInput) -> s.Exposures:
    client = resolve_client(ctx, p.client_id)
    as_of = pd.Timestamp(p.as_of_date) if p.as_of_date else None
    val = valuation(ctx.data, client.client_id, as_of)
    groupings = [p.by] if p.by else list(GROUPINGS)
    return s.Exposures(
        client_id=client.client_id,
        reference_currency=client.currency,
        as_of_date=val.as_of.date(),
        total_value=money(val.total_value),
        exposures_pct={
            g: {str(k): pct0(float(v)) for k, v in exposures(val, g).items()} for g in groupings
        },
    )


@cache
def _stats(data: PortfolioData) -> pd.DataFrame:
    return instrument_stats(data)


def _instrument_row(r: dict[str, object]) -> s.InstrumentRow:
    return s.InstrumentRow(
        ticker=str(r["ticker"]),
        name=str(r["name"]),
        asset_class=str(r["asset_class"]),
        sub_asset_class=str(r["sub_asset_class"]),
        region=str(r["region"]),
        currency=str(r["currency"]),
        ter_pct=round(float(str(r["ter_pct"])), 3),
        esg_score=int(str(r["esg_score"])),
        volatility_1y_pct=round(float(str(r["volatility_1y_pct"])), 2),
        return_1y_pct=round(float(str(r["return_1y_pct"])), 2),
    )


@registry.register(
    "screen_universe",
    "Screen the investment universe with structured filters (asset class, sub-asset class, "
    "region, currency, maximum TER, minimum ESG score, maximum 1-year volatility). Returns "
    "matching instruments with their TER, ESG score, 1-year volatility and return.",
    s.ScreenUniverseInput,
    s.ScreenResult,
)
def screen_universe(ctx: ToolContext, p: s.ScreenUniverseInput) -> s.ScreenResult:
    criteria = ScreenCriteria(
        asset_class=p.asset_class,
        sub_asset_class=p.sub_asset_class,
        region=p.region,
        currency=p.currency,
        max_ter_pct=p.max_ter_pct,
        min_esg_score=p.min_esg_score,
        max_volatility_pct=p.max_volatility_pct,
        sort_by=p.sort_by,
    )
    result = screen(_stats(ctx.data), criteria)
    used = p.model_dump(exclude_none=True, exclude={"limit", "sort_by"})
    return s.ScreenResult(
        criteria=used,
        match_count=len(result),
        instruments=[
            _instrument_row({str(k): v for k, v in r.items()})
            for r in result.head(p.limit).to_dict("records")
        ],
    )


@registry.register(
    "search_instruments",
    "Free-text search of the investment universe by name and description (any language), "
    "e.g. 'cheap Swiss government bonds' or 'actions durables'. Returns the best matches "
    "with a relevance score.",
    s.SearchInstrumentsInput,
    s.SearchResult,
)
def search_instruments(ctx: ToolContext, p: s.SearchInstrumentsInput) -> s.SearchResult:
    searcher = ctx.searcher or BM25Searcher(list(ctx.data.instruments.values()))
    ctx.searcher = searcher
    hits = []
    for ticker, score in searcher.search(p.query, p.limit):
        inst = ctx.data.instruments[ticker]
        hits.append(
            s.SearchHit(
                ticker=ticker,
                name=inst.name,
                asset_class=inst.asset_class,
                sub_asset_class=inst.sub_asset_class,
                region=inst.region,
                currency=inst.currency,
                ter_pct=inst.ter_pct,
                score=score,
                description=inst.description,
            )
        )
    return s.SearchResult(query=p.query, method=searcher.name, hits=hits)
