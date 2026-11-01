"""Input and output models of the tools.

Number conventions (repeated in the tool descriptions so the model knows them):
- fields ending in `_pct` are percentages, rounded to 2 decimals (4.23 means 4.23 %);
- monetary amounts are in the client's reference currency, rounded to 2 decimals;
- ratios without unit (information ratio, HHI) are rounded to 4 decimals.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pi.analytics.periods import PERIOD_HELP

ClientRef = Field(description="Client id (e.g. 'C01') or exact client name.")
PeriodField = Field(default="YTD", description=f"Period: {PERIOD_HELP}.")
StartField = Field(default=None, description="Optional start date (overrides `period`).")
EndField = Field(default=None, description="Optional end date, defaults to the latest date.")
Segmentation = Literal["asset_class", "region"]
Grouping = Literal["asset_class", "sub_asset_class", "region", "currency"]
TxType = Literal["deposit", "withdrawal", "buy", "sell", "dividend", "fee"]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Output(BaseModel):
    pass


class PeriodOut(Output):
    label: str
    start_date: date = Field(description="Performance is measured from the close of this date.")
    end_date: date
    truncated: bool = Field(description="True if the period starts before the data history.")


# --- inputs ----------------------------------------------------------------------------------


class ListClientsInput(ToolInput):
    pass


class GetPortfolioInput(ToolInput):
    client_id: str = ClientRef
    as_of_date: date | None = Field(default=None, description="Valuation date, default latest.")


class GetTransactionsInput(ToolInput):
    client_id: str = ClientRef
    period: str | None = PeriodField
    start_date: date | None = StartField
    end_date: date | None = EndField
    types: list[TxType] | None = Field(default=None, description="Filter on transaction types.")
    ticker: str | None = Field(default=None, description="Filter on one instrument.")
    limit: int = Field(default=30, ge=1, le=200, description="Maximum rows, most recent first.")


class GetPerformanceInput(ToolInput):
    client_id: str = ClientRef
    period: str | None = PeriodField
    start_date: date | None = StartField
    end_date: date | None = EndField


class GetAttributionInput(GetPerformanceInput):
    by: Segmentation = Field(default="asset_class", description="Segmentation of the analysis.")


class GetRiskReportInput(ToolInput):
    client_id: str = ClientRef
    period: str | None = Field(default="1Y", description=f"Period: {PERIOD_HELP}.")
    start_date: date | None = StartField
    end_date: date | None = EndField


class CompareToBenchmarkInput(GetAttributionInput):
    pass


class GetExposuresInput(ToolInput):
    client_id: str = ClientRef
    by: Grouping | None = Field(default=None, description="One grouping, or all if omitted.")
    as_of_date: date | None = Field(default=None, description="Valuation date, default latest.")


class ScreenUniverseInput(ToolInput):
    asset_class: Literal["equity", "fixed_income", "alternatives", "cash"] | None = None
    sub_asset_class: str | None = Field(
        default=None, description="e.g. government, corporate, high_yield, gold, real_estate, esg."
    )
    region: str | None = Field(
        default=None,
        description="Switzerland, Europe, North America, Emerging Markets, Japan or Global.",
    )
    currency: Literal["CHF", "EUR", "USD"] | None = None
    max_ter_pct: float | None = Field(default=None, ge=0, description="Maximum TER in percent.")
    min_esg_score: int | None = Field(default=None, ge=0, le=100)
    max_volatility_pct: float | None = Field(
        default=None, gt=0, description="Maximum 1-year annualised volatility in percent."
    )
    sort_by: Literal["ter_pct", "esg_score", "volatility_1y_pct", "return_1y_pct"] = "ter_pct"
    limit: int = Field(default=10, ge=1, le=50)


class SearchInstrumentsInput(ToolInput):
    query: str = Field(min_length=2, description="Free-text description of what is sought.")
    limit: int = Field(default=5, ge=1, le=20)


# --- outputs ---------------------------------------------------------------------------------


class ClientSummary(Output):
    client_id: str
    name: str
    reference_currency: str
    mandate: str
    profile: str
    total_value: float


class ClientList(Output):
    banker_id: str
    banker_name: str
    as_of_date: date
    clients: list[ClientSummary]


class Position(Output):
    ticker: str
    name: str
    asset_class: str
    region: str
    currency: str
    quantity: float
    price: float = Field(description="In the instrument currency.")
    market_value: float = Field(description="In the client's reference currency.")
    weight_pct: float


class Portfolio(Output):
    client_id: str
    client_name: str
    reference_currency: str
    mandate: str
    as_of_date: date
    total_value: float
    positions: list[Position]


class Transaction(Output):
    date: date
    type: str
    ticker: str | None
    quantity: float
    price: float | None
    amount: float = Field(description="Cash impact in the reference currency (+ in, - out).")


class TransactionList(Output):
    client_id: str
    reference_currency: str
    period: PeriodOut
    total_count: int
    shown: int
    totals_by_type: dict[str, float]
    transactions: list[Transaction]


class Performance(Output):
    client_id: str
    reference_currency: str
    period: PeriodOut
    twr_pct: float = Field(description="Time-weighted return, net of fees.")
    twr_annualised_pct: float | None
    mwr_pct: float = Field(description="Money-weighted return (IRR) over the period.")
    mwr_annualised_pct: float | None
    benchmark_return_pct: float
    active_return_pct: float = Field(description="TWR minus benchmark return.")
    start_value: float
    end_value: float
    net_flows: float
    income: float
    fees: float
    investment_gain: float


class AttributionSegment(Output):
    segment: str
    avg_portfolio_weight_pct: float
    benchmark_weight_pct: float
    portfolio_return_pct: float
    benchmark_return_pct: float
    allocation_pct: float
    selection_pct: float
    interaction_pct: float
    total_effect_pct: float


class Attribution(Output):
    client_id: str
    reference_currency: str
    period: PeriodOut
    method: str
    by: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    active_return_pct: float
    allocation_pct: float
    selection_pct: float
    interaction_pct: float
    fees_pct: float
    segments: list[AttributionSegment]


class RiskReport(Output):
    client_id: str
    reference_currency: str
    period: PeriodOut
    observations: int
    volatility_pct: float
    benchmark_volatility_pct: float
    max_drawdown_pct: float
    drawdown_peak_date: date | None
    drawdown_trough_date: date | None
    tracking_error_pct: float
    information_ratio: float
    var_95_1d_pct: float = Field(description="1-day historical VaR (loss, positive number).")
    var_99_1d_pct: float
    cvar_95_1d_pct: float
    cvar_99_1d_pct: float
    ex_ante_volatility_pct: float
    hhi: float
    top5_weight_pct: float
    top10_weight_pct: float
    largest_position: str
    largest_position_weight_pct: float
    currency_exposure_pct: dict[str, float]


class BenchmarkRow(Output):
    segment: str
    portfolio_weight_pct: float
    benchmark_weight_pct: float
    active_weight_pct: float
    portfolio_return_pct: float
    benchmark_return_pct: float | None
    portfolio_contribution_pct: float
    benchmark_contribution_pct: float


class BenchmarkComparison(Output):
    client_id: str
    reference_currency: str
    mandate: str
    period: PeriodOut
    by: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    fees_pct: float
    rows: list[BenchmarkRow]


class Exposures(Output):
    client_id: str
    reference_currency: str
    as_of_date: date
    total_value: float
    exposures_pct: dict[str, dict[str, float]]


class InstrumentRow(Output):
    ticker: str
    name: str
    asset_class: str
    sub_asset_class: str
    region: str
    currency: str
    ter_pct: float
    esg_score: int
    volatility_1y_pct: float
    return_1y_pct: float


class ScreenResult(Output):
    criteria: dict[str, object]
    match_count: int
    instruments: list[InstrumentRow]


class SearchHit(Output):
    ticker: str
    name: str
    asset_class: str
    sub_asset_class: str
    region: str
    currency: str
    ter_pct: float
    score: float
    description: str


class SearchResult(Output):
    query: str
    method: str
    hits: list[SearchHit]
