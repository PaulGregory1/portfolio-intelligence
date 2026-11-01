"""Grounding check: every number in the final answer must be traceable to a tool output.

Numbers are extracted with their display precision, so "4.5%" is matched against a tool
value of 4.49 (rounding tolerance) but not against 4.62. Supported formats: Swiss
(1'234'567.89), English (1,234,567.89), continental (1.234.567,89 and 1 234 567,89),
decimal commas (4,23 %), signs, currencies, percentages, basis points and magnitude
suffixes (k, M, Mio, mn, Mrd, bn, million, milliard).

Obvious non-financial numbers are ignored: dates, years, quarter and period labels (Q3,
1Y), small counts, and VaR confidence levels (95 %, 99 %).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"  # 2026-09-30
    r"|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"  # 30.09.2026, 30/09/26
    r"|\b\d{4}[-/]\d{1,2}\b(?![.,]\d)"  # 2026-09
)

_NUMBER = re.compile(
    r"(?<![\w.,'’])"
    r"(?P<sign>[+\-−±])?\s?"
    r"(?P<num>"
    r"\d{1,3}(?:['’]\d{3})+(?:[.,]\d+)?"  # Swiss
    r"|\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?|\d{1,3}\.\d{3},\d+"  # continental with dots
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # English
    r"|\d{1,3}(?:[   ]\d{3})+(?:,\d+)?"  # continental with spaces
    r"|\d+(?:[.,]\d+)?"  # plain
    r")"
    r"(?P<suffix>\s?(?:%|pct\b|percent\b|pour\s?cent\b|bps\b|bp\b|pb\b"
    r"|Mio\.?|mio\.?|Mrd\.?|mn\b|millions?\b|milliards?\b|billions?\b|bn\b|[kKM]\b))?"
    r"(?![\w])"
)

_CURRENCY = re.compile(r"(CHF|EUR|USD|Fr\.|SFr|\$|€)", re.IGNORECASE)
_VAR_CONTEXT = re.compile(r"c?var|confiance|confidence|niveau|level|quantile", re.IGNORECASE)

_MULTIPLIERS = {
    "k": 1e3,
    "mio": 1e6,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "mrd": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "milliard": 1e9,
    "milliards": 1e9,
}


@dataclass(frozen=True)
class ExtractedNumber:
    raw: str
    value: float  # signed value, magnitude suffix applied, percent sign removed
    decimals: int  # display precision
    unit: str  # "percent", "amount" (currency nearby) or "number"
    scale: float  # magnitude multiplier applied (1, 1e3, 1e6, 1e9)
    ignored: str | None = None  # reason if this is not a financial figure


def _parse(num: str) -> tuple[float, int]:
    """Value and number of decimals of the numeric part, whatever the separators."""
    compact = re.sub(r"[   '’]", "", num)
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", compact) and (
        compact.count(".") > 1 or "," in compact
    ):
        compact = compact.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", compact) and not re.fullmatch(
        r"\d+,\d{1,2}", compact
    ):
        compact = compact.replace(",", "")
    else:
        compact = compact.replace(",", ".")
    decimals = len(compact.split(".")[1]) if "." in compact else 0
    return float(compact), decimals


def extract_numbers(text: str) -> list[ExtractedNumber]:
    cleaned = _DATE.sub(lambda m: " " * len(m.group()), text)
    out: list[ExtractedNumber] = []
    for m in _NUMBER.finditer(cleaned):
        raw = m.group().strip()
        value, decimals = _parse(m.group("num"))
        suffix = (m.group("suffix") or "").strip().lower().rstrip(".")
        suffix = re.sub(r"\s+", "", suffix)
        sign = -1.0 if m.group("sign") in ("-", "−") else 1.0
        window = cleaned[max(0, m.start() - 12) : m.end() + 12]
        has_currency = bool(_CURRENCY.search(window))

        unit, scale = "number", 1.0
        if suffix in ("%", "pct", "percent", "pourcent"):
            unit = "percent"
        elif suffix in ("bp", "bps", "pb"):
            unit, value, decimals = "percent", value / 100.0, decimals + 2
        elif suffix in _MULTIPLIERS:
            scale = _MULTIPLIERS[suffix]
            unit = "amount"
        elif has_currency:
            unit = "amount"

        ignored = None
        is_int = decimals == 0
        if unit == "number" and is_int and 1900 <= value <= 2100:
            ignored = "year"
        elif unit == "number" and is_int and value < 100:
            ignored = "small count"
        elif suffix == "m" and is_int and value in (1, 3, 6) and not has_currency:
            ignored = "period label"
        elif unit == "percent" and value in (90, 95, 97.5, 99) and _VAR_CONTEXT.search(window):
            ignored = "confidence level"
        out.append(ExtractedNumber(raw, sign * value * scale, decimals, unit, scale, ignored))
    return out


def collect_values(payload: Any) -> list[float]:
    """Every numeric value found (recursively) in tool outputs."""
    values: list[float] = []
    if isinstance(payload, bool):
        return values
    if isinstance(payload, int | float):
        values.append(float(payload))
    elif isinstance(payload, dict):
        for v in payload.values():
            values.extend(collect_values(v))
    elif isinstance(payload, list | tuple):
        for v in payload:
            values.extend(collect_values(v))
    return values


def is_grounded(number: ExtractedNumber, values: list[float]) -> float | None:
    """The tool value matching `number` within its display precision, if any.

    Signs are compared in absolute value because the answer often carries the sign in words
    ("underperformed by 3.25%" for an active return of -3.25). A percentage also matches a
    decimal value (4.23% <-> 0.0423).
    """
    tolerance = 0.5 * 10.0 ** (-number.decimals) * number.scale * 1.001 + 1e-9
    candidates = [(abs(number.value), tolerance)]
    if number.unit == "percent":
        candidates.append((abs(number.value) / 100.0, tolerance / 100.0))
    for v in values:
        for target, tol in candidates:
            if abs(abs(v) - target) <= tol:
                return v
    return None


@dataclass
class GroundingReport:
    numbers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def checked(self) -> list[dict[str, Any]]:
        return [n for n in self.numbers if n["ignored"] is None]

    @property
    def ungrounded_values(self) -> list[str]:
        return [n["raw"] for n in self.checked if not n["grounded"]]

    @property
    def grounding_rate(self) -> float | None:
        checked = self.checked
        if not checked:
            return None
        return sum(1 for n in checked if n["grounded"]) / len(checked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numbers": self.numbers,
            "checked": len(self.checked),
            "ungrounded": self.ungrounded_values,
            "grounding_rate": self.grounding_rate,
        }


def check_grounding(answer: str, tool_outputs: list[dict[str, Any]]) -> GroundingReport:
    values = collect_values(tool_outputs)
    report = GroundingReport()
    for number in extract_numbers(answer):
        match = None if number.ignored else is_grounded(number, values)
        report.numbers.append(
            {
                **asdict(number),
                "grounded": number.ignored is None and match is not None,
                "matched_value": match,
            }
        )
    return report
