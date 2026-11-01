"""Free-text instrument search: BM25 by default, embeddings as an optional plug-in.

Descriptions are in English while bankers often ask in French or German, so queries go
through a small multilingual synonym table before BM25 scoring. Documents are enriched with
their metadata (asset class, region, currency, cost bucket) so that a query such as
"obligations suisses peu chères" finds cheap CHF bond funds.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from pi.data.universe import Instrument

_STOPWORDS_TEXT = """
a an and are as at be by for from in is it of on the to with
le la les un une des du de d l et ou en au aux pour avec sur dans qui que
der die das ein eine und oder mit fur von im zu
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

# Query-side translations (accents already stripped, plural "s" removed).
SYNONYMS: dict[str, list[str]] = {
    "obligation": ["bond"],
    "anleihe": ["bond"],
    "suisse": ["swiss", "switzerland", "chf"],
    "schweiz": ["swiss", "switzerland", "chf"],
    "schweizer": ["swiss", "switzerland", "chf"],
    "action": ["equity", "stock"],
    "aktie": ["equity", "stock"],
    "or": ["gold"],
    "gold": ["gold"],
    "immobilier": ["real", "estate", "property"],
    "immobilie": ["real", "estate", "property"],
    "emergent": ["emerging"],
    "durable": ["sustainable", "esg"],
    "nachhaltig": ["sustainable", "esg"],
    "responsable": ["sustainable", "esg"],
    "cher": ["cheap", "cost"],
    "chere": ["cheap", "cost"],
    "marche": ["market"],
    "frai": ["cost"],
    "gunstig": ["cheap", "cost"],
    "etat": ["government"],
    "souverain": ["government", "sovereign"],
    "entreprise": ["corporate"],
    "rendement": ["yield", "income"],
    "dividende": ["dividend"],
    "technologie": ["technology"],
    "sante": ["healthcare"],
    "inflation": ["inflation"],
    "americain": ["us", "america"],
    "europeen": ["europe", "european"],
    "japon": ["japan"],
    "liquidite": ["cash", "money", "market"],
    "monetaire": ["money", "market"],
    "dollar": ["usd"],
    "euro": ["eur", "euro"],
    "franc": ["chf"],
}
# "peu chère" means cheap; drop the negation so it does not dilute the query.
DROP = frozenset({"peu", "bon"})


def normalise(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[a-z0-9&]+", text)
    out = []
    for tok in tokens:
        if tok in STOPWORDS or tok in DROP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def expand_query(text: str) -> list[str]:
    tokens = normalise(text)
    expanded: list[str] = []
    for tok in tokens:
        expanded.append(tok)
        expanded.extend(SYNONYMS.get(tok, []))
    return expanded


_ASSET_WORDS = {
    "equity": "equity stock share",
    "fixed_income": "bond fixed income",
    "alternatives": "alternative",
    "cash": "cash money market",
}


def document_text(inst: Instrument) -> str:
    cost = "cheap low cost" if inst.ter_pct <= 0.2 else ""
    return " ".join(
        [
            inst.name,
            inst.description,
            _ASSET_WORDS[inst.asset_class],
            inst.sub_asset_class.replace("_", " "),
            inst.region,
            inst.currency,
            "esg sustainable" if inst.esg_score >= 80 else "",
            cost,
        ]
    )


class Searcher(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return (ticker, score) pairs, best first."""
        ...


class BM25Searcher:
    name = "bm25"

    def __init__(self, instruments: list[Instrument]) -> None:
        self.tickers = [i.ticker for i in instruments]
        self.index = BM25Okapi([normalise(document_text(i)) for i in instruments])

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        scores = np.asarray(self.index.get_scores(expand_query(query)), dtype=float)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(self.tickers[k], round(float(scores[k]), 4)) for k in order if scores[k] > 0]


class EmbeddingSearcher:
    """Cosine similarity over embeddings from any OpenAI-compatible `/embeddings` endpoint.

    Optional and disabled by default; the embedding model slug must be one the endpoint
    actually serves (check with `pi models`).
    """

    name = "embeddings"

    def __init__(self, instruments: list[Instrument], client: object, model: str) -> None:
        from openai import OpenAI

        assert isinstance(client, OpenAI)
        self.client, self.model = client, model
        self.tickers = [i.ticker for i in instruments]
        self.matrix = self._embed([document_text(i) for i in instruments])

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vectors = np.array([d.embedding for d in resp.data], dtype=float)
        normed: np.ndarray = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return normed

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        scores = self.matrix @ self._embed([query])[0]
        order = np.argsort(-scores)[:limit]
        return [(self.tickers[k], round(float(scores[k]), 4)) for k in order]
