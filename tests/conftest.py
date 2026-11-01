from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pi.data.generate import GeneratedData, generate
from pi.data.store import PortfolioData, write_database

SEED = 42
END = date(2026, 10, 30)


@pytest.fixture(scope="session")
def generated(tmp_path_factory: pytest.TempPathFactory) -> GeneratedData:
    return generate(SEED, END, offline=True, cache_dir=tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def db_path(generated: GeneratedData, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "portfolio.duckdb"
    write_database(generated, path)
    return path


@pytest.fixture(scope="session")
def data(db_path: Path) -> PortfolioData:
    return PortfolioData.load(db_path)
