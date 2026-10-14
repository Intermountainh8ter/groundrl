from __future__ import annotations

from pathlib import Path

import pytest

from groundrl.dataset import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(DATA)
