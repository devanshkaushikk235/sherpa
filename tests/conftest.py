import pandas as pd
import pytest

from src.clean.series_cleaner import clean_series
from src.ingest import synthetic


@pytest.fixture
def clean_game():
    daily = synthetic.make_daily(500, "2022-01-01", seed=1, shift_days=(200,), shift_size=0.5, decay=10)
    return clean_series(daily, "g", 1)
