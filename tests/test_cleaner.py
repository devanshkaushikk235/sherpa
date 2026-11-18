import numpy as np
import pandas as pd

from src import schemas
from src.clean.series_cleaner import clean_series
from src.ingest import synthetic


def _daily(n=100):
    return synthetic.make_daily(n, "2022-01-01", seed=0)


def test_schema_and_continuity(clean_game):
    schemas.validate_series(clean_game)
    assert clean_game["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_short_gap_imputed_long_gap_not():
    d = _daily()
    d.loc[10:12, "y_peak"] = np.nan     # 3 days -> imputed
    d.loc[40:44, "y_peak"] = np.nan     # 5 days -> left missing
    s = clean_series(d, "g", 1, max_gap_days=3)
    assert s.loc[10:12, "is_imputed"].all() and s.loc[10:12, "y_peak"].notna().all()
    assert not s.loc[40:44, "is_imputed"].any() and s.loc[40:44, "y_peak"].isna().all()
    assert s.loc[40:44, "is_outage"].all()


def test_missing_dates_are_reindexed():
    d = _daily().drop(index=[20, 21, 60])
    s = clean_series(d, "g", 1)
    schemas.validate_series(s)
    assert len(s) == 100 and s.loc[[20, 21], "is_imputed"].all()


def test_zero_and_dip_flagged_but_upward_spike_kept():
    d = _daily()
    d.loc[30, "y_peak"] = 0.0
    d.loc[50, "y_peak"] = d["y_peak"].median() * 0.05      # outage dip
    d.loc[70, "y_peak"] = d["y_peak"].median() * 5         # legit launch spike
    s = clean_series(d, "g", 1)
    assert s.loc[[30, 50], "is_outage"].all()
    assert not s.loc[70, "is_outage"] and s.loc[70, "y_peak"] > d["y_peak"].median() * 4


def test_no_imputation_at_edges():
    d = _daily()
    d.loc[0:1, "y_peak"] = np.nan
    s = clean_series(d, "g", 1)
    assert s.loc[0:1, "y_peak"].isna().all() and not s.loc[0:1, "is_imputed"].any()
