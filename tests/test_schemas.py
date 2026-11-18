import pandas as pd
import pytest

from src import schemas


def test_series_rejects_gap(clean_game):
    with pytest.raises(ValueError, match="missing or irregular"):
        schemas.validate_series(clean_game.drop(index=10))


def test_series_rejects_wrong_dtype(clean_game):
    bad = clean_game.copy()
    bad["is_imputed"] = bad["is_imputed"].astype(int)
    with pytest.raises(ValueError, match="dtype"):
        schemas.validate_series(bad)


def test_series_rejects_duplicates(clean_game):
    with pytest.raises(ValueError):
        schemas.validate_series(pd.concat([clean_game, clean_game.iloc[[5]]]))


def test_patches_rejects_bad_type():
    p = pd.DataFrame({"game_id": ["g"], "patch_id": ["x"], "date": pd.to_datetime(["2022-01-01"]),
                      "title": ["t"], "magnitude": [0.1], "patch_type": ["huge"], "overlaps_prev": [False]})
    with pytest.raises(ValueError, match="major"):
        schemas.validate_patches(p)
