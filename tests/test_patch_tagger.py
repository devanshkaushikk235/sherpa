import pandas as pd

from src import schemas
from src.clean.patch_tagger import tag_patches
from src.ingest import synthetic


def test_big_shift_is_major_small_is_minor(clean_game):
    start = clean_game["date"].iloc[0]
    c = synthetic.make_patch_candidates("g", [start + pd.Timedelta(days=200), start + pd.Timedelta(days=350)])
    p = tag_patches(c, clean_game, verbose=False)
    schemas.validate_patches(p)
    assert p.loc[0, "patch_type"] == "major" and p.loc[0, "magnitude"] > 0.2
    assert p.loc[1, "patch_type"] == "minor" and abs(p.loc[1, "magnitude"]) < 0.2


def test_overlap_flag(clean_game):
    start = clean_game["date"].iloc[0]
    c = synthetic.make_patch_candidates("g", [start + pd.Timedelta(days=d) for d in (100, 105, 130)])
    p = tag_patches(c, clean_game, verbose=False)
    assert list(p["overlaps_prev"]) == [False, True, False]


def test_edge_patches_dropped(clean_game):
    start, end = clean_game["date"].iloc[0], clean_game["date"].iloc[-1]
    c = synthetic.make_patch_candidates("g", [start + pd.Timedelta(days=3), end - pd.Timedelta(days=2),
                                              start + pd.Timedelta(days=100)])
    p = tag_patches(c, clean_game, verbose=False)
    assert len(p) == 1


def test_force_major_and_regex(clean_game):
    start = clean_game["date"].iloc[0]
    c = synthetic.make_patch_candidates("g", [start + pd.Timedelta(days=100), start + pd.Timedelta(days=150)],
                                        ["Tiny fix", "New Season launch"])
    c.loc[0, "force_major"] = True
    p = tag_patches(c, clean_game, major_regex="new season", verbose=False)
    assert (p["patch_type"] == "major").all()


def test_same_day_duplicates_collapse(clean_game):
    start = clean_game["date"].iloc[0]
    d = start + pd.Timedelta(days=100)
    c = synthetic.make_patch_candidates("g", [d, d])
    assert len(tag_patches(c, clean_game, verbose=False)) == 1
