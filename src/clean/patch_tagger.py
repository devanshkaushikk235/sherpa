import re

import pandas as pd

OUT_COLS = ["game_id", "patch_id", "date", "title", "magnitude", "patch_type", "overlaps_prev"]


def tag_patches(candidates: pd.DataFrame, series: pd.DataFrame, window_days: int = 7,
                major_threshold: float = 0.20, overlap_days: int = 7,
                major_regex: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """M = (mean y_peak over days 0..W-1 - mean over days -W..-1) / mean pre.
    Patches without full pre/post windows (or with >2 NaN days in either) are dropped and counted.
    NOTE: patch_type uses the outcome (M), so use it for stratification, not as a predictor."""
    rows, dropped = [], 0
    for gid, p in candidates.groupby("game_id"):
        s = series[series["game_id"] == gid].set_index("date")["y_peak"].sort_index()
        p = p.sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
        gap = p["date"].diff().dt.days
        for i, r in p.iterrows():
            d = r["date"]
            pre = s.loc[d - pd.Timedelta(days=window_days): d - pd.Timedelta(days=1)]
            post = s.loc[d: d + pd.Timedelta(days=window_days - 1)]
            if (len(pre) < window_days or len(post) < window_days
                    or pre.isna().sum() > 2 or post.isna().sum() > 2 or not pre.mean() > 0):
                dropped += 1
                continue
            m = float((post.mean() - pre.mean()) / pre.mean())
            by_regex = bool(major_regex and re.search(major_regex, str(r["title"]), flags=re.I))
            major = abs(m) >= major_threshold or bool(r.get("force_major", False)) or by_regex
            rows.append({
                "game_id": gid, "patch_id": f"{gid}-{d:%Y%m%d}", "date": d, "title": str(r["title"])[:200],
                "magnitude": m, "patch_type": "major" if major else "minor",
                "overlaps_prev": bool(pd.notna(gap.iloc[i]) and gap.iloc[i] <= overlap_days),
            })
    if verbose and dropped:
        print(f"  patch_tagger: dropped {dropped} patches lacking full pre/post windows")
    out = pd.DataFrame(rows, columns=OUT_COLS)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out["magnitude"] = out["magnitude"].astype("float64")
    out["overlaps_prev"] = out["overlaps_prev"].astype(bool)
    out["game_id"] = out["game_id"].astype("object")
    return out
