"""Patch-date candidates from the official Steam News API (no key needed), plus a manual CSV.
Developer announcements are NOT all patches: the regex filter is a first pass, review the output.
Manual override: data/raw/<game_id>/patches_manual.csv with columns date,title[,force_major]."""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
UA = "liveops-telemetry-engine/1.0 (LiveOps Telemetry Client)"
CAND_COLS = ["game_id", "date", "title", "force_major"]


def fetch_news(app_id: int, raw_dir: Path, count: int = 100, max_pages: int = 60,
               sleep: float = 2.0, timeout: int = 30) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    items, enddate = [], None
    for _ in range(max_pages):
        params = {"appid": app_id, "count": count, "maxlength": 0}
        if enddate is not None:
            params["enddate"] = enddate
        r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        news = r.json().get("appnews", {}).get("newsitems", [])
        if not news:
            break
        items.extend(news)
        new_end = min(n["date"] for n in news) - 1
        if enddate is not None and new_end >= enddate:
            break
        enddate = new_end
        if len(news) < count:
            break
        time.sleep(sleep)
    path = raw_dir / "steam_news.json"
    path.write_text(json.dumps(items))
    return path


def parse_news(path: Path, game_id: str, feeds: list[str] | None = None) -> pd.DataFrame:
    items = json.loads(Path(path).read_text())
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=CAND_COLS)
    if feeds:
        df = df[df["feedname"].isin(feeds)]
    out = pd.DataFrame({
        "game_id": game_id,
        "date": pd.to_datetime(df["date"], unit="s", utc=True).dt.tz_localize(None).dt.normalize(),
        "title": df["title"].astype(str),
        "force_major": False,
    })
    return out


def load_patch_candidates(raw_dir: Path, game_id: str, patch_regex: str,
                          feeds: list[str] | None = None) -> pd.DataFrame:
    parts = []
    news = raw_dir / "steam_news.json"
    if news.exists():
        n = parse_news(news, game_id, feeds)
        n = n[n["title"].str.contains(patch_regex, case=False, regex=True)]
        parts.append(n)
    manual = raw_dir / "patches_manual.csv"
    if manual.exists():
        m = pd.read_csv(manual, parse_dates=["date"])
        m["game_id"] = game_id
        m["force_major"] = m["force_major"].fillna(False).astype(bool) if "force_major" in m else False
        parts.append(m[CAND_COLS])
    if not parts:
        return pd.DataFrame(columns=CAND_COLS)
    return pd.concat(parts, ignore_index=True)
