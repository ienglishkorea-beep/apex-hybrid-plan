from __future__ import annotations

import os
from io import StringIO
from typing import List

import pandas as pd
import requests


OUT_PATH = os.getenv("OUT_PATH", "data/universe.csv")
TIMEOUT = 20

HEADERS = {"User-Agent": "Mozilla/5.0"}

WIKI_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
]

EXCLUDE_SECTOR_KEYWORDS = [
    "REAL ESTATE",
]

EXCLUDE_INDUSTRY_KEYWORDS = [
    "REIT",
    "MORTGAGE REIT",
    "PROPERTY MANAGEMENT",
    "REAL ESTATE",
    "REAL ESTATE SERVICES",
    "REAL ESTATE DEVELOPMENT",
    "BIOTECH",
    "BIOTECHNOLOGY",
    "DRUG MANUFACTURERS - SPECIALTY & GENERIC",
    "DRUG MANUFACTURERS",
    "PHARMACEUTICAL",
]

EXCLUDE_NAME_KEYWORDS = [
    "ACQUISITION",
    "CAPITAL TRUST",
    "CHINA",
    "ADR",
    "SPAC",
    "FUND",
    "ETF",
    "REIT",
]

ALLOW_DOTTED = {"BRK.B"}


def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if t in ALLOW_DOTTED:
        return t
    return t.replace(".", "-")


def contains_any(text: str, keywords: List[str]) -> bool:
    text = str(text).upper()
    return any(k in text for k in keywords)


def fetch_table(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    if not tables:
        raise RuntimeError(f"No tables found at {url}")
    return tables[0]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    ticker_col = None
    name_col = None
    sector_col = None
    industry_col = None

    for c in df.columns:
        lc = c.lower()
        if lc in ("symbol", "ticker"):
            ticker_col = c
        elif "security" in lc or "company" in lc or "name" in lc:
            if name_col is None:
                name_col = c
        elif "gics sector" in lc or lc == "sector":
            sector_col = c
        elif "gics sub-industry" in lc or "sub-industry" in lc or lc == "industry":
            industry_col = c

    if ticker_col is None:
        raise RuntimeError("Ticker column not found")
    if name_col is None:
        raise RuntimeError("Name column not found")
    if sector_col is None:
        raise RuntimeError("Sector column not found")
    if industry_col is None:
        raise RuntimeError("Industry column not found")

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].astype(str).map(normalize_ticker)
    out["name"] = df[name_col].astype(str).str.strip()
    out["market_cap"] = pd.NA
    out["security_type"] = "Common Stock"
    out["country"] = "United States"
    out["sector"] = df[sector_col].astype(str).str.strip()
    out["industry"] = df[industry_col].astype(str).str.strip()
    return out


def main() -> None:
    frames = []
    for url in WIKI_URLS:
        df = fetch_table(url)
        frames.append(standardize_columns(df))

    universe = pd.concat(frames, ignore_index=True)
    universe = universe.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    # 기본 구조 제외
    universe = universe[~universe["ticker"].str.contains(r"/|\^", regex=True, na=False)].copy()

    # BRK.B는 허용, 나머지 점 포함 티커 제거
    keep_mask = []
    for t in universe["ticker"]:
        raw = str(t).replace("-", ".")
        if "." in raw and raw not in ALLOW_DOTTED:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    universe = universe[keep_mask].copy()

    # 이름/섹터/업종 제외
    universe = universe[
        ~universe["name"].map(lambda x: contains_any(x, EXCLUDE_NAME_KEYWORDS))
    ].copy()

    universe = universe[
        ~universe["sector"].map(lambda x: contains_any(x, EXCLUDE_SECTOR_KEYWORDS))
    ].copy()

    universe = universe[
        ~universe["industry"].map(lambda x: contains_any(x, EXCLUDE_INDUSTRY_KEYWORDS))
    ].copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    universe = universe.sort_values(["sector", "industry", "ticker"]).reset_index(drop=True)
    universe.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] saved: {OUT_PATH}")
    print(f"[INFO] count: {len(universe)}")


if __name__ == "__main__":
    main()
