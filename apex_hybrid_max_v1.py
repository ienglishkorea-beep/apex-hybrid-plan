from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =========================================================
# ENV / CONFIG
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "output/apex_hybrid_max_candidates.csv")

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0") or "0")  # 0 = unlimited
MIN_PRICE = float(os.getenv("MIN_PRICE", "12"))
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "1500000000"))  # 1.5B
MIN_DOLLAR_VOLUME_20D = float(os.getenv("MIN_DOLLAR_VOLUME_20D", "20000000"))  # 20M

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "420"))
TOP_WATCHLIST = int(os.getenv("TOP_WATCHLIST", "20"))
MAX_POSITIONS_NORMAL = int(os.getenv("MAX_POSITIONS_NORMAL", "6"))
MAX_POSITIONS_HALF = int(os.getenv("MAX_POSITIONS_HALF", "4"))

BREAKOUT_LOOKBACK_FAST = int(os.getenv("BREAKOUT_LOOKBACK_FAST", "20"))
BREAKOUT_LOOKBACK_SLOW = int(os.getenv("BREAKOUT_LOOKBACK_SLOW", "55"))
PULLBACK_LOOKBACK = int(os.getenv("PULLBACK_LOOKBACK", "10"))
PULLBACK_MAX_DRAWDOWN_PCT = float(os.getenv("PULLBACK_MAX_DRAWDOWN_PCT", "0.06"))  # 6%

BREAKOUT_CONFIRM_BUFFER = float(os.getenv("BREAKOUT_CONFIRM_BUFFER", "0.001"))  # 0.1%
VOLUME_CONFIRM_MULTIPLIER = float(os.getenv("VOLUME_CONFIRM_MULTIPLIER", "1.5"))
MAX_EXTENSION_FROM_50MA = float(os.getenv("MAX_EXTENSION_FROM_50MA", "0.25"))  # +25%
GAP_UP_BLOCK_PCT = float(os.getenv("GAP_UP_BLOCK_PCT", "0.05"))  # +5%

STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.08"))  # -8%
PYRAMID_STEP_1 = float(os.getenv("PYRAMID_STEP_1", "0.08"))
PYRAMID_STEP_2 = float(os.getenv("PYRAMID_STEP_2", "0.16"))

BREADTH_STRONG = float(os.getenv("BREADTH_STRONG", "60"))
BREADTH_WEAK = float(os.getenv("BREADTH_WEAK", "40"))

SPY_TICKER = "SPY"
QQQ_TICKER = "QQQ"

# score weights
W_RET_6M = 20.0
W_RET_3M = 15.0
W_RET_1M = 10.0
W_ACCEL_1 = 15.0
W_ACCEL_2 = 10.0
W_HIGH_PROX = 15.0
W_RS = 10.0
W_VOLUME = 5.0
W_TREND_QUALITY = 10.0

# output control
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") == "1"


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class MarketRegime:
    spy_close: float
    spy_sma50: float
    spy_sma200: float
    qqq_close: float
    qqq_sma200: float
    breadth_pct: float
    regime_score: int
    mode: str  # normal / half / defensive
    note: str


@dataclass
class CandidateRow:
    ticker: str
    name: str
    close: float
    market_cap: float
    dollar_volume_20d: float
    ret_6m_pct: float
    ret_3m_pct: float
    ret_1m_pct: float
    accel_1: float
    accel_2: float
    rs_3m_pct: float
    dist_52w_high_pct: float
    ext_from_50ma_pct: float
    avg_vol_20: float
    vol_ratio: float
    setup_type: str
    breakout_level: float
    entry_price: float
    stop_price: float
    pyramid_1_price: float
    pyramid_2_price: float
    risk_reward_note: str
    total_score: float
    score_ret_6m: float
    score_ret_3m: float
    score_ret_1m: float
    score_accel_1: float
    score_accel_2: float
    score_high_prox: float
    score_rs: float
    score_volume: float
    score_trend_quality: float


# =========================================================
# UTIL
# =========================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def send_telegram_message(text: str) -> None:
    if not SEND_TELEGRAM:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception:
        pass


def load_universe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"ticker", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV missing columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    else:
        df["market_cap"] = np.nan

    if MAX_SYMBOLS > 0:
        df = df.head(MAX_SYMBOLS).copy()

    return df.reset_index(drop=True)


def download_price_history(tickers: List[str], period_days: int) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out: Dict[str, pd.DataFrame] = {}

    if raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            cols = raw.get(ticker)
            if cols is None:
                continue
            df = cols.copy()
            if "Close" not in df.columns:
                continue
            df = df.dropna(subset=["Close"]).copy()
            if not df.empty:
                out[ticker] = df
    else:
        # single ticker fallback
        t = tickers[0]
        df = raw.copy()
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"]).copy()
            if not df.empty:
                out[t] = df

    return out


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b is None or b == 0 or pd.isna(a) or pd.isna(b):
        return default
    return float(a) / float(b)


def pct_change(a: float, b: float) -> float:
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (float(a) / float(b) - 1.0) * 100.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_from_bounds(value: float, lo: float, hi: float, max_points: float) -> float:
    if pd.isna(value):
        return 0.0
    if hi == lo:
        return max_points if value >= hi else 0.0
    x = clamp((value - lo) / (hi - lo), 0.0, 1.0)
    return x * max_points


def latest(series: pd.Series, n_back: int = 0) -> float:
    s = series.dropna()
    if len(s) <= n_back:
        return np.nan
    return float(s.iloc[-1 - n_back])


def rolling_high(series: pd.Series, window: int, exclude_current: bool = False) -> float:
    s = series.dropna()
    if len(s) < window + (1 if exclude_current else 0):
        return np.nan
    if exclude_current:
        s = s.iloc[:-1]
    return float(s.iloc[-window:].max())


def avg_dollar_volume(df: pd.DataFrame, window: int = 20) -> float:
    if "Close" not in df.columns or "Volume" not in df.columns:
        return np.nan
    dv = df["Close"] * df["Volume"]
    return float(dv.tail(window).mean())


# =========================================================
# MARKET REGIME
# =========================================================

def compute_breadth_from_universe(price_map: Dict[str, pd.DataFrame], tickers: List[str]) -> float:
    valid = 0
    above = 0

    for ticker in tickers:
        df = price_map.get(ticker)
        if df is None or len(df) < 60:
            continue
        close = latest(df["Close"])
        sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
        if pd.isna(close) or pd.isna(sma50):
            continue
        valid += 1
        if close > sma50:
            above += 1

    if valid == 0:
        return np.nan
    return above / valid * 100.0


def compute_market_regime(price_map: Dict[str, pd.DataFrame], universe_tickers: List[str]) -> MarketRegime:
    spy = price_map.get(SPY_TICKER)
    qqq = price_map.get(QQQ_TICKER)

    if spy is None or qqq is None:
        raise RuntimeError("Missing SPY/QQQ price history for regime.")

    if len(spy) < 220 or len(qqq) < 220:
        raise RuntimeError("Not enough SPY/QQQ history for regime.")

    spy_close = latest(spy["Close"])
    spy_sma50 = float(spy["Close"].rolling(50).mean().iloc[-1])
    spy_sma200 = float(spy["Close"].rolling(200).mean().iloc[-1])

    qqq_close = latest(qqq["Close"])
    qqq_sma200 = float(qqq["Close"].rolling(200).mean().iloc[-1])

    breadth_pct = compute_breadth_from_universe(price_map, universe_tickers)

    regime_score = 0
    if spy_close > spy_sma200:
        regime_score += 1
    if spy_sma50 > spy_sma200:
        regime_score += 1
    if qqq_close > qqq_sma200:
        regime_score += 1

    if regime_score >= 2 and breadth_pct >= BREADTH_STRONG:
        mode = "normal"
        note = "정상 공격"
    elif regime_score >= 2 and breadth_pct >= BREADTH_WEAK:
        mode = "half"
        note = "절반 공격"
    else:
        mode = "defensive"
        note = "방어 모드"

    return MarketRegime(
        spy_close=round(spy_close, 2),
        spy_sma50=round(spy_sma50, 2),
        spy_sma200=round(spy_sma200, 2),
        qqq_close=round(qqq_close, 2),
        qqq_sma200=round(qqq_sma200, 2),
        breadth_pct=round(float(breadth_pct), 2),
        regime_score=regime_score,
        mode=mode,
        note=note,
    )


# =========================================================
# CANDIDATE CALCULATION
# =========================================================

def passes_universe_hardcut(df: pd.DataFrame, market_cap_value: float) -> bool:
    if len(df) < 220:
        return False

    close = latest(df["Close"])
    if pd.isna(close) or close < MIN_PRICE:
        return False

    if not pd.isna(market_cap_value) and market_cap_value < MIN_MARKET_CAP:
        return False

    dv20 = avg_dollar_volume(df, 20)
    if pd.isna(dv20) or dv20 < MIN_DOLLAR_VOLUME_20D:
        return False

    return True


def calc_setup_type(df: pd.DataFrame) -> Tuple[str, float]:
    close = latest(df["Close"])
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    breakout_20 = rolling_high(high, BREAKOUT_LOOKBACK_FAST, exclude_current=True)
    breakout_55 = rolling_high(high, BREAKOUT_LOOKBACK_SLOW, exclude_current=True)

    avg_vol_20 = float(volume.tail(20).mean())
    today_vol = latest(volume)

    # A급: 20/55일 돌파
    if not pd.isna(breakout_20) and close > breakout_20 * (1 + BREAKOUT_CONFIRM_BUFFER):
        if today_vol >= avg_vol_20 * VOLUME_CONFIRM_MULTIPLIER:
            return "A_FAST_BREAKOUT", breakout_20

    if not pd.isna(breakout_55) and close > breakout_55 * (1 + BREAKOUT_CONFIRM_BUFFER):
        if today_vol >= avg_vol_20 * VOLUME_CONFIRM_MULTIPLIER:
            return "A_SLOW_BREAKOUT", breakout_55

    # B급: 얕은 눌림 후 재돌파
    recent_high = float(high.tail(PULLBACK_LOOKBACK).max())
    recent_low = float(low.tail(PULLBACK_LOOKBACK).min())
    pullback_depth = safe_div(recent_high - recent_low, recent_high, default=np.nan)

    if not pd.isna(breakout_20) and not pd.isna(pullback_depth):
        if pullback_depth <= PULLBACK_MAX_DRAWDOWN_PCT and close > recent_high * (1 + BREAKOUT_CONFIRM_BUFFER):
            return "B_PULLBACK_REBREAK", recent_high

    return "WATCH", np.nan


def calc_candidate(
    ticker: str,
    name: str,
    df: pd.DataFrame,
    market_cap_value: float,
    spy_df: pd.DataFrame,
) -> Optional[CandidateRow]:
    close = latest(df["Close"])
    high = df["High"]
    volume = df["Volume"]

    sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    sma150 = float(df["Close"].rolling(150).mean().iloc[-1])

    if pd.isna(close) or pd.isna(sma50) or pd.isna(sma150):
        return None

    # hardcuts
    if not (close > sma50 and sma50 > sma150):
        return None

    high_52w = float(high.tail(252).max())
    if pd.isna(high_52w) or high_52w <= 0:
        return None

    dist_52w_high_pct = pct_change(close, high_52w)
    if dist_52w_high_pct < -10.0:
        return None

    ext_from_50ma_pct = pct_change(close, sma50)
    if ext_from_50ma_pct > MAX_EXTENSION_FROM_50MA * 100:
        return None

    # returns
    close_21 = latest(df["Close"], 21)
    close_63 = latest(df["Close"], 63)
    close_126 = latest(df["Close"], 126)

    ret_1m_pct = pct_change(close, close_21)
    ret_3m_pct = pct_change(close, close_63)
    ret_6m_pct = pct_change(close, close_126)

    if any(pd.isna(x) for x in [ret_1m_pct, ret_3m_pct, ret_6m_pct]):
        return None

    # acceleration
    base_3m_monthly = ret_3m_pct / 3.0 if ret_3m_pct != 0 else np.nan
    base_6m_quarterly = ret_6m_pct / 2.0 if ret_6m_pct != 0 else np.nan

    accel_1 = safe_div(ret_1m_pct, base_3m_monthly, default=np.nan)
    accel_2 = safe_div(ret_3m_pct, base_6m_quarterly, default=np.nan)

    # RS vs SPY
    spy_close = latest(spy_df["Close"])
    spy_63 = latest(spy_df["Close"], 63)
    spy_ret_3m = pct_change(spy_close, spy_63)
    rs_3m_pct = ret_3m_pct - spy_ret_3m

    # volume
    avg_vol_20 = float(volume.tail(20).mean())
    vol_ratio = safe_div(latest(volume), avg_vol_20, default=np.nan)

    # setup
    setup_type, breakout_level = calc_setup_type(df)

    # trend quality
    trend_quality_raw = 0
    trend_quality_raw += 1 if close > sma50 else 0
    trend_quality_raw += 1 if sma50 > sma150 else 0
    trend_quality_raw += 1 if dist_52w_high_pct >= -10 else 0

    score_ret_6m = score_from_bounds(ret_6m_pct, 10, 80, W_RET_6M)
    score_ret_3m = score_from_bounds(ret_3m_pct, 5, 40, W_RET_3M)
    score_ret_1m = score_from_bounds(ret_1m_pct, 2, 20, W_RET_1M)

    score_accel_1 = score_from_bounds(accel_1, 0.7, 1.8, W_ACCEL_1)
    score_accel_2 = score_from_bounds(accel_2, 0.7, 1.6, W_ACCEL_2)

    # closer to high is better, dist is negative near 0
    high_prox_metric = max(0.0, 10.0 + dist_52w_high_pct)  # 10 when at high, 0 when -10%
    score_high_prox = score_from_bounds(high_prox_metric, 0, 10, W_HIGH_PROX)

    score_rs = score_from_bounds(rs_3m_pct, 0, 25, W_RS)
    score_volume = score_from_bounds(vol_ratio, 1.0, 2.5, W_VOLUME)
    score_trend_quality = score_from_bounds(trend_quality_raw, 0, 3, W_TREND_QUALITY)

    total_score = (
        score_ret_6m
        + score_ret_3m
        + score_ret_1m
        + score_accel_1
        + score_accel_2
        + score_high_prox
        + score_rs
        + score_volume
        + score_trend_quality
    )

    entry_price = breakout_level * (1 + BREAKOUT_CONFIRM_BUFFER) if not pd.isna(breakout_level) else close
    stop_price = entry_price * (1 - STOP_LOSS_PCT)
    pyramid_1_price = entry_price * (1 + PYRAMID_STEP_1)
    pyramid_2_price = entry_price * (1 + PYRAMID_STEP_2)

    risk_reward_note = "진입 후 +8%, +16% 피라미딩 / -8% 손절"

    return CandidateRow(
        ticker=ticker,
        name=name,
        close=round(close, 2),
        market_cap=float(market_cap_value) if not pd.isna(market_cap_value) else np.nan,
        dollar_volume_20d=round(avg_dollar_volume(df, 20), 2),
        ret_6m_pct=round(ret_6m_pct, 2),
        ret_3m_pct=round(ret_3m_pct, 2),
        ret_1m_pct=round(ret_1m_pct, 2),
        accel_1=round(accel_1, 2) if not pd.isna(accel_1) else np.nan,
        accel_2=round(accel_2, 2) if not pd.isna(accel_2) else np.nan,
        rs_3m_pct=round(rs_3m_pct, 2),
        dist_52w_high_pct=round(dist_52w_high_pct, 2),
        ext_from_50ma_pct=round(ext_from_50ma_pct, 2),
        avg_vol_20=round(avg_vol_20, 2),
        vol_ratio=round(vol_ratio, 2) if not pd.isna(vol_ratio) else np.nan,
        setup_type=setup_type,
        breakout_level=round(breakout_level, 2) if not pd.isna(breakout_level) else np.nan,
        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
        pyramid_1_price=round(pyramid_1_price, 2),
        pyramid_2_price=round(pyramid_2_price, 2),
        risk_reward_note=risk_reward_note,
        total_score=round(total_score, 2),
        score_ret_6m=round(score_ret_6m, 2),
        score_ret_3m=round(score_ret_3m, 2),
        score_ret_1m=round(score_ret_1m, 2),
        score_accel_1=round(score_accel_1, 2),
        score_accel_2=round(score_accel_2, 2),
        score_high_prox=round(score_high_prox, 2),
        score_rs=round(score_rs, 2),
        score_volume=round(score_volume, 2),
        score_trend_quality=round(score_trend_quality, 2),
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def build_candidates(universe: pd.DataFrame, price_map: Dict[str, pd.DataFrame], regime: MarketRegime) -> pd.DataFrame:
    spy_df = price_map[SPY_TICKER]
    rows: List[Dict] = []

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market_cap_value = row["market_cap"]

        if ticker in (SPY_TICKER, QQQ_TICKER):
            continue

        df = price_map.get(ticker)
        if df is None:
            continue

        if not passes_universe_hardcut(df, market_cap_value):
            continue

        candidate = calc_candidate(
            ticker=ticker,
            name=name,
            df=df,
            market_cap_value=market_cap_value,
            spy_df=spy_df,
        )
        if candidate is None:
            continue

        # 과열 갭 진입 금지 근사치: 오늘 시가가 breakout 대비 +5% 이상이면 제외
        day_open = latest(df["Open"])
        if not pd.isna(candidate.breakout_level) and not pd.isna(day_open):
            if day_open >= candidate.breakout_level * (1 + GAP_UP_BLOCK_PCT):
                continue

        rows.append(asdict(candidate))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # setup rank preference
    setup_rank = {
        "A_SLOW_BREAKOUT": 3,
        "A_FAST_BREAKOUT": 2,
        "B_PULLBACK_REBREAK": 1,
        "WATCH": 0,
    }
    out["setup_rank"] = out["setup_type"].map(setup_rank).fillna(0)
    out = out.sort_values(
        by=["setup_rank", "total_score", "ret_1m_pct", "rs_3m_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # regime-based max candidates
    if regime.mode == "normal":
        top_n = TOP_WATCHLIST
    elif regime.mode == "half":
        top_n = min(TOP_WATCHLIST, 12)
    else:
        top_n = min(TOP_WATCHLIST, 6)

    return out.head(top_n).copy()


def save_output(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_summary_message(regime: MarketRegime, df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("Apex Hybrid Max v1")
    lines.append(f"시각: {utc_now()}")
    lines.append("")
    lines.append(
        f"시장레짐: {regime.note} | SPY {regime.spy_close:.2f} / 50MA {regime.spy_sma50:.2f} / 200MA {regime.spy_sma200:.2f}"
    )
    lines.append(
        f"QQQ {regime.qqq_close:.2f} / 200MA {regime.qqq_sma200:.2f} | Breadth {regime.breadth_pct:.2f}%"
    )
    lines.append("")

    if df.empty:
        lines.append("후보 없음")
        return "\n".join(lines)

    lines.append(f"후보 수: {len(df)}")
    lines.append("상위 후보:")

    limit = min(6, len(df))
    for i in range(limit):
        row = df.iloc[i]
        lines.append(
            f"{i+1}. {row['ticker']} {row['name']} | 점수 {row['total_score']:.1f} | "
            f"셋업 {row['setup_type']} | 진입 {row['entry_price']:.2f} | 손절 {row['stop_price']:.2f} | "
            f"+8% {row['pyramid_1_price']:.2f} | +16% {row['pyramid_2_price']:.2f}"
        )

    if regime.mode == "normal":
        lines.append("")
        lines.append(f"운용: 최대 {MAX_POSITIONS_NORMAL}종목 정상 공격")
    elif regime.mode == "half":
        lines.append("")
        lines.append(f"운용: 최대 {MAX_POSITIONS_HALF}종목 절반 공격")
    else:
        lines.append("")
        lines.append("운용: 방어 모드, 신규진입 중단 권장")

    return "\n".join(lines)


def main() -> None:
    universe = load_universe(UNIVERSE_CSV)

    base_tickers = [SPY_TICKER, QQQ_TICKER]
    all_tickers = sorted(set(base_tickers + universe["ticker"].tolist()))

    print(f"[INFO] Downloading data for {len(all_tickers)} tickers...")
    price_map = download_price_history(all_tickers, LOOKBACK_DAYS)

    if SPY_TICKER not in price_map or QQQ_TICKER not in price_map:
        raise RuntimeError("SPY/QQQ download failed.")

    regime = compute_market_regime(price_map, universe["ticker"].tolist())
    candidates = build_candidates(universe, price_map, regime)

    save_output(candidates, OUTPUT_CSV)

    message = build_summary_message(regime, candidates)
    print(message)
    send_telegram_message(message)

    print("")
    print(f"[INFO] Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
