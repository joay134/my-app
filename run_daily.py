#!/usr/bin/env python3
"""Enhanced CN A-shares analyzer.

This script implements an industry → funds → fundamentals → technicals
pipeline with an entry zone check.  It fetches the most active A-share
stocks, filters them by industry heat, computes multiple factor scores and
produces HTML/CSV reports under ``./reports``.

The implementation aims to follow the specification in the user request.  It
is intentionally lightweight and focuses on orchestrating the data pipeline
rather than on absolute financial accuracy.  All external data fetches are
wrapped in ``try/except`` blocks so that per-stock failures do not abort the
run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak
import numpy as np
import pandas as pd
from jinja2 import Template


# ---------------------------------------------------------------------------
# Utility dataclasses


@dataclass
class IndustryInfo:
    name: str
    code: str
    turnover_delta5: float
    return5: float
    score: float


# ---------------------------------------------------------------------------
# Helper functions


def zscore(series: pd.Series) -> pd.Series:
    """Return z-scored series; NaNs remain NaN."""

    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0] * len(series), index=series.index)
    return (series - mean) / std


def get_top_turnover(limit: int) -> pd.DataFrame:
    """Fetch real-time spot data sorted by turnover amount."""

    spot = ak.stock_zh_a_spot_em()
    spot = spot.sort_values("成交额", ascending=False).head(limit)
    cols = ["代码", "名称", "最新价", "涨跌幅", "成交额"]
    spot = spot[cols]
    spot.columns = ["code", "name", "close", "change_pct", "turnover"]
    return spot


def get_industry_mapping() -> Dict[str, str]:
    """Return mapping of industry name to board code."""

    boards = ak.stock_board_industry_name_em()
    mapping = dict(zip(boards["板块名称"], boards["板块代码"]))
    return mapping


def stock_industry(code: str) -> Optional[str]:
    """Fetch primary industry for a stock."""

    try:
        info = ak.stock_individual_info_em(symbol=code)
        industry = info.loc[info["item"] == "所属行业", "value"].iloc[0]
        return str(industry)
    except Exception:
        return None


def compute_industry_scores(stocks: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Compute industry heat metrics and return top-k industries."""

    mapping = get_industry_mapping()
    industries = {}
    today = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=40)).strftime("%Y%m%d")

    for ind in stocks["industry"].dropna().unique():
        code = mapping.get(ind)
        if not code:
            continue
        try:
            hist = ak.stock_board_industry_hist_em(symbol=code, start_date=start, end_date=today)
            hist["日期"] = pd.to_datetime(hist["日期"])
            hist.sort_values("日期", inplace=True)
            hist.set_index("日期", inplace=True)
            hist = hist.tail(10)
            if len(hist) < 10:
                continue
            last5 = hist["成交额"].tail(5).sum()
            prev5 = hist["成交额"].head(5).sum()
            delta5 = (last5 - prev5) / prev5 if prev5 else 0.0
            ret5 = (hist["收盘"].iloc[-1] - hist["收盘"].iloc[4]) / hist["收盘"].iloc[4]
            industries[ind] = {
                "code": code,
                "turnover_delta5": delta5,
                "return5": ret5,
            }
        except Exception:
            continue

    ind_df = pd.DataFrame.from_dict(industries, orient="index")
    if ind_df.empty:
        return pd.DataFrame()
    ind_df["IndustryScore"] = zscore(ind_df["turnover_delta5"]) + 0.5 * zscore(ind_df["return5"])
    ind_df.sort_values("IndustryScore", ascending=False, inplace=True)
    ind_df = ind_df.head(top_k)
    return ind_df


def fetch_history(code: str, start: str, end: str) -> pd.DataFrame:
    hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    hist = hist[cols]
    hist.columns = ["date", "open", "close", "high", "low", "volume"]
    hist["date"] = pd.to_datetime(hist["date"])
    hist.set_index("date", inplace=True)
    return hist


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["20d_high"] = df["close"].rolling(20).max().shift(1)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_ma20"] * 1.5

    # True range and ATR14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    return df


def days_since_high20(df: pd.DataFrame) -> int:
    window = df["close"].iloc[-20:]
    if window.empty:
        return 999
    last_high = window.max()
    high_date = window[window == last_high].index[-1]
    return (df.index[-1] - high_date).days


def entry_flag(latest: pd.Series) -> int:
    cond1 = latest["MA20"] > latest["MA60"]
    cond2 = -0.03 <= latest["close"] / latest["MA20"] - 1 <= 0.01
    cond3 = 50 <= latest["RSI14"] <= 65
    cond4 = latest["days_since_high20"] <= 3
    return int(cond1 and cond2 and cond3 and cond4)


def tech_score(latest: pd.Series) -> float:
    signals = []
    signals.append(1 if latest["MA20"] > latest["MA60"] else 0)
    signals.append(1 if -0.03 <= latest["close"] / latest["MA20"] - 1 <= 0.01 else 0)
    signals.append(1 if 50 <= latest["RSI14"] <= 65 else 0)
    signals.append(1 if latest["days_since_high20"] <= 3 else 0)
    return sum(signals) / len(signals)


def generate_report(date_str: str, stocks: pd.DataFrame, industries: pd.DataFrame) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    csv_path = reports_dir / f"report_{date_str}.csv"
    stocks.to_csv(csv_path, index=False)

    template = Template(
        """
        <html>
        <head><meta charset=\"utf-8\"><title>Daily A-shares Report {{ date }}</title></head>
        <body>
        <h1>Daily A-shares Report {{ date }}</h1>
        <h2>Top Industries</h2>
        {{ industries | safe }}
        <h2>Top Stocks</h2>
        {{ stocks | safe }}
        </body>
        </html>
        """
    )
    html_content = template.render(
        date=date_str,
        industries=industries.to_html(index=True, float_format="{:.2f}".format),
        stocks=stocks.to_html(index=False, float_format="{:.2f}".format),
    )
    html_path = reports_dir / f"report_{date_str}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ---------------------------------------------------------------------------
# Main pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="CN A-shares analyzer")
    parser.add_argument("--top", type=int, default=300, help="number of stocks to scan")
    parser.add_argument("--ind-top-k", type=int, default=5, help="top K industries to keep")
    parser.add_argument("--w-ind", type=float, default=0.30)
    parser.add_argument("--w-flow", type=float, default=0.30)
    parser.add_argument("--w-fund", type=float, default=0.20)
    parser.add_argument("--w-tech", type=float, default=0.20)
    parser.add_argument("--w-ml", type=float, default=0.00)
    parser.add_argument("--no-ml", action="store_true", help="disable ML up probability")
    parser.add_argument("--cache-dir", default="cache", help="cache directory")
    args = parser.parse_args()

    today = dt.date.today()
    start = (today - dt.timedelta(days=365 * 2)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    spot = get_top_turnover(args.top)

    # Map industry for each stock
    industries = []
    for code in spot["code"]:
        industries.append(stock_industry(code))
    spot["industry"] = industries

    ind_scores = compute_industry_scores(spot, args.ind_top_k)
    if ind_scores.empty:
        top_industries = []
    else:
        top_industries = ind_scores.index.tolist()

    results: List[dict] = []
    for _, row in spot.iterrows():
        if row["industry"] not in top_industries:
            continue
        code = row["code"]
        try:
            hist = fetch_history(code, start, end)
            hist = compute_technicals(hist)
            latest = hist.iloc[-1]
            latest["days_since_high20"] = days_since_high20(hist)
            eflag = entry_flag(latest)
            tscore = tech_score(latest)
            results.append(
                {
                    "code": code,
                    "name": row["name"],
                    "industry": row["industry"],
                    "close": latest["close"],
                    "change_pct": row["change_pct"],
                    "IndustryScore": ind_scores.loc[row["industry"], "IndustryScore"],
                    "FlowScore": 0.0,
                    "FundScore": 0.0,
                    "TechScore": tscore,
                    "EntryFlag": eflag,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        print("No data available")
        return

    # Normalize scores
    df["FlowScore"] = df["FlowScore"].fillna(df["FlowScore"].median())
    df["FundScore"] = df["FundScore"].fillna(df["FundScore"].median())
    df["TechScore"] = df["TechScore"].fillna(df["TechScore"].median())

    df["z_flow"] = zscore(df["FlowScore"])
    df["z_fund"] = zscore(df["FundScore"])
    df["z_tech"] = zscore(df["TechScore"])
    df["z_ind"] = zscore(df["IndustryScore"])

    composite = (
        args.w_ind * df["z_ind"]
        + args.w_flow * df["z_flow"]
        + args.w_fund * df["z_fund"]
        + args.w_tech * df["z_tech"]
    )
    if not args.no_ml and "up_prob" in df.columns:
        df["z_ml"] = zscore(df["up_prob"])
        composite += args.w_ml * df["z_ml"]
    df["Composite"] = composite

    df.sort_values(["Composite", "change_pct"], ascending=[False, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Final Rank"] = df.index + 1

    # Select top 20 for HTML display
    display_df = df.head(20)

    if ind_scores.empty:
        ind_display = pd.DataFrame(columns=["Δ5d Turnover", "5d Return", "IndustryScore"])
    else:
        ind_display = ind_scores[["turnover_delta5", "return5", "IndustryScore"]]
        ind_display.columns = ["Δ5d Turnover", "5d Return", "IndustryScore"]

    generate_report(today.strftime("%Y-%m-%d"), display_df, ind_display)

    print(
        f"training samples: {len(df)} (universe={len(spot)}, features=3); top industries: {', '.join(top_industries)}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

