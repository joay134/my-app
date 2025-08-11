#!/usr/bin/env python3
"""Daily A-shares analyzer.

Fetches the top turnover A-share stocks, computes technical indicators
and generates CSV/HTML reports under ``./reports``.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import List

import akshare as ak
import numpy as np
import pandas as pd
from jinja2 import Template


TOP_N = 20  # number of stocks in final report
SCAN_LIMIT = 50  # number of stocks to scan initially


def get_top_turnover(limit: int = SCAN_LIMIT) -> pd.DataFrame:
    """Return real-time spot data sorted by turnover amount."""
    spot = ak.stock_zh_a_spot_em()
    spot = spot.sort_values("成交额", ascending=False).head(limit)
    spot = spot[["代码", "名称", "成交额"]]
    spot.columns = ["code", "name", "turnover"]
    return spot


def fetch_history(code: str, start: str, end: str) -> pd.DataFrame:
    hist = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    hist = hist[["日期", "收盘", "成交量"]]
    hist.columns = ["date", "close", "volume"]
    hist["date"] = pd.to_datetime(hist["date"])
    hist.set_index("date", inplace=True)
    return hist


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA60"] = df["close"].rolling(window=60).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["20d_high"] = df["close"].rolling(window=20).max().shift(1)
    df["breakout"] = df["close"] > df["20d_high"]

    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    df["vol_spike"] = df["volume"] > df["vol_ma20"] * 1.5

    return df


def score_row(row: pd.Series) -> int:
    score = 0
    if row.get("breakout"):
        score += 2
    if row.get("vol_spike"):
        score += 1
    if row.get("close") > row.get("MA20"):
        score += 1
    if row.get("MACD") > row.get("MACD_signal"):
        score += 1
    if row.get("RSI") > 50:
        score += 1
    return score


def generate_report(df: pd.DataFrame, date_str: str) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    csv_path = reports_dir / f"report_{date_str}.csv"
    df.to_csv(csv_path, index=False)

    template = Template(
        """
        <html>
        <head><meta charset="utf-8"><title>Daily A-shares Report {{ date }}</title></head>
        <body>
        <h1>Daily A-shares Report {{ date }}</h1>
        {{ table | safe }}
        </body>
        </html>
        """
    )
    html_content = template.render(date=date_str, table=df.to_html(index=False))
    html_path = reports_dir / f"report_{date_str}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main() -> None:
    today = dt.date.today()
    start = (today - dt.timedelta(days=365 * 2)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    spot = get_top_turnover()
    results: List[dict] = []
    for _, row in spot.iterrows():
        code = row["code"]
        try:
            hist = fetch_history(code, start, end)
            hist = compute_indicators(hist)
            latest = hist.iloc[-1]
            score = score_row(latest)
            results.append(
                {
                    "code": code,
                    "name": row["name"],
                    "turnover": row["turnover"],
                    "close": latest["close"],
                    "MA20": latest["MA20"],
                    "MA60": latest["MA60"],
                    "MACD": latest["MACD"],
                    "RSI": latest["RSI"],
                    "breakout": latest["breakout"],
                    "vol_spike": latest["vol_spike"],
                    "score": score,
                }
            )
        except Exception as err:  # pragma: no cover - network/API errors
            print(f"Failed to process {code}: {err}")
            continue

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("score", ascending=False).head(TOP_N)

    generate_report(df_result, today.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()
