#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CN A-shares daily analyzer with robust ML (Logistic Regression) and fallbacks.

Enhancements vs previous:
- More robust data fetching with retries/backoff
- Longer lookback (default 900 days) and lower per-stock min rows (default 60)
- Train once over cross-section, reuse histories to score latest Up Prob
- Graceful degradation: if ML fails, Up Prob = NaN and we rank by rule score only
- Chinese headers in CSV/HTML; parameter block printed to HTML
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
from jinja2 import Template
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------- Utils --------------------------------- #

def _backoff_sleep(attempt: int) -> None:
    # Exponential backoff with jitter
    time.sleep(min(2.0 * (1.5 ** attempt) + np.random.rand() * 0.5, 8.0))


def safe_call(fn, retries: int = 3, **kwargs):
    """Call akshare function with retries/backoff; return None if fails."""
    last_err = None
    for i in range(retries):
        try:
            return fn(**kwargs)
        except Exception as e:
            last_err = e
            _backoff_sleep(i)
    # print for CI logs
    print(f"[warn] {fn.__name__} failed after retries: {last_err}")
    return None


def zscore(series: pd.Series) -> pd.Series:
    m = series.mean()
    s = series.std(ddof=0)
    if s == 0 or (isinstance(s, float) and np.isnan(s)):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - m) / s


# ------------------------- Data Fetching ------------------------------ #

def get_top_turnover(limit: int) -> pd.DataFrame:
    """Fetch real-time spot data sorted by turnover amount."""
    spot = safe_call(ak.stock_zh_a_spot_em, retries=3)
    if spot is None or spot.empty:
        return pd.DataFrame(columns=["code", "name", "close", "change_pct", "turnover"])
    spot = spot.sort_values("成交额", ascending=False).head(limit)
    cols = ["代码", "名称", "最新价", "涨跌幅", "成交额"]
    spot = spot[cols].rename(
        columns={"代码": "code", "名称": "name", "最新价": "close", "涨跌幅": "change_pct", "成交额": "turnover"}
    )
    return spot


def get_industry_mapping() -> Dict[str, str]:
    boards = safe_call(ak.stock_board_industry_name_em, retries=3)
    if boards is None or boards.empty:
        return {}
    return dict(zip(boards["板块名称"], boards["板块代码"]))


def stock_industry(code: str) -> Optional[str]:
    """Try to get industry of a given code. Return None if fails."""
    info = safe_call(ak.stock_individual_info_em, retries=3, symbol=code)
    if info is None or info.empty:
        return None
    try:
        return str(info.loc[info["item"] == "所属行业", "value"].iloc[0])
    except Exception:
        return None


def fetch_history(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    df = safe_call(
        ak.stock_zh_a_hist,
        retries=3,
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if df is None or df.empty:
        return None
    # Standardize cols
    try:
        df = df[["日期", "开盘", "收盘", "最高", "最低", "成交量"]].copy()
        df.columns = ["date", "open", "close", "high", "low", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None


def compute_industry_scores(stocks: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Compute industry heat (Δ5日成交额 + 5日涨幅) and return top-k industries."""
    if stocks.empty:
        return pd.DataFrame()
    mapping = get_industry_mapping()
    end = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=40)).strftime("%Y%m%d")

    inds = {}
    for ind in stocks["industry"].dropna().unique():
        code = mapping.get(ind)
        if not code:
            continue
        hist = safe_call(ak.stock_board_industry_hist_em, retries=3, symbol=code, start_date=start, end_date=end)
        if hist is None or hist.empty:
            continue
        try:
            hist["日期"] = pd.to_datetime(hist["日期"])
            hist = hist.sort_values("日期").tail(10)
            if len(hist) < 10:
                continue
            last5 = hist["成交额"].tail(5).sum()
            prev5 = hist["成交额"].head(5).sum()
            delta5 = (last5 - prev5) / prev5 if prev5 else 0.0
            ret5 = (hist["收盘"].iloc[-1] - hist["收盘"].iloc[4]) / max(hist["收盘"].iloc[4], 1e-9)
            inds[ind] = {"code": code, "turnover_delta5": delta5, "return5": ret5}
        except Exception:
            continue
    if not inds:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(inds, orient="index")
    df["IndustryScore"] = zscore(df["turnover_delta5"]) + 0.5 * zscore(df["return5"])
    df.sort_values("IndustryScore", ascending=False, inplace=True)
    return df.head(max(1, top_k))


# ------------------------- Technicals & Features ---------------------- #

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA20/MA60/RSI14/20日最高/量均/ATR/MACD"""
    out = df.copy()
    out["MA20"] = out["close"].rolling(20).mean()
    out["MA60"] = out["close"].rolling(60).mean()

    # RSI14
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out["RSI14"] = 100 - (100 / (1 + rs))

    # 20-day high (shift 1 day to avoid leakage)
    out["high20"] = out["close"].rolling(20).max().shift(1)
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    out["vol_spike"] = out["volume"] > out["vol_ma20"] * 1.5

    # ATR14
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    # MACD (12,26,9)
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = diff - dea

    return out


def days_since_high20(df: pd.DataFrame) -> int:
    window = df["close"].iloc[-20:]
    if window.empty:
        return 999
    last_high = window.max()
    high_date = window[window == last_high].index[-1]
    return int((df.index[-1] - high_date).days)


def entry_flag(latest: pd.Series, ma20_band_low: float, ma20_band_high: float,
               rsi_low: float, rsi_high: float, days_since_high20_max: int) -> int:
    cond1 = latest["MA20"] > latest["MA60"]
    cond2 = ma20_band_low <= latest["close"] / latest["MA20"] - 1 <= ma20_band_high
    cond3 = rsi_low <= latest["RSI14"] <= rsi_high
    cond4 = latest["days_since_high20"] <= days_since_high20_max
    return int(cond1 and cond2 and cond3 and cond4)


def tech_score(latest: pd.Series, ma20_band_low: float, ma20_band_high: float,
               rsi_low: float, rsi_high: float, days_since_high20_max: int) -> float:
    s = 0
    s += 1 if latest["MA20"] > latest["MA60"] else 0
    s += 1 if ma20_band_low <= latest["close"] / latest["MA20"] - 1 <= ma20_band_high else 0
    s += 1 if rsi_low <= latest["RSI14"] <= rsi_high else 0
    s += 1 if latest["days_since_high20"] <= days_since_high20_max else 0
    return s / 4.0


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build lag-1 features and next-day direction label."""
    X = pd.DataFrame(index=df.index)

    X["ret1"] = df["close"].pct_change().shift(1)
    X["ret5"] = df["close"].pct_change(5).shift(1)
    X["vol_change"] = df["volume"].pct_change().shift(1)
    X["macd_hist"] = df["MACD_HIST"].shift(1)
    X["rsi14"] = df["RSI14"].shift(1)
    X["close_ma20"] = (df["close"] / df["MA20"]).shift(1)
    X["close_ma60"] = (df["close"] / df["MA60"]).shift(1)
    X["gap_to_high20"] = (df["close"] / df["high20"] - 1).shift(1)

    # Label: next-day positive return
    y = (df["close"].pct_change().shift(-1) > 0).astype(int)

    Xy = pd.concat([X, y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y = Xy["y"].astype(int)
    X = Xy.drop(columns=["y"])
    return X, y


# ------------------------- ML Training -------------------------------- #

def fit_cross_section(codes: List[str],
                      start_date: str,
                      end_date: str,
                      ml_min_rows: int) -> Tuple[Optional[Pipeline], Dict[str, pd.DataFrame], int, int]:
    """Fit LR on cross-section; return model, hist_map, n_samples, n_stocks_used."""
    X_all, y_all = [], []
    hist_map: Dict[str, pd.DataFrame] = {}
    used = 0

    for code in codes:
        df = fetch_history(code, start_date, end_date)
        if df is None or len(df) < ml_min_rows:
            continue
        df = add_technicals(df)
        X, y = build_features(df)
        if len(X) < ml_min_rows:
            continue
        hist_map[code] = df
        X_all.append(X)
        y_all.append(y)
        used += 1

    if not X_all:
        return None, hist_map, 0, 0

    X_all = pd.concat(X_all, axis=0)
    y_all = pd.concat(y_all, axis=0)

    try:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        )
        model.fit(X_all, y_all)
        print(f"training samples: {len(X_all)}, pos_rate={y_all.mean():.2f}, stocks={used}")
        return model, hist_map, int(len(X_all)), int(used)
    except Exception as e:
        print(f"[warn] ML training failed: {e}")
        return None, hist_map, 0, used


# ------------------------- Report ------------------------------------ #

def generate_report(date_str: str,
                    params: Dict[str, str],
                    industries: pd.DataFrame,
                    stocks: pd.DataFrame) -> None:
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    # CSV
    csv_path = reports / f"report_{date_str}.csv"
    stocks.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # HTML
    tmpl = Template("""
    <html><head>
      <meta charset="utf-8">
      <title>每日选股报告 {{ date }}</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; line-height: 1.5; padding: 16px; }
        table { border-collapse: collapse; width: 100%; margin: 12px 0; }
        th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
        th { background: #f6f8fa; }
        td:first-child, th:first-child { text-align: left; }
        h1 { margin: 0 0 8px 0; }
        .sub { color: #666; font-size: 13px; }
      </style>
    </head><body>
      <h1>每日选股报告 {{ date }}</h1>

      <h3>参数</h3>
      <table>
        <tbody>
        {% for k,v in params.items() %}
          <tr><td style="text-align:left">{{k}}</td><td>{{v}}</td></tr>
        {% endfor %}
        </tbody>
      </table>

      <h3>行业热度</h3>
      {{ industries_html | safe }}

      <h3>股票列表</h3>
      {{ stocks_html | safe }}

      <p class="sub">声明：仅供研究，不构成投资建议。</p>
    </body></html>
    """)

    industries_html = industries.to_html(
        index=True, float_format=lambda x: f"{x:.2f}", border=0, justify="center"
    )
    stocks_html = stocks.to_html(
        index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else x, border=0
    )

    html = tmpl.render(date=date_str, params=params, industries_html=industries_html, stocks_html=stocks_html)
    (reports / f"report_{date_str}.html").write_text(html, encoding="utf-8")


# ------------------------- Main -------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="A-shares daily analyzer with robust ML")
    # universe & industry
    parser.add_argument("--top", type=int, default=300, help="候选数量 top")
    parser.add_argument("--ind-top-k", type=int, default=5, help="行业保留 ind-top-k")
    parser.add_argument("--per-industry-top", type=int, default=2, help="每行业上限 per-industry-top")

    # entry rule
    parser.add_argument("--entry-only", action="store_true", help="只展示入场=1 的票")
    parser.add_argument("--ma20-band-low", type=float, default=-0.03)
    parser.add_argument("--ma20-band-high", type=float, default=0.01)
    parser.add_argument("--rsi-low", type=float, default=50.0)
    parser.add_argument("--rsi-high", type=float, default=65.0)
    parser.add_argument("--days-since-high20-max", type=int, default=3)

    # rule weights
    parser.add_argument("--w-ind", type=float, default=0.30)
    parser.add_argument("--w-flow", type=float, default=0.30)
    parser.add_argument("--w-fund", type=float, default=0.20)
    parser.add_argument("--w-tech", type=float, default=0.20)

    # ML
    parser.add_argument("--no-ml", action="store_true", help="禁用 ML")
    parser.add_argument("--w-ml", type=float, default=0.40, help="最终排名中 UpProb 权重（0~1）")
    parser.add_argument("--ml-min-rows", type=int, default=60, help="单票最少样本")
    parser.add_argument("--ml-lookback-days", type=int, default=900, help="回看天数（训练窗口）")

    args = parser.parse_args()

    today = dt.date.today()
    start_hist = (today - dt.timedelta(days=int(args.ml_lookback_days))).strftime("%Y%m%d")
    end_hist = today.strftime("%Y%m%d")

    # 1) Universe
    spot = get_top_turnover(args.top)
    if spot.empty:
        print("No spot data, exit.")
        return

    industries = [stock_industry(c) for c in spot["code"]]
    spot["industry"] = industries

    # 2) Industry heat & filter
    ind_scores = compute_industry_scores(spot, args.ind_top_k)
    top_ind_list = ind_scores.index.tolist() if not ind_scores.empty else []

    cand = spot[spot["industry"].isin(top_ind_list)].copy()
    if cand.empty:
        print("No candidates in hot industries.")
        return

    # 3) ML training (cross-section)
    model, hist_map, n_samples, n_used = (None, {}, 0, 0)
    ml_enabled = not args.no_ml
    if ml_enabled:
        model, hist_map, n_samples, n_used = fit_cross_section(
            codes=cand["code"].tolist(),
            start_date=start_hist,
            end_date=end_hist,
            ml_min_rows=args.ml_min_rows,
        )

    # 4) Score each candidate by rule + (optional) ML UpProb
    results = []
    for _, row in cand.iterrows():
        code = row["code"]
        ind = row["industry"]
        ind_score = ind_scores.loc[ind, "IndustryScore"] if ind in ind_scores.index else 0.0

        # history for this code (use trained map if exists, otherwise fetch fresh)
        if code in hist_map:
            df = hist_map[code]
        else:
            df = fetch_history(code, start_hist, end_hist)
            if df is None:
                continue
            df = add_technicals(df)

        if df.empty:
            continue

        latest = df.iloc[-1].copy()
        latest["days_since_high20"] = days_since_high20(df)

        ef = entry_flag(
            latest,
            args.ma20_band_low,
            args.ma20_band_high,
            args.rsi_low,
            args.rsi_high,
            args.days_since_high20_max,
        )
        ts = tech_score(
            latest,
            args.ma20_band_low,
            args.ma20_band_high,
            args.rsi_low,
            args.rsi_high,
            args.days_since_high20_max,
        )

        # rule-only composite (flow/fund placeholders=0，可自行接入净流/财报后改)
        flow_s, fund_s = 0.0, 0.0
        z_ind, z_flow, z_fund, z_tech = ind_score, flow_s, fund_s, ts  # 先不做 z 标准化，统一相对值
        rule_score = args.w_ind * z_ind + args.w_flow * z_flow + args.w_fund * z_fund + args.w_tech * z_tech

        up_prob = np.nan
        if ml_enabled and model is not None:
            try:
                X_last, _ = build_features(df)
                if len(X_last) >= 1:
                    p = model.predict_proba(X_last.tail(1))[:, 1][0]
                    up_prob = float(p)
            except Exception:
                up_prob = np.nan

        results.append(
            {
                "代码": code,
                "名称": row["name"],
                "行业": ind,
                "收盘": float(latest["close"]),
                "涨跌幅": float(row["change_pct"]),
                "行业得分": float(ind_score),
                "资金分": float(flow_s),
                "基本面分": float(fund_s),
                "技术分": float(ts),
                "入场信号": int(ef),
                "上涨概率(Up Prob)": up_prob,
                "规则分": float(rule_score),
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        print("No results to rank.")
        return

    # 5) Rank
    # 规则分取 zscore，UpProb 正向，最终排名 = (1-w_ml)*rank(规则分) + w_ml*rank(UpProb)
    df["规则分_z"] = zscore(df["规则分"])
    if df["上涨概率(Up Prob)"].notna().sum() > 0 and ml_enabled:
        r_rule = df["规则分_z"].rank(ascending=False, method="min")
        r_ml = df["上涨概率(Up Prob)"].rank(ascending=False, method="min")
        final_rank = (1 - args.w_ml) * r_rule + args.w_ml * r_ml
    else:
        final_rank = df["规则分_z"].rank(ascending=False, method="min")

    df["综合分"] = df["规则分_z"]
    df["最终排名"] = final_rank.astype(int)

    # entry-only
    if args.entry_only:
        df = df[df["入场信号"] == 1]

    df = df.sort_values(["最终排名", "综合分"], ascending=[True, False]).reset_index(drop=True)

    # 6) Industry display
    if ind_scores.empty:
        ind_display = pd.DataFrame(columns=["Δ5日成交额", "5日涨幅", "行业得分", "样本数"])
    else:
        ind_display = ind_scores[["turnover_delta5", "return5", "IndustryScore"]].copy()
        ind_display.columns = ["Δ5日成交额", "5日涨幅", "行业得分"]
        # 样本数
        cnt = df.groupby("行业")["代码"].count()
        ind_display["样本数"] = cnt.reindex(ind_display.index).fillna(0).astype(int)

    # 7) Report
    params = {
        "运行时间": today.strftime("%Y-%m-%d"),
        "候选数量 top": str(args.top),
        "行业保留 ind-top-k": str(args.ind_top_k),
        "每行业上限 per-industry-top": str(args.per_industry_top),
        "只看入场 entry-only": "是" if args.entry_only else "否",
        "MA20带宽": f"{args.ma20_band_low:.1%} ~ {args.ma20_band_high:.1%}",
        "RSI区间": f"{args.rsi_low:.0f} ~ {args.rsi_high:.0f}",
        "距20日新高上限": str(args.days_since_high20_max),
        "权重 w-ind/w-flow/w-fund/w-tech": f"{args.w_ind}/{args.w_flow}/{args.w_fund}/{args.w_tech}",
        "ML(上表权重)": "启用" if ml_enabled else "禁用",
        "训练样本(条)": str(n_samples),
        "参与训练股票数": str(n_used),
        "回看天数": str(args.ml_lookback_days),
        "过盈秩算法": f"(1-w_ml)*rank(规则分)+w_ml*rank(UpProb), w_ml={args.w_ml:.2f}",
    }

    # 重命名中文列顺序
    show_cols = [
        "代码", "名称", "行业", "收盘", "涨跌幅",
        "行业得分", "资金分", "基本面分", "技术分", "入场信号",
        "上涨概率(Up Prob)", "综合分", "最终排名",
    ]
    for c in show_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[show_cols]

    generate_report(today.strftime("%Y-%m-%d"), params, ind_display, df)

    print("Done. Top 5:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
