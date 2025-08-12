#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CN A-shares daily analyzer (industry -> funds -> fundamentals -> technicals),
optionally adds a next-day up-probability model (Logistic Regression).

- Robust to per-stock failures (won't abort the whole run)
- Outputs CSV/HTML under ./reports
- Chinese headers for the final HTML report
"""

from __future__ import annotations

# --------------------------- 内存/并发“保险丝” ---------------------------
import os

# 限制底层 BLAS 线程，避免 CPU/内存过度竞争（非常重要）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# -----------------------------------------------------------------------

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from jinja2 import Template

import akshare as ak


# --------------------------- sklearn 延迟导入 ---------------------------
def _lazy_import_sklearn():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    return Pipeline, StandardScaler, LogisticRegression


# --------------------------- 小工具 ---------------------------

def zscore(series: pd.Series) -> pd.Series:
    """z-score；若方差为 0 则返回 0 序列"""
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


# --------------------------- 数据获取 ---------------------------

def get_top_turnover(limit: int) -> pd.DataFrame:
    """按成交额排序取前 N"""
    spot = ak.stock_zh_a_spot_em()
    spot = spot.sort_values("成交额", ascending=False).head(limit)
    cols = ["代码", "名称", "最新价", "涨跌幅", "成交额"]
    spot = spot[cols].copy()
    spot.columns = ["code", "name", "close", "change_pct", "turnover"]
    return spot


def get_industry_mapping() -> Dict[str, str]:
    """行业中文名 -> 行业板块代码"""
    boards = ak.stock_board_industry_name_em()
    return dict(zip(boards["板块名称"], boards["板块代码"]))


def stock_industry(code: str) -> Optional[str]:
    """查询个股所属行业（中文）"""
    try:
        info = ak.stock_individual_info_em(symbol=code)
        industry = info.loc[info["item"] == "所属行业", "value"].iloc[0]
        return str(industry)
    except Exception:
        return None


def compute_industry_scores(stocks: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    简易行业热度：
    - 最近 10 天成交额滚动窗口，对比最近 5 天与前 5 天增幅
    - 5 日收盘涨幅
    - z-score 加权
    """
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
    """复权日线"""
    hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    hist = hist[cols].copy()
    hist.columns = ["date", "open", "close", "high", "low", "volume"]
    hist["date"] = pd.to_datetime(hist["date"])
    hist.set_index("date", inplace=True)
    hist[["open", "close", "high", "low", "volume"]] = hist[["open", "close", "high", "low", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return hist


# --------------------------- 技术指标 & 入场规则 ---------------------------

def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """MA20/MA60、RSI14、20日最高(shift1)、ATR14"""
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["high20"] = df["close"].rolling(20).max().shift(1)

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
    return int((df.index[-1] - high_date).days)


def entry_flag(latest: pd.Series) -> int:
    cond1 = safe_float(latest.get("MA20")) > safe_float(latest.get("MA60"))
    cond2 = -0.03 <= latest["close"] / latest.get("MA20", np.nan) - 1 <= 0.01
    cond3 = 50 <= safe_float(latest.get("RSI14")) <= 65
    cond4 = safe_float(latest.get("days_since_high20")) <= 3
    return int(bool(cond1 and cond2 and cond3 and cond4))


def tech_score(latest: pd.Series) -> float:
    s = 0
    s += 1 if safe_float(latest.get("MA20")) > safe_float(latest.get("MA60")) else 0
    s += 1 if -0.03 <= latest["close"] / latest.get("MA20", np.nan) - 1 <= 0.01 else 0
    s += 1 if 50 <= safe_float(latest.get("RSI14")) <= 65 else 0
    s += 1 if safe_float(latest.get("days_since_high20")) <= 3 else 0
    return s / 4.0


# --------------------------- ML 特征/训练 ---------------------------

def build_features_for_ml(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    用前一日特征预测下一日方向：
      ret1, ret5, volume_change(与ma5对比), rsi14, close/ma20-1, close/ma60-1, gap_to_20d_high(close/high20 - 1)
    目标: next_day_return > 0
    需要 >= 120 行有效数据
    """
    d = df.copy()
    d["ret1"] = d["close"].pct_change(1)
    d["ret5"] = d["close"].pct_change(5)
    d["vol_ma5"] = d["volume"].rolling(5).mean()
    d["vol_change"] = d["volume"] / d["vol_ma5"] - 1
    d["rsi14"] = d["RSI14"]
    d["c_ma20"] = d["close"] / d["MA20"] - 1
    d["c_ma60"] = d["close"] / d["MA60"] - 1
    d["gap_20h"] = d["close"] / d["high20"] - 1

    # 目标：下一日收益
    d["next_ret"] = d["close"].pct_change().shift(-1)
    d["y"] = (d["next_ret"] > 0).astype(np.int8)

    feats = ["ret1", "ret5", "vol_change", "rsi14", "c_ma20", "c_ma60", "gap_20h"]
    d = d[feats + ["y"]].dropna().iloc[:-1]  # 最后一行没 next_ret
    if len(d) < 120:
        return np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.int8)

    X = d[feats].astype(np.float32).values
    y = d["y"].astype(np.int8).values
    return X, y


def last_lagged_feature_row(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    取“预测当天”的一行特征（全部用 T-1 的信息）：
    注意：build_features_for_ml 已经把特征对齐到 T-1，
    所以这里只需要取倒数第一行（对应上一交易日）。
    """
    d = df.copy()
    d["ret1"] = d["close"].pct_change(1)
    d["ret5"] = d["close"].pct_change(5)
    d["vol_ma5"] = d["volume"].rolling(5).mean()
    d["vol_change"] = d["volume"] / d["vol_ma5"] - 1
    d["rsi14"] = d["RSI14"]
    d["c_ma20"] = d["close"] / d["MA20"] - 1
    d["c_ma60"] = d["close"] / d["MA60"] - 1
    d["gap_20h"] = d["close"] / d["high20"] - 1
    feats = ["ret1", "ret5", "vol_change", "rsi14", "c_ma20", "c_ma60", "gap_20h"]
    d = d[feats].dropna()
    if d.empty:
        return None
    return d.iloc[-1].astype(np.float32).values.reshape(1, -1)


def fit_cross_section(codes: List[str], start_date: str, end_date: str,
                      args) -> Optional[object]:
    """
    横截面合并训练：
    - 对每只股票取历史 -> 指标 -> 特征 -> (X,y)
    - 合并后做 float32 & int8，并随机下采样到 --max-train-samples
    - 训练 Logistic Regression
    """
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for code in codes:
        try:
            hist = fetch_history(code, start_date, end_date)
            hist = compute_technicals(hist)
            X, y = build_features_for_ml(hist)
            if len(X):
                X_all.append(X)
                y_all.append(y)
        except Exception:
            continue

    if not X_all:
        print("[ML] no training samples")
        return None

    X_all = np.concatenate(X_all, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(y_all, axis=0).astype(np.int8, copy=False)

    # -------- 内存关键：随机下采样 + 打印样本量 --------
    n = len(X_all)
    max_n = int(getattr(args, "max_train_samples", 60000))
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]
        print(f"[ML] training samples capped: {n} -> {len(X_all)}")
    else:
        print(f"[ML] training samples: {n}")

    try:
        Pipeline, StandardScaler, LogisticRegression = _lazy_import_sklearn()
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs"  # 通用稳定
            ))
        ])
        model.fit(X_all, y_all)
        return model
    except Exception as e:
        print(f"[ML] training failed: {e}")
        return None


# --------------------------- 报表 ---------------------------

def generate_report(date_str: str,
                    params: Dict[str, str],
                    industries: pd.DataFrame,
                    stocks: pd.DataFrame) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True, parents=True)

    # CSV
    stocks.to_csv(reports_dir / f"report_{date_str}.csv", index=False, encoding="utf-8-sig")

    # HTML
    template = Template(
        """
        <html><head><meta charset="utf-8"><title>每日选股报告 {{ date }}</title>
        <style>
          body{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC","Hiragino Sans GB","Microsoft YaHei", Helvetica, Arial, sans-serif; padding: 12px;}
          h1{font-size:22px;}
          table{border-collapse: collapse; width: 100%; font-size: 13px;}
          th,td{border:1px solid #ddd; padding:6px; text-align:right;}
          th:first-child, td:first-child{text-align:left}
          .note{color:#888; font-size:12px}
          .section{margin-top:14px;}
        </style></head>
        <body>
        <h1>每日选股报告 {{ date }}</h1>

        <div class="section">
          <table>
            <tr><th style="text-align:left">参数</th><th style="text-align:left">取值</th></tr>
            {% for k,v in params.items() %}
              <tr><td style="text-align:left">{{k}}</td><td style="text-align:left">{{v}}</td></tr>
            {% endfor %}
          </table>
        </div>

        <div class="section">
          <h3>行业热度</h3>
          {{ industries | safe }}
        </div>

        <div class="section">
          <h3>股票列表</h3>
          {{ stocks | safe }}
        </div>

        <div class="note section">
          说明：本报告仅用于研究，不构成投资建议。
        </div>
        </body></html>
        """
    )

    html = template.render(
        date=date_str,
        params=params,
        industries=industries.to_html(index=True, float_format="{:.2f}".format),
        stocks=stocks.to_html(index=False, float_format="{:.2f}".format),
    )
    with open(reports_dir / f"report_{date_str}.html", "w", encoding="utf-8") as f:
        f.write(html)


# --------------------------- 主流程 ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CN A-shares analyzer")

    # 宇宙池/行业/入场
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--ind-top-k", type=int, default=5)
    parser.add_argument("--per-industry-top", type=int, default=3)
    parser.add_argument("--entry-only", action="store_true", help="仅展示入场信号=1 的票")

    # 权重
    parser.add_argument("--w-ind", type=float, default=0.30)
    parser.add_argument("--w-flow", type=float, default=0.30)
    parser.add_argument("--w-fund", type=float, default=0.20)
    parser.add_argument("--w-tech", type=float, default=0.20)
    parser.add_argument("--w-ml", type=float, default=0.00)

    # ML 相关
    parser.add_argument("--ml-lookback-days", type=int, default=720)
    parser.add_argument("--ml-min-rows", type=int, default=50)
    parser.add_argument("--eval-days", type=int, default=40)
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=60000,
                        help="最大训练样本上限，超出则随机下采样（降内存）")

    args = parser.parse_args()

    today = dt.date.today()
    start_rule = (today - dt.timedelta(days=365 * 2)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    # 1) 取成交额 Top 池
    spot = get_top_turnover(args.top)

    # 2) 映射行业
    spot["industry"] = [stock_industry(c) for c in spot["code"]]

    # 3) 行业热度打分 & 选取 Top K 行业
    ind_scores = compute_industry_scores(spot, args.ind_top_k)
    top_inds = ind_scores.index.tolist() if not ind_scores.empty else []
    if not top_inds:
        print("[INFO] 无法计算行业热度或为空，直接退出。")
        return

    # 4) 只保留这些行业 & 行业内再保留前 N
    pool = spot[spot["industry"].isin(top_inds)].copy()
    pool = pool.sort_values("turnover", ascending=False)
    pool = pool.groupby("industry").head(args.per_industry_top).reset_index(drop=True)

    # 5) 训练 ML（可选）
    model = None
    if not args.no-ml and args.w_ml > 0:
        ml_start = (today - dt.timedelta(days=int(args.ml_lookback_days))).strftime("%Y%m%d")
        try:
            model = fit_cross_section(pool["code"].tolist(), ml_start, end, args)
        except Exception as e:
            print(f"[ML] fit failed: {e}")
            model = None

    # 6) 对每只股票跑技术 + 入场信号 +（可选）up_prob
    results: List[dict] = []
    for _, row in pool.iterrows():
        code = row["code"]
        try:
            hist = fetch_history(code, start_rule, end)
            hist = compute_technicals(hist)
            latest = hist.iloc[-1].copy()
            latest["days_since_high20"] = days_since_high20(hist)
            eflag = entry_flag(latest)
            tscore = tech_score(latest)

            up_prob = np.nan
            if model is not None:
                x_last = last_lagged_feature_row(hist)
                if x_last is not None:
                    try:
                        up_prob = float(model.predict_proba(x_last)[0, 1])
                    except Exception:
                        up_prob = np.nan

            results.append({
                "代码": code,
                "名称": row["name"],
                "行业": row["industry"],
                "收盘": latest["close"],
                "涨跌幅": row["change_pct"],
                "行业得分": ind_scores.loc[row["industry"], "IndustryScore"] if row["industry"] in ind_scores.index else 0.0,
                "资金分": 0.0,
                "基本面分": 0.0,
                "技术分": tscore,
                "入场信号": int(eflag),
                "上涨概率(Up Prob)": round(up_prob, 4) if up_prob == up_prob else np.nan,  # 保留 4 位
                "raw_up_prob": up_prob,
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        print("[INFO] 无候选股票")
        params = {
            "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "候选数量 top": str(args.top),
            "行业保留 ind-top-k": str(args.ind_top_k),
            "每行业上限 per-industry-top": str(args.per_industry_top),
            "仅入场 entry-only": "是" if args.entry_only else "否",
        }
        generate_report(today.strftime("%Y-%m-%d"), params, pd.DataFrame(), pd.DataFrame())
        return

    # 7) 规则分标准化 + 组合分
    for c in ["行业得分", "资金分", "基本面分", "技术分"]:
        if c not in df:
            df[c] = 0.0
        df[c] = df[c].fillna(df[c].median())
        df[f"z_{c}"] = zscore(df[c])

    composite = (args.w_ind * df["z_行业得分"] +
                 args.w_flow * df["z_资金分"] +
                 args.w_fund * df["z_基本面分"] +
                 args.w_tech * df["z_技术分"])

    if ("raw_up_prob" in df.columns) and df["raw_up_prob"].notna().any() and args.w_ml > 0:
        df["z_ml"] = zscore(df["raw_up_prob"].fillna(df["raw_up_prob"].median()))
        composite = composite + args.w_ml * df["z_ml"]

    df["综合分"] = composite
    df = df.sort_values(["综合分", "涨跌幅"], ascending=[False, False]).reset_index(drop=True)
    df["最终排名"] = df.index + 1

    if args.entry_only:
        df = df[df["入场信号"] == 1].reset_index(drop=True)

    # 展示列 & 中文抬头
    show_cols = ["代码", "名称", "行业", "收盘", "涨跌幅", "行业得分", "资金分", "基本面分", "技术分", "入场信号", "上涨概率(Up Prob)", "综合分", "最终排名"]
    for c in show_cols:
        if c not in df.columns:
            df[c] = np.nan
    df_display = df[show_cols].copy()

    # 行业展示
    if ind_scores.empty:
        ind_display = pd.DataFrame(columns=["Δ5日成交额", "5日涨幅", "行业得分"])
    else:
        ind_display = ind_scores[["turnover_delta5", "return5", "IndustryScore"]].copy()
        ind_display.columns = ["Δ5日成交额", "5日涨幅", "行业得分"]

    # 参数说明
    params = {
        "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "候选数量 top": str(args.top),
        "行业保留 ind-top-k": str(args.ind_top_k),
        "每行业上限 per-industry-top": str(args.per_industry_top),
        "仅入场 entry-only": "是" if args.entry_only else "否",
        "MA20带宽": "-3.00%～1.00%",
        "RSI区间": "50～65",
        "距20日新高上限": "3",
        "权重 w-ind/w-flow/w-fund/w-tech/w-ml": f"{args.w_ind}/{args.w_flow}/{args.w_fund}/{args.w_tech}/{args.w_ml}",
        "ML(样本截断)": f"{args.max_train_samples}",
    }

    generate_report(today.strftime("%Y-%m-%d"), params, ind_display, df_display)

    print(f"[DONE] candidates: {len(df_display)}; industries: {', '.join(ind_scores.index.tolist())}")


if __name__ == "__main__":
    main()
