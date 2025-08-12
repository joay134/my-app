#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CN A-shares analyzer with relaxed entry rule, tunable weights, and
a lightweight historical snapshot evaluation.

变更要点：
1) 入场阈值放宽（默认不硬筛；加 --entry-only 才启用硬过滤）：
   - MA20 > MA60
   - close / MA20 ∈ [-4%, +2%]
   - RSI14 ∈ [45, 70]
   - 20日新高 ≤ 5 天
2) 可调权重（含 --w-ml=0.30 默认值），ML 真参与排序
3) 轻量历史回测快照：对“最近 N 天”的样本，用本次训练好的模型计算
   - 按天 Top10/Top20 命中率（y = next_day_return>0）
   - 阈值命中率（proba>0.55 / 0.60）
   打印在日志，并写入 HTML 报告“参数/快照”表。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
from jinja2 import Template

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ----------------------------- 基础工具 -----------------------------

def zscore(s: pd.Series) -> pd.Series:
    m, std = s.mean(), s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0] * len(s), index=s.index)
    return (s - m) / std


def safe_to_datetime(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


# --------------------------- 市场侧数据 ----------------------------

def get_top_turnover(limit: int) -> pd.DataFrame:
    """按成交额排序取前 N 名"""
    spot = ak.stock_zh_a_spot_em()
    spot = spot.sort_values("成交额", ascending=False).head(limit)
    cols = ["代码", "名称", "最新价", "涨跌幅", "成交额"]
    spot = spot[cols]
    spot.columns = ["code", "name", "close", "change_pct", "turnover"]
    return spot


def stock_industry(code: str) -> Optional[str]:
    """获取单票行业"""
    try:
        info = ak.stock_individual_info_em(symbol=code)
        return str(info.loc[info["item"] == "所属行业", "value"].iloc[0])
    except Exception:
        return None


def get_industry_mapping() -> Dict[str, str]:
    """行业中文名 → 行业板块代码"""
    boards = ak.stock_board_industry_name_em()
    return dict(zip(boards["板块名称"], boards["板块代码"]))


def compute_industry_scores(stocks: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """行业热度：最近 10 天成交额动量 + 5 日涨幅"""
    mapping = get_industry_mapping()
    today = dt.date.today()
    start = (today - dt.timedelta(days=40)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    rec = {}
    for ind in stocks["industry"].dropna().unique():
        code = mapping.get(ind)
        if not code:
            continue
        try:
            hist = ak.stock_board_industry_hist_em(symbol=code, start_date=start, end_date=end)
            hist["日期"] = pd.to_datetime(hist["日期"])
            hist.sort_values("日期", inplace=True)
            hist = hist.tail(10)
            if len(hist) < 10:
                continue
            last5 = hist["成交额"].tail(5).sum()
            prev5 = hist["成交额"].head(5).sum()
            delta5 = (last5 - prev5) / prev5 if prev5 else 0.0
            ret5 = (hist["收盘"].iloc[-1] - hist["收盘"].iloc[4]) / hist["收盘"].iloc[4]
            rec[ind] = {"turnover_delta5": delta5, "return5": ret5}
        except Exception:
            continue

    ind_df = pd.DataFrame(rec).T
    if ind_df.empty:
        return ind_df
    ind_df["IndustryScore"] = zscore(ind_df["turnover_delta5"]) + 0.5 * zscore(ind_df["return5"])
    ind_df.sort_values("IndustryScore", ascending=False, inplace=True)
    return ind_df.head(top_k)


def fetch_history(code: str, start: str, end: str) -> pd.DataFrame:
    """日线前复权"""
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    df = df[cols]
    df.columns = ["date", "open", "close", "high", "low", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """核心技术指标"""
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rs = pd.Series(gain, index=df.index).rolling(14).mean() / pd.Series(loss, index=df.index).rolling(14).mean()
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["20d_high"] = df["close"].rolling(20).max().shift(1)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_ma20"] * 1.5

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    return df


def days_since_high20(df: pd.DataFrame) -> int:
    win = df["close"].iloc[-20:]
    if win.empty:
        return 999
    last_high = win.max()
    high_date = win[win == last_high].index[-1]
    return (df.index[-1] - high_date).days


# --------------------------- 入场与评分 ----------------------------

def entry_flag_relaxed(latest: pd.Series) -> int:
    """
    放宽版入场（默认用于加分，不做硬过滤）：
    - MA20>MA60
    - close/MA20 ∈ [-4%, +2%]
    - RSI ∈ [45, 70]
    - 20日新高 ≤ 5 天
    """
    if pd.isna(latest["MA20"]) or pd.isna(latest["MA60"]) or pd.isna(latest["RSI14"]):
        return 0
    cond1 = latest["MA20"] > latest["MA60"]
    cond2 = -0.04 <= latest["close"] / latest["MA20"] - 1 <= 0.02
    cond3 = 45 <= latest["RSI14"] <= 70
    cond4 = latest.get("days_since_high20", 999) <= 5
    return int(cond1 and cond2 and cond3 and cond4)


def tech_score(latest: pd.Series) -> float:
    """规则分（0~1），与 relaxed 条件一致"""
    sig = [
        1 if latest["MA20"] > latest["MA60"] else 0,
        1 if -0.04 <= latest["close"] / latest["MA20"] - 1 <= 0.02 else 0,
        1 if 45 <= latest["RSI14"] <= 70 else 0,
        1 if latest.get("days_since_high20", 999) <= 5 else 0,
    ]
    return sum(sig) / 4.0


# -------------------------- ML 特征与训练 --------------------------

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    用前一日特征预测 next_day_return>0
    特征：ret1, ret5, volume_change, macd_hist(近似), rsi14, close/ma20, close/ma60, gap_to_20d_high
    """
    out = df.copy()

    out["ret1"] = out["close"].pct_change(1)
    out["ret5"] = out["close"].pct_change(5)

    out["vol_chg"] = out["volume"].pct_change(5)

    # 简化 MACD-Hist：EMA12-EMA26 差的变化近似
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (dif - dea)

    out["rsi14"] = out["RSI14"]
    out["c_ma20"] = out["close"] / out["MA20"] - 1
    out["c_ma60"] = out["close"] / out["MA60"] - 1
    out["gap_h20"] = out["close"] / out["20d_high"] - 1

    # y：下一日收益 > 0
    out["next_ret"] = out["close"].pct_change().shift(-1)
    y = (out["next_ret"] > 0).astype(int)

    feats = ["ret1", "ret5", "vol_chg", "macd_hist", "rsi14", "c_ma20", "c_ma60", "gap_h20"]
    X = out[feats]
    # 用 t-1 的特征预测 t 的方向：整体向下移 1
    X = X.shift(1)
    # 清 NaN
    good = ~X.isna().any(axis=1) & ~y.isna()
    return X.loc[good], y.loc[good], out.index.to_series().loc[good]


def fit_cross_section(
    codes: List[str],
    start_date: str,
    end_date: str,
    lookback_days: int,
    min_rows: int,
    eval_days: int = 60,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    跨票拼接训练 + 轻量历史快照（最近 eval_days 的样本）
    返回：
      model: 已拟合 Pipeline
      snap:  DataFrame[date, code, y, proba]（最近 eval_days 内的样本）
    """
    X_all, y_all, meta = [], [], []
    for code in codes:
        try:
            hist = fetch_history(code, start_date, end_date)
            hist = compute_technicals(hist)
            X, y, idx = build_features(hist)
            if len(X) < min_rows:
                continue
            # 截取 lookback_days
            if lookback_days > 0:
                X = X.tail(lookback_days)
                y = y.loc[X.index]
                idx = idx.loc[X.index]
            X_all.append(X)
            y_all.append(y)
            meta.append(pd.DataFrame({"date": idx, "code": code, "y": y}))
        except Exception:
            continue

    if not X_all:
        raise RuntimeError("训练样本为空，可能数据拉取失败或过滤过严。")

    X_all = pd.concat(X_all, axis=0)
    y_all = pd.concat(y_all, axis=0)
    meta = pd.concat(meta, axis=0)
    # 对齐索引
    meta = meta.loc[X_all.index]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(X_all.values, y_all.values)

    # 轻量历史快照：取“最近 eval_days 天”的样本做统计
    if eval_days > 0:
        # 找出最近 eval_days 个交易日（按日期去重）
        dates = meta["date"].drop_duplicates().sort_values().tail(eval_days)
        snap_idx = meta["date"].isin(dates)
        proba = model.predict_proba(X_all.loc[snap_idx].values)[:, 1]
        snap = meta.loc[snap_idx].copy()
        snap["proba"] = proba
    else:
        snap = pd.DataFrame(columns=["date", "code", "y", "proba"])

    print(f"training samples: {len(X_all)} (unique days ~{meta['date'].nunique()})")
    return model, snap


def snapshot_metrics(snap: pd.DataFrame) -> Dict[str, float]:
    """
    从 snap（最近 N 天样本）计算：
      - Top10/Top20（按天截面）命中率
      - 阈值 0.55/0.60 命中率（整体）
    """
    if snap.empty:
        return {"top10_win": np.nan, "top20_win": np.nan, "thr055": np.nan, "thr060": np.nan, "days": 0}

    by_day = []
    for d, g in snap.groupby("date"):
        g = g.sort_values("proba", ascending=False)
        top10 = g.head(10)
        top20 = g.head(20)
        if len(top10) > 0:
            by_day.append(
                {
                    "date": d,
                    "top10_win": top10["y"].mean(),
                    "top20_win": top20["y"].mean(),
                }
            )
    df_day = pd.DataFrame(by_day)
    top10_win = df_day["top10_win"].mean() if not df_day.empty else np.nan
    top20_win = df_day["top20_win"].mean() if not df_day.empty else np.nan

    thr055 = snap.loc[snap["proba"] > 0.55, "y"].mean() if (snap["proba"] > 0.55).any() else np.nan
    thr060 = snap.loc[snap["proba"] > 0.60, "y"].mean() if (snap["proba"] > 0.60).any() else np.nan

    return {"top10_win": top10_win, "top20_win": top20_win, "thr055": thr055, "thr060": thr060, "days": df_day.shape[0]}


# ------------------------------- 报告 -------------------------------

def generate_report(
    date_str: str,
    params_table: pd.DataFrame,
    ind_table: pd.DataFrame,
    stocks_table: pd.DataFrame,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    csv_path = reports_dir / f"report_{date_str}.csv"
    stocks_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    template = Template(
        """
        <html>
        <head>
          <meta charset="utf-8" />
          <title>每日选股报告 {{ date }}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
            table { border-collapse: collapse; width: 100%; margin: 8px 0; }
            th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
            th:first-child, td:first-child { text-align: left; }
            h1,h2{ margin: 8px 0; }
            .muted { color: #666; font-size: 12px; }
          </style>
        </head>
        <body>
          <h1>每日选股报告 {{ date }}</h1>

          <h2>参数 & 快照</h2>
          {{ params | safe }}

          <h2>行业热度</h2>
          {{ inds | safe }}

          <h2>股票列表</h2>
          {{ stocks | safe }}

          <p class="muted">
            说明：行业分=行业动量与5日涨幅的加权；综合分=加权规则分与模型概率分的排序融合。<br/>
            历史快照为“使用本次模型对最近N天样本”的近似评估，仅用于风格确认，非完整回测结果。
          </p>
        </body>
        </html>
        """
    )
    html = template.render(
        date=date_str,
        params=params_table.to_html(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x),
        inds=ind_table.to_html(index=True, float_format="{:.2f}".format),
        stocks=stocks_table.to_html(index=False, float_format="{:.2f}".format),
    )
    (reports_dir / f"report_{date_str}.html").write_text(html, encoding="utf-8")


# ------------------------------- 主流程 ------------------------------

def main():
    p = argparse.ArgumentParser(description="CN A-shares analyzer")

    # 池子与行业
    p.add_argument("--top", type=int, default=300)
    p.add_argument("--ind-top-k", type=int, default=5)
    p.add_argument("--per-industry-top", type=int, default=2)

    # 权重
    p.add_argument("--w-ind", type=float, default=0.30)
    p.add_argument("--w-flow", type=float, default=0.00)  # 目前未用，可留作扩展
    p.add_argument("--w-fund", type=float, default=0.00)  # 目前未用，可留作扩展
    p.add_argument("--w-tech", type=float, default=0.40)
    p.add_argument("--w-ml", type=float, default=0.30)

    # ML/评估
    p.add_argument("--ml-lookback-days", type=int, default=1200)
    p.add_argument("--ml-min-rows", type=int, default=50)
    p.add_argument("--eval-days", type=int, default=60)   # 历史快照窗口

    # 筛选
    p.add_argument("--entry-only", action="store_true", help="仅保留满足放宽入场条件的票（可能导致无票）")

    args = p.parse_args()

    today = dt.date.today()
    start = (today - dt.timedelta(days=365 * 3)).strftime("%Y%m%d")  # 拉 3 年，训练再截断
    end = today.strftime("%Y%m%d")

    # 1) 池子
    spot = get_top_turnover(args.top)
    spot["industry"] = [stock_industry(c) for c in spot["code"]]

    # 2) 行业热度
    ind_scores = compute_industry_scores(spot, args.ind_top_k)
    top_inds = ind_scores.index.tolist() if not ind_scores.empty else []

    # 3) 训练 ML（用池子里股票）
    codes = spot["code"].tolist()
    model, snap = fit_cross_section(
        codes=codes,
        start_date=start,
        end_date=end,
        lookback_days=args.ml_lookback_days,
        min_rows=args.ml_min_rows,
        eval_days=args.eval_days,
    )
    snap_stat = snapshot_metrics(snap)
    print(f"[snapshot] days={snap_stat['days']}, "
          f"Top10={snap_stat['top10_win']:.2%} Top20={snap_stat['top20_win']:.2%}, "
          f"thr>0.55={snap_stat['thr055']:.2%} thr>0.60={snap_stat['thr060']:.2%}")

    # 4) 逐票打分
    results = []
    for _, row in spot.iterrows():
        if top_inds and row["industry"] not in top_inds:
            continue
        code = row["code"]
        try:
            hist = fetch_history(code, start, end)
            hist = compute_technicals(hist)
            if len(hist) < 120:
                continue
            latest = hist.iloc[-1]
            latest["days_since_high20"] = days_since_high20(hist)
            tscore = tech_score(latest)

            # ML 概率：用最近一行(t-1)特征
            X, y, idx = build_features(hist)
            up_prob = np.nan
            if not X.empty:
                x_last = X.tail(1).values
                up_prob = float(model.predict_proba(x_last)[:, 1][0])

            eflag = entry_flag_relaxed(latest)
            results.append(
                {
                    "code": code,
                    "name": row["name"],
                    "industry": row["industry"],
                    "close": latest["close"],
                    "change_pct": row["change_pct"],
                    "IndustryScore": ind_scores.loc[row["industry"], "IndustryScore"] if row["industry"] in ind_scores.index else 0.0,
                    "RuleScore": tscore,
                    "UpProb": round(up_prob, 4) if pd.notna(up_prob) else np.nan,
                    "Entry": eflag,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        # 输出“无候选”的报告，附参数与快照数据
        params = pd.DataFrame(
            [
                ["运行时间", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["候选数量 top", args.top],
                ["行业保留 ind-top-k", args.ind_top_k],
                ["每行业上限 per-industry-top", args.per_industry_top],
                ["只入场 entry-only", "是" if args.entry_only else "否"],
                ["权重 w-ind/w-tech/w-ml", f"{args.w_ind}/{args.w_tech}/{args.w_ml}"],
                ["历史快照天数", args.eval_days],
                ["快照 Top10/Top20", f"{snap_stat['top10_win']:.2%}/{snap_stat['top20_win']:.2%}"],
                ["快照 thr>0.55/0.60", f"{snap_stat['thr055']:.2%}/{snap_stat['thr060']:.2%}"],
            ],
            columns=["参数", "取值"],
        )
        generate_report(
            today.strftime("%Y-%m-%d"),
            params_table=params,
            ind_table=(ind_scores[["turnover_delta5", "return5", "IndustryScore"]] if not ind_scores.empty else pd.DataFrame()),
            stocks_table=pd.DataFrame(columns=["无候选股票"]),
        )
        print("No candidates after filters. Report generated with parameters only.")
        return

    # 5) 规则+ML 融合排序
    df["z_ind"] = zscore(df["IndustryScore"])
    df["z_rule"] = zscore(df["RuleScore"])
    if df["UpProb"].notna().any():
        df["z_ml"] = zscore(df["UpProb"].fillna(df["UpProb"].median()))
    else:
        df["z_ml"] = 0.0

    composite = args.w_ind * df["z_ind"] + args.w_tech * df["z_rule"] + args.w_ml * df["z_ml"]
    df["Composite"] = composite

    # 硬过滤（可选）
    if args.entry_only:
        df = df[df["Entry"] == 1]
        if df.empty:
            print("Entry-only 过滤后无股票。")

    # 行业内限额
    if args.per_industry_top > 0 and "industry" in df.columns:
        out_rows = []
        for ind, g in df.sort_values("Composite", ascending=False).groupby("industry", as_index=False):
            out_rows.append(g.head(args.per_industry_top))
        df = pd.concat(out_rows, axis=0)

    df.sort_values("Composite", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["最终排名"] = range(1, len(df) + 1)

    # 6) 输出报告（参数 + 快照 + 行业 + 股票）
    params = pd.DataFrame(
        [
            ["运行时间", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["候选数量 top", args.top],
            ["行业保留 ind-top-k", args.ind_top_k],
            ["每行业上限 per-industry-top", args.per_industry_top],
            ["只入场 entry-only", "是" if args.entry_only else "否"],
            ["RSI区间", "45~70"],
            ["距20日新高上限(天)", "≤5"],
            ["相对MA20区间", "-4% ~ +2%"],
            ["权重 w-ind/w-tech/w-ml", f"{args.w_ind}/{args.w_tech}/{args.w_ml}"],
            ["历史快照天数", args.eval_days],
            ["快照 Top10/Top20", f"{snap_stat['top10_win']:.2%}/{snap_stat['top20_win']:.2%}"],
            ["快照 thr>0.55/0.60", f"{snap_stat['thr055']:.2%}/{snap_stat['thr060']:.2%}"],
        ],
        columns=["参数", "取值"],
    )

    # 行业表
    if not ind_scores.empty:
        ind_display = ind_scores[["turnover_delta5", "return5", "IndustryScore"]].copy()
        ind_display.columns = ["Δ5日成交额", "5日涨幅", "行业得分"]
    else:
        ind_display = pd.DataFrame()

    # 股票列重命名（中文）
    show = df[[
        "code", "name", "industry", "close", "change_pct",
        "IndustryScore", "RuleScore", "UpProb", "Entry", "Composite", "最终排名"
    ]].copy()
    show.columns = ["代码", "名称", "行业", "收盘", "涨跌幅", "行业分", "规则分", "上涨概率(UpProb)", "入场信号", "综合分", "最终排名"]

    generate_report(
        today.strftime("%Y-%m-%d"),
        params_table=params,
        ind_table=ind_display,
        stocks_table=show,
    )
    print("Report generated.")


if __name__ == "__main__":
    main()
