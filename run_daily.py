#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A 股日选股分析（行业→资金→基本面→技术，含入场/每行业TopN，中文报表、网络容错、参数摘要、行业样本数、财务多源兜底、次日上涨概率）
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import akshare as ak
import numpy as np
import pandas as pd
from jinja2 import Template

# --- 机器学习 ---
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception as _:
    SKLEARN_OK = False


# ---------------------- 通用 ----------------------

def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0] * len(series), index=series.index)
    return (series - mean) / std


# ---------------------- 成交额池（网络容错） ----------------------

def get_top_turnover(limit: int) -> pd.DataFrame:
    """东财→新浪降级，统一输出: code, name, close, change_pct, turnover(元)"""
    # Eastmoney
    try:
        df = ak.stock_zh_a_spot_em()
        if "成交额" in df.columns:
            df["turnover"] = pd.to_numeric(df["成交额"], errors="coerce")
        elif "成交额(元)" in df.columns:
            df["turnover"] = pd.to_numeric(df["成交额(元)"], errors="coerce")
        elif "amount" in df.columns:
            df["turnover"] = pd.to_numeric(df["amount"], errors="coerce")
        else:
            df["turnover"] = pd.NA

        out = df.rename(columns={"代码": "code", "名称": "name", "最新价": "close", "涨跌幅": "change_pct"})
        out = out.dropna(subset=["turnover"]).sort_values("turnover", ascending=False).head(limit)
        for c in ["code", "name", "close", "change_pct"]:
            if c not in out.columns:
                out[c] = pd.NA
        return out[["code", "name", "close", "change_pct", "turnover"]]
    except Exception as e:
        print(f"[warn] Eastmoney spot failed: {e}")

    # Sina
    try:
        df2 = ak.stock_zh_a_spot()
        out2 = pd.DataFrame()
        out2["code"] = df2.get("symbol", df2.get("代码"))
        out2["name"] = df2.get("name", df2.get("名称"))
        out2["close"] = pd.to_numeric(df2.get("trade", df2.get("最新价")), errors="coerce")
        out2["change_pct"] = pd.to_numeric(df2.get("changepercent", df2.get("涨跌幅")), errors="coerce")
        if "amount" in df2.columns:
            out2["turnover"] = pd.to_numeric(df2["amount"], errors="coerce") * 1e4  # 万→元
        elif "成交额" in df2.columns:
            out2["turnover"] = pd.to_numeric(df2["成交额"], errors="coerce")
        else:
            out2["turnover"] = pd.NA

        out2 = out2.dropna(subset=["turnover"]).sort_values("turnover", ascending=False).head(limit)
        for c in ["code", "name", "close", "change_pct"]:
            if c not in out2.columns:
                out2[c] = pd.NA
        return out2[["code", "name", "close", "change_pct", "turnover"]]
    except Exception as e2:
        print(f"[error] Fallback (Sina) failed: {e2}")
        return pd.DataFrame(columns=["code", "name", "close", "change_pct", "turnover"])


# ---------------------- 行业映射/热度（双兜底） ----------------------

def _board_name_code_map() -> Dict[str, str]:
    """行业名称 -> 板块代码 的映射。东财→同花顺兜底。"""
    try:
        b = ak.stock_board_industry_name_em()
        if not b.empty:
            return dict(zip(b["板块名称"], b["板块代码"]))
    except Exception as e:
        print(f"[warn] industry_name_em failed: {e}")

    try:
        if hasattr(ak, "stock_board_industry_name_ths"):
            b = ak.stock_board_industry_name_ths()
            if not b.empty:
                if {"板块名称", "板块代码"} <= set(b.columns):
                    return dict(zip(b["板块名称"], b["板块代码"]))
                elif {"name", "code"} <= set(b.columns):
                    return dict(zip(b["name"], b["code"]))
    except Exception as e2:
        print(f"[warn] industry_name_ths failed: {e2}")
    return {}


def map_industry_for_codes(codes: List[str]) -> Dict[str, str]:
    """
    通过“行业板块成分”反向映射：code -> 行业名称。
    任一数据源失败时，返回 {}，让上层优雅退化。
    """
    code_left: Set[str] = set(codes)
    result: Dict[str, str] = {}

    boards = pd.DataFrame()
    try:
        boards = ak.stock_board_industry_name_em()
    except Exception as e:
        print(f"[warn] stock_board_industry_name_em failed: {e}")

    if boards.empty and hasattr(ak, "stock_board_industry_name_ths"):
        try:
            boards = ak.stock_board_industry_name_ths()
        except Exception as e2:
            print(f"[warn] stock_board_industry_name_ths failed: {e2}")

    if boards.empty:
        return {}

    if "板块名称" in boards.columns:
        name_col = "板块名称"
    elif "name" in boards.columns:
        name_col = "name"
    else:
        return {}

    for _, row in boards.iterrows():
        bname = str(row[name_col])
        cons = pd.DataFrame()

        try:
            cons = ak.stock_board_industry_cons_em(symbol=bname)
        except Exception:
            pass

        if cons.empty and hasattr(ak, "stock_board_industry_cons_ths"):
            try:
                cons = ak.stock_board_industry_cons_ths(symbol=bname)
            except Exception:
                pass

        if cons.empty:
            continue

        code_col = None
        for c in ["代码", "code", "股票代码", "symbol"]:
            if c in cons.columns:
                code_col = c
                break
        if code_col is None:
            continue

        cons_codes = set(cons[code_col].astype(str).str.zfill(6))
        hit = cons_codes & code_left
        if hit:
            for c in hit:
                result[c] = bname
            code_left -= hit
            if not code_left:
                break

    return result


def compute_industry_scores(spot: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Δ5日成交额 + 5日涨幅 的 z 分，返回 top_k 行业，并附样本数 count。"""
    if spot.empty or "industry" not in spot.columns or spot["industry"].dropna().empty:
        return pd.DataFrame()

    try:
        name_code = _board_name_code_map()
    except Exception as e:
        print(f"[warn] _board_name_code_map error: {e}")
        name_code = {}

    if not name_code:
        print("[warn] industry mapping unavailable, skip industry scoring.")
        return pd.DataFrame()

    industries = {}
    today = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=40)).strftime("%Y%m%d")

    # 统计样本数
    counts = spot.groupby("industry", dropna=False)["code"].nunique()

    for ind in spot["industry"].dropna().unique():
        code = name_code.get(ind)
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
            industries[ind] = {"turnover_delta5": delta5, "return5": ret5, "count": int(counts.get(ind, 0))}
        except Exception:
            continue

    ind_df = pd.DataFrame.from_dict(industries, orient="index")
    if ind_df.empty:
        return pd.DataFrame()
    ind_df["IndustryScore"] = zscore(ind_df["turnover_delta5"]) + 0.5 * zscore(ind_df["return5"])
    ind_df.sort_values("IndustryScore", ascending=False, inplace=True)
    return ind_df.head(top_k)


# ---------------------- 历史/技术（含MACD） ----------------------

def fetch_history(code: str, start: str, end: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    df = df[cols]
    df.columns = ["date", "open", "close", "high", "low", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    # RSI14
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["RSI14"] = 100 - (100 / (1 + rs))

    # 量均线、20日高
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["20d_high"] = df["close"].rolling(20).max().shift(1)

    # ATR14
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()],
        axis=1
    ).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # MACD（12,26,9），hist = dif - dea
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    df["MACD_hist"] = dif - dea

    return df


def days_since_high20(df: pd.DataFrame) -> int:
    window = df["close"].iloc[-20:]
    if window.empty:
        return 999
    last_high = window.max()
    high_date = window[window == last_high].index[-1]
    return (df.index[-1] - high_date).days


def entry_flag(latest: pd.Series, band_low: float, band_high: float, rsi_low: float, rsi_high: float, dsh20_max: int) -> int:
    cond1 = latest["MA20"] > latest["MA60"]
    cond2 = band_low <= latest["close"] / latest["MA20"] - 1 <= band_high
    cond3 = rsi_low <= latest["RSI14"] <= rsi_high
    cond4 = int(latest.get("days_since_high20", 999)) <= dsh20_max
    return int(cond1 and cond2 and cond3 and cond4)


def tech_score(latest: pd.Series, band_low: float, band_high: float, rsi_low: float, rsi_high: float, dsh20_max: int) -> float:
    s = 0
    s += 1 if latest["MA20"] > latest["MA60"] else 0
    s += 1 if band_low <= latest["close"] / latest["MA20"] - 1 <= band_high else 0
    s += 1 if rsi_low <= latest["RSI14"] <= rsi_high else 0
    s += 1 if int(latest.get("days_since_high20", 999)) <= dsh20_max else 0
    return s / 4.0


# ---------------------- 资金（Flow） ----------------------

def northbound_top10_hits(code: str, days: int = 10) -> int:
    """最近 N 天北向 Top10 命中次数（接口可用则统计；否则返回 0）"""
    try:
        end = dt.date.today().strftime("%Y%m%d")
        start = (dt.date.today() - dt.timedelta(days=days * 2)).strftime("%Y%m%d")
        df = ak.stock_hsgt_top10_em(start_date=start, end_date=end)
        code6 = str(code).zfill(6)
        code_col = "代码" if "代码" in df.columns else ("stock_code" if "stock_code" in df.columns else None)
        if code_col is None:
            return 0
        return int((df[code_col].astype(str).str.zfill(6) == code6).sum())
    except Exception:
        return 0


# ---------------------- 基本面（多源兜底） ----------------------

def _latest_from_row(row_df: pd.DataFrame) -> Optional[float]:
    if row_df is None or row_df.empty:
        return None
    cols = [c for c in row_df.columns if c not in ("指标", "科目", "item")]
    for c in cols:  # 列一般按最近→最远
        v = pd.to_numeric(row_df[c], errors="coerce")
        if not pd.isna(v).all():
            val = v.iloc[0]
            if pd.notna(val):
                return float(val)
    return None


def _pick_from_table(df: pd.DataFrame, keys: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    col = "指标" if "指标" in df.columns else ("科目" if "科目" in df.columns else ("item" if "item" in df.columns else None))
    if col is None:
        return None
    mask = False
    for kw in keys:
        mask = mask | df[col].astype(str).str.contains(kw)
    row = df.loc[mask]
    if row.empty:
        return None
    return _latest_from_row(row.iloc[[0]])


def _try_fin_tables(code: str) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    # 1) 东财-财务分析指标
    try:
        t1 = ak.stock_financial_analysis_indicator(symbol=str(code).zfill(6))
        if t1 is not None and not t1.empty:
            tables.append(t1)
    except Exception as e:
        print(f"[warn] financial_analysis_indicator failed: {e}")
    # 2) 东财-财务摘要（若可用）
    try:
        if hasattr(ak, "stock_financial_abstract"):
            t2 = ak.stock_financial_abstract(symbol=str(code).zfill(6))
            if t2 is not None and not t2.empty:
                tables.append(t2)
    except Exception as e2:
        print(f"[warn] financial_abstract failed: {e2}")
    # 3) 新浪-财务报表摘要（若可用）
    try:
        if hasattr(ak, "stock_financial_report_sina"):
            t3 = ak.stock_financial_report_sina(stock=str(code).zfill(6))
            if t3 is not None and not t3.empty:
                tables.append(t3)
    except Exception as e3:
        print(f"[warn] financial_report_sina failed: {e3}")
    return tables


def fetch_fundamentals(code: str) -> dict:
    out = {"roe": 0.0, "rev_yoy": 0.0, "profit_yoy": 0.0, "debt_ratio": 0.0}
    tables = _try_fin_tables(code)
    if not tables:
        return out

    key_map = {
        "roe": ["净资产收益率", "ROE", "净资产收益率-加权", "ROE(加权)"],
        "rev_yoy": ["营业总收入同比", "主营业务收入同比", "营业收入同比", "营业总收入增长率", "营业收入增长率"],
        "profit_yoy": ["净利润同比", "归母净利润同比", "净利润增长率", "归属于母公司股东的净利润同比增长"],
        "debt_ratio": ["资产负债率", "负债率", "资产负债率(%)", "资产负债率-平均"],
    }
    for k, kws in key_map.items():
        val = None
        for tb in tables:
            val = _pick_from_table(tb, kws)
            if val is not None:
                break
        if val is not None:
            out[k] = float(val)
    return out


# ---------------------- ML 特征 & 训练 ----------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    用单个股票的历史（已含compute_technicals）构造特征：
    特征均滞后1日：ret1, ret5, volume_change, macd_hist, rsi14,
                 close/ma20-1, close/ma60-1, gap_to_20d_high(close/high20-1)
    目标：next_day_return > 0
    """
    if df is None or df.empty:
        return pd.DataFrame()

    feats = pd.DataFrame(index=df.index)
    # 收益
    feats["ret1"] = df["close"].pct_change(1)
    feats["ret5"] = df["close"].pct_change(5)
    # 量能比
    feats["volume_change"] = df["volume"] / df["vol_ma20"] - 1
    # MACD 柱
    feats["macd_hist"] = df["MACD_hist"]
    # RSI
    feats["rsi14"] = df["RSI14"]
    # 价格相对均线
    feats["close_ma20"] = df["close"] / df["MA20"] - 1
    feats["close_ma60"] = df["close"] / df["MA60"] - 1
    # 距离20日高
    feats["gap_to_20d_high"] = df["close"] / df["20d_high"] - 1

    # 特征滞后一天
    feats = feats.shift(1)

    # 标签：次日涨跌
    next_ret = df["close"].pct_change(1).shift(-1)
    feats["y"] = (next_ret > 0).astype(int)

    # 丢空
    feats = feats.dropna()
    return feats


def fit_cross_section(hist_map: Dict[str, pd.DataFrame], min_rows: int = 120) -> Optional[Pipeline]:
    """把多个股票的特征拼接训练一个逻辑回归模型；返回 sklearn Pipeline。"""
    if not SKLEARN_OK:
        print("[warn] scikit-learn 不可用，跳过 ML。")
        return None

    X_list, y_list = [], []
    for code, h in hist_map.items():
        if h is None or h.empty:
            continue
        feats = build_features(h)
        if feats.shape[0] < min_rows:
            continue
        y = feats["y"].values.astype(int)
        X = feats.drop(columns=["y"]).values
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        print("[warn] ML 无足够样本，跳过。")
        return None

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"training samples: {len(X_all)}")

    try:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
        model.fit(X_all, y_all)
        return model
    except Exception as e:
        print(f"[warn] ML 训练失败：{e}")
        return None


def predict_up_prob_last(df: pd.DataFrame, model: Optional[Pipeline]) -> Optional[float]:
    """用单只股票的‘最后一行特征’做预测（等于对“明天”的概率预测）；失败返回 None"""
    if model is None or df is None or df.empty:
        return None
    feats = build_features(df)
    if feats.empty:
        return None
    X_last = feats.drop(columns=["y"]).iloc[[-1]].values
    try:
        prob = model.predict_proba(X_last)[0, 1]
        return float(prob)
    except Exception:
        return None


# ---------------------- 报告 ----------------------

def _param_summary(args, total_universe: int, kept_universe: int, ml_enabled: bool, train_samples: int) -> pd.DataFrame:
    return pd.DataFrame([
        ["运行时间", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["候选数量 top", args.top],
        ["行业保留 ind-top-k", args.ind_top_k],
        ["每行业上限 per-industry-top", args.per_industry_top],
        ["只看入场 entry-only", "是" if args.entry_only else "否"],
        ["MA20带宽", f"{args.ma20_band_low:.2%} ~ {args.ma20_band_high:.2%}"],
        ["RSI区间", f"{args.rsi_low} ~ {args.rsi_high}"],
        ["距20日新高上限", args.days_since_high20_max],
        ["权重 w-ind/w-flow/w-fund/w-tech", f"{args.w_ind}/{args.w_flow}/{args.w_fund}/{args.w_tech}"],
        ["ML(上涨概率)", "启用" if ml_enabled else "停用"],
        ["训练样本(条)", train_samples],
        ["原始池大小", total_universe],
        ["过滤后样本", kept_universe],
    ], columns=["参数", "取值"])


def generate_report(date_str: str, stocks: pd.DataFrame, industries: pd.DataFrame, params_df: pd.DataFrame) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    zh_map = {
        "code": "代码", "name": "名称", "industry": "行业",
        "close": "收盘", "change_pct": "涨跌幅",
        "IndustryScore": "行业得分", "FlowScore": "资金分", "FundScore": "基本面分",
        "TechScore": "技术分", "EntryFlag": "入场信号",
        "up_prob": "上涨概率(Up Prob)",
        "Composite": "综合分", "Final Rank": "最终排名",
    }
    out_cols = [
        "code", "name", "industry", "close", "change_pct",
        "IndustryScore", "FlowScore", "FundScore", "TechScore", "EntryFlag",
        "up_prob", "Composite", "Final Rank",
    ]
    df_csv = stocks.copy()
    for c in out_cols:
        if c not in df_csv.columns:
            df_csv[c] = np.nan
    df_csv = df_csv[out_cols].rename(columns=zh_map)
    df_csv.to_csv(reports_dir / f"report_{date_str}.csv", index=False, encoding="utf-8-sig")

    # 行业热度表：中文列名
    ind_display = industries.rename(
        columns={"turnover_delta5": "Δ5日成交额", "return5": "5日涨幅", "IndustryScore": "行业得分", "count": "样本数"}
    )

    html = Template("""
<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><title>每日选股报告 {{ date }}</title>
<style>
body{font-family:"Noto Sans SC","Microsoft YaHei","PingFang SC",Arial,sans-serif;margin:24px;line-height:1.6}
h1{font-size:22px;margin:0 0 12px} h2{font-size:18px;margin:18px 0 8px}
table{border-collapse:collapse;width:100%} th,td{border:1px solid #eaeaea;padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
.card{border:1px solid #eaeaea;border-radius:8px;padding:12px;margin:6px 0;background:#fafafa}
.hint{color:#666;font-size:12px}
</style>
</head><body>
<h1>每日选股报告 {{ date }}</h1>

<div class="card">
  <b>本次参数摘要</b>
  {{ params | safe }}
  <div class="hint">说明：若“基本面分”为0，多半是数据源暂不可用；ML 若停用或样本不足时，“上涨概率”列为空，综合排名仅依赖因子。</div>
</div>

<h2>行业热度</h2>
{{ industries | safe }}

<h2>股票列表</h2>
{{ stocks | safe }}
</body></html>""").render(
        date=date_str,
        params=params_df.to_html(index=False),
        industries=ind_display.to_html(float_format="{:.2f}".format),
        stocks=df_csv.to_html(index=False, float_format="{:.2f}".format),
    )
    with open(reports_dir / f"report_{date_str}.html", "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------- 主流程 ----------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=300)
    p.add_argument("--ind-top-k", type=int, default=5)
    p.add_argument("--per-industry-top", type=int, default=2)
    p.add_argument("--entry-only", action="store_true")

    # 入场阈值
    p.add_argument("--ma20-band-low", type=float, default=-0.03)
    p.add_argument("--ma20-band-high", type=float, default=0.01)
    p.add_argument("--rsi-low", type=float, default=50)
    p.add_argument("--rsi-high", type=float, default=65)
    p.add_argument("--days-since-high20-max", type=int, default=3)

    # 权重
    p.add_argument("--w-ind", type=float, default=0.30)
    p.add_argument("--w-flow", type=float, default=0.30)
    p.add_argument("--w-fund", type=float, default=0.20)
    p.add_argument("--w-tech", type=float, default=0.20)
    p.add_argument("--w-ml", type=float, default=0.40)  # 仅用于 rank 融合时的可读性（实际用 0.6/0.4 的名次融合）

    # 机器学习
    p.add_argument("--no-ml", action="store_true", help="禁用上涨概率模型")
    args = p.parse_args()

    today = dt.date.today()
    start = (today - dt.timedelta(days=365 * 2)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    # 1) 成交额池
    spot = get_top_turnover(args.top)
    if spot is None or len(spot) == 0:
        print("[warn] 无法获取成交额池，优雅退出。")
        out_dir = "reports"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([{"提示": "数据源暂不可达，请稍后重试"}]).to_csv(
            os.path.join(out_dir, "placeholder.csv"), index=False, encoding="utf-8-sig"
        )
        with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write("<meta charset='utf-8'><h3>数据源暂不可达，请稍后重试。</h3>")
        return

    # 2) 行业映射
    code_list = spot["code"].astype(str).str.zfill(6).tolist()
    code_to_industry = map_industry_for_codes(code_list)
    spot["industry"] = spot["code"].map(code_to_industry)

    # 3) 行业热度
    ind_scores = compute_industry_scores(spot, args.ind_top_k)
    top_industries = list(ind_scores.index) if not ind_scores.empty else []

    # 4) 逐股：历史/技术 + 资金原始因子 + 基本面原始因子
    rows = []
    hist_cache: Dict[str, pd.DataFrame] = {}
    for _, r in spot.iterrows():
        # 仅保留热门行业；若没有行业映射/行业打分，则不过滤
        if top_industries and r["industry"] not in top_industries:
            continue
        code = str(r["code"]).zfill(6)
        try:
            hist = fetch_history(code, start, end)
            if hist.empty:
                continue
            hist = compute_technicals(hist)
            hist["days_since_high20"] = days_since_high20(hist)
            latest = hist.iloc[-1]

            # 技术分/入场
            eflag = entry_flag(
                latest, args.ma20_band_low, args.ma20_band_high,
                args.rsi_low, args.rsi_high, args.days_since_high20_max
            )
            tscore = tech_score(
                latest, args.ma20_band_low, args.ma20_band_high,
                args.rsi_low, args.rsi_high, args.days_since_high20_max
            )

            # 资金原始因子
            vol_ratio = float(latest["volume"] / latest["vol_ma20"]) if latest.get("vol_ma20") and latest["vol_ma20"] else 0.0
            turnover = float(r.get("turnover", 0.0))
            hsgt_hits = northbound_top10_hits(code, days=10)

            # 基本面原始因子（多源兜底）
            fin = fetch_fundamentals(code)

            hist_cache[code] = hist  # 训练时复用
            rows.append({
                "code": code, "name": r["name"], "industry": r["industry"],
                "close": latest["close"], "change_pct": r.get("change_pct", np.nan),
                "IndustryScore": ind_scores.loc[r["industry"], "IndustryScore"] if not ind_scores.empty and r["industry"] in ind_scores.index else 0.0,
                "TechScore": tscore, "EntryFlag": eflag,
                # flow raw
                "vol_ratio": vol_ratio, "turnover_raw": turnover, "hsgt_hits": hsgt_hits,
                # fund raw
                "roe": fin["roe"], "rev_yoy": fin["rev_yoy"], "profit_yoy": fin["profit_yoy"], "debt_ratio": fin["debt_ratio"],
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("[warn] 没有可展示的数据。")
        out_dir = "reports"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([{"提示": "没有符合条件的股票或数据不足"}]).to_csv(
            os.path.join(out_dir, "placeholder_empty.csv"), index=False, encoding="utf-8-sig"
        )
        return

    # 只看入场
    if args.entry_only:
        df = df[df["EntryFlag"] == 1].copy()
        if df.empty:
            print("[info] 只看入场：无满足条件的个股。")
            out_dir = "reports"
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame([{"提示": "只看入场：本次无满足条件的个股"}]).to_csv(
                os.path.join(out_dir, "placeholder_entry_only.csv"), index=False, encoding="utf-8-sig"
            )
            return

    # 5) 资金分/基本面分（z 标准化后线性组合）
    for col in ["vol_ratio", "turnover_raw", "hsgt_hits"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["FlowScore"] = zscore(df["vol_ratio"]) + 0.5 * zscore(df["turnover_raw"]) + 0.5 * zscore(df["hsgt_hits"])

    for col in ["roe", "rev_yoy", "profit_yoy", "debt_ratio"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["FundScore"] = zscore(df["roe"]) + 0.5 * zscore(df["rev_yoy"]) + 0.5 * zscore(df["profit_yoy"]) - 0.5 * zscore(df["debt_ratio"])

    # 6) 归一化综合分
    for col in ["TechScore", "IndustryScore", "FlowScore", "FundScore"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(df[col].median()) if not df[col].dropna().empty else 0.0

    df["z_tech"] = zscore(df["TechScore"])
    df["z_ind"] = zscore(df["IndustryScore"])
    df["z_flow"] = zscore(df["FlowScore"])
    df["z_fund"] = zscore(df["FundScore"])

    composite = args.w_ind * df["z_ind"] + args.w_flow * df["z_flow"] + args.w_fund * df["z_fund"] + args.w_tech * df["z_tech"]
    df["Composite"] = composite

    # 7) 训练 ML（若启用）
    model = None
    train_samples = 0
    ml_enabled = (not args.no_ml) and SKLEARN_OK
    if ml_enabled:
        model = fit_cross_section(hist_cache, min_rows=120)
        if model is None:
            ml_enabled = False
        else:
            # 统计训练样本数量（打印时由 fit_cross_section 打过日志，这里再估算一遍）
            for h in hist_cache.values():
                feats = build_features(h)
                if feats.shape[0] >= 120:
                    train_samples += feats.shape[0]

    # 8) 逐股计算 Up Prob
    df["up_prob"] = np.nan
    if ml_enabled and model is not None:
        for i, r in df.iterrows():
            code = r["code"]
            h = hist_cache.get(code)
            prob = predict_up_prob_last(h, model) if h is not None else None
            if prob is not None:
                df.at[i, "up_prob"] = prob

    # 9) 每行业 TopN 截断 → 再整体排序
    df.sort_values("Composite", ascending=False, inplace=True)
    if args.per_industry_top and args.per_industry_top > 0 and df["industry"].notna().any():
        df = (
            df.groupby("industry", dropna=False, group_keys=False)
            .apply(lambda x: x.head(args.per_industry_top))
            .reset_index(drop=True)
        )

    # 融合排名：0.6 * 综合分名次 + 0.4 * Up Prob 名次（Up Prob 缺失则只用综合分名次）
    rank_comp = df["Composite"].rank(ascending=False, method="min")
    if df["up_prob"].notna().any():
        rank_prob = df["up_prob"].rank(ascending=False, method="min")
        final_score = 0.6 * rank_comp + 0.4 * rank_prob
    else:
        final_score = rank_comp

    df["Final Rank"] = final_score.rank(ascending=True, method="min").astype(int)
    df.sort_values(["Final Rank", "Composite"], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 10) 报表（带参数摘要）
    display_df = df.head(20)
    # 行业显示
    ind_display = (
        ind_scores[["turnover_delta5", "return5", "IndustryScore", "count"]]
        if not ind_scores.empty
        else pd.DataFrame(columns=["Δ5日成交额", "5日涨幅", "行业得分", "样本数"])
    )
    params_df = _param_summary(
        args,
        total_universe=len(spot),
        kept_universe=len(df),
        ml_enabled=ml_enabled,
        train_samples=train_samples
    )
    generate_report(today.strftime("%Y-%m-%d"), display_df, ind_display, params_df)

    print(
        f"done. universe={len(spot)}, after filters={len(df)}, "
        f"top industries: {', '.join(ind_display.index.astype(str)) if not ind_display.empty else 'N/A'}"
    )


if __name__ == "__main__":
    main()
