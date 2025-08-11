#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A 股日选股分析（行业→资金→基本面→技术，含入场/每行业TopN，中文报表、网络容错）

- 成交额池：东财→新浪降级；失败则优雅退出并写入占位报告
- 行业映射：用“行业板块成分”反向映射，Eastmoney/同花顺双兜底；拿不到行业时自动跳过行业打分
- 资金分 FlowScore：量能放大（量/20日均量）+ 成交额 + 北向Top10命中（若接口可用）
- 基本面 FundScore：ROE、营收同比、归母净利同比、资产负债率（负债率越低越好）
- 技术分 TechScore：MA20>MA60、价格落在MA20附近带、RSI区间、距20日新高≤阈值
- 入场阈值参数化、只看入场、每行业TopN（仅在 industry 有效时生效）
- 报表中文，空池/失败不报红
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
        # 兼容不同字段名
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
            # 新浪 amount 单位为“万”，换算为元
            out2["turnover"] = pd.to_numeric(df2["amount"], errors="coerce") * 1e4
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
    """
    行业名称 -> 板块代码 的映射。
    先试东财；失败再试同花顺；都失败返回 {}（让上层优雅跳过行业打分）。
    """
    # 1) Eastmoney
    try:
        b = ak.stock_board_industry_name_em()
        if not b.empty:
            return dict(zip(b["板块名称"], b["板块代码"]))
    except Exception as e:
        print(f"[warn] industry_name_em failed: {e}")

    # 2) THS
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
    # 先试东财
    try:
        boards = ak.stock_board_industry_name_em()
    except Exception as e:
        print(f"[warn] stock_board_industry_name_em failed: {e}")

    # 再试 THS
    if boards.empty and hasattr(ak, "stock_board_industry_name_ths"):
        try:
            boards = ak.stock_board_industry_name_ths()
        except Exception as e2:
            print(f"[warn] stock_board_industry_name_ths failed: {e2}")

    if boards.empty:
        return {}

    # 统一行业名称列
    if "板块名称" in boards.columns:
        name_col = "板块名称"
    elif "name" in boards.columns:
        name_col = "name"
    else:
        return {}

    # 逐行业取成分，匹配当前池子的 code
    for _, row in boards.iterrows():
        bname = str(row[name_col])
        cons = pd.DataFrame()

        # 东财成分
        try:
            cons = ak.stock_board_industry_cons_em(symbol=bname)
        except Exception:
            pass

        # THS 成分
        if cons.empty and hasattr(ak, "stock_board_industry_cons_ths"):
            try:
                cons = ak.stock_board_industry_cons_ths(symbol=bname)
            except Exception:
                pass

        if cons.empty:
            continue

        # 统一代码列
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
    """Δ5日成交额 + 5日涨幅 的 z 分，返回 top_k 行业。拿不到映射则直接跳过。"""
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
            industries[ind] = {"turnover_delta5": delta5, "return5": ret5}
        except Exception:
            continue

    ind_df = pd.DataFrame.from_dict(industries, orient="index")
    if ind_df.empty:
        return pd.DataFrame()
    ind_df["IndustryScore"] = zscore(ind_df["turnover_delta5"]) + 0.5 * zscore(ind_df["return5"])
    ind_df.sort_values("IndustryScore", ascending=False, inplace=True)
    return ind_df.head(top_k)


# ---------------------- 历史/技术 ----------------------

def fetch_history(code: str, start: str, end: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    df = df[cols]
    df.columns = ["date", "open", "close", "high", "low", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["20d_high"] = df["close"].rolling(20).max().shift(1)

    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()],
        axis=1
    ).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
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


# ---------------------- 基本面（Fund） ----------------------

def _latest_numeric(row_df: pd.DataFrame) -> Optional[float]:
    """从财务指标宽表里取最近一期有值的数字"""
    if row_df.empty:
        return None
    cols = [c for c in row_df.columns if c != "指标"]
    for c in cols:  # 列一般按最近→最远
        v = pd.to_numeric(row_df[c], errors="coerce")
        if not pd.isna(v).all():
            val = v.iloc[0]
            if pd.notna(val):
                return float(val)
    return None


def fetch_fundamentals(code: str) -> dict:
    """返回 {'roe':..., 'rev_yoy':..., 'profit_yoy':..., 'debt_ratio':...}；失败返回 0"""
    out = {"roe": 0.0, "rev_yoy": 0.0, "profit_yoy": 0.0, "debt_ratio": 0.0}
    try:
        df = ak.stock_financial_analysis_indicator(symbol=str(code).zfill(6))
        # 兼容不同表头（取包含关键词的行）
        def pick(keywords: List[str]) -> Optional[float]:
            mask = False
            for kw in keywords:
                mask = mask | df["指标"].astype(str).str.contains(kw)
            row = df.loc[mask]
            if row.empty:
                return None
            return _latest_numeric(row.iloc[[0]])

        roe = pick(["净资产收益率", "ROE"])
        rev = pick(["营业总收入同比", "营业总收入增长"])
        prof = pick(["净利润同比", "归母净利润同比"])
        debt = pick(["资产负债率", "负债率"])

        if roe is not None:
            out["roe"] = float(roe)
        if rev is not None:
            out["rev_yoy"] = float(rev)
        if prof is not None:
            out["profit_yoy"] = float(prof)
        if debt is not None:
            out["debt_ratio"] = float(debt)
    except Exception:
        pass
    return out


# ---------------------- 报告 ----------------------

def generate_report(date_str: str, stocks: pd.DataFrame, industries: pd.DataFrame) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    zh_map = {
        "code": "代码", "name": "名称", "industry": "行业",
        "close": "收盘", "change_pct": "涨跌幅",
        "IndustryScore": "行业得分", "FlowScore": "资金分", "FundScore": "基本面分",
        "TechScore": "技术分", "EntryFlag": "入场信号",
        "Composite": "综合分", "Final Rank": "最终排名",
    }
    out_cols = [
        "code", "name", "industry", "close", "change_pct",
        "IndustryScore", "FlowScore", "FundScore", "TechScore", "EntryFlag",
        "Composite", "Final Rank",
    ]
    df_csv = stocks.copy()
    for c in out_cols:
        if c not in df_csv.columns:
            df_csv[c] = np.nan
    df_csv = df_csv[out_cols].rename(columns=zh_map)
    df_csv.to_csv(reports_dir / f"report_{date_str}.csv", index=False, encoding="utf-8-sig")

    html = Template("""
<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><title>每日选股报告 {{ date }}</title>
<style>
body{font-family:"Noto Sans SC","Microsoft YaHei","PingFang SC",Arial,sans-serif;margin:24px;line-height:1.6}
h1{font-size:22px;margin:0 0 12px} h2{font-size:18px;margin:18px 0 8px}
table{border-collapse:collapse;width:100%} th,td{border:1px solid #eaeaea;padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
</style>
</head><body>
<h1>每日选股报告 {{ date }}</h1>
<h2>行业热度</h2>
{{ industries | safe }}
<h2>股票列表</h2>
{{ stocks | safe }}
</body></html>""").render(
        date=date_str,
        industries=industries.rename(
            columns={"turnover_delta5": "Δ5日成交额", "return5": "5日涨幅", "IndustryScore": "行业得分"}
        ).to_html(float_format="{:.2f}".format),
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
    p.add_argument("--w-ml", type=float, default=0.00)
    p.add_argument("--no-ml", action="store_true")

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
            latest = hist.iloc[-1]
            latest["days_since_high20"] = days_since_high20(hist)

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

            # 基本面原始因子
            fin = fetch_fundamentals(code)

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
    # FlowScore = z(vol_ratio) + 0.5*z(turnover_raw) + 0.5*z(hsgt_hits)
    for col in ["vol_ratio", "turnover_raw", "hsgt_hits"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["FlowScore"] = zscore(df["vol_ratio"]) + 0.5 * zscore(df["turnover_raw"]) + 0.5 * zscore(df["hsgt_hits"])

    # FundScore = z(roe) + 0.5*z(rev_yoy) + 0.5*z(profit_yoy) - 0.5*z(debt_ratio)
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
    # 预留 ML：
    if not args.no_ml and "up_prob" in df.columns:
        df["z_ml"] = zscore(df["up_prob"])
        composite += args.w_ml * df["z_ml"]
    df["Composite"] = composite

    # 7) 每行业 TopN 截断 → 再整体排序（只有 industry 有效时才截断）
    df.sort_values("Composite", ascending=False, inplace=True)
    if args.per_industry_top and args.per_industry_top > 0 and df["industry"].notna().any():
        df = (
            df.groupby("industry", dropna=False, group_keys=False)
            .apply(lambda x: x.head(args.per_industry_top))
            .reset_index(drop=True)
        )
    df.sort_values(["Composite", "change_pct"], ascending=[False, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Final Rank"] = df.index + 1

    # 8) 报表
    display_df = df.head(20)
    ind_display = (
        ind_scores[["turnover_delta5", "return5", "IndustryScore"]]
        if not ind_scores.empty
        else pd.DataFrame(columns=["Δ5日成交额", "5日涨幅", "行业得分"])
    )
    generate_report(today.strftime("%Y-%m-%d"), display_df, ind_display)

    print(
        f"done. universe={len(spot)}, after filters={len(df)}, "
        f"top industries: {', '.join(top_industries) if top_industries else 'N/A'}"
    )


if __name__ == "__main__":
    main()
