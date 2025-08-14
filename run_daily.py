#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight A-shares daily analyzer (T-trading friendly)

- 快速：默认仅拉近 180 个交易日的 K 线，不做 ML。
- 稳健：行业热度取数失败时自动降级为「仅按成交额池」。
- 可调：入场条件参数化，支持宽松/严格两种风格。
- 必产出：即使 0 只，也会输出 HTML 报告，写明原因与参数。

依赖：akshare, pandas, numpy, jinja2
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
from jinja2 import Template


# =========================
# 工具与指标
# =========================

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(), s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def safe_call(fn, *args, retries: int = 2, sleep: float = 0.8, **kwargs):
    """小型重试封装。"""
    last = None
    for _ in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise last


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 df 上附加常用技术指标。index 为日期，包含 close/high/low/volume。"""
    out = df.copy()
    out["MA20"] = out["close"].rolling(20).mean()
    out["MA60"] = out["close"].rolling(60).mean()
    out["RSI14"] = compute_rsi(out["close"], 14)
    out["High20"] = out["close"].rolling(20).max()
    out["days_since_high20"] = (
        out["close"].rolling(20).apply(lambda x: int((len(x) - 1) - np.argmax(x[::-1])), raw=False)
    )
    out["VOL_MA20"] = out["volume"].rolling(20).mean()
    out["VOL_RATIO"] = out["volume"] / out["VOL_MA20"]
    out["ATR14"] = compute_atr(out["high"], out["low"], out["close"], 14)
    # ATR 百分比（相对收盘）
    out["ATR_PCT"] = (out["ATR14"] / out["close"]).replace([np.inf, -np.inf], np.nan)
    return out


def entry_signal(
    latest: pd.Series,
    ma20_band: Tuple[float, float] = (-0.04, 0.02),
    rsi_band: Tuple[float, float] = (45, 70),
    max_days_since_high20: int = 4,
    min_vol_ratio: float = 0.8,
) -> Tuple[int, Dict[str, float]]:
    """返回 (signal, diagnostics)；signal=1 表示满足入场条件（做T友好版）"""
    diag = {}

    # 相对 MA20 偏离
    ma20 = latest.get("MA20", np.nan)
    close = latest.get("close", np.nan)
    if pd.isna(ma20) or ma20 == 0 or pd.isna(close):
        return 0, {"reason": "MA/close缺失"}

    dev = close / ma20 - 1.0
    diag["near_ma20_pct"] = dev

    rsi = latest.get("RSI14", np.nan)
    diag["rsi14"] = rsi

    dsh = latest.get("days_since_high20", np.nan)
    diag["days_since_high20"] = dsh

    vol_ratio = latest.get("VOL_RATIO", np.nan)
    diag["vol_ratio"] = vol_ratio

    cond1 = (ma20_band[0] <= dev <= ma20_band[1])
    cond2 = (rsi_band[0] <= rsi <= rsi_band[1])
    cond3 = (dsh <= max_days_since_high20) if pd.notna(dsh) else False
    cond4 = (vol_ratio >= min_vol_ratio) if pd.notna(vol_ratio) else False

    signal = int(cond1 and cond2 and cond3 and cond4)
    diag["passed"] = signal
    return signal, diag


# =========================
# 数据获取
# =========================

def get_spot_top_by_turnover(n: int) -> pd.DataFrame:
    spot = safe_call(ak.stock_zh_a_spot_em)
    # 保留常用列
    cols = ["代码", "名称", "最新价", "涨跌幅", "成交额"]
    spot = spot[cols].copy()
    spot.columns = ["code", "name", "close_rt", "change_pct_rt", "turnover"]
    spot = spot.sort_values("turnover", ascending=False).head(n)
    return spot.reset_index(drop=True)


def get_industry_name(code: str) -> Optional[str]:
    try:
        info = safe_call(ak.stock_individual_info_em, symbol=code)
        row = info.loc[info["item"] == "所属行业"]
        if row.empty:
            return None
        return str(row["value"].iloc[0])
    except Exception:
        return None


def get_board_code_map() -> Dict[str, str]:
    try:
        boards = safe_call(ak.stock_board_industry_name_em)
        return dict(zip(boards["板块名称"], boards["板块代码"]))
    except Exception:
        return {}


def get_board_hist(board_code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        df = safe_call(ak.stock_board_industry_hist_em, symbol=board_code, start_date=start, end_date=end)
        df = df.copy()
        df["日期"] = pd.to_datetime(df["日期"])
        df.sort_values("日期", inplace=True)
        df.set_index("日期", inplace=True)
        return df
    except Exception:
        return None


def get_kline(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        df = safe_call(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        use = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
        df = df[use].copy()
        df.columns = ["date", "open", "close", "high", "low", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)
        return df
    except Exception:
        return None


# =========================
# 行业热度与评分
# =========================

def compute_industry_heat(stocks: pd.DataFrame, top_k: int, start: str, end: str) -> pd.DataFrame:
    """返回行业热度表（可能为空）。"""
    name2code = get_board_code_map()
    if not name2code:
        return pd.DataFrame()

    out_rows = []
    for ind in sorted(set(stocks["industry"].dropna())):
        code = name2code.get(ind)
        if not code:
            continue
        hist = get_board_hist(code, start, end)
        if hist is None or len(hist) < 10:
            continue
        last = hist.tail(10)
        if len(last) < 10:
            continue
        last5 = last["成交额"].tail(5).sum()
        prev5 = last["成交额"].head(5).sum()
        delta5 = (last5 - prev5) / prev5 if prev5 else 0.0
        ret5 = (last["收盘"].iloc[-1] - last["收盘"].iloc[4]) / last["收盘"].iloc[4]
        out_rows.append(
            {"industry": ind, "Δ5d成交额": delta5, "5日涨幅": ret5, "样本数": (stocks["industry"] == ind).sum()}
        )

    if not out_rows:
        return pd.DataFrame()

    ind_df = pd.DataFrame(out_rows).set_index("industry")
    ind_df["行业得分"] = zscore(ind_df["Δ5d成交额"]) + 0.5 * zscore(ind_df["5日涨幅"])
    ind_df.sort_values("行业得分", ascending=False, inplace=True)
    if top_k > 0:
        ind_df = ind_df.head(top_k)
    return ind_df


# =========================
# 主流程
# =========================

def build_and_rank(
    top: int,
    ind_top_k: int,
    per_industry_top: int,
    lookback_days: int,
    ma20_band: Tuple[float, float],
    rsi_band: Tuple[float, float],
    max_days_since_high20: int,
    min_vol_ratio: float,
    entry_only: bool,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """返回 (候选结果df, 行业热度df, 诊断信息)"""
    diag = {}
    today = dt.date.today()
    end = today.strftime("%Y%m%d")
    start = (today - dt.timedelta(days=lookback_days)).strftime("%Y%m%d")

    # 1) 取前 top 成交额股票
    spot = get_spot_top_by_turnover(top)
    diag["spot_size"] = str(len(spot))

    # 2) 补行业
    inds = []
    for code in spot["code"]:
        inds.append(get_industry_name(code))
    spot["industry"] = inds

    # 3) 行业热度（失败则降级）
    ind_heat = compute_industry_heat(spot, ind_top_k, start, end)
    diag["industry_heat_rows"] = str(len(ind_heat))
    if ind_heat.empty:
        # 降级：不过滤行业
        chosen_codes = spot["code"].tolist()
        top_industries = []
        diag["degrade"] = "行业热度为空，使用成交额池"
    else:
        top_industries = ind_heat.index.tolist()
        chosen_codes = (
            spot.loc[spot["industry"].isin(top_industries)]
            .groupby("industry", group_keys=False)
            .apply(lambda d: d.head(per_industry_top))["code"]
            .tolist()
        )
        if not chosen_codes:
            chosen_codes = spot["code"].tolist()
            diag["degrade"] = "按行业截取后为空，退回成交额池"

    # 4) 拉 K 线并做技术打分
    rows = []
    for code in chosen_codes:
        try:
            k = get_kline(code, start, end)
            if k is None or len(k) < 65:  # 至少要足够算 MA/RSI
                continue
            f = tech_features(k)
            latest = f.iloc[-1]
            sig, d = entry_signal(
                latest,
                ma20_band=ma20_band,
                rsi_band=rsi_band,
                max_days_since_high20=max_days_since_high20,
                min_vol_ratio=min_vol_ratio,
            )
            # 技术分（简单合计；也可按权重）
            tech_score = 0.0
            tech_score += 1.0 if latest["MA20"] > latest["MA60"] else 0.0
            dev = latest["close"] / latest["MA20"] - 1.0
            tech_score += 1.0 if (ma20_band[0] <= dev <= ma20_band[1]) else 0.0
            tech_score += 1.0 if (rsi_band[0] <= latest["RSI14"] <= rsi_band[1]) else 0.0
            tech_score += 1.0 if (latest["days_since_high20"] <= max_days_since_high20) else 0.0
            tech_score += 0.5 if (latest["VOL_RATIO"] >= max(1.0, min_vol_ratio)) else 0.0
            # ATR_PCT 越大，波动越充足（做T友好）
            atr_pct = float(latest["ATR_PCT"]) if pd.notna(latest["ATR_PCT"]) else np.nan

            rt = spot.loc[spot["code"] == code].head(1)
            rows.append(
                {
                    "代码": code,
                    "名称": rt["name"].iloc[0] if not rt.empty else "",
                    "行业": rt["industry"].iloc[0] if (not rt.empty and pd.notna(rt["industry"].iloc[0])) else "",
                    "收盘": float(latest["close"]),
                    "涨跌幅%": float(rt["change_pct_rt"].iloc[0]) if not rt.empty else np.nan,
                    "成交额(亿)": float(rt["turnover"].iloc[0]) / 1e8 if not rt.empty else np.nan,
                    "近MA20偏离%": round(dev * 100, 2) if pd.notna(dev) else np.nan,
                    "RSI14": float(latest["RSI14"]),
                    "距20日新高天数": int(latest["days_since_high20"]) if pd.notna(latest["days_since_high20"]) else np.nan,
                    "量比(20)": float(latest["VOL_RATIO"]),
                    "ATR14%": round(atr_pct * 100, 2) if pd.notna(atr_pct) else np.nan,
                    "入场信号": int(sig),
                    "技术分": float(tech_score),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, ind_heat, diag

    # 合成分：做T强调流动性(成交额)、波动(ATR)、技术形态
    df["z_turn"] = zscore(df["成交额(亿)"])
    df["z_atr"] = zscore(df["ATR14%"])
    df["z_tech"] = zscore(df["技术分"])
    df["综合分"] = 0.45 * df["z_turn"] + 0.25 * df["z_atr"] + 0.30 * df["z_tech"]

    # 行业加成（如果有热度）
    if not ind_heat.empty:
        ind_score_map = ind_heat["行业得分"].to_dict()
        df["行业得分"] = df["行业"].map(ind_score_map).fillna(0.0)
        df["综合分"] += 0.15 * zscore(df["行业得分"])
    else:
        df["行业得分"] = 0.0

    # 最终排序
    if entry_only:
        # 仅看入场=1 的票
        base = df[df["入场信号"] == 1].copy()
        if base.empty:
            base = df.copy()
            diag["entry_only_fallback"] = "入场=1为空，改用全部"
    else:
        base = df.copy()

    base.sort_values(["综合分", "技术分", "成交额(亿)"], ascending=[False, False, False], inplace=True)
    base.reset_index(drop=True, inplace=True)
    base["最终排名"] = np.arange(1, len(base) + 1)

    return base, ind_heat, diag


def render_html(
    out_dir: Path,
    result: pd.DataFrame,
    ind_heat: pd.DataFrame,
    params: Dict[str, str],
    title: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = dt.date.today().strftime("%Y-%m-%d")
    html_path = out_dir / f"report_{date_str}.html"
    csv_path = out_dir / f"report_{date_str}.csv"

    # 导出 CSV（全量结果）
    if not result.empty:
        result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    tpl = Template(
        """
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>{{ title }} {{ date }}</title>
          <style>
            body {font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,'Noto Sans','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif; margin:16px;}
            table {border-collapse: collapse; width: 100%; margin: 10px 0;}
            th, td {border: 1px solid #ddd; padding: 6px 8px; font-size: 12px;}
            th {background: #f6f8fa; text-align: left;}
            .note {color:#666; font-size:12px;}
          </style>
        </head>
        <body>
          <h2>{{ title }} — {{ date }}</h2>

          <h3>参数</h3>
          <table>
            <tbody>
              {% for k, v in params.items() %}
              <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
              {% endfor %}
            </tbody>
          </table>

          <h3>行业热度</h3>
          {% if ind_html %}
            {{ ind_html | safe }}
          {% else %}
            <div class="note">本日无法计算行业热度或为空，已退化为成交额池。</div>
          {% endif %}

          <h3>股票候选</h3>
          {% if res_html %}
            {{ res_html | safe }}
          {% else %}
            <div class="note">本日无候选；可放宽 MA20/RSI/新高天数/量比阈值，或扩大成交额池。</div>
          {% endif %}

          <div class="note">* 本报告仅供研究参考，不构成投资建议。做T需严格风控：设定固定止损/止盈，并控制仓位。</div>
        </body>
        </html>
        """
    )

    ind_html = (
        ind_heat.rename(
            columns={"Δ5d成交额": "Δ5日成交额", "5日涨幅": "5日涨幅", "行业得分": "行业得分", "样本数": "样本数"}
        ).to_html(float_format=lambda x: f"{x:.2f}", border=0)
        if not ind_heat.empty
        else ""
    )

    show_cols = [
        "最终排名",
        "代码",
        "名称",
        "行业",
        "收盘",
        "涨跌幅%",
        "成交额(亿)",
        "近MA20偏离%",
        "RSI14",
        "距20日新高天数",
        "量比(20)",
        "ATR14%",
        "行业得分",
        "技术分",
        "综合分",
        "入场信号",
    ]
    res_html = (
        result[show_cols].to_html(index=False, float_format=lambda x: f"{x:.2f}", border=0)
        if not result.empty
        else ""
    )

    html = tpl.render(
        title=title,
        date=date_str,
        params=params,
        ind_html=ind_html,
        res_html=res_html,
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


def main():
    parser = argparse.ArgumentParser(description="Daily A-shares analyzer (T-friendly)")
    parser.add_argument("--top", type=int, default=350, help="成交额池规模（默认350）")
    parser.add_argument("--ind-top-k", type=int, default=8, help="取前K个行业（默认8）")
    parser.add_argument("--per-industry-top", type=int, default=3, help="各行业保留前N只（默认3）")
    parser.add_argument("--lookback-days", type=int, default=180, help="K线回看天数（默认180）")
    parser.add_argument("--out", default="reports", help="报告输出目录（默认 reports）")

    # 做T阈值（宽松；可通过参数收紧）
    parser.add_argument("--ma20-band-low", type=float, default=-0.04, help="相对MA20下界（默认-4%）")
    parser.add_argument("--ma20-band-high", type=float, default=0.02, help="相对MA20上界（默认+2%）")
    parser.add_argument("--rsi-low", type=float, default=45, help="RSI下界（默认45）")
    parser.add_argument("--rsi-high", type=float, default=70, help="RSI上界（默认70）")
    parser.add_argument("--max-days-since-high20", type=int, default=4, help="距20日新高最大天数（默认4）")
    parser.add_argument("--min-vol-ratio", type=float, default=0.8, help="量比阈值（默认0.8）")
    parser.add_argument("--entry-only", action="store_true", help="仅保留入场=1 的股票（默认否）")

    args = parser.parse_args()

    out_dir = Path(args.out)

    result, ind_heat, diag = build_and_rank(
        top=args.top,
        ind_top_k=args.ind_top_k,
        per_industry_top=args.per_industry_top,
        lookback_days=args.lookback_days,
        ma20_band=(args.ma20_band_low, args.ma20_band_high),
        rsi_band=(args.rsi_low, args.rsi_high),
        max_days_since_high20=args.max_days_since_high20,
        min_vol_ratio=args.min_vol_ratio,
        entry_only=args.entry_only,
        out_dir=out_dir,
    )

    params = {
        "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "候选数量 top": str(args.top),
        "行业保留 ind-top-k": str(args.ind_top_k),
        "各行业上限 per-industry-top": str(args.per_industry_top),
        "仅看入场 entry-only": "是" if args.entry_only else "否",
        "MA20带宽": f"{args.ma20_band_low:.1%} ~ {args.ma20_band_high:.1%}",
        "RSI区间": f"{args.rsi_low:.0f} ~ {args.rsi_high:.0f}",
        "距20日新高上限": str(args.max_days_since_high20),
        "量比下限": f"{args.min_vol_ratio:.2f}",
        "K线回看天数": str(args.lookback_days),
        "spot数量": diag.get("spot_size", "-"),
        "行业热度行数": diag.get("industry_heat_rows", "0"),
        "降级说明": diag.get("degrade", diag.get("entry_only_fallback", "")),
    }

    html_path = render_html(out_dir, result, ind_heat, params, title="每日T交易候选（轻量版）")

    # 控制台回显
    print(f"[OK] 报告已生成: {html_path}")
    if result.empty:
        print("[WARN] 本日无候选；建议放宽阈值或扩大成交额池。")


if __name__ == "__main__":
    # 为了兼容 CI 上的 python -m py_compile
    try:
        main()
    except Exception as e:
        # 打印信息但不抛出巨量 Traceback，方便在 Actions 日志定位
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
