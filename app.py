# app.py
# A-shares "Longtou" strategy assistant - Streamlit (CSV backend, no external API)
# 优化版：热点板块/梯队复盘/情绪晋级率&炸板率/次日接力股（T+1）/合力分权重可调

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="龙头战法助手（CSV）", layout="wide")
st.title("龙头战法助手（CSV 数据源）· 优化版")

# ------------------ 基础工具 ------------------
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

# ------------------ CSV 读取（缓存） ------------------
@st.cache_data(ttl=3600)
def load_prices_csv(folder: str) -> pd.DataFrame:
    p = Path(folder) / "prices.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p, dtype={"code": str})
    need = {"code","date","open","high","low","close","volume"}
    if not need.issubset(df.columns):
        st.error(f"prices.csv 必须包含列：{need}")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["code","date","close"])
    return df.sort_values(["code","date"]).reset_index(drop=True)

@st.cache_data(ttl=3600)
def load_spot_csv(folder: str) -> pd.DataFrame:
    p = Path(folder) / "spot.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p, dtype={"code": str})
    if "code" not in df.columns or "name" not in df.columns:
        st.warning("spot.csv 至少需要列：code,name")
        return pd.DataFrame()
    for c in ["turnover","pct_chg","close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_industry_map_csv(folder: str) -> pd.DataFrame:
    p = Path(folder) / "industry_map.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p, dtype={"code": str})
    if not {"code","industry"}.issubset(df.columns):
        st.warning("industry_map.csv 需要列：code,industry")
        return pd.DataFrame()
    return df[["code","industry"]]

@st.cache_data(ttl=3600)
def load_limits_csv(folder: str) -> pd.DataFrame:
    # limits.csv（可选）：code,limit_pct  例：000001,10 ；创业板/科创可填 20
    p = Path(folder) / "limits.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p, dtype={"code": str})
    if not {"code","limit_pct"}.issubset(df.columns):
        st.warning("limits.csv 需要列：code,limit_pct")
        return pd.DataFrame()
    df["limit_pct"] = pd.to_numeric(df["limit_pct"], errors="coerce")
    return df.dropna(subset=["limit_pct"])

# ------------------ 指标衍生 ------------------
def enrich_daily(prices: pd.DataFrame, spot: pd.DataFrame, limits: pd.DataFrame, default_limit: float) -> pd.DataFrame:
    """日级衍生：pct_chg / up_limit / touched_limit / limit_price / prev_close / limit_pct"""
    df = prices.copy()
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    df["pct_chg"] = (df["close"] / df["prev_close"] - 1) * 100.0

    # 每只股票的涨停幅度
    limit_map = dict(zip(limits["code"], limits["limit_pct"])) if not limits.empty else {}
    st_set = set()
    if not spot.empty and "name" in spot.columns:
        st_set = set(spot.loc[spot["name"].fillna("").str.contains("ST"), "code"].tolist())

    def limit_for(code):
        if code in limit_map: return float(limit_map[code])
        if code in st_set:    return 5.0
        return default_limit

    df["limit_pct"] = df["code"].map(limit_for)
    df["limit_price"] = df["prev_close"] * (1 + df["limit_pct"] / 100.0)
    # 判定允许微小误差
    df["up_limit"] = (df["pct_chg"] >= (df["limit_pct"] - 0.25))
    df["touched_limit"] = (df["high"] >= df["limit_price"] * 0.999)

    # 连板数 L
    d = df[["code","date","up_limit"]].sort_values(["code","date"]).copy()
    def streak(series):
        out = []; cnt = 0
        for v in series:
            if bool(v): cnt += 1
            else: cnt = 0
            out.append(cnt)
        return out
    df["L"] = d.groupby("code")["up_limit"].transform(lambda s: pd.Series(streak(s))).values
    return df

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """当日 5日均量 / 量比（近似换手强度）"""
    vol_ma5 = df.groupby("code")["volume"].rolling(5).mean().reset_index(level=0, drop=True)
    df2 = df.copy()
    df2["VOL_MA5"] = vol_ma5
    return df2

# 情绪面板（增强：晋级率、炸板率）
def market_emotion(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["code","date"]).copy()
    # 昨日涨停
    d["up_yest"] = d.groupby("code")["up_limit"].shift(1)

    agg = d.groupby("date").agg(
        up_count = ("up_limit","sum"),
        touch_count = ("touched_limit","sum"),
        max_L = ("L","max"),
    )
    agg["seal_rate"] = agg["up_count"] / agg["touch_count"].replace(0, np.nan)
    # 炸板率（摸板未封）
    agg["open_fail_rate"] = (agg["touch_count"] - agg["up_count"]) / agg["touch_count"].replace(0, np.nan)

    # 连板晋级率：昨日涨停→今日继续涨停
    adv = d[d["up_yest"] == True].groupby("date")["up_limit"].sum()
    base = d.groupby("date")["up_yest"].sum()  # 昨日涨停家数（参与晋级的基数）
    agg["advance_rate"] = adv / base.replace(0, np.nan)

    # 昨日涨停 → 次日开盘溢价（中位）
    d["next_open"] = d.groupby("code")["open"].shift(-1)
    d["next_open_premium"] = (d["next_open"] / d["close"] - 1) * 100.0
    prev_up = d[d["up_limit"]].groupby("date")["next_open_premium"].median()
    agg["next_open_median%"] = prev_up

    return agg.reset_index().sort_values("date")

# 行业强度（近 10 根K：5日收益 + 量能增幅 中位）
def compute_industry_strength(prices: pd.DataFrame, indmap: pd.DataFrame, last_n_days: int = 15) -> pd.DataFrame:
    if prices.empty or indmap.empty: return pd.DataFrame()
    latest_date = prices["date"].max()
    use = prices[prices["date"] >= (latest_date - pd.Timedelta(days=last_n_days))].copy()

    def per_code(g):
        g = g.sort_values("date").tail(12)
        if len(g) < 10:
            return pd.Series({"ret5": np.nan, "vol_delta5": np.nan})
        ret5 = (g["close"].iloc[-1] / g["close"].iloc[-6] - 1)
        last5 = g["volume"].iloc[-5:].sum()
        prev5 = g["volume"].iloc[-10:-5].sum()
        vol_delta5 = (last5 - prev5) / prev5 if prev5 else np.nan
        return pd.Series({"ret5": ret5, "vol_delta5": vol_delta5})

    mat = use.groupby("code").apply(per_code).reset_index()
    mat = mat.merge(indmap, on="code", how="left").dropna(subset=["industry"])
    if mat.empty: return pd.DataFrame()
    ind = mat.groupby("industry")[["ret5","vol_delta5"]].median()
    ind["IndustryScore"] = zscore(ind["vol_delta5"].fillna(0)) + 0.5 * zscore(ind["ret5"].fillna(0))
    ind = ind.sort_values("IndustryScore", ascending=False)
    return ind

# 当日热点板块（按涨停/摸板/封板率 + 强度打分）
def hot_sectors_today(daily: pd.DataFrame, indmap: pd.DataFrame) -> pd.DataFrame:
    if indmap.empty: return pd.DataFrame()
    latest = daily["date"].max()
    today = daily[daily["date"] == latest].merge(indmap, on="code", how="left")
    if today.empty: return pd.DataFrame()

    g = today.groupby("industry").agg(
        up_count=("up_limit","sum"),
        touch_count=("touched_limit","sum"),
        max_L=("L","max")
    )
    g["seal_rate"] = g["up_count"] / g["touch_count"].replace(0, np.nan)
    # 强度 = z(封板率) + 0.6*z(涨停家数)
    g["Strength"] = zscore(g["seal_rate"].fillna(0)) + 0.6 * zscore(g["up_count"].fillna(0))
    return g.sort_values(["Strength","up_count","seal_rate"], ascending=[False,False,False])

# 梯队复盘（当日 L>=1）
def ladder_today(daily: pd.DataFrame, indmap: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    latest = daily["date"].max()
    t = daily[daily["date"] == latest][["code","L","up_limit","touched_limit","pct_chg","close","volume"]]
    t = t[t["L"] >= 1].copy()
    if t.empty: return t
    # 近似成交额排序
    t["approx_turnover"] = t["close"] * t["volume"]
    t = t.merge(indmap, on="code", how="left")
    if not spot.empty and "name" in spot.columns:
        t = t.merge(spot[["code","name"]], on="code", how="left")
    else:
        t["name"] = t["code"]
    t["梯队"] = t["L"].apply(lambda x: f"L{int(x)}")
    t = t.sort_values(["L","approx_turnover","pct_chg"], ascending=[False,False,False]).reset_index(drop=True)
    t["Rank"] = np.arange(1, len(t)+1)
    return t[["Rank","code","name","industry","L","梯队","pct_chg","up_limit","touched_limit"]]

# 龙头候选与合力分（权重可调）
def longtou_candidates(
    daily_with_volma: pd.DataFrame,  # 含 VOL_MA5
    spot: pd.DataFrame,
    indmap: pd.DataFrame,
    ind_strength: pd.DataFrame,
    top_codes: int,
    ind_top_k: int,
    per_industry_top: int,
    w_ind: float, w_L: float, w_turn: float, w_co: float
):
    latest_date = daily_with_volma["date"].max()

    # 当日池（优先 spot.turnover；否则 close*volume）
    today = daily_with_volma[daily_with_volma["date"] == latest_date].copy()
    today_turn = today[["code","close","volume"]].copy()
    today_turn["approx_turnover"] = today_turn["close"] * today_turn["volume"]

    if not spot.empty and "turnover" in spot.columns:
        pool = spot[["code","name","turnover"]].copy()
        pool = pool.merge(today_turn[["code","approx_turnover"]], on="code", how="left")
        pool["sort_key"] = pool["turnover"].fillna(pool["approx_turnover"])
    else:
        pool = today_turn.copy()
        pool["name"] = pool["code"]
        pool["sort_key"] = pool["approx_turnover"]
    pool = pool.sort_values("sort_key", ascending=False).head(top_codes)

    # 合并当日必要列
    cols_need = ["code","pct_chg","up_limit","touched_limit","L","high","close","open","volume","VOL_MA5"]
    d_today = today[cols_need]
    base = pool.merge(d_today, on="code", how="left").merge(indmap, on="code", how="left")

    # 行业过滤 & 共振
    if ind_top_k > 0 and not ind_strength.empty:
        keep_inds = set(ind_strength.head(ind_top_k).index)
        base = base[base["industry"].isin(keep_inds)].copy()
        ind_map_score = ind_strength["IndustryScore"].to_dict()
        base["industry_score"] = base["industry"].map(ind_map_score).fillna(0.0)
    else:
        base["industry_score"] = 0.0

    if not base.empty:
        counts = base.groupby("industry")["up_limit"].sum().rename("ind_up_count").reset_index()
        base = base.merge(counts, on="industry", how="left")
    else:
        base["ind_up_count"] = np.nan

    # 换手强度 & 二板确认
    base["turnover_ratio"] = base["volume"] / base["VOL_MA5"]
    base["is_rotation_up"] = (base["up_limit"] == True) & (base["turnover_ratio"] >= 1.0)

    # 连板强度分
    def l_score(x):
        x = float(x or 0)
        if x >= 4: return 1.5
        if x == 3: return 1.3
        if x == 2: return 1.0
        if x == 1: return 0.5
        return 0.0
    base["L_score"] = base["L"].map(l_score)

    # 昨日涨停（用于二板确认）
    yest = daily_with_volma[daily_with_volma["date"] == (latest_date - pd.Timedelta(days=1))][["code","up_limit"]].rename(
        columns={"up_limit":"up_yest"}
    )
    base = base.merge(yest, on="code", how="left")
    base["二板确认"] = ((base["up_yest"] == True) & (base["L"] >= 2)).astype(int)

    # 合力分（权重可调）
    base["z_ind"] = zscore(base["industry_score"])
    base["z_turnover_ratio"] = zscore(base["turnover_ratio"])
    base["z_L"] = zscore(base["L_score"])
    base["z_ind_up"] = zscore(base["ind_up_count"].fillna(0))

    # 归一后防止用户输入全0
    w_sum = max(w_ind + w_L + w_turn + w_co, 1e-9)
    w_ind, w_L, w_turn, w_co = w_ind/w_sum, w_L/w_sum, w_turn/w_sum, w_co/w_sum

    base["Cohesion"] = (
        w_ind  * base["z_ind"] +
        w_L    * base["z_L"] +
        w_turn * base["z_turnover_ratio"] +
        w_co   * base["z_ind_up"]
    )

    # 行业内名额限制
    base = base.sort_values(["Cohesion","turnover_ratio","pct_chg"], ascending=[False,False,False]).copy()
    if per_industry_top > 0 and not base.empty:
        base["ind_rank"] = base.groupby("industry").cumcount() + 1
        base = base[base["ind_rank"] <= per_industry_top].copy()
        base.drop(columns=["ind_rank"], inplace=True)

    # 排序与展示
    base = base.sort_values(["Cohesion","L","turnover_ratio"], ascending=[False,False,False]).reset_index(drop=True)
    show = ["code","name","industry","L","二板确认","is_rotation_up","turnover_ratio","ind_up_count","pct_chg","Cohesion"]
    for c in show:
        if c not in base.columns: base[c] = np.nan
    base["Rank"] = np.arange(1, len(base)+1)
    return base

# 次日接力股（T+1）清单
def t_plus_1_list(res: pd.DataFrame) -> pd.DataFrame:
    if res.empty: return res
    t = res.copy()
    core = ((t["L"] >= 2) & (t["is_rotation_up"] == True)) | (t["二板确认"] == 1)
    t = t[core].copy()
    if t.empty: return t

    def reason(row):
        rs = []
        if row["L"] >= 3: rs.append(f"L{int(row['L'])}高度")
        elif row["L"] == 2: rs.append("二板")
        if row["二板确认"] == 1: rs.append("二板确认")
        if row["is_rotation_up"] == True: rs.append("换手充分涨停")
        if (row.get("ind_up_count",0) or 0) >= 2: rs.append("板块共振")
        return "、".join(rs) if rs else "—"

    t["逻辑"] = t.apply(reason, axis=1)
    t["建议"] = t["L"].apply(lambda x: f"关注冲击L{int(x)+1}" if pd.notna(x) else "观察")
    keep = ["Rank","code","name","industry","L","Cohesion","逻辑","建议","turnover_ratio","ind_up_count","pct_chg"]
    return t.sort_values(["Cohesion","L","turnover_ratio"], ascending=[False,False,False])[keep]

# ------------------ 侧边栏参数 ------------------
with st.sidebar:
    st.header("数据目录与基础阈值")
    data_dir = st.text_input("CSV 目录", "data")
    default_limit = st.number_input("默认涨停幅度 %（主板10；创业/科创20 可用 limits.csv 覆盖）", 5.0, 30.0, 10.0, step=0.5)
    top_codes = st.number_input("成交额池规模（Top N）", 100, 2000, 400, step=50)
    ind_top_k = st.number_input("行业保留 K（0=不按行业过滤）", 0, 50, 10, step=1)
    per_ind_top = st.number_input("每行业保留 M", 1, 20, 3, step=1)

    st.markdown("---")
    st.header("合力分权重（可调）")
    w_ind  = st.slider("行业强度权重", 0.0, 1.0, 0.35, 0.05)
    w_L    = st.slider("连板强度权重", 0.0, 1.0, 0.35, 0.05)
    w_turn = st.slider("换手强度权重", 0.0, 1.0, 0.20, 0.05)
    w_co   = st.slider("板块共振权重", 0.0, 1.0, 0.10, 0.05)

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("刷新缓存", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# ------------------ 读取与加工 ------------------
prices = load_prices_csv(data_dir)
spot   = load_spot_csv(data_dir)
indmap = load_industry_map_csv(data_dir)
limits = load_limits_csv(data_dir)

if prices.empty:
    st.error("未找到有效的 prices.csv（必需）。请放入 data/prices.csv 后重试。")
    st.stop()

daily = enrich_daily(prices, spot, limits, default_limit)
daily = volume_features(daily)

latest_date = daily["date"].max()
st.write(f"最新交易日：**{latest_date.date() if pd.notna(latest_date) else 'NA'}**")

# ------------------ 各模块计算 ------------------
emo = market_emotion(daily)
inds = compute_industry_strength(prices, indmap)
hot = hot_sectors_today(daily, indmap)
ladder = ladder_today(daily, indmap, spot)
res = longtou_candidates(
    daily_with_volma=daily,
    spot=spot,
    indmap=indmap,
    ind_strength=inds if not inds.empty else pd.DataFrame(),
    top_codes=int(top_codes),
    ind_top_k=int(ind_top_k),
    per_industry_top=int(per_ind_top),
    w_ind=w_ind, w_L=w_L, w_turn=w_turn, w_co=w_co
)
t1 = t_plus_1_list(res)

# ------------------ Tabs ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "数据体检", "情绪仪表盘", "热点板块 & 梯队", "龙头候选（合力分）", "次日接力股（T+1）"
])

# ---- 1) 数据体检 ----
with tab1:
    st.subheader("数据体检")
    n_codes = prices["code"].nunique()
    dup = prices.duplicated(["code","date"]).sum()
    latest_cov = prices[prices["date"]==latest_date]["code"].nunique() / max(n_codes,1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("股票数", n_codes)
    c2.metric("总行数", len(prices))
    c3.metric("重复(code,date)", dup)
    c4.metric("最新日覆盖率", f"{latest_cov:.1%}")
    if dup > 0: st.warning("存在重复行，请按 code+date 去重。")
    if latest_cov < 0.9: st.warning("最新交易日覆盖率 <90%，请检查数据是否完整。")

    if spot.empty:
        st.info("未提供 spot.csv：将用 close×volume 近似成交额排序。")
    if indmap.empty:
        st.info("未提供 industry_map.csv：行业强度/热点板块将自动跳过。")
    if not limits.empty:
        st.caption(f"limits.csv 生效代码数：{limits['code'].nunique()}")

# ---- 2) 情绪仪表盘 ----
with tab2:
    st.subheader("市场情绪（近 60 交易日）")
    if emo.empty:
        st.info("情绪数据不足。")
    else:
        tail = emo.tail(60)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("昨日涨停家数", int(tail["up_count"].iloc[-1]))
        c2.metric("封板率", f"{(tail['seal_rate'].iloc[-1]*100):.1f}%")
        c3.metric("连板高度", int(tail["max_L"].iloc[-1]))
        c4.metric("晋级率", f"{(tail['advance_rate'].iloc[-1]*100):.1f}%" if pd.notna(tail["advance_rate"].iloc[-1]) else "NA")
        c5.metric("炸板率", f"{(tail['open_fail_rate'].iloc[-1]*100):.1f}%" if pd.notna(tail["open_fail_rate"].iloc[-1]) else "NA")
        st.dataframe(tail, use_container_width=True)

# ---- 3) 热点板块 & 梯队 ----
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("热点板块（当日）")
        if hot.empty:
            st.info("缺少 industry_map.csv 或当日无有效数据。")
        else:
            show = hot.reset_index()[["industry","up_count","touch_count","seal_rate","max_L","Strength"]]
            st.dataframe(show, use_container_width=True)
    with c2:
        st.subheader("连板梯队（当日 L≥1）")
        if ladder.empty:
            st.info("当日无连板股。")
        else:
            st.dataframe(ladder, use_container_width=True)
            csv = ladder.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载梯队 CSV", data=csv, file_name="ladder_today.csv", mime="text/csv")

# ---- 4) 龙头候选（合力分） ----
with tab4:
    st.subheader("龙头候选（合力分）")
    if res.empty:
        st.warning("本日无候选；可检查数据是否最新，或放宽 Top/行业过滤。")
    else:
        show_cols = ["Rank","code","name","industry","L","二板确认","is_rotation_up","turnover_ratio","ind_up_count","pct_chg","Cohesion"]
        st.dataframe(res[show_cols], use_container_width=True)
        csv = res[show_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载候选 CSV", data=csv, file_name="longtou_candidates.csv", mime="text/csv")
    st.caption("合力分 = 行业强度 + 连板强度 + 换手强度 + 板块共振（侧栏可调权重）。二板确认=昨日涨停且今日L≥2。")

# ---- 5) 次日接力股（T+1） ----
with tab5:
    st.subheader("次日接力股（T+1 低吸/接力备选）")
    if t1.empty:
        st.info("按当前规则未筛到备选；可调整权重或提高 top N。")
    else:
        st.dataframe(t1, use_container_width=True)
        csv = t1.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载 T+1 清单", data=csv, file_name="t_plus_1_list.csv", mime="text/csv")
    st.caption("规则：优先二板确认、或 L≥2 且换手充分（涨停日量 ≥ 5日均量），并结合板块共振。仅供研究参考，不构成投资建议。")
