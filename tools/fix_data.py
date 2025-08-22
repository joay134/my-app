# tools/fix_data.py
import os, sys, argparse, pandas as pd
from datetime import datetime

DATA = "data"
PR = os.path.join(DATA, "prices.csv")
SP = os.path.join(DATA, "spot.csv")

def add_suffix(code: str) -> str:
    c = str(code).strip()
    if not c: return c
    if '.' in c: return c  # already suffixed
    # 简单规则：6/9打头=>SH；其余（0/2/3）=>SZ
    if c.startswith(("6","9")): ex = "SH"
    else: ex = "SZ"
    return f"{c}.{ex}"

def drop_suffix(code: str) -> str:
    return str(code).split('.')[0] if isinstance(code,str) else code

def norm_date(s):
    try:
        return pd.to_datetime(s, errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return pd.NaT

def fix_prices(style):
    if not os.path.exists(PR): return
    df = pd.read_csv(PR, dtype={"code":str})
    need = ["code","date","open","high","low","close","volume"]
    miss = [c for c in need if c not in df.columns]
    if miss: 
        print(f"[prices] missing columns: {miss}"); return
    if style=="with-suffix":
        df["code"] = df["code"].map(add_suffix)
    else:
        df["code"] = df["code"].map(drop_suffix)

    df["date"] = df["date"].map(norm_date)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["code","date","close"]).sort_values(["code","date"]).drop_duplicates(["code","date"])
    df.to_csv(PR, index=False, encoding="utf-8-sig")
    print(f"[prices] wrote {PR}, rows={len(df)}")

def fix_spot(style):
    if not os.path.exists(SP): return
    df = pd.read_csv(SP, dtype={"code":str})
    # 允许列稍有不同，只要核心列在就处理
    rename = {"pctchg":"pct_chg","latest":"close"}  # 容错
    for k,v in rename.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)
    need = ["code","name","turnover","pct_chg","close"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"[spot] missing columns: {miss}"); return

    if style=="with-suffix":
        df["code"] = df["code"].map(add_suffix)
    else:
        df["code"] = df["code"].map(drop_suffix)

    for c in ["turnover","pct_chg","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["code","close"]).drop_duplicates(["code"])
    df.to_csv(SP, index=False, encoding="utf-8-sig")
    print(f"[spot] wrote {SP}, rows={len(df)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["with-suffix","no-suffix"], default="with-suffix")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    fix_prices(args.style)
    fix_spot(args.style)
