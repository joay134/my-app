# scripts/ths_http.py
import os, sys, json, time, math, argparse, datetime as dt
from pathlib import Path
import requests
import pandas as pd

TOK_URLS = [
    "https://quantapi.10jqka.com.cn/api/v1/get_access_token",
    "https://quantapi.51ifind.com/api/v1/get_access_token",
    "https://ft.10jqka.com.cn/api/v1/get_access_token",
]

SPOT_URL = "https://ft.10jqka.com.cn/ds_service/api/v1/real_time_quotation"
HIS_URL  = "https://ft.10jqka.com.cn/api/v1/cmd_history_quotation"

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def get_access_token(refresh_token: str, timeout=20) -> str:
    last_err = None
    for url in TOK_URLS:
        try:
            r = requests.post(url, headers={
                "Content-Type": "application/json",
                "refresh_token": refresh_token
            }, timeout=timeout)
            j = r.json()
            if isinstance(j, dict) and j.get("data", {}).get("access_token"):
                return j["data"]["access_token"]
            last_err = j
        except Exception as e:
            last_err = e
    raise RuntimeError(f"get_access_token failed: {last_err}")

def resolve_codes(env_codes: str | None) -> list[str]:
    if env_codes:
        return [c.strip() for c in env_codes.replace(";", ",").split(",") if c.strip()]
    p = Path("data/codes.txt")
    if p.exists():
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ["000001.SZ", "600000.SH"]  # minimal sample

def write_dir():
    Path("data").mkdir(parents=True, exist_ok=True)

def pull_spot(access_token: str, codes: list[str], batch=200, timeout=20) -> pd.DataFrame:
    headers = {"Content-Type": "application/json", "access_token": access_token}
    indicators = "open,high,low,latest,amount,pctchg,name"
    rows = []
    for group in chunks(codes, batch):
        payload = {"codes": ",".join(group), "indicators": indicators}
        r = requests.post(SPOT_URL, json=payload, headers=headers, timeout=timeout)
        j = r.json()
        part = j.get("data", [])
        if isinstance(part, list):
            rows.extend(part)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.rename(columns={
        "latest": "close",
        "amount": "turnover",
        "pctchg": "pct_chg"
    }, inplace=True)
    cols = [c for c in ["code","name","turnover","close","pct_chg","open","high","low"] if c in df.columns]
    out = df[cols].copy()
    out.to_csv("data/spot.csv", index=False, encoding="utf-8-sig")
    return out

def load_history_template(path="data/history_form.json") -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except Exception:
        return None

def normalize_history_response(j: dict) -> pd.DataFrame:
    if not isinstance(j, dict):
        return pd.DataFrame()
    # common shapes handler
    if "data" in j and isinstance(j["data"], list):
        data = j["data"]
        # shape A: flat list of dicts with columns including 'code','date','open','high','low','close','volume'
        if data and isinstance(data[0], dict) and any(k in data[0] for k in ("date","trade_date")):
            df = pd.DataFrame(data)
            if "trade_date" in df.columns and "date" not in df.columns:
                df.rename(columns={"trade_date":"date"}, inplace=True)
            return df

        # shape B: [{"code": "...", "items": [{"date": "...", "...": ...}, ...]}]
        if data and isinstance(data[0], dict) and "items" in data[0]:
            all_rows = []
            for g in data:
                code = g.get("code")
                items = g.get("items", [])
                for row in items:
                    row = dict(row)
                    row["code"] = code
                    all_rows.append(row)
            return pd.DataFrame(all_rows)

        # shape C: [{"code":"...", "datas":[...]}]
        if data and isinstance(data[0], dict) and "datas" in data[0]:
            all_rows = []
            for g in data:
                code = g.get("code")
                items = g.get("datas", [])
                for row in items:
                    row = dict(row)
                    row["code"] = code
                    all_rows.append(row)
            return pd.DataFrame(all_rows)
    return pd.DataFrame()

def pull_history_by_cmd(access_token: str, codes: list[str],
                        start: str, end: str,
                        form_template: dict,
                        out_csv="data/prices.csv",
                        batch=60, timeout=40) -> pd.DataFrame:
    headers = {"Content-Type": "application/json", "access_token": access_token}
    frames = []
    for group in chunks(codes, batch):
        form = json.loads(json.dumps(form_template, ensure_ascii=False))
        # try different field names for codes/time as templates may differ
        for k in ("codes", "Codes", "secCodes"):
            if k in form:
                form[k] = ",".join(group)
        for k in ("startdate","startDate","beginDate"):
            if k in form:
                form[k] = start
        for k in ("enddate","endDate","stopDate"):
            if k in form:
                form[k] = end
        r = requests.post(HIS_URL, json=form, headers=headers, timeout=timeout)
        j = r.json()
        df = normalize_history_response(j)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # unify typical column names if present
    rename_map = {
        "open_price":"open", "high_price":"high", "low_price":"low", "close_price":"close",
        "volume_amount":"volume", "turnover":"amount"
    }
    for a,b in rename_map.items():
        if a in df.columns and b not in df.columns:
            df.rename(columns={a:b}, inplace=True)
    keep = [c for c in ["code","date","open","high","low","close","volume","amount"] if c in df.columns]
    df = df[keep].copy()
    df.sort_values(["code","date"], inplace=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=(dt.date.today()-dt.timedelta(days=370)).isoformat())
    ap.add_argument("--end", type=str, default=dt.date.today().isoformat())
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--no-spot", action="store_true")
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()

    write_dir()
    refresh_token = os.getenv("THS_REFRESH_TOKEN", "").strip()
    if not refresh_token:
        raise SystemExit("env THS_REFRESH_TOKEN is required")

    access = get_access_token(refresh_token, timeout=args.timeout)
    codes = resolve_codes(os.getenv("THS_CODES"))

    if not args.no_spot:
        spot = pull_spot(access, codes, batch=min(args.batch,200), timeout=args.timeout)
        print(f"spot rows: {len(spot)} -> data/spot.csv")

    if not args.no_history:
        tmpl = load_history_template("data/history_form.json")
        if tmpl is None:
            print("history_form.json not found, skip history. Put your cmd_history_quotation JSON into data/history_form.json")
        else:
            his = pull_history_by_cmd(
                access, codes, args.start, args.end, tmpl,
                out_csv="data/prices.csv", batch=min(args.batch,60),
                timeout=max(args.timeout,40)
            )
            print(f"history rows: {len(his)} -> data/prices.csv")

if __name__ == "__main__":
    main()
