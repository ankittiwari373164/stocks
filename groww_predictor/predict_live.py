"""
LIVE PREDICTION — run this once at 09:25 IST.

Reads the cumulative first-10-minute volume/turnover for every symbol in
the universe via the Groww live-quote API, attaches each symbol's
historical context, scores with the model, and writes the ranked
prediction (with the single most-traded pick) to data/predictions.json.
"""
from __future__ import annotations
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from .config import CFG
from .groww_client import GrowwClient
from .universe import build_universe
from .features import FEATURE_COLS
from .model import rank_day


def _parse_ohlc(payload: dict) -> dict:
    o = payload.get("open"); h = payload.get("high")
    l = payload.get("low"); c = payload.get("close")
    if o is None and isinstance(payload.get("ohlc"), str):
        nums = dict(re.findall(r"(open|high|low|close):\s*([0-9.]+)", payload["ohlc"]))
        o = float(nums.get("open", 0)); h = float(nums.get("high", 0))
        l = float(nums.get("low", 0)); c = float(nums.get("close", 0))
    return {"open": o or 0, "high": h or 0, "low": l or 0, "close": c or 0}


def _context_table() -> pd.DataFrame:
    """Latest leak-free context per symbol from the training dataset."""
    cols = ["symbol", "open_share_hist", "prev_full_vol", "prev_full_turnover",
            "avg5_full_vol", "avg20_full_vol", "avg20_full_turnover", "std20_full_vol"]
    if CFG.dataset_path.exists():
        ds = pd.read_pickle(CFG.dataset_path)
        ctx = ds.sort_values("date").groupby("symbol").last().reset_index()
        keep = [c for c in cols if c in ctx.columns]
        return ctx[keep]
    if CFG.shares_path.exists():
        return pd.read_pickle(CFG.shares_path)
    return pd.DataFrame(columns=cols)


def predict_live(client: GrowwClient | None = None, top_n: int = 10) -> dict:
    client = client or GrowwClient()
    uni = build_universe(client)
    ctx = _context_table()
    gs = ctx["open_share_hist"].median() if ("open_share_hist" in ctx and len(ctx)) else None
    global_share = float(gs) if (gs is not None and np.isfinite(gs)) else 0.06  # sane default until dataset is built

    rows = []
    for sym in uni["trading_symbol"]:
        try:
            q = client.live_quote(sym)
            vol = float(q.get("volume") or 0)
            if vol <= 0:
                continue
            avg_price = float(q.get("average_price") or q.get("last_price") or 0)
            ohlc = _parse_ohlc(q)
            last = float(q.get("last_price") or ohlc["close"] or avg_price)
            day_change = float(q.get("day_change") or 0)
            prev_close = last - day_change if last else 0
            rows.append({
                "symbol": sym,
                "open_vol": vol,
                "open_vwap": avg_price or last,
                "open_turnover": vol * (avg_price or last),
                "open_first": ohlc["open"] or last,
                "open_high": ohlc["high"], "open_low": ohlc["low"],
                "open_ret": (last / ohlc["open"] - 1.0) if ohlc["open"] else 0.0,
                "open_range_pct": ((ohlc["high"] - ohlc["low"]) / ohlc["open"]) if ohlc["open"] else 0.0,
                "last_price": last, "prev_close": prev_close,
            })
        except Exception as e:  # noqa
            print(f"[live] {sym}: {e}")

    if not rows:
        raise RuntimeError("No live quotes returned — is the market open and token valid?")

    df = pd.DataFrame(rows).merge(ctx, on="symbol", how="left")
    # context cols may arrive as object dtype (empty merge) — coerce to float
    for c in ["open_share_hist", "prev_full_vol", "prev_full_turnover", "avg5_full_vol",
              "avg20_full_vol", "avg20_full_turnover", "std20_full_vol"]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_share_hist"] = df["open_share_hist"].fillna(global_share).clip(1e-4, 0.9)

    # rebuild the projection + derived features with today's live numbers
    df["proj_full_vol"] = df["open_vol"] / df["open_share_hist"]
    df["proj_full_turnover"] = df["proj_full_vol"] * df["open_vwap"]
    df["vol_zscore"] = ((df["open_vol"] - df["avg20_full_vol"] * df["open_share_hist"])
                        / (df["std20_full_vol"] * df["open_share_hist"] + 1e-9)).fillna(0)
    df["gap_pct"] = (df["open_first"] / df["prev_close"] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["dow"] = datetime.now().weekday()
    for c in FEATURE_COLS:
        if c not in df:
            df[c] = 0.0

    ranked = rank_day(df)
    top = ranked.head(top_n)

    # confidence = separation of #1 from the field, squashed to 0-100
    s = ranked["score"].to_numpy()
    if len(s) > 1 and np.std(s) > 0:
        gap = (s[0] - s[1]) / (np.std(s) + 1e-9)
        confidence = float(np.clip(50 + gap * 20, 5, 99))
    else:
        confidence = 50.0

    pick = ranked.iloc[0]

    def rnd(x, n=2):
        """Round, but turn NaN/inf into None so the result is always JSON-valid."""
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        return round(xf, n) if np.isfinite(xf) else None

    result = {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "window": f"{CFG.open_from}-{CFG.open_to} IST",
        "metric": CFG.rank_metric,
        "model": CFG.model_mode,
        "scanned": int(len(ranked)),
        "most_traded_pick": {
            "symbol": pick["symbol"],
            "confidence_pct": rnd(confidence, 1),
            "open_vol": int(pick["open_vol"]),
            "open_turnover_cr": rnd(pick["open_turnover"] / 1e7),
            "proj_full_turnover_cr": rnd(pick["proj_full_turnover"] / 1e7),
            "open_ret_pct": rnd(pick["open_ret"] * 100),
            "groww_url": f"https://groww.in/stocks/{str(pick['symbol']).lower()}",
        },
        "top": [
            {
                "rank": int(r.pred_rank),
                "symbol": r.symbol,
                "open_turnover_cr": rnd(r.open_turnover / 1e7),
                "proj_full_turnover_cr": rnd(r.proj_full_turnover / 1e7),
                "open_ret_pct": rnd(r.open_ret * 100),
                "score": rnd(float(r.score), 4),
            }
            for r in top.itertuples()
        ],
    }
    # allow_nan=False guarantees we never write invalid JSON; rnd() already nulls non-finite
    CFG.predictions_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["most_traded_pick"], indent=2))
    print(f"[live] full ranking written -> {CFG.predictions_path}")
    return result


if __name__ == "__main__":
    predict_live()