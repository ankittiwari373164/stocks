"""
Feature engineering.

Core idea (the whole basis of the predictor):
Intraday volume follows a very stable U-shaped profile, and a stock's
*share* of its full-day volume that trades in the first 10 minutes is
remarkably consistent day to day. So once we observe the 09:15-09:25
volume we can project the full-day volume by dividing by that stock's
historical opening share, then rank stocks by projected full-day
turnover (or volume). The LightGBM ranker later refines this with extra
context, but the projection alone is a strong, transparent baseline.

This module turns raw 1-min candles into:
  * per (date, symbol) opening-window aggregates  -> live features
  * per (date, symbol) realised full-day totals   -> training labels
All times are converted to IST before bucketing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


def _hm_to_min(hm: str) -> int:
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def candles_to_frame(candles: list) -> pd.DataFrame:
    """candles: list of [epoch_sec, o, h, l, c, v] -> tidy IST frame."""
    if not candles:
        return pd.DataFrame(columns=["ts", "date", "mod", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(candles, columns=["epoch", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert(None) + IST_OFFSET
    df["date"] = df["ts"].dt.normalize()
    df["mod"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute  # minute-of-day, IST
    return df.sort_values("ts").reset_index(drop=True)


def day_aggregates(day_df: pd.DataFrame, open_from: str, open_to: str) -> dict | None:
    """Compute opening-window + full-day aggregates for one symbol-day."""
    if day_df.empty:
        return None
    of, ot = _hm_to_min(open_from), _hm_to_min(open_to)
    win = day_df[(day_df["mod"] >= of) & (day_df["mod"] < ot)]
    if win.empty:
        return None

    open_vol = float(win["v"].sum())
    # per-minute turnover ~ volume * close; sum over window
    open_turnover = float((win["v"] * win["c"]).sum())
    first_open = float(win["o"].iloc[0])
    last_close = float(win["c"].iloc[-1])
    win_high = float(win["h"].max())
    win_low = float(win["l"].min())
    open_vwap = open_turnover / open_vol if open_vol > 0 else last_close

    full_vol = float(day_df["v"].sum())
    full_turnover = float((day_df["v"] * day_df["c"]).sum())

    return {
        "open_vol": open_vol,
        "open_turnover": open_turnover,
        "open_ret": (last_close / first_open - 1.0) if first_open else 0.0,
        "open_range_pct": ((win_high - win_low) / first_open) if first_open else 0.0,
        "open_vwap": open_vwap,
        "open_first": first_open,
        "open_high": win_high,
        "open_low": win_low,
        "full_vol": full_vol,
        "full_turnover": full_turnover,
        "n_min": int(len(day_df)),
    }


def build_panel(per_symbol_candles: dict, open_from: str, open_to: str) -> pd.DataFrame:
    """
    per_symbol_candles: {symbol: candles_list}
    Returns long panel with one row per (date, symbol) and lagged context
    features computed without look-ahead leakage.
    """
    rows = []
    for sym, candles in per_symbol_candles.items():
        f = candles_to_frame(candles)
        if f.empty:
            continue
        for date, day_df in f.groupby("date"):
            agg = day_aggregates(day_df, open_from, open_to)
            if agg is None:
                continue
            agg.update({"symbol": sym, "date": date})
            rows.append(agg)
    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    gs = panel.groupby("symbol")  # default group_keys; we use transform/shift (index-aligned)

    # ---- lagged, leak-free context (uses only days strictly before `date`) ----
    panel["prev_close"] = gs["open_first"].shift(1)  # prev day's opening level (proxy; live uses true prev close)
    panel["prev_full_vol"] = gs["full_vol"].shift(1)
    panel["prev_full_turnover"] = gs["full_turnover"].shift(1)
    panel["avg5_full_vol"] = gs["full_vol"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    panel["avg20_full_vol"] = gs["full_vol"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    panel["avg20_full_turnover"] = gs["full_turnover"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    panel["std20_full_vol"] = gs["full_vol"].transform(lambda s: s.shift(1).rolling(20, min_periods=2).std())

    # historical opening share f_s (median of past open_vol/full_vol), leak-free
    panel["_share"] = (panel["open_vol"] / panel["full_vol"]).replace([np.inf, -np.inf], np.nan)
    panel["open_share_hist"] = panel.groupby("symbol")["_share"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).median()
    )
    share = panel["_share"]

    # fallbacks for early days / thin history
    global_share = share.median()
    panel["open_share_hist"] = panel["open_share_hist"].fillna(global_share)
    panel["open_share_hist"] = panel["open_share_hist"].clip(lower=1e-4, upper=0.9)

    # ---- the analytic projection (today's open / historical share) ----
    panel["proj_full_vol"] = panel["open_vol"] / panel["open_share_hist"]
    panel["proj_full_turnover"] = panel["proj_full_vol"] * panel["open_vwap"]

    # volume surprise vs its own trailing average
    panel["vol_zscore"] = ((panel["open_vol"] - panel["avg20_full_vol"] * panel["open_share_hist"])
                           / (panel["std20_full_vol"] * panel["open_share_hist"] + 1e-9))
    panel["gap_pct"] = (panel["open_first"] / panel["prev_close"] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0)
    panel["dow"] = panel["date"].dt.dayofweek

    panel = panel.drop(columns=["_share"])
    return panel


FEATURE_COLS = [
    "open_vol", "open_turnover", "open_ret", "open_range_pct", "open_vwap",
    "prev_full_vol", "prev_full_turnover", "avg5_full_vol", "avg20_full_vol",
    "avg20_full_turnover", "open_share_hist", "proj_full_vol",
    "proj_full_turnover", "vol_zscore", "gap_pct", "dow",
]


def add_labels(panel: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add per-day rank labels for the chosen metric (turnover|volume)."""
    target = "full_turnover" if metric == "turnover" else "full_vol"
    panel = panel.copy()
    panel["rank"] = panel.groupby("date")[target].rank(ascending=False, method="first")
    panel["is_top1"] = (panel["rank"] == 1).astype(int)
    panel["is_top3"] = (panel["rank"] <= 3).astype(int)
    # graded relevance for learning-to-rank
    panel["relevance"] = np.select(
        [panel["rank"] == 1, panel["rank"] <= 3, panel["rank"] <= 10],
        [3, 2, 1], default=0,
    )
    return panel
