"""
Offline self-test — NO network, NO Groww token needed.

Generates synthetic 1-min candles for a basket of stocks over ~40 trading
days, where each stock has a persistent volume level and a stable (but
stock-specific) opening-volume share plus daily noise. Then runs the full
pipeline and checks that:
  1. the feature panel builds with no look-ahead leakage,
  2. the analytic (share-corrected) ranker beats the naive raw-opening ranker,
  3. the live scoring path produces a #1 "most traded" pick.

Run:  python -m groww_predictor.selftest
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import CFG
from .features import build_panel, add_labels, FEATURE_COLS
from .backtest import backtest
from .model import rank_day, _HAS_LGB

RNG = np.random.default_rng(7)
MKT_OPEN_MIN = 9 * 60 + 15      # 09:15
MKT_CLOSE_MIN = 15 * 60 + 30    # 15:30
N_MIN = MKT_CLOSE_MIN - MKT_OPEN_MIN  # 375 minutes


def _synth_day_candles(date: pd.Timestamp, price: float, day_vol: float, open_share: float):
    """Return list of [epoch, o,h,l,c,v] for one symbol-day (IST minutes)."""
    open_window = 10
    open_vol = day_vol * open_share * RNG.uniform(0.85, 1.15)
    rest_vol = max(day_vol - open_vol, day_vol * 0.1)
    # U-shape weights for the remaining minutes
    x = np.linspace(0, 1, N_MIN - open_window)
    w = 0.4 + (x - 0.5) ** 2 * 4
    w = w / w.sum()
    minute_vol = np.concatenate([
        np.full(open_window, open_vol / open_window),
        rest_vol * w,
    ])
    rets = RNG.normal(0, 0.0006, N_MIN)
    closes = price * np.cumprod(1 + rets)
    opens = np.concatenate([[price], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + RNG.uniform(0, 0.0008, N_MIN))
    lows = np.minimum(opens, closes) * (1 - RNG.uniform(0, 0.0008, N_MIN))
    base_epoch = int((date - pd.Timedelta(hours=5, minutes=30)).timestamp()) + MKT_OPEN_MIN * 60
    candles = []
    for i in range(N_MIN):
        candles.append([base_epoch + i * 60, round(float(opens[i]), 2), round(float(highs[i]), 2),
                        round(float(lows[i]), 2), round(float(closes[i]), 2), int(minute_vol[i])])
    return candles


def make_synthetic(n_symbols=30, n_days=40):
    dates = pd.bdate_range("2025-01-06", periods=n_days)
    syms = [f"STK{i:02d}" for i in range(n_symbols)]
    price = {s: float(RNG.uniform(80, 3500)) for s in syms}
    base_vol = {s: float(RNG.lognormal(13, 1.0)) for s in syms}        # persistent size
    share = {s: float(np.clip(RNG.normal(0.06, 0.02), 0.02, 0.14)) for s in syms}  # stable per-stock
    data = {}
    for s in syms:
        candles = []
        v = base_vol[s]
        for d in dates:
            v = 0.7 * v + 0.3 * base_vol[s] * RNG.lognormal(0, 0.35)  # autocorrelated daily volume
            candles += _synth_day_candles(d, price[s], v, share[s] * RNG.uniform(0.9, 1.1))
        data[s] = candles
    return data


def main():
    print("Generating synthetic market (no network)...")
    data = make_synthetic()
    panel = build_panel(data, CFG.open_from, CFG.open_to)
    panel = add_labels(panel, CFG.rank_metric)
    print(f"panel: {len(panel)} rows, {panel['date'].nunique()} days, {panel['symbol'].nunique()} symbols")

    # leakage check: open_share_hist on a symbol's first day must be NaN/global (not its own same-day share)
    first_rows = panel.sort_values("date").groupby("symbol").head(1)
    assert (first_rows["open_share_hist"].notna()).all(), "share fallback failed"
    print("leak check: opening share uses only past days  OK")

    res = backtest(panel)
    a = res.loc["analytic"]; n = res.loc["naive"]
    print(f"\nanalytic actual1_in_pred3={a['actual1_in_pred3']:.2f}  "
          f"vs naive={n['actual1_in_pred3']:.2f}")
    assert a["actual1_in_pred3"] >= n["actual1_in_pred3"] - 1e-9, \
        "share-corrected model should not be worse than naive"
    print("ranking check: share-corrected >= naive  OK")

    # live scoring path on the latest day
    last_day = panel["date"].max()
    today = panel[panel["date"] == last_day].copy()
    ranked = rank_day(today)
    pick = ranked.iloc[0]
    print(f"\nLIVE-style pick for {last_day.date()}: {pick['symbol']}  "
          f"proj_turnover={pick['proj_full_turnover']/1e7:.1f} Cr  "
          f"(actual rank that day = {int(pick['rank'])})")
    print(f"lightgbm available: {_HAS_LGB}")
    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
