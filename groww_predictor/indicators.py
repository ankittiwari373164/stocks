"""
Standard technical indicators, computed from a daily OHLC series.

These are the indicators traders expect to see (RSI, MACD, ATR, ADX) plus
VWAP positioning. Honest note: indicators are deterministic functions of past
price — they organise information, they do not add a crystal ball. They feed
the signal and the target/stop sizing; the tracker measures whether any of it
actually works.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _series(x) -> pd.Series:
    return pd.Series(np.asarray(x, dtype="float64"))


def rsi(closes, n: int = 14) -> float:
    c = _series(closes)
    if len(c) < n + 1:
        return float("nan")
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def macd(closes, fast=12, slow=26, sig=9) -> dict:
    c = _series(closes)
    if len(c) < slow + sig:
        return {"macd": float("nan"), "signal": float("nan"), "hist": float("nan")}
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return {"macd": float(line.iloc[-1]), "signal": float(signal.iloc[-1]),
            "hist": float(line.iloc[-1] - signal.iloc[-1])}


def atr(highs, lows, closes, n: int = 14) -> float:
    h, l, c = _series(highs), _series(lows), _series(closes)
    if len(c) < n + 1:
        return float("nan")
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/n, adjust=False).mean().iloc[-1])


def adx(highs, lows, closes, n: int = 14) -> float:
    h, l, c = _series(highs), _series(lows), _series(closes)
    if len(c) < 2 * n:
        return float("nan")
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean() / (atr_ + 1e-12)
    mdi = 100 * pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean() / (atr_ + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return float(dx.ewm(alpha=1/n, adjust=False).mean().iloc[-1])


def compute_all(highs, lows, closes, last_price: float = None, vwap: float = None) -> dict:
    """Return the latest indicator readings + simple bull/bear votes in [-1,1]."""
    closes = list(closes)
    if last_price is not None:
        closes = closes + [last_price]          # include today's live price
        highs = list(highs) + [max(last_price, highs[-1] if len(highs) else last_price)]
        lows = list(lows) + [min(last_price, lows[-1] if len(lows) else last_price)]
    r = rsi(closes); m = macd(closes); a = atr(highs, lows, closes); adx_v = adx(highs, lows, closes)

    votes = []
    if r == r:  # not NaN
        votes.append((50 - r) / 50)            # <50 oversold→bullish, >50→bearish-ish (mild)
    if m["hist"] == m["hist"]:
        votes.append(np.tanh(m["hist"] / (abs(closes[-1]) * 0.005 + 1e-9)))
    if vwap and last_price:
        votes.append(np.tanh((last_price / vwap - 1) * 100))
    ind_score = float(np.clip(np.mean(votes), -1, 1)) if votes else 0.0
    return {
        "rsi": None if r != r else round(r, 1),
        "macd_hist": None if m["hist"] != m["hist"] else round(m["hist"], 3),
        "atr": None if a != a else round(a, 2),
        "adx": None if adx_v != adx_v else round(adx_v, 1),
        "ind_score": round(ind_score, 3),
    }