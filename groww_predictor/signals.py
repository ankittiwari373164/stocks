"""
Buy/Sell lean and expected day High/Low range.

IMPORTANT, READ THIS:
  * The expected High/Low *range* is the defensible part. Volatility is
    persistent day to day, so projecting a day's range from recent daily
    ranges is reasonably well-calibrated (we measure it in the backtest).
  * The Buy/Sell *direction* is NOT reliably predictable intraday. This is a
    transparent momentum lean, not a forecast. Treat the backtested hit-rate
    (typically ~50-55%) as the truth, and never as a guarantee.

None of this is investment advice.
"""
from __future__ import annotations
import numpy as np


def directional_lean(gap_pct: float, open_ret: float, pos_in_range: float,
                     vol_z: float = 0.0) -> tuple[str, float, float]:
    """
    Combine opening signals into a lean.
      gap_pct       : open vs previous close (overnight gap)
      open_ret      : move across the opening window (09:15->09:25)
      pos_in_range  : where the last price sits in the opening range, 0..1
                      (1 = at the highs = bullish, 0 = at the lows = bearish)
      vol_z         : opening-volume surprise (conviction multiplier)
    Returns (label, score, confidence_pct). score in roughly [-1, 1].
    """
    g = np.tanh((gap_pct or 0) * 50)        # ±2% gap ~ ±0.76
    m = np.tanh((open_ret or 0) * 80)       # opening momentum
    p = (np.clip(pos_in_range, 0, 1) - 0.5) * 2 if pos_in_range == pos_in_range else 0.0
    conviction = 1.0 + 0.3 * np.tanh(abs(vol_z or 0))   # heavier opening volume = a bit more conviction

    score = float(np.clip((0.45 * m + 0.30 * g + 0.25 * p) * conviction, -1, 1))
    if score > 0.15:
        label = "BUY"
    elif score < -0.15:
        label = "SELL"
    else:
        label = "NEUTRAL"
    # confidence is the *strength of the lean*, NOT a probability of being right
    confidence = float(np.clip(abs(score) * 100, 0, 95))
    return label, score, round(confidence, 1)


def expected_high_low(anchor_price: float, hist_range_pct: float,
                      open_high: float = None, open_low: float = None,
                      lean_score: float = 0.0) -> dict:
    """
    Project the day's High/Low as a band around the current price.
      anchor_price   : current price (last traded)
      hist_range_pct : the stock's typical full-day (high-low)/open
      open_high/low  : opening-window extremes (the band can't be tighter than these)
      lean_score     : -1..1, nudges the band asymmetrically in the lean's direction
    The band is centred on the anchor and sized by the historical range, then
    skewed slightly by the lean. It is an EXPECTATION, not a guarantee.
    """
    hr = max(float(hist_range_pct or 0.02), 0.002)
    half = anchor_price * hr * 0.6      # ~1.2x the median daily range, total band
    # skew: a bullish lean lifts the high more than the low (and vice-versa)
    skew = float(np.clip(lean_score, -1, 1)) * 0.35
    exp_high = anchor_price + half * (1 + skew)
    exp_low = anchor_price - half * (1 - skew)
    # never inside the opening range already printed
    if open_high:
        exp_high = max(exp_high, open_high)
    if open_low:
        exp_low = min(exp_low, open_low)
    return {
        "expected_high": round(float(exp_high), 2),
        "expected_low": round(float(exp_low), 2),
        "expected_range_pct": round(hr * 100, 2),
    }