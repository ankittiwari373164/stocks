"""
Track record — the honest "accuracy" engine.

Every live prediction is logged. After the market closes, each past
prediction is scored against what ACTUALLY happened (fetched from Groww
historical candles for that day), producing a real, rolling hit-rate for:
  * direction  (did the BUY/SELL call match the close vs entry?)
  * band       (did the day stay inside the predicted High/Low?)
  * target/stop(which got hit first — target, stop, or neither?)

This is the number to trust before risking money — not the live "conviction".
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta

from .config import CFG


def _load() -> list:
    if CFG.track_path.exists():
        try:
            return json.loads(CFG.track_path.read_text())
        except Exception:  # noqa
            return []
    return []


def _save(records: list):
    CFG.track_path.write_text(json.dumps(records, indent=2, allow_nan=False))


def record_prediction(result: dict):
    """Append today's pick + signal to the track log (one row per day)."""
    p = result.get("most_traded_pick", {})
    today = datetime.now().strftime("%Y-%m-%d")
    recs = _load()
    if any(r.get("date") == today for r in recs):   # one record per day
        recs = [r for r in recs if r.get("date") != today]
    recs.append({
        "date": today,
        "pick": p.get("symbol"),
        "signal": p.get("signal", "NEUTRAL"),
        "entry": p.get("last_price"),
        "target": p.get("target"),
        "stop": p.get("stop"),
        "expected_high": p.get("expected_high"),
        "expected_low": p.get("expected_low"),
        "scored": False,
    })
    _save(recs)


def score_pending(client=None):
    """Score past, unscored predictions using that day's actual candle."""
    from .groww_client import GrowwClient
    recs = _load()
    pending = [r for r in recs if not r.get("scored") and r.get("date") < datetime.now().strftime("%Y-%m-%d")]
    if not pending:
        return {"scored_now": 0, **summary()}
    client = client or GrowwClient()
    n = 0
    for r in pending:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d")
            candles = client.historical_candles(r["pick"], d, d + timedelta(days=1), interval_minutes=1440)
            if not candles:
                continue
            _, o, h, l, c, _v = candles[-1]
            entry = r.get("entry") or o
            r["actual_close"] = c
            r["actual_high"] = h
            r["actual_low"] = l
            r["move_pct"] = round((c / entry - 1) * 100, 2) if entry else None
            if r["signal"] == "BUY":
                r["direction_correct"] = c > entry
            elif r["signal"] == "SELL":
                r["direction_correct"] = c < entry
            else:
                r["direction_correct"] = None
            r["band_held"] = (r.get("expected_high") is not None and r.get("expected_low") is not None
                              and h <= r["expected_high"] and l >= r["expected_low"])
            # which came first is unknown from a daily candle; record hit/touch instead
            if r.get("target") is not None and r.get("stop") is not None:
                if r["signal"] == "BUY":
                    r["target_hit"] = h >= r["target"]; r["stop_hit"] = l <= r["stop"]
                elif r["signal"] == "SELL":
                    r["target_hit"] = l <= r["target"]; r["stop_hit"] = h >= r["stop"]
            r["scored"] = True
            n += 1
        except Exception as e:  # noqa
            print(f"[tracker] could not score {r.get('date')} {r.get('pick')}: {e}")
    _save(recs)
    return {"scored_now": n, **summary()}


def summary() -> dict:
    recs = _load()
    scored = [r for r in recs if r.get("scored")]
    directional = [r for r in scored if r.get("direction_correct") is not None]
    banded = [r for r in scored if "band_held" in r]
    def rate(xs, key):
        xs = [r for r in xs if r.get(key) is not None]
        return round(sum(1 for r in xs if r[key]) / len(xs), 3) if xs else None
    return {
        "total_logged": len(recs),
        "total_scored": len(scored),
        "direction_hit_rate": rate(directional, "direction_correct"),
        "directional_n": len(directional),
        "band_coverage": rate(banded, "band_held"),
        "recent": recs[-15:][::-1],
    }