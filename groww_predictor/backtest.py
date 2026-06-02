"""
Backtest the predictor on held-out days and report accuracy honestly.

Compares three rankers on the same out-of-sample days:
  * naive      : rank by raw opening turnover/volume (no share correction)
  * analytic   : rank by projected full-day turnover/volume (share-corrected)
  * lgbm       : the trained LambdaRank model (if available)

Metrics per ranker:
  top1_acc          predicted #1 == actual #1
  actual1_in_pred3  actual #1 is within our predicted top-3   (the practical metric)
  top3_jaccard      overlap of predicted vs actual top-3
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import CFG
from .features import FEATURE_COLS, add_labels
from .model import train_ranker, analytic_score, _HAS_LGB


def _naive_score(df):
    col = "open_turnover" if CFG.rank_metric == "turnover" else "open_vol"
    return df[col].fillna(0).to_numpy()


def _eval_ranker(test: pd.DataFrame, score_fn) -> dict:
    top1 = a1in3 = jac = n = 0
    for _, day in test.groupby("date"):
        if len(day) < 3:
            continue
        d = day.copy()
        d["s"] = score_fn(d)
        pred = d.sort_values("s", ascending=False)
        pred_top1 = pred.iloc[0]["symbol"]
        pred_top3 = set(pred.head(3)["symbol"])
        actual_top1 = d.loc[d["rank"] == 1, "symbol"].iloc[0]
        actual_top3 = set(d.loc[d["rank"] <= 3, "symbol"])
        top1 += int(pred_top1 == actual_top1)
        a1in3 += int(actual_top1 in pred_top3)
        jac += len(pred_top3 & actual_top3) / len(pred_top3 | actual_top3)
        n += 1
    n = max(n, 1)
    return {"days": n, "top1_acc": top1 / n, "actual1_in_pred3": a1in3 / n,
            "top3_jaccard": jac / n}


def signal_backtest(panel: pd.DataFrame) -> dict:
    """Honestly measure the buy/sell lean and the High/Low range on the same data.
       - direction hit-rate: does the lean's sign match the rest-of-day move?
       - range coverage: did the actual day stay inside the expected band?
    """
    from .signals import directional_lean, expected_high_low
    need = {"gap_pct", "open_ret", "open_high", "open_low", "open_last",
            "rest_of_day_ret", "day_high", "day_low", "hist_range_pct"}
    if not need.issubset(panel.columns):
        return {}
    d = panel.dropna(subset=list(need)).copy()
    if d.empty:
        return {}

    rng = (d["open_high"] - d["open_low"]).replace(0, np.nan)
    pos = ((d["open_last"] - d["open_low"]) / rng).clip(0, 1).fillna(0.5)

    hits = directional = 0
    covered = 0
    for (g, m, p, vz, ror, dh, dl, do, hr) in zip(
        d["gap_pct"], d["open_ret"], pos, d.get("vol_zscore", pd.Series(0, index=d.index)),
        d["rest_of_day_ret"], d["day_high"], d["day_low"], d["day_open"], d["hist_range_pct"]):
        label, score, _ = directional_lean(g, m, p, vz)
        if label != "NEUTRAL":
            directional += 1
            hits += int((score > 0 and ror > 0) or (score < 0 and ror < 0))
        band = expected_high_low(do, hr, None, None, score)   # full-day band centred on the open
        covered += int(dh <= band["expected_high"] and dl >= band["expected_low"])

    n = len(d)
    return {
        "direction_hit_rate": round(hits / directional, 3) if directional else None,
        "directional_signals_pct": round(directional / n, 3),
        "range_coverage": round(covered / n, 3),
        "samples": int(n),
    }


def backtest(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel is None:
        panel = pd.read_pickle(CFG.dataset_path)
    if "rank" not in panel.columns:
        panel = add_labels(panel, CFG.rank_metric)
    panel = panel.dropna(subset=FEATURE_COLS).sort_values("date")

    days = np.sort(panel["date"].unique())
    if len(days) < 6:
        print("[backtest] not enough days for a meaningful split")
        cut = days[max(len(days) - 1, 0)]
    else:
        cut = days[int(len(days) * 0.8)]
    train, test = panel[panel["date"] < cut], panel[panel["date"] >= cut]

    results = {
        "naive": _eval_ranker(test, _naive_score),
        "analytic": _eval_ranker(test, analytic_score),
    }
    if _HAS_LGB:
        booster = train_ranker(train)
        if booster is not None:
            def lgb_score(d):
                return booster.predict(d[FEATURE_COLS].fillna(0),
                                       num_iteration=booster.best_iteration)
            results["lgbm"] = _eval_ranker(test, lgb_score)

    out = pd.DataFrame(results).T
    print("\n=== BACKTEST (out-of-sample) ===")
    print(f"metric={CFG.rank_metric}  test_days={results['analytic']['days']}")
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- pick the ranker to serve in `auto` mode ----
    # primary metric: actual #1 inside our predicted top-3 (the practical target);
    # tie-break prefers the more robust/transparent model so we don't chase noise.
    from .model import save_best_model
    preference = {"analytic": 0, "naive": 1, "lgbm": 2}  # lower = preferred on ties
    best = sorted(
        results.keys(),
        key=lambda k: (-results[k]["actual1_in_pred3"], preference.get(k, 9)),
    )[0]
    save_best_model(best, {
        "metric": CFG.rank_metric,
        "actual1_in_pred3": round(results[best]["actual1_in_pred3"], 4),
        "test_days": int(results[best]["days"]),
    })
    print(f"[backtest] auto-mode will serve: {best.upper()} "
          f"(actual#1-in-top3={results[best]['actual1_in_pred3']:.2f} over "
          f"{results[best]['days']} days)")

    # honest measurement of the buy/sell lean + High/Low range
    sig = signal_backtest(panel)
    if sig:
        print("\n=== SIGNAL BACKTEST (buy/sell is weak by nature — read these honestly) ===")
        print(f"  direction hit-rate : {sig['direction_hit_rate']}  "
              f"(on {sig['directional_signals_pct']*100:.0f}% of days a non-neutral lean fired)")
        print(f"  High/Low coverage  : {sig['range_coverage']}  "
              f"(fraction of days the real day stayed inside the predicted band)")
        print(f"  samples            : {sig['samples']}")
    return out


if __name__ == "__main__":
    backtest()