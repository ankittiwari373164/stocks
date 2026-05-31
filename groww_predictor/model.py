"""
Two models, one interface.

1. Analytic projector (always available, no training):
   score = projected full-day turnover (or volume) = open / open_share_hist * price.

2. LightGBM LambdaRank (optional, recommended):
   learns to rank stocks within each day using all features, trained on
   graded relevance (3=#1, 2=top3, 1=top10). Falls back to the analytic
   score if the model file is missing or lightgbm isn't installed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import CFG
from .features import FEATURE_COLS

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


def analytic_score(df: pd.DataFrame) -> np.ndarray:
    col = "proj_full_turnover" if CFG.rank_metric == "turnover" else "proj_full_vol"
    return df[col].fillna(0).to_numpy()


def train_ranker(panel: pd.DataFrame, num_round: int = 400) -> "lgb.Booster | None":
    if not _HAS_LGB:
        print("[model] lightgbm not installed — analytic model only")
        return None
    panel = panel.dropna(subset=FEATURE_COLS + ["relevance"]).sort_values("date")
    if panel["date"].nunique() < 8:
        print("[model] too few days to train a ranker — using analytic model")
        return None

    # time-based split: last 20% of days for validation
    days = np.sort(panel["date"].unique())
    cut = days[int(len(days) * 0.8)]
    tr, va = panel[panel["date"] < cut], panel[panel["date"] >= cut]

    def group_sizes(d):
        return d.groupby("date").size().to_numpy()

    dtr = lgb.Dataset(tr[FEATURE_COLS], label=tr["relevance"], group=group_sizes(tr))
    dva = lgb.Dataset(va[FEATURE_COLS], label=va["relevance"], group=group_sizes(va), reference=dtr)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
    }
    booster = lgb.train(
        params, dtr, num_boost_round=num_round, valid_sets=[dva],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    booster.save_model(str(CFG.model_path))
    print(f"[model] LightGBM ranker saved -> {CFG.model_path} "
          f"(best_iter={booster.best_iteration})")
    return booster


def load_ranker() -> "lgb.Booster | None":
    if not _HAS_LGB or not CFG.model_path.exists():
        return None
    try:
        return lgb.Booster(model_file=str(CFG.model_path))
    except Exception as e:  # noqa
        print(f"[model] could not load ranker: {e}")
        return None


def score(df: pd.DataFrame) -> np.ndarray:
    """Return a ranking score per row, honouring MODEL_MODE."""
    mode = CFG.model_mode
    if mode == "analytic":
        return analytic_score(df)
    if mode in ("auto", "lgbm"):
        booster = load_ranker()
        if booster is not None:
            X = df[FEATURE_COLS].fillna(0)
            return booster.predict(X, num_iteration=booster.best_iteration)
        if mode == "lgbm":
            print("[model] lgbm requested but no model file — using analytic")
    return analytic_score(df)


def rank_day(df_today: pd.DataFrame) -> pd.DataFrame:
    """Add `score` and `pred_rank` to a single-day frame, sorted best first."""
    out = df_today.copy()
    out["score"] = score(out)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["pred_rank"] = np.arange(1, len(out) + 1)
    return out
