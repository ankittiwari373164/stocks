"""
Build the historical training/backtest dataset from Groww 1-min candles.

Memory-efficient: fetches one symbol at a time, reduces it to a handful of
per-day aggregate rows, and frees the raw candles immediately. This keeps
peak memory tiny so the build completes inside small hosts (e.g. Render
free, 512 MB) without OOM restarts.

Saves:
  data/training_dataset.pkl   (features + labels, one row per symbol-day)
  data/opening_shares.pkl     (per-symbol opening share, for live use)
"""
from __future__ import annotations
from datetime import datetime, timedelta
import os
import sys
import gc
import pandas as pd

from .config import CFG
from .groww_client import GrowwClient
from .universe import build_universe
from .features import symbol_day_rows, finalize_panel, add_labels


def build_dataset(client: GrowwClient | None = None, max_symbols: int | None = None) -> pd.DataFrame:
    client = client or GrowwClient()
    uni = build_universe(client)
    symbols = uni["trading_symbol"].tolist()

    # cap symbols for memory/time on small hosts (env DATASET_MAX_SYMBOLS, 0 = all)
    cap = max_symbols if max_symbols is not None else int(os.environ.get("DATASET_MAX_SYMBOLS", "0") or 0)
    if cap and cap > 0:
        symbols = symbols[:cap]
        print(f"[dataset] capped to {cap} symbols for this build", flush=True)

    end = datetime.now()
    start = end - timedelta(days=CFG.history_days)

    rows: list[dict] = []
    for i, sym in enumerate(symbols, 1):
        try:
            candles = client.historical_candles(sym, start, end, interval_minutes=1)
            rows.extend(symbol_day_rows(sym, candles, CFG.open_from, CFG.open_to))
            n = len(candles)
            del candles            # free immediately — keeps memory flat
            if i % 10 == 0 or i == len(symbols):
                print(f"[dataset] {i}/{len(symbols)} {sym}: {n} candles | rows so far {len(rows)}", flush=True)
        except Exception as e:  # noqa
            print(f"[dataset] {i}/{len(symbols)} {sym}: ERROR {e}", flush=True)
        if i % 25 == 0:
            gc.collect()

    panel = finalize_panel(rows)
    if panel.empty:
        print("[dataset] empty panel — check credentials / market data access", flush=True)
        return panel
    panel = add_labels(panel, CFG.rank_metric)
    panel.to_pickle(CFG.dataset_path)

    shares = panel.groupby("symbol")["open_share_hist"].last().rename("open_share_hist").reset_index()
    shares.to_pickle(CFG.shares_path)

    print(f"[dataset] saved {len(panel)} rows -> {CFG.dataset_path}", flush=True)
    print(f"[dataset] saved {len(shares)} opening shares -> {CFG.shares_path}", flush=True)
    return panel


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_dataset(max_symbols=n)