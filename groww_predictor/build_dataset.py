"""
Build the historical training/backtest dataset from Groww 1-min candles.

For each symbol in the universe we pull 1-min candles for the last
HISTORY_DAYS, then build the leak-free feature panel and labels.
Saves:
  data/training_dataset.parquet   (features + labels, one row per symbol-day)
  data/opening_shares.parquet      (per-symbol median opening share, for live use)
"""
from __future__ import annotations
from datetime import datetime, timedelta
import sys
import pandas as pd

from .config import CFG
from .groww_client import GrowwClient
from .universe import build_universe
from .features import build_panel, add_labels


def build_dataset(client: GrowwClient | None = None, max_symbols: int | None = None) -> pd.DataFrame:
    client = client or GrowwClient()
    uni = build_universe(client)
    symbols = uni["trading_symbol"].tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]

    end = datetime.now()
    start = end - timedelta(days=CFG.history_days)

    per_symbol = {}
    for i, sym in enumerate(symbols, 1):
        try:
            candles = client.historical_candles(sym, start, end, interval_minutes=1)
            if candles:
                per_symbol[sym] = candles
            print(f"[dataset] {i}/{len(symbols)} {sym}: {len(candles)} candles", flush=True)
        except Exception as e:  # noqa
            print(f"[dataset] {i}/{len(symbols)} {sym}: ERROR {e}", flush=True)

    panel = build_panel(per_symbol, CFG.open_from, CFG.open_to)
    if panel.empty:
        print("[dataset] empty panel — check credentials / market data access")
        return panel
    panel = add_labels(panel, CFG.rank_metric)
    panel.to_pickle(CFG.dataset_path)

    # persist per-symbol opening share for the live predictor
    shares = (panel.groupby("symbol")["open_share_hist"].last().rename("open_share_hist").reset_index())
    shares.to_pickle(CFG.shares_path)

    print(f"[dataset] saved {len(panel)} rows -> {CFG.dataset_path}")
    print(f"[dataset] saved {len(shares)} opening shares -> {CFG.shares_path}")
    return panel


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_dataset(max_symbols=n)
