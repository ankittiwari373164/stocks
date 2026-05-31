"""
Build the stock universe to scan.

Default = NSE F&O underlyings (~190 names). Rationale: the single
"most traded" stock of the day is almost always one of the highly
liquid F&O names, so scanning these covers the answer while keeping
the live scan well within rate limits.
"""
from __future__ import annotations
import io
import pandas as pd

from .config import CFG

# Static fallback used if the instruments CSV can't be downloaded.
# Top liquid NSE names — not exhaustive, but a safe minimum.
FALLBACK = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY", "TCS", "AXISBANK",
    "KOTAKBANK", "ITC", "LT", "BHARTIARTL", "TATAMOTORS", "TATASTEEL", "WIPRO",
    "BAJFINANCE", "HINDALCO", "ADANIENT", "ADANIPORTS", "MARUTI", "SUNPHARMA",
    "ONGC", "POWERGRID", "NTPC", "COALINDIA", "JSWSTEEL", "VEDL", "IDEA",
    "YESBANK", "PNB", "BANKBARODA", "ZOMATO", "PAYTM", "IRFC", "GAIL",
]


def build_universe(client=None) -> pd.DataFrame:
    """Return DataFrame[trading_symbol, exchange, segment] for the live scan."""
    cfg = CFG
    if cfg.universe == "CUSTOM" and cfg.custom_symbols:
        df = pd.DataFrame({"trading_symbol": cfg.custom_symbols})
        df["exchange"], df["segment"] = "NSE", "CASH"
        df.to_csv(cfg.universe_path, index=False)
        return df

    try:
        text = client.instruments_csv() if client else None
        if text is None:
            raise RuntimeError("no client")
        inst = pd.read_csv(io.StringIO(text), low_memory=False)
        inst.columns = [c.strip() for c in inst.columns]

        if cfg.universe == "FNO":
            # underlyings that have F&O contracts = the liquid set
            fno = inst[(inst["segment"] == "FNO") & (inst["exchange"] == "NSE")]
            symbols = sorted(fno["underlying_symbol"].dropna().unique().tolist())
            # keep only those that exist as a tradable CASH/EQ instrument
            cash = inst[(inst["segment"] == "CASH") & (inst["exchange"] == "NSE")
                        & (inst.get("series", "EQ") == "EQ")]
            cash_syms = set(cash["trading_symbol"].astype(str))
            symbols = [s for s in symbols if s in cash_syms]
        else:  # NIFTY200 or anything else -> all NSE EQ cash (then trimmed by caller)
            cash = inst[(inst["segment"] == "CASH") & (inst["exchange"] == "NSE")
                        & (inst.get("series", "EQ") == "EQ")]
            symbols = sorted(cash["trading_symbol"].astype(str).unique().tolist())

        if not symbols:
            raise RuntimeError("empty universe from CSV")
    except Exception as e:  # noqa
        print(f"[universe] falling back to static list ({e})")
        symbols = FALLBACK

    df = pd.DataFrame({"trading_symbol": symbols})
    df["exchange"], df["segment"] = "NSE", "CASH"
    df.to_csv(cfg.universe_path, index=False)
    print(f"[universe] {len(df)} symbols ({cfg.universe})")
    return df
