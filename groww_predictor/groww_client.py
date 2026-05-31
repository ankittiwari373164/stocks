"""
Thin REST client for the Groww Trading API.

Only the read-only market-data endpoints we need are implemented:
  - authentication (access-token / api-key+secret / api-key+TOTP)
  - historical 1-min candles            (/v1/historical/candle/range)
  - live quote with cumulative volume   (/v1/live-data/quote)
  - live LTP (batch)                     (/v1/live-data/ltp)

Docs: https://groww.in/trade-api/docs/curl
"""
from __future__ import annotations
import time
import hashlib
import threading
from collections import deque
from datetime import datetime, timedelta

import requests

from .config import CFG


class RateLimiter:
    """Token-bucket-ish limiter: at most `per_sec` per second and `per_min` per minute."""
    def __init__(self, per_sec: int, per_min: int):
        self.per_sec, self.per_min = per_sec, per_min
        self._sec, self._min = deque(), deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.time()
                while self._sec and now - self._sec[0] > 1.0:
                    self._sec.popleft()
                while self._min and now - self._min[0] > 60.0:
                    self._min.popleft()
                if len(self._sec) < self.per_sec and len(self._min) < self.per_min:
                    self._sec.append(now)
                    self._min.append(now)
                    return
                # how long to wait
                wait = 0.05
                if len(self._sec) >= self.per_sec:
                    wait = max(wait, 1.0 - (now - self._sec[0]))
                if len(self._min) >= self.per_min:
                    wait = max(wait, 60.0 - (now - self._min[0]))
            time.sleep(wait)


class GrowwClient:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.s = requests.Session()
        self.token = self._authenticate()
        self.s.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "X-API-VERSION": cfg.api_version,
        })
        # Live Data type: 10/s, 300/min (shared by quote/ltp/ohlc)
        self.live_rl = RateLimiter(per_sec=9, per_min=290)
        # Historical / non-trading is more generous; keep it polite
        self.hist_rl = RateLimiter(per_sec=9, per_min=290)

    # ---------------------------------------------------------------- auth
    def _authenticate(self) -> str:
        cfg = self.cfg
        if cfg.access_token:
            return cfg.access_token
        if cfg.api_key and cfg.api_secret:
            return self._token_from_secret()
        if cfg.api_key and cfg.totp_secret:
            return self._token_from_totp()
        raise RuntimeError(
            "No Groww credentials found. Set GROWW_ACCESS_TOKEN, or "
            "GROWW_API_KEY+GROWW_API_SECRET, or GROWW_API_KEY+GROWW_TOTP_SECRET in .env"
        )

    def _post_token(self, body: dict) -> str:
        r = requests.post(
            f"{self.cfg.base_url}/v1/token/api/access",
            headers={"Authorization": f"Bearer {self.cfg.api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        tok = data.get("token") or data.get("payload", {}).get("token")
        if not tok:
            raise RuntimeError(f"Token generation failed: {data}")
        return tok

    def _token_from_secret(self) -> str:
        ts = str(int(time.time()))
        checksum = hashlib.sha256((self.cfg.api_secret + ts).encode()).hexdigest()
        return self._post_token({"key_type": "approval", "checksum": checksum, "timestamp": ts})

    def _token_from_totp(self) -> str:
        import pyotp
        totp = pyotp.TOTP(self.cfg.totp_secret).now()
        return self._post_token({"key_type": "totp", "totp": totp})

    # ------------------------------------------------------------- requests
    def _get(self, path: str, params: dict, rl: RateLimiter, tries: int = 4):
        last = None
        for i in range(tries):
            rl.acquire()
            try:
                r = self.s.get(f"{self.cfg.base_url}{path}", params=params, timeout=15)
                if r.status_code == 429:
                    time.sleep(1.5 * (i + 1)); continue
                r.raise_for_status()
                j = r.json()
                if j.get("status") == "FAILURE":
                    raise RuntimeError(j.get("error"))
                return j.get("payload", {})
            except Exception as e:  # noqa
                last = e
                time.sleep(0.6 * (i + 1))
        raise RuntimeError(f"GET {path} failed after {tries} tries: {last}")

    # ----------------------------------------------------------- historical
    def historical_candles(self, trading_symbol: str, start: datetime, end: datetime,
                           interval_minutes: int = 1, exchange="NSE", segment="CASH"):
        """
        Returns list of [epoch_sec, open, high, low, close, volume].
        Automatically chunks 1-min requests into <=7-day windows (API limit).
        """
        max_span = {1: 7, 5: 15, 10: 30, 60: 150, 240: 365, 1440: 1080}.get(interval_minutes, 7)
        out = []
        cur = start
        while cur < end:
            chunk_end = min(cur + timedelta(days=max_span), end)
            payload = self._get(
                "/v1/historical/candle/range",
                {
                    "exchange": exchange, "segment": segment,
                    "trading_symbol": trading_symbol,
                    "start_time": cur.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "interval_in_minutes": str(interval_minutes),
                },
                self.hist_rl,
            )
            out.extend(payload.get("candles", []) or [])
            cur = chunk_end
        return out

    # ----------------------------------------------------------------- live
    def live_quote(self, trading_symbol: str, exchange="NSE", segment="CASH") -> dict:
        """Full snapshot incl. cumulative `volume` and `average_price` (VWAP)."""
        return self._get(
            "/v1/live-data/quote",
            {"exchange": exchange, "segment": segment, "trading_symbol": trading_symbol},
            self.live_rl,
        )

    def instruments_csv(self) -> str:
        r = requests.get(self.cfg.instruments_url, timeout=30)
        r.raise_for_status()
        return r.text
