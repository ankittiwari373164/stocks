# Groww Most-Traded-Stock Predictor

Predicts **which stock will be the most traded on NSE today**, using the
real **Groww Trading API**, by reading the first 10 minutes of the session
(09:15–09:25 IST) and projecting the full day.

This replaces the prediction logic in your existing Node `server.js`, which
had two fundamental problems described below.

---

## Why the old approach couldn't work (and this one can)

Your previous `server.js`:

1. **Predicted the wrong thing.** "Most traded" is a *volume / turnover*
   question, but `predictFromDeviation()` scored *price direction*
   (BUY/SELL/HOLD from % move + news sentiment). A stock can be the most
   traded while barely moving in price. The target was never volume.
2. **Used delayed, unofficial data.** It pulled quotes by *scraping Yahoo
   Finance* and Groww web pages (`__NEXT_DATA__`). NSE quotes from Yahoo are
   delayed and the scrape breaks whenever Groww changes its markup.
3. **Had magic-number heuristics with no validation.** Thresholds like
   `dev10 > 2 && combined > 0.55` were hand-tuned and never backtested, so
   "most accurate" was a claim, not a measurement.

This package fixes all three: it predicts **full-day turnover/volume**, uses
the **official Groww API** (real-time `volume` + historical 1-min candles),
and **backtests** itself on your own data so accuracy is measured, not
asserted.

### The method (and why it's sound)

Intraday volume on NSE follows a very stable U-shaped profile, and the
*share* of a stock's full-day volume that trades in the first 10 minutes is
remarkably consistent day to day. So:

```
projected_full_day_volume[stock] = observed_09:15-09:25_volume[stock]
                                    ─────────────────────────────────────
                                    that stock's historical opening share
```

Rank stocks by `projected_full_day_turnover = projected_volume × VWAP` and
the #1 is the predicted most-traded stock. This share-correction is what
makes it better than naively ranking by raw opening volume (which is biased
by each stock's idiosyncratic opening behaviour).

On top of that transparent baseline, an optional **LightGBM LambdaRank**
model learns to combine the projection with extra context (previous-day
volume, 5/20-day averages, opening volatility, gap, volume z-score, day of
week) for sharper ranking. If the model file or library is absent, the code
**automatically falls back to the analytic projection** — it always runs.

---

## What's in here

```
groww_predictor/
  config.py          settings + credentials (from .env)
  groww_client.py    Groww REST client: auth, historical candles, live quote, rate-limited
  universe.py        builds the scan list (NSE F&O underlyings by default)
  features.py        opening-window features + full-day labels (leak-free)
  build_dataset.py   pulls ~60d of 1-min candles -> training_dataset.parquet
  model.py           analytic projector + LightGBM ranker (train/load/score)
  backtest.py        walk-forward top-1 / top-3 accuracy vs baselines
  predict_live.py    THE 09:25 PREDICTION -> data/predictions.json
  serve.py           optional FastAPI service (/predict, /predict/run, /backtest)
  selftest.py        offline pipeline test on synthetic data (no token needed)
requirements.txt
.env.example
node_integration.js  drop-in routes for your existing Node dashboard
run_daily.sh         cron driver
```

---

## Setup

```bash
cd groww_predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env with your Groww credentials
```

> **Windows (PowerShell) note:** use `python` (not `python3`), run each
> command on its own line (PowerShell doesn't accept `&&`), and the venv
> is optional. If `python` opens the Microsoft Store, use `py` instead.
> Data is stored as `.pkl`, so no `pyarrow` install is needed.
>
> ```powershell
> pip install -r requirements.txt
> copy .env.example .env
> python -m groww_predictor.selftest
> ```

**Credentials** — pick one mode in `.env`:
- `GROWW_ACCESS_TOKEN` — simplest, but expires daily at 6 AM IST.
- `GROWW_API_KEY` + `GROWW_API_SECRET` — auto-generates a token each run.
- `GROWW_API_KEY` + `GROWW_TOTP_SECRET` — same, via TOTP.

> A Groww **Trading API subscription** with market-data access is required.
> The API supports NSE/BSE CASH + FNO.

Verify everything works **without** any credentials first:

```bash
python3 -m groww_predictor.selftest
```

---

## Daily workflow

```bash
# 1) once (and weekly): build history + train + see measured accuracy
python3 -m groww_predictor.build_dataset      # ~60 days of 1-min candles
python3 -m groww_predictor.backtest           # prints top-1 / top-3 accuracy

# 2) every trading day at 09:25 IST: the prediction
python3 -m groww_predictor.predict_live       # prints the #1 pick + writes predictions.json
```

Automate with cron (already wired in `run_daily.sh`):

```
25 9 * * 1-5  cd /path/to/groww_predictor && ./run_daily.sh >> data/run.log 2>&1
```

Example `predictions.json`:

```json
{
  "as_of": "2026-05-31T09:25:11",
  "metric": "turnover",
  "scanned": 187,
  "most_traded_pick": {
    "symbol": "RELIANCE",
    "confidence_pct": 78.4,
    "open_turnover_cr": 312.5,
    "proj_full_turnover_cr": 5210.0,
    "open_ret_pct": 0.83,
    "groww_url": "https://groww.in/stocks/reliance"
  },
  "top": [ ... ranked list ... ]
}
```

---

## Deploy live on Render (free tier + UptimeRobot)

Render fits this app natively: an always-on Python process (no serverless
timeout that would kill the 40–60s opening scan) running the daily 09:25 IST
prediction **inside the same process** on a timer.

> **Why not Vercel:** its serverless functions time out (10s Hobby / 60s
> Pro) before the scan finishes, have an ephemeral filesystem so the
> prediction/dataset wouldn't persist, and make the `pandas`+`lightgbm`
> bundle awkward. Vercel is for frontends, not scheduled stateful Python.

The free plan sleeps after 15 minutes idle, so we keep it awake with an
UptimeRobot ping every 5 minutes. While it's awake, the in-process scheduler
fires the 09:25 prediction on its own. (UptimeRobot only *keeps it alive* —
it can't trigger at a precise time; that's the scheduler's job.)

### Steps

1. Push this folder to a **GitHub repo** (the repo root must contain
   `requirements.txt` and the `groww_predictor/` package). `.env` and
   `data/` are already git-ignored — never commit secrets.
2. On Render: **New → Blueprint**, point it at the repo. It reads
   `render.yaml` and creates a **free** web service (no disk).
3. In the service's **Environment** tab, set your secrets:
   `GROWW_API_KEY` and `GROWW_API_SECRET` (key+secret mode is best for a
   bot — the plain access token expires daily at 6 AM). Deploy.
4. First boot builds the history automatically (`RUN_DATASET_ON_BOOT=1`).
5. Copy your service URL, e.g. `https://groww-predictor.onrender.com`.

### Keep it awake with UptimeRobot

1. Create a free account at uptimerobot.com.
2. **Add New Monitor** → type **HTTP(s)**.
3. URL: `https://<your-app>.onrender.com/health`
4. **Monitoring interval: 5 minutes** (must be under Render's 15-min idle
   limit). Save.

That's it — the service stays up 24/7, and the scheduler fires the 09:25 IST
prediction every weekday. Open your Render URL to see the dashboard.

### Free-tier realities (be aware)

- **Ephemeral filesystem.** Free tier has no persistent disk, so on any
  restart/redeploy the dataset/model are wiped and rebuilt on next boot
  (a few minutes; that's why `HISTORY_DAYS` is set to 30 here). The live
  prediction still works during a rebuild via the analytic fallback.
- **750 free instance-hours/month per workspace.** 24/7 ≈ 730 hrs, under
  the cap — but only if this is your *only* always-on free service.
- **Render may restart free services at any time**, causing a ~1-min cold
  start. The scheduler's misfire grace (10 min) means a restart near 09:25
  still runs the prediction once it's back.

### Upgrading later (optional, rock-solid)

Switch `plan: free` → `plan: starter` (~$7/mo) in `render.yaml` and add a
disk so data survives restarts — then UptimeRobot isn't needed:

```yaml
    plan: starter
    disk:
      name: data
      mountPath: /var/data
      sizeGB: 1
    # and set DATA_DIR=/var/data, HISTORY_DAYS=60
```

---


## Dashboard

A full visual dashboard ships with the predictor. Start the service:

```bash
uvicorn groww_predictor.serve:app --port 8000
# Windows: python -m uvicorn groww_predictor.serve:app --port 8000
```

Open **http://localhost:8000** in a browser. You get:

- the predicted **most-traded stock** headline with confidence and projected
  full-day turnover,
- a **"Run live prediction"** button (calls the API for a fresh 09:25 scan),
- the full **ranked table** (opening vs projected full-day turnover, move %),
- a **backtest accuracy** panel comparing naive / analytic / LightGBM.

Until you've added credentials and run a prediction, the dashboard shows
helpful empty states telling you exactly what to run next.

---

## Plugging into your existing Node dashboard

Run the predictor as a service and point Node at it:

```bash
uvicorn groww_predictor.serve:app --port 8000
```

Then paste the routes from `node_integration.js` into your `server.js`
(they replace the old `/api/mtf/predictions` and `/api/mtf/analyze`
handlers). Your dashboard keeps working; the data behind it is now real.

---

## Configuration knobs (`.env`)

| Key | Meaning |
|---|---|
| `RANK_METRIC` | `turnover` (value, recommended) or `volume` (shares) |
| `UNIVERSE` | `FNO` (≈190 liquid names) / `NIFTY200` / `CUSTOM` |
| `MODEL_MODE` | `auto` / `analytic` / `lgbm` |
| `HISTORY_DAYS` | how much 1-min history to train/backtest on |
| `OPEN_FROM`,`OPEN_TO` | the opening window, default 09:15–09:25 |

---

## Honest limitations

- **It is a conditional prediction.** It does not know the future; it
  projects the day from the first 10 minutes. Because opening volume is
  highly predictive of full-day volume, the *top-3* hit rate is typically
  strong — but run `backtest.py` on **your** data to see the real number
  before trusting it. Don't treat the printed confidence as a probability.
- **`volume` from the live quote is cumulative for the day**, so call
  `predict_live` close to 09:25 (not 10:30) for the opening-window read to
  be correct.
- **Not investment advice.** This identifies likely *activity*, not a
  profitable trade. Position sizing and risk are on you.
- Data access depends on your Groww API subscription tier and rate limits
  (Live Data: 10/s, 300/min — the universe scan stays within this).
