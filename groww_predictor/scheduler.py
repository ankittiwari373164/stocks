"""
In-process scheduler (used when deployed on an always-on host like Render).

Enable by setting ENABLE_SCHEDULER=1. Jobs run in IST:
  * 09:25 Mon-Fri  -> live prediction          (predict_live)
  * 18:00 Sunday   -> rebuild dataset + backtest (build_dataset, backtest)

Imports of apscheduler are lazy so local dev works without it installed.
A single uvicorn worker is assumed (the default); do not run --workers >1
or the jobs would be scheduled multiple times.
"""
from __future__ import annotations
import os
import threading

from .config import CFG


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _job_predict():
    try:
        from .predict_live import predict_live
        predict_live()
    except Exception as e:  # noqa
        print(f"[sched] predict_live failed: {e}", flush=True)


def _job_dataset():
    try:
        from .build_dataset import build_dataset
        from .backtest import backtest
        build_dataset()
        backtest()
    except Exception as e:  # noqa
        print(f"[sched] dataset rebuild failed: {e}", flush=True)


def _job_score():
    try:
        from .tracker import score_pending
        print("[sched] scoring past predictions:", score_pending())
    except Exception as e:  # noqa
        print(f"[sched] scoring failed: {e}", flush=True)


def start_scheduler():
    """Start background jobs if ENABLE_SCHEDULER is set. Returns the scheduler or None."""
    if not _truthy(os.environ.get("ENABLE_SCHEDULER", "0")):
        print("[sched] disabled (set ENABLE_SCHEDULER=1 to enable)", flush=True)
        return None

    # Build the dataset once on first boot if missing (and asked to).
    if _truthy(os.environ.get("RUN_DATASET_ON_BOOT", "0")) and not CFG.dataset_path.exists():
        print("[sched] no dataset found — building on boot (background)…", flush=True)
        threading.Thread(target=_job_dataset, daemon=True).start()

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except Exception as e:  # noqa
        print(f"[sched] apscheduler not installed ({e}) — scheduler off", flush=True)
        return None

    sched = BackgroundScheduler(timezone="Asia/Kolkata")
    sched.add_job(_job_predict, CronTrigger(day_of_week="mon-fri", hour=9, minute=25),
                  id="predict", misfire_grace_time=600, coalesce=True)
    sched.add_job(_job_score, CronTrigger(day_of_week="mon-fri", hour=16, minute=0),
                  id="score", misfire_grace_time=7200, coalesce=True)
    sched.add_job(_job_dataset, CronTrigger(day_of_week="sun", hour=18, minute=0),
                  id="dataset", misfire_grace_time=3600, coalesce=True)
    sched.start()
    print("[sched] started — predict 09:25, score 16:00 (Mon-Fri IST), dataset rebuild Sun 18:00 IST", flush=True)
    return sched