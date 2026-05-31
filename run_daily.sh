#!/usr/bin/env bash
# Daily driver. Cron this at 09:25 IST on weekdays:
#   25 9 * * 1-5  cd /path/to/groww_predictor && ./run_daily.sh >> data/run.log 2>&1
set -e
cd "$(dirname "$0")"

# Rebuild the training dataset once a week (Mondays) so context stays fresh.
if [ "$(date +%u)" = "1" ]; then
  echo "[run_daily] Monday — rebuilding dataset + retraining"
  python3 -m groww_predictor.build_dataset
  python3 -m groww_predictor.backtest || true
fi

# Live prediction for today.
python3 -m groww_predictor.predict_live
