#!/usr/bin/env bash
# Autonomous momentum paper-bot runner. Designed for a daily cron after the US
# market close. Idempotent: safe to run more than once per day (it won't
# double-trade or duplicate history). Logs each tick to paper_bot.log.
#
# Example crontab entry (weekdays ~17:10 ET; adjust to your timezone):
#   10 17 * * 1-5  /home/mrblackreal/Projects/market-net/run_paper_bot.sh
set -euo pipefail
cd "$(dirname "$0")"
{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
  ./venv/bin/python -u main.py --mode paper --capital 10000
} >> paper_bot.log 2>&1
