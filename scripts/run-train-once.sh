#!/usr/bin/env bash
set -euo pipefail
mkdir -p /data/runs /data/status
cd /app
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
log="/data/runs/train-${run_id}.log"
ln -sfn "$log" /data/status/train.log
python3 train.py 2>&1 | tee "$log"
