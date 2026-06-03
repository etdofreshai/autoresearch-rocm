#!/usr/bin/env bash
set -euo pipefail
mkdir -p /data/cache /data/runs /data/status
export HOME=/data
export AUTORESEARCH_CACHE_DIR=${AUTORESEARCH_CACHE_DIR:-/data/cache}
cd /app
{
  echo "AUTORESEARCH_ROCM_START $(date -Is)"
  echo "devices:"; ls -l /dev/kfd /dev/dri 2>&1 || true
  echo "rocminfo:"; rocminfo 2>&1 | grep -E 'Marketing Name|Name:|gfx|Unable|Error' | head -80 || true
  echo "torch:"; python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('hip', getattr(torch.version, 'hip', None))
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY
  echo "config: seq=${AUTORESEARCH_MAX_SEQ_LEN:-} depth=${AUTORESEARCH_DEPTH:-} dbs=${AUTORESEARCH_DEVICE_BATCH_SIZE:-} total=${AUTORESEARCH_TOTAL_BATCH_SIZE:-} time=${AUTORESEARCH_TIME_BUDGET:-} shards=${AUTORESEARCH_NUM_SHARDS:-}"
} | tee /data/status/startup.txt

cat > /data/status/index.html <<'HTML'
<!doctype html><meta charset="utf-8"><title>Autoresearch ROCm</title>
<h1>Autoresearch ROCm</h1>
<ul>
<li><a href="/startup.txt">startup diagnostics</a></li>
<li><a href="/prepare.log">prepare log</a></li>
<li><a href="/train.log">latest train log</a></li>
</ul>
HTML

if [[ "${AUTORESEARCH_PREPARE_ON_START:-1}" == "1" ]]; then
  (python3 prepare.py --num-shards "${AUTORESEARCH_NUM_SHARDS:-2}" --download-workers "${AUTORESEARCH_DOWNLOAD_WORKERS:-2}" 2>&1 | tee /data/status/prepare.log) || true
fi

if [[ "${AUTORESEARCH_TRAIN_ON_START:-0}" == "1" ]]; then
  /app/scripts/run-train-once.sh || true
fi

cd /data/status
exec python3 -m http.server 8080
