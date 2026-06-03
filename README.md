# autoresearch-rocm

ROCm/Dokploy wrapper for Karpathy-style autoresearch on ETzMinisforumX1Pro's AMD Radeon 890M.

This repo packages the AMD-compatible autoresearch fork with:

- `rocm/pytorch:latest` base image
- smaller default hyperparameters for the 890M iGPU
- persistent `/data` cache/runs volume
- startup diagnostics served on port `8080`
- `scripts/run-train-once.sh` for a bounded experiment run

## Dokploy Compose GPU block

Use Dokploy **Docker Compose** service, not Application/Swarm:

```yaml
devices:
  - /dev/kfd
  - /dev/dri
group_add:
  - "44"
  - "993"
security_opt:
  - seccomp=unconfined
ipc: host
```

## Runtime env

Defaults are intentionally small:

- `AUTORESEARCH_MAX_SEQ_LEN=512`
- `AUTORESEARCH_TIME_BUDGET=60`
- `AUTORESEARCH_VOCAB_SIZE=4096`
- `AUTORESEARCH_DEPTH=2`
- `AUTORESEARCH_DEVICE_BATCH_SIZE=8`
- `AUTORESEARCH_TOTAL_BATCH_SIZE=16384`
- `AUTORESEARCH_NUM_SHARDS=2`

Set `AUTORESEARCH_TRAIN_ON_START=1` to run a training experiment on container startup.
