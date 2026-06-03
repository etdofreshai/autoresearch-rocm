FROM rocm/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    HOME=/data \
    AUTORESEARCH_CACHE_DIR=/data/cache \
    AUTORESEARCH_MAX_SEQ_LEN=512 \
    AUTORESEARCH_TIME_BUDGET=60 \
    AUTORESEARCH_EVAL_TOKENS=262144 \\
    AUTORESEARCH_VOCAB_SIZE=4096 \
    AUTORESEARCH_WINDOW_PATTERN=L \
    AUTORESEARCH_TOTAL_BATCH_SIZE=16384 \
    AUTORESEARCH_DEPTH=2 \
    AUTORESEARCH_DEVICE_BATCH_SIZE=8 \
    AUTORESEARCH_NUM_SHARDS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml prepare.py train.py program.md README.md progress.png analysis.ipynb .python-version /app/
COPY scripts/ /app/scripts/

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install uv \
    && uv pip install --system -r pyproject.toml

VOLUME /data
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/entrypoint.sh"]
