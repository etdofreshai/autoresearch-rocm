#!/usr/bin/env python3
"""Local controller for a Dokploy-hosted autoresearch keep/discard loop.

The loop asks Codex to make one bounded train.py-only experiment locally,
pushes the candidate branch, deploys the Dokploy ROCm runner against that
branch, parses the public train.log, keeps improvements and reverts failures.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
BRANCH = os.environ.get("AUTORESEARCH_BRANCH", "autoresearch/rocm-live")
COMPOSE_ID = os.environ.get("AUTORESEARCH_COMPOSE_ID", "x70FFfB7hr85GgsmD86a2")
MANAGER_SERVER_ID = os.environ.get("DOKPLOY_SERVER_ID", "YpYKF1N_VLu1fXwlK3RBW")
RUNNER_URL = os.environ.get("AUTORESEARCH_RUNNER_URL", "https://autoresearch-runner.etdofresh.com")
RESULTS = REPO / "results.tsv"
CONTROL_LOG = REPO / "control-loop.log"
TIME_BUDGET = os.environ.get("AUTORESEARCH_TIME_BUDGET", "300")
MAX_ITERS = int(os.environ.get("AUTORESEARCH_MAX_ITERS", "1000000"))

BASELINE_VAL = 1.821447


def log(msg: str):
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with CONTROL_LOG.open("a") as f:
        f.write(line + "\n")


def run(cmd, timeout=None, check=True, env=None):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    if env is None:
        env = os.environ.copy()
    env.setdefault('GIT_ASKPASS', '/tmp/git-askpass-hermes')
    p = subprocess.run(cmd, cwd=REPO, shell=isinstance(cmd, str), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout, env=env)
    if p.stdout:
        with CONTROL_LOG.open("a") as f:
            f.write(p.stdout[-12000:] + "\n")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {cmd}\n{p.stdout[-2000:]}")
    return p.stdout


def load_env():
    for p in [Path('/root/.hermes/.env'), Path('/root/workspace/.env')]:
        if p.exists():
            for line in p.read_text(errors='ignore').splitlines():
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def dokploy_headers():
    load_env()
    base = os.environ.get('DOKPLOY_URL', 'https://app.dokploy.com').rstrip('/')
    tok = os.environ.get('DOKPLOY_TOKEN') or os.environ.get('DOKPLOY_API_KEY')
    if not tok:
        raise RuntimeError('missing DOKPLOY_TOKEN/DOKPLOY_API_KEY')
    return base, {
        'x-api-key': tok,
        'accept': 'application/json',
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0',
        'origin': base,
        'referer': base + '/',
    }


def get_compose():
    base, h = dokploy_headers()
    inp = urllib.parse.quote(json.dumps({'json': {'composeId': COMPOSE_ID}}))
    r = requests.get(f'{base}/api/trpc/compose.one?input={inp}', headers=h, timeout=30)
    r.raise_for_status()
    return r.json()['result']['data']['json']


def update_compose_and_deploy(auto_run: bool):
    data = get_compose()
    cf = data['composeFile']
    # Make sure runtime clone uses the live experiment branch.
    cf = re.sub(r'git clone(?: --branch [^ ]+)? https://github.com/etdofreshai/autoresearch-rocm.git /tmp/app',
                f'git clone --branch {BRANCH} https://github.com/etdofreshai/autoresearch-rocm.git /tmp/app', cf)
    # Enable/disable prepare/train on startup and pin experiment config.
    replacements = {
        r'AUTORESEARCH_PREPARE_ON_START: "[01]"': f'AUTORESEARCH_PREPARE_ON_START: "{1 if auto_run else 0}"',
        r'AUTORESEARCH_TRAIN_ON_START: "[01]"': f'AUTORESEARCH_TRAIN_ON_START: "{1 if auto_run else 0}"',
        r'AUTORESEARCH_NUM_SHARDS: "\d+"': 'AUTORESEARCH_NUM_SHARDS: "10"',
        r'AUTORESEARCH_TIME_BUDGET: "\d+"': f'AUTORESEARCH_TIME_BUDGET: "{TIME_BUDGET}"',
        r'AUTORESEARCH_EVAL_TOKENS: "\d+"': 'AUTORESEARCH_EVAL_TOKENS: "524288"',
        r'AUTORESEARCH_DEPTH: "\d+"': 'AUTORESEARCH_DEPTH: "4"',
        r'AUTORESEARCH_DEVICE_BATCH_SIZE: "\d+"': 'AUTORESEARCH_DEVICE_BATCH_SIZE: "8"',
    }
    for pat, repl in replacements.items():
        cf = re.sub(pat, repl, cf)
    base, h = dokploy_headers()
    body = {'json': {'composeId': COMPOSE_ID, 'composeFile': cf, 'name': data['name'],
                     'description': f'Autoresearch autonomous loop branch {BRANCH}',
                     'composeType': 'docker-compose'}}
    r = requests.post(f'{base}/api/trpc/compose.update', headers=h, json=body, timeout=30)
    r.raise_for_status()
    r = requests.post(f'{base}/api/trpc/compose.deploy', headers=h, json={'json': {'composeId': COMPOSE_ID}}, timeout=30)
    r.raise_for_status()


def fetch(path, timeout=15):
    r = requests.get(RUNNER_URL.rstrip('/') + path, timeout=timeout, verify=False)
    if r.status_code >= 400:
        return None
    return r.text


def wait_for_result(deadline_seconds=900):
    start = time.time()
    last = ''
    while time.time() - start < deadline_seconds:
        txt = fetch('/train.log')
        if txt:
            last = txt
            if 'train exit=0' in txt or 'train exit=' in txt:
                return txt
        time.sleep(20)
    return last


def parse_result(txt):
    def grab(name):
        m = re.search(rf'^{name}:\s+([0-9.]+)', txt, re.M)
        return float(m.group(1)) if m else None
    return {
        'val_bpb': grab('val_bpb'),
        'peak_vram_mb': grab('peak_vram_mb'),
        'num_steps': grab('num_steps'),
        'total_tokens_M': grab('total_tokens_M'),
        'train_exit': (m.group(1) if (m := re.search(r'train exit=(\d+)', txt)) else None),
    }


def current_best():
    best = BASELINE_VAL
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines()[1:]:
            parts = line.split('\t')
            if len(parts) >= 4 and parts[3] == 'keep':
                try: best = min(best, float(parts[1]))
                except ValueError: pass
    return best


def append_result(commit, metrics, status, desc):
    if not RESULTS.exists():
        RESULTS.write_text('commit\tval_bpb\tmemory_gb\tstatus\tdescription\n')
        append_result('baseline', {'val_bpb': BASELINE_VAL, 'peak_vram_mb': 808.3}, 'keep', 'verified depth=4 ROCm baseline')
    val = metrics.get('val_bpb') or 0.0
    mem = (metrics.get('peak_vram_mb') or 0.0) / 1024.0
    with RESULTS.open('a') as f:
        f.write(f'{commit[:7]}\t{val:.6f}\t{mem:.1f}\t{status}\t{desc[:160].replace(chr(9), " ")}\n')


def setup_git():
    run('git fetch origin main', timeout=120)
    branches = run('git branch --list ' + BRANCH, timeout=30)
    if branches.strip():
        run('git checkout ' + BRANCH, timeout=60)
    else:
        run('git checkout -b ' + BRANCH + ' origin/main', timeout=60)
    run('git reset --hard HEAD', timeout=60)
    run('git clean -fd -- . ":!results.tsv" ":!control-loop.log"', timeout=60)
    run('git push -u origin ' + BRANCH, timeout=120)


def codex_experiment(best):
    prompt = f"""
You are running Karpathy autoresearch on an AMD ROCm Radeon 890M via Dokploy.
Read README.md, program.md, prepare.py, and train.py as needed.
Make exactly one experiment by editing ONLY train.py. Do not edit prepare.py, Dockerfile, compose, scripts, or results.tsv.
Current best val_bpb is {best:.6f}; lower is better. Baseline config is depth=4, seq=512, batch=8, total batch=16384, 5-minute run.
Prefer simple changes likely to improve validation bpb on small AMD iGPU: optimizer/lr schedule/model architecture/batch sizing inside train.py.
Do not run training locally; the GPU run happens remotely after you commit.
Commit your train.py change with a concise message describing the hypothesis.
""".strip()
    env = os.environ.copy()
    env['GIT_ASKPASS'] = '/tmp/git-askpass-hermes'
    run(['codex', 'exec', '--full-auto', prompt], timeout=900, env=env)
    changed = run('git diff --stat HEAD~1..HEAD -- train.py && git diff --name-only HEAD~1..HEAD', timeout=60)
    if 'train.py' not in changed:
        raise RuntimeError('Codex did not commit a train.py experiment')
    commit = run('git rev-parse --short=12 HEAD', timeout=30).strip()
    desc = run('git log -1 --pretty=%s', timeout=30).strip()
    run('GIT_ASKPASS=/tmp/git-askpass-hermes git push origin ' + BRANCH, timeout=120)
    return commit, desc


def main():
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    setup_git()
    if not RESULTS.exists():
        append_result('baseline', {'val_bpb': BASELINE_VAL, 'peak_vram_mb': 808.3}, 'keep', 'verified depth=4 ROCm baseline')
    log(f'LAUNCH branch={BRANCH} best={current_best():.6f}')
    update_compose_and_deploy(auto_run=False)
    for i in range(1, MAX_ITERS + 1):
        best_before = current_best()
        base_commit = run('git rev-parse --short=12 HEAD', timeout=30).strip()
        log(f'ITER {i} start base={base_commit} best={best_before:.6f}')
        try:
            commit, desc = codex_experiment(best_before)
            update_compose_and_deploy(auto_run=True)
            txt = wait_for_result(deadline_seconds=1200)
            metrics = parse_result(txt or '')
            if metrics.get('train_exit') != '0' or metrics.get('val_bpb') is None:
                status = 'crash'
                append_result(commit, metrics, status, desc)
                run('git reset --hard ' + base_commit, timeout=60)
                run('GIT_ASKPASS=/tmp/git-askpass-hermes git push --force-with-lease origin ' + BRANCH, timeout=120)
                log(f'ITER {i} crash/revert metrics={metrics}')
            elif metrics['val_bpb'] < best_before:
                status = 'keep'
                append_result(commit, metrics, status, desc)
                log(f'ITER {i} KEEP {metrics}')
            else:
                status = 'discard'
                append_result(commit, metrics, status, desc)
                run('git reset --hard ' + base_commit, timeout=60)
                run('GIT_ASKPASS=/tmp/git-askpass-hermes git push --force-with-lease origin ' + BRANCH, timeout=120)
                log(f'ITER {i} discard {metrics}')
            update_compose_and_deploy(auto_run=False)
        except Exception as e:
            log(f'ITER {i} controller_error={e!r}')
            try:
                run('git reset --hard ' + base_commit, timeout=60, check=False)
                run('GIT_ASKPASS=/tmp/git-askpass-hermes git push --force-with-lease origin ' + BRANCH, timeout=120, check=False)
                update_compose_and_deploy(auto_run=False)
            except Exception as ee:
                log(f'recovery_error={ee!r}')
            time.sleep(60)

if __name__ == '__main__':
    main()
