# CollabSQL

Collaborative distributed Text-to-SQL with **DQCP** + **EFPL**.

```
Offline EFPL (EFPL/) → frozen π_θ*
Online DQCP (collabsql/) → Coordinator → Data Agents → Align / Synthesize
```

## Layout

```
EFPL/                 # offline EFPL / GRPO training
collabsql/            # online DQCP method code
data/
  bird_dev/           # BIRD-dev pack
  spider_test/        # Spider-test pack
  spider_dk/          # Spider-DK pack
results/              # Table-2 summaries
requirements.txt
```

## Results (Table 2)

| Method | Bird-dev | Spider-test | Spider-DK |
|--------|---------:|------------:|----------:|
| CollabSQL | 77.34 | 87.92 | 67.29 |
| CollabSQL (Hor.) | 73.00 | 82.74 | 63.62 |

Files: `results/collabsql_centralized.json`, `results/collabsql_horizontal_n4.json`.

## Bundled datasets

Each pack shares the same layout:

```
data/<name>/
  databases/                     # full SQLite DBs for EX
  fragments/horizontal_n4/       # 4-agent horizontal shards + plan.json
  <name>.parquet                 # eval split
  cache/step80_<name>_@16_result.parquet
```

| Pack | Queries | DBs | Size (approx.) |
|------|--------:|----:|---------------:|
| `bird_dev` | 1534 | 11 | ~2.5G |
| `spider_test` | 2147 | 40 | ~40M |
| `spider_dk` | 532 | 10 | ~125M |

Original dataset licenses apply (BIRD / Spider / Spider-DK).

## Quick start (cached, no GPU)

```bash
pip install -r requirements.txt

# Spider-test smoke
PYTHONPATH=. python collabsql/runner/run_collabsql.py \
  --input data/spider_test/spider_test.parquet \
  --fragments-dir data/spider_test/fragments/horizontal_n4 \
  --db-root data/spider_test/databases \
  --backend cached \
  --cached-parquet data/spider_test/cache/step80_spider_test_@16_result.parquet \
  --topology horizontal_n4 \
  --max-queries 5 \
  --output outputs/smoke_spider_test.json

# BIRD-dev
PYTHONPATH=. python collabsql/runner/run_collabsql.py \
  --input data/bird_dev/bird_dev.parquet \
  --fragments-dir data/bird_dev/fragments/horizontal_n4 \
  --db-root data/bird_dev/databases \
  --backend cached \
  --cached-parquet data/bird_dev/cache/step80_bird_dev_@16_result.parquet \
  --topology horizontal_n4 \
  --max-queries 5 \
  --output outputs/smoke_bird_dev.json

# Spider-DK
PYTHONPATH=. python collabsql/runner/run_collabsql.py \
  --input data/spider_dk/spider_dk.parquet \
  --fragments-dir data/spider_dk/fragments/horizontal_n4 \
  --db-root data/spider_dk/databases \
  --backend cached \
  --cached-parquet data/spider_dk/cache/step80_spider_dk_@16_result.parquet \
  --topology horizontal_n4 \
  --max-queries 5 \
  --output outputs/smoke_spider_dk.json
```

## Online inference (live LLM / API)

```bash
export COLLABSQL_MODEL=/path/to/efpl_consolidated_checkpoint
# optional: COLLABSQL_LLM_API_BASE / COLLABSQL_LLM_MODEL_NAME

PYTHONPATH=. python collabsql/runner/run_collabsql.py \
  --input data/spider_test/spider_test.parquet \
  --fragments-dir data/spider_test/fragments/horizontal_n4 \
  --db-root data/spider_test/databases \
  --backend api \
  --topology horizontal_n4 \
  --max-queries 5 \
  --output outputs/run.json
```

HTTP multi-process: `collabsql/distributed/` + `collabsql/runner/run_distributed_http.py`
(`--dataset bird_dev|spider_test|spider_dk`).

Rebuild BIRD fragments (optional):

```bash
PYTHONPATH=. python collabsql/scripts/build_distributed_bird.py \
  --db-root data/bird_dev/databases \
  --output-root data/bird_dev/fragments \
  --num-agents 4 --seed 42 --topology horizontal
```

## EFPL training

```bash
cd EFPL
# Install.md (CUDA 12.4, uv, Ray 2.43, Python 3.12)
# edit efpl_train.sh: BASE_MODEL, DB_PATH, CKPT_PATH, parquet paths
bash efpl_train.sh
```

Defaults: \(N_g{=}3\), \(T_{\max}{=}5\), \(\eta{=}1{\times}10^{-6}\), \(\varepsilon_c{=}0.2\), \(\beta{=}0\).

## License

MIT — see `LICENSE`. `EFPL/verl` retains upstream Apache-2.0. Datasets retain original licenses.

## Env

| Variable | Role |
|----------|------|
| `COLLABSQL_DB_ROOT` | SQLite root (default `data/spider_test/databases`) |
| `COLLABSQL_MODEL` | EFPL checkpoint |
| `COLLABSQL_LLM_API_BASE` | OpenAI-compatible API |
| `COLLABSQL_LLM_MODEL_NAME` | Served model id |
| `COLLABSQL_LLM_COORDINATOR` | `auto` / `1` / `0` |
| `COLLABSQL_LLM_COMPOSITION` | `auto` / `1` / `0` |
| `WANDB_API_KEY` | Optional EFPL logging |
