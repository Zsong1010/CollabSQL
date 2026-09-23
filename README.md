# CollabSQL

Collaborative distributed Text-to-SQL with **DQCP** + **EFPL**.
Title: A Collaborative Text-to-SQL Framework with Reusable Policy Learning for Autonomous Distributed Databases
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



## Bundled datasets

Each pack shares the same layout:

```
data/<name>/
  databases/                     # full SQLite DBs for EX
  fragments/horizontal_n4/       # 4-agent horizontal shards + plan.json
  <name>.parquet                 # eval split
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

Rebuild BIRD fragments :

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

## Experimental Environment

All experiments are conducted on a server equipped with:
- GPU: NVIDIA H200, 141 GB HBM3e
- Python: 3.12
- CUDA: 12.4
- Ray: 2.43

## License

MIT — see `LICENSE`. `EFPL/verl` retains upstream Apache-2.0. Datasets retain original licenses.


