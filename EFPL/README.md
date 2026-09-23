# EFPL — Execution-Feedback Policy Learning

Offline training stack for CollabSQL’s reusable Text-to-SQL policy \(\pi_{\theta^*}\) (paper Step 0).

Built on verl with multi-turn SQL execution feedback and GRPO.

## Quick start

```bash
cd EFPL
# See Install.md for CUDA / uv / Ray setup

# Configure paths inside efpl_train.sh:
#   BASE_MODEL, DB_PATH, CKPT_PATH, DATA_DIR, train parquet paths

bash efpl_train.sh
```

## Hyperparameters (default)

| Symbol | Value | Script variable |
|--------|------:|-----------------|
| \(N_g\) | 3 | `N_AGENT` |
| \(T_{\max}\) | 5 | `N_TURNS` |
| \(\eta\) | \(1\times10^{-6}\) | `LR` |
| \(\varepsilon_c\) | 0.2 | `CLIP_LOW` / `CLIP_HIGH` |
| \(\beta\) | 0 | `USE_KL_LOSS=False` |

Entry: `efpl_train.sh` → `python -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` and `task_type=sql`.

## Layout

| Path | Role |
|------|------|
| `efpl_train.sh` | Training launch script |
| `Install.md` | Environment install |
| `verl/` | Training framework + `llm_sql_agent` SQL rollout |

After training, consolidate the checkpoint and point CollabSQL online inference to it via `COLLABSQL_MODEL` / `--backend api`.
