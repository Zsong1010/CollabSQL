## Pre-requisites

- [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- `build-essential` (for `torch-memory-saver`)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation)
- `python` 3.12
- `ray` 2.43.0

```bash
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
```

## Installation

```bash
cd CollabSQL_Open/EFPL
uv lock                 # first time / after dependency edits
uv sync --extra sql
```

Dry run:

```bash
uv run --isolated --extra sql python -c 'import ray; ray.init(); print("Success!")'
```

## Launch EFPL training

Edit `efpl_train.sh` (`BASE_MODEL`, `DB_PATH`, `CKPT_PATH`, parquet paths), then:

```bash
bash efpl_train.sh
```
