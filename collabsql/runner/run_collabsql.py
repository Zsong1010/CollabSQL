#!/usr/bin/env python3
"""
CollabSQL evaluation runner (method-only open release).

Example (cached EFPL trajectories — no GPU required):
  PYTHONPATH=. python collabsql/runner/run_collabsql.py \\
    --input data/spider_test/spider_test.parquet \\
    --fragments-dir data/spider_test/fragments/horizontal_n4 \\
    --db-root data/spider_test/databases \\
    --backend cached \\
    --cached-parquet data/spider_test/cache/step80_spider_test_@16_result.parquet \\
    --max-queries 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collabsql.aggregator.merger import MergeResult  # noqa: E402
from collabsql.config import CollabSqlConfig  # noqa: E402
from collabsql.constants import METHOD_NAME  # noqa: E402
from collabsql.evaluation.metrics import count_messages  # noqa: E402
from collabsql.pipeline import run_dqcp_pipeline  # noqa: E402
from collabsql.runner.distributed_run import _extract_gt, build_agents, build_backend  # noqa: E402
from collabsql.utils.eval_mode import evaluate_ex  # noqa: E402
from collabsql.utils.plan_loader import find_plan  # noqa: E402
from collabsql.utils.prompt_utils import extract_question  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    config = CollabSqlConfig()
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} runner")
    parser.add_argument("--input", required=True, help="Eval parquet (e.g. spider_test.parquet)")
    parser.add_argument("--fragments-dir", required=True, help="Fragment plan root for one topology")
    parser.add_argument("--db-root", default=str(config.default_db_root))
    parser.add_argument("--backend", choices=["cached", "api", "subprocess"], default="cached")
    parser.add_argument("--cached-parquet", default="", help="EFPL cached trajectories for --backend cached")
    parser.add_argument("--topology", default="horizontal_n4")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "run.json"))
    args = parser.parse_args(argv)

    config = CollabSqlConfig(default_db_root=Path(args.db_root))
    backend = build_backend(
        config=config,
        backend=args.backend,
        cached_parquet=Path(args.cached_parquet) if args.cached_parquet else None,
        topology=args.topology,
    )

    df = pd.read_parquet(args.input)
    if args.max_queries > 0:
        df = df.head(args.max_queries)

    fragments_dir = Path(args.fragments_dir)
    results: list[dict] = []
    correct = 0
    for qid, row in tqdm(df.iterrows(), total=len(df), desc=METHOD_NAME):
        db_id = str(row["db_id"])
        prompt = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
        question = extract_question(prompt)
        gt_sql = _extract_gt(row)
        plan = find_plan(fragments_dir, db_id)
        if plan is None:
            results.append(
                {
                    "question_id": int(qid),
                    "db_id": db_id,
                    "error": "missing_plan",
                    "execution_accuracy": 0,
                }
            )
            continue

        agents = build_agents(plan, backend)
        pipe = run_dqcp_pipeline(
            plan=plan,
            backend=backend,
            prompt=prompt,
            db_id=db_id,
            data_source=str(row.get("data", "spider")),
            question=question,
            question_index=int(qid),
            agents=agents,
        )
        eval_db = Path(args.db_root) / db_id / f"{db_id}.sqlite"
        ex, meta = (
            evaluate_ex(
                merge=pipe.merge,
                agent_results=[],
                ground_truth_sql=gt_sql,
                eval_db_path=str(eval_db),
            )
            if gt_sql
            else (0, {})
        )
        correct += int(ex)
        msgs = count_messages(pipe.dqcp_messages)
        results.append(
            {
                "question_id": int(qid),
                "db_id": db_id,
                "question": question,
                "ground_truth_sql": gt_sql,
                "predicted_sql": meta.get("predicted_sql", pipe.merge.predicted_sql),
                "execution_accuracy": int(ex),
                "message_count": msgs,
                "e2e_latency_ms": pipe.e2e_latency_ms,
                "coord_latency_ms": pipe.coord_latency_ms,
                "composition_latency_ms": pipe.composition_latency_ms,
                "decompose_mode": pipe.decompose_mode,
                "composition_mode": getattr(pipe.merge, "composition_mode", None),
            }
        )

    total = len(results)
    summary = {
        "method": METHOD_NAME,
        "topology": args.topology,
        "backend": args.backend,
        "correct": correct,
        "total": total,
        "execution_accuracy_pct": round(100.0 * correct / total, 2) if total else 0.0,
        "messages_per_query": round(
            sum(int(r.get("message_count") or 0) for r in results) / total, 2
        )
        if total
        else 0.0,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"EX: {summary['execution_accuracy_pct']}% ({correct}/{total})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
