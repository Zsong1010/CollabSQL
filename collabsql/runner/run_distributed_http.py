#!/usr/bin/env python3
"""
Host-side evaluator for the containerized CollabSQL deployment.

Calls Coordinator over HTTP (which calls Data Agents over HTTP), then evaluates
EX on the integrated source DB locally (evaluator machine only — not Coordinator).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

from collabsql.aggregator.merger import MergeResult  # noqa: E402
from collabsql.evaluation.metrics import execution_accuracy  # noqa: E402
from collabsql.runner.distributed_run import _extract_gt  # noqa: E402
from collabsql.utils.eval_mode import evaluate_ex  # noqa: E402
from collabsql.utils.prompt_utils import extract_question  # noqa: E402


DATASET_PRESETS = {
    "bird_dev": {
        "input": ROOT / "data" / "bird_dev" / "bird_dev.parquet",
        "db_root": ROOT / "data" / "bird_dev" / "databases",
        "data_source": "bird",
        "fragments": ROOT / "data" / "bird_dev" / "fragments" / "horizontal_n4",
    },
    "spider_test": {
        "input": ROOT / "data" / "spider_test" / "spider_test.parquet",
        "db_root": ROOT / "data" / "spider_test" / "databases",
        "data_source": "spider",
        "fragments": ROOT / "data" / "spider_test" / "fragments" / "horizontal_n4",
    },
    "spider_dk": {
        "input": ROOT / "data" / "spider_dk" / "spider_dk.parquet",
        "db_root": ROOT / "data" / "spider_dk" / "databases",
        "data_source": "spider_dk",
        "fragments": ROOT / "data" / "spider_dk" / "fragments" / "horizontal_n4",
    },
}


def _http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        raw = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        data = raw
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_source_db(db_root: Path, db_id: str) -> Path:
    candidates = [
        db_root / db_id / f"{db_id}.sqlite",
        db_root / db_id / f"{db_id}.sqlite3",
        db_root / db_id / "sqlite_db",
        db_root / f"{db_id}.sqlite",
    ]
    for c in candidates:
        if c.exists():
            return c
    return db_root / db_id / f"{db_id}.sqlite"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CollabSQL Docker distributed deployment")
    p.add_argument("--dataset", choices=sorted(DATASET_PRESETS), required=True)
    p.add_argument("--coordinator-url", default="http://127.0.0.1:8200")
    p.add_argument("--max-queries", type=int, default=0, help="0 = all")
    p.add_argument("--query-list", default="", help="Optional JSON list of {question_id}")
    p.add_argument("--output", default="", help="Output JSON path")
    p.add_argument("--db-root", default="", help="Override source DB root")
    p.add_argument("--input", default="", help="Override input parquet")
    args = p.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    input_path = Path(args.input) if args.input else preset["input"]
    db_root = Path(args.db_root) if args.db_root else preset["db_root"]
    out_path = (
        Path(args.output)
        if args.output
        else ROOT
        / "data"
        / "collabsql"
        / "results"
        / "distributed_http"
        / f"collabsql_{args.dataset}_random_n4.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wait for coordinator
    health_url = args.coordinator_url.rstrip("/") + "/health"
    for i in range(60):
        try:
            h = _http_json(health_url)
            if h.get("ok"):
                break
        except Exception:
            time.sleep(2)
    else:
        raise SystemExit(f"Coordinator not healthy at {health_url}")

    df = pd.read_parquet(input_path)
    indices = list(range(len(df)))
    if args.query_list:
        qids = [int(x["question_id"]) for x in json.loads(Path(args.query_list).read_text())]
        indices = qids
    if args.max_queries and args.max_queries > 0:
        indices = indices[: args.max_queries]

    results: list[dict[str, Any]] = []
    for qid in tqdm(indices, desc=f"distributed-http/{args.dataset}"):
        row = df.iloc[int(qid)]
        db_id = str(row["db_id"])
        prompt = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
        question = extract_question(prompt)
        gt_sql = _extract_gt(row)
        source_db = _resolve_source_db(db_root, db_id)

        t_client = time.perf_counter()
        try:
            resp = _http_json(
                args.coordinator_url.rstrip("/") + "/query",
                {
                    "db_id": db_id,
                    "question": question,
                    "prompt": prompt,
                    "question_index": int(qid),
                    "data_source": preset["data_source"],
                },
            )
            client_err = ""
        except Exception as e:
            resp = {
                "success": False,
                "predicted_sql": "",
                "columns": [],
                "rows": [],
                "metrics": {},
                "dqcp_messages": [],
                "error": str(e),
            }
            client_err = str(e)
        client_ms = (time.perf_counter() - t_client) * 1000.0

        metrics = dict(resp.get("metrics") or {})
        merge = MergeResult(
            success=bool(resp.get("success")),
            columns=list(resp.get("columns") or []),
            rows=list(resp.get("rows") or []),
            predicted_sql=str(resp.get("predicted_sql") or ""),
            error=str(resp.get("error") or client_err),
            composition_mode=str(resp.get("composition_mode") or ""),
        )

        # EX on host with source DB (Coordinator never sees this DB)
        t_sql = time.perf_counter()
        if source_db.exists() and gt_sql:
            ex, ex_meta = evaluate_ex(
                merge=merge,
                agent_results=[],
                ground_truth_sql=gt_sql,
                eval_db_path=str(source_db),
                eval_mode="source_db",
            )
        else:
            ex, ex_meta = 0, {"eval_mode": "source_db", "error": f"missing source db or gt: {source_db}"}
            # still time a no-op path
        sql_ms = float(ex_meta.get("sql_latency_ms") or (time.perf_counter() - t_sql) * 1000.0)

        results.append(
            {
                "question_id": int(qid),
                "db_id": db_id,
                "question": question,
                "ground_truth_sql": gt_sql,
                "predicted_sql": merge.predicted_sql,
                "execution_accuracy": int(ex),
                "success": bool(resp.get("success")),
                "composition_mode": resp.get("composition_mode"),
                "decompose_mode": resp.get("decompose_mode"),
                "participating_agents": resp.get("participating_agents"),
                "e2e_latency_ms": metrics.get("e2e_latency_ms"),
                "coord_latency_ms": metrics.get("coord_latency_ms"),
                "composition_latency_ms": metrics.get("composition_latency_ms"),
                "sql_latency_ms": round(sql_ms, 3),
                "client_wall_ms": round(client_ms, 3),
                "messages_per_query": metrics.get("messages_per_query"),
                "communication_bytes": metrics.get("communication_bytes"),
                "http_request_bytes": metrics.get("http_request_bytes"),
                "http_response_bytes": metrics.get("http_response_bytes"),
                "http_requests": metrics.get("http_requests"),
                "token_usage": metrics.get("token_usage"),
                "dqcp_messages": resp.get("dqcp_messages") or [],
                "error": resp.get("error") or client_err,
            }
        )

    ex_stats = execution_accuracy(results)
    # reuse message counter from stored per-query field
    msg_avg = round(
        sum(float(r.get("messages_per_query") or 0) for r in results) / max(len(results), 1), 2
    )
    bytes_avg = round(
        sum(float(r.get("communication_bytes") or 0) for r in results) / max(len(results), 1), 2
    )

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "method": "CollabSQL (Docker HTTP distributed)",
        "topology": "random_n4",
        "dataset": args.dataset,
        "coordinator_url": args.coordinator_url,
        "correct": ex_stats["correct"],
        "total": ex_stats["total"],
        "execution_accuracy_pct": ex_stats["execution_accuracy_pct"],
        "messages_per_query": msg_avg,
        "communication_bytes_per_query": bytes_avg,
        "latency": {
            "e2e_ms_mean": _mean("e2e_latency_ms"),
            "coord_ms_mean": _mean("coord_latency_ms"),
            "composition_ms_mean": _mean("composition_latency_ms"),
            "sql_ms_mean": _mean("sql_latency_ms"),
            "client_wall_ms_mean": _mean("client_wall_ms"),
        },
        "http": {
            "requests_mean": _mean("http_requests"),
            "request_bytes_mean": _mean("http_request_bytes"),
            "response_bytes_mean": _mean("http_response_bytes"),
            "total_bytes_mean": bytes_avg,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    summary_path = out_path.with_name("summary.json")
    slim = {k: v for k, v in summary.items() if k != "results"}
    summary_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
    print(json.dumps(slim, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
