#!/usr/bin/env python3
"""Build Distributed-BIRD (4-node horizontal / vertical / random topologies)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collabsql.data.distributed_bird import TOPOLOGIES, build_distributed_bird  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Distributed-BIRD dataset")
    parser.add_argument("--db-root", default=str(ROOT / "data" / "bird_dev" / "databases"))
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "bird_dev" / "fragments"),
    )
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topology",
        choices=[*TOPOLOGIES, "all"],
        default="all",
        help="Which topology to build (default: all three)",
    )
    parser.add_argument("--db-id", action="append", dest="db_ids", help="Limit to specific db_id(s)")
    parser.add_argument(
        "--query-parquet",
        default=str(ROOT / "data" / "bird_dev" / "bird_dev.parquet"),
        help="BIRD parquet for query-aware vertical column affinity",
    )
    parser.add_argument(
        "--query-list",
        default="",
        help="Optional query list JSON for affinity stats",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if plans exist")
    args = parser.parse_args()

    topologies = TOPOLOGIES if args.topology == "all" else (args.topology,)
    built = build_distributed_bird(
        db_root=Path(args.db_root),
        output_root=Path(args.output_root),
        num_agents=args.num_agents,
        seed=args.seed,
        topologies=topologies,
        db_ids=args.db_ids,
        skip_existing=not args.force,
        query_parquet=Path(args.query_parquet) if args.query_parquet else None,
        query_list=Path(args.query_list) if args.query_list else None,
    )

    print(f"Distributed-BIRD built under {args.output_root}")
    for topo, dbs in built.items():
        print(f"  {topo}_n{args.num_agents}: {len(dbs)} databases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
