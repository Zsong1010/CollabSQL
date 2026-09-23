"""
Distributed-BIRD: BIRD partitioned across K nodes for CollabSQL evaluation.

Topology A (horizontal): row-wise split, full table schema per node.
Topology B (vertical):   column-wise split, primary keys replicated.
Topology C (random):     mixed row/column/table assignment.
"""
from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SplitType = Literal["horizontal", "vertical", "random"]
TOPOLOGIES: tuple[SplitType, ...] = ("horizontal", "vertical", "random")


@dataclass
class AgentFragment:
    agent_id: str
    db_path: Path
    tables: list[str]
    description: str
    split_type: SplitType
    metadata: dict = field(default_factory=dict)


@dataclass
class FragmentationPlan:
    db_id: str
    source_db: Path
    split_type: SplitType
    num_agents: int
    agents: list[AgentFragment]
    seed: int = 42


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _copy_schema_subset(
    source: Path,
    target: Path,
    table_rows: dict[str, list[int]] | None = None,
    table_columns: dict[str, list[str]] | None = None,
) -> None:
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(str(source))
    src.text_factory = lambda b: b.decode("utf-8", errors="replace")
    dst = sqlite3.connect(str(target))
    dst.text_factory = str
    dst.execute("PRAGMA foreign_keys=OFF")

    tables = [
        r[0]
        for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    for table in tables:
        col_info = src.execute(f'PRAGMA table_info("{table}")').fetchall()
        all_cols = [c[1] for c in col_info]

        if table_columns and table in table_columns:
            cols = table_columns[table]
        else:
            cols = all_cols

        pk_cols = [c[1] for c in col_info if c[5] and c[1] in cols]
        pk_cols.sort(key=lambda name: next(c[5] for c in col_info if c[1] == name))

        col_defs = []
        for c in col_info:
            if c[1] not in cols:
                continue
            if len(pk_cols) == 1 and c[1] == pk_cols[0]:
                col_defs.append(f'"{c[1]}" {c[2] or "TEXT"} PRIMARY KEY')
            else:
                col_defs.append(f'"{c[1]}" {c[2] or "TEXT"}')

        if len(pk_cols) > 1:
            pk_list = ", ".join(f'"{n}"' for n in pk_cols)
            col_defs.append(f"PRIMARY KEY ({pk_list})")

        dst.execute(f'CREATE TABLE "{table}" ({", ".join(col_defs)})')

        col_list = ", ".join(f'"{c}"' for c in cols)
        rows = src.execute(f'SELECT {col_list} FROM "{table}"').fetchall()

        if table_rows and table in table_rows:
            indices = set(table_rows[table])
            rows = [rows[i] for i in range(len(rows)) if i in indices]

        placeholders = ", ".join(["?"] * len(cols))
        dst.executemany(f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})', rows)

    dst.commit()
    src.close()
    dst.close()


def horizontal_split(
    source_db: Path,
    output_dir: Path,
    db_id: str,
    num_agents: int,
    seed: int = 42,
) -> FragmentationPlan:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(source_db))
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    row_assignments: dict[str, dict[str, list[int]]] = {f"A{i+1}": {} for i in range(num_agents)}

    for table in tables:
        n_rows = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        agent_row_indices: dict[str, list[int]] = {f"A{i+1}": [] for i in range(num_agents)}
        for idx in range(n_rows):
            agent = f"A{rng.randint(1, num_agents)}"
            agent_row_indices[agent].append(idx)
        for agent_id, indices in agent_row_indices.items():
            row_assignments[agent_id][table] = indices

    conn.close()

    agents: list[AgentFragment] = []
    for i in range(num_agents):
        agent_id = f"A{i+1}"
        agent_db = output_dir / f"{db_id}_{agent_id}.sqlite"
        _copy_schema_subset(source_db, agent_db, table_rows=row_assignments[agent_id])
        agents.append(
            AgentFragment(
                agent_id=agent_id,
                db_path=agent_db,
                tables=tables,
                description=f"Agent {agent_id} holds a horizontal partition of {db_id} database rows.",
                split_type="horizontal",
                metadata={"row_assignments": row_assignments[agent_id]},
            )
        )

    return FragmentationPlan(
        db_id=db_id,
        source_db=source_db,
        split_type="horizontal",
        num_agents=num_agents,
        agents=agents,
        seed=seed,
    )


def vertical_split(
    source_db: Path,
    output_dir: Path,
    db_id: str,
    num_agents: int,
    seed: int = 42,
    *,
    column_cooccurrence: Counter | None = None,
) -> FragmentationPlan:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(source_db))
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    agent_columns: dict[str, dict[str, list[str]]] = {f"A{i+1}": {} for i in range(num_agents)}
    agent_load = [0] * num_agents
    pair_counter = column_cooccurrence or Counter()

    for table in tables:
        col_info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        all_cols = [c[1] for c in col_info]
        pk_cols = [c[1] for c in col_info if c[5]]
        fk_cols = [fk[3] for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()]
        key_cols = list(dict.fromkeys(pk_cols + fk_cols))
        non_key_cols = [c for c in all_cols if c not in key_cols]

        rng.shuffle(non_key_cols)
        table_lower = table.lower()
        assigned_on_agent: dict[int, list[str]] = {i: [] for i in range(num_agents)}

        for col in non_key_cols:
            best_agent = 0
            best_score = -1.0
            for agent_idx in range(num_agents):
                score = agent_load[agent_idx] * 0.05
                col_key = (table_lower, col)
                for other_col in assigned_on_agent[agent_idx]:
                    other_key = (table_lower, other_col)
                    score += pair_counter.get(tuple(sorted((col_key, other_key))), 0)
                if score > best_score:
                    best_score = score
                    best_agent = agent_idx
            assigned_on_agent[best_agent].append(col)
            agent_load[best_agent] += 1

        for i in range(num_agents):
            cols = list(dict.fromkeys(key_cols + assigned_on_agent[i]))
            if cols:
                agent_columns[f"A{i+1}"][table] = cols

    conn.close()

    agents: list[AgentFragment] = []
    for i in range(num_agents):
        agent_id = f"A{i+1}"
        agent_db = output_dir / f"{db_id}_{agent_id}.sqlite"
        cols = agent_columns[agent_id]
        if not cols:
            continue
        _copy_schema_subset(source_db, agent_db, table_columns=cols)
        agents.append(
            AgentFragment(
                agent_id=agent_id,
                db_path=agent_db,
                tables=list(cols.keys()),
                description=f"Agent {agent_id} holds vertical column partitions of {db_id}.",
                split_type="vertical",
                metadata={"columns": cols},
            )
        )

    return FragmentationPlan(
        db_id=db_id,
        source_db=source_db,
        split_type="vertical",
        num_agents=num_agents,
        agents=agents,
        seed=seed,
    )


def random_split(
    source_db: Path,
    output_dir: Path,
    db_id: str,
    num_agents: int,
    seed: int = 42,
) -> FragmentationPlan:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(source_db))
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    agent_data: dict[str, dict] = {f"A{i+1}": {"rows": {}, "cols": {}} for i in range(num_agents)}

    for table in tables:
        mode = rng.choice(["horizontal", "vertical"])
        col_info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        all_cols = [c[1] for c in col_info]
        n_rows = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

        if mode == "horizontal":
            for idx in range(n_rows):
                agent = f"A{rng.randint(1, num_agents)}"
                agent_data[agent]["rows"].setdefault(table, []).append(idx)
        else:
            pk_cols = [c[1] for c in col_info if c[5]]
            fk_cols = [fk[3] for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()]
            key_cols = list(dict.fromkeys(pk_cols + fk_cols))
            non_key = [c for c in all_cols if c not in key_cols]
            rng.shuffle(non_key)
            for i, col in enumerate(non_key):
                agent = f"A{(i % num_agents) + 1}"
                agent_data[agent]["cols"].setdefault(table, list(key_cols))
                if col not in agent_data[agent]["cols"][table]:
                    agent_data[agent]["cols"][table].append(col)

    conn.close()

    agents: list[AgentFragment] = []
    for i in range(num_agents):
        agent_id = f"A{i+1}"
        agent_db = output_dir / f"{db_id}_{agent_id}.sqlite"
        rows = agent_data[agent_id]["rows"] or None
        cols = agent_data[agent_id]["cols"] or None
        # Always materialize A1..An so dynamic-join prefix subsets are well-defined.
        if not rows and not cols:
            rows = {t: [] for t in tables}
        _copy_schema_subset(source_db, agent_db, table_rows=rows, table_columns=cols)
        held_tables = list(dict.fromkeys(list((cols or {}).keys()) + list((rows or {}).keys())))
        agents.append(
            AgentFragment(
                agent_id=agent_id,
                db_path=agent_db,
                tables=held_tables,
                description=f"Agent {agent_id} holds a random mixed partition of {db_id}.",
                split_type="random",
                metadata={
                    "row_tables": list((rows or {}).keys()),
                    "col_tables": list((cols or {}).keys()),
                    "columns": cols or {},
                },
            )
        )

    return FragmentationPlan(
        db_id=db_id,
        source_db=source_db,
        split_type="random",
        num_agents=num_agents,
        agents=agents,
        seed=seed,
    )


def fragment_database(
    source_db: Path,
    output_dir: Path,
    db_id: str,
    split_type: SplitType,
    num_agents: int,
    seed: int = 42,
    *,
    column_cooccurrence: Counter | None = None,
) -> FragmentationPlan:
    if split_type == "horizontal":
        return horizontal_split(source_db, output_dir, db_id, num_agents, seed)
    if split_type == "vertical":
        return vertical_split(
            source_db,
            output_dir,
            db_id,
            num_agents,
            seed,
            column_cooccurrence=column_cooccurrence,
        )
    return random_split(source_db, output_dir, db_id, num_agents, seed)


def save_plan(plan: FragmentationPlan, path: Path, *, repo_root: Path | None = None) -> None:
    root = repo_root or _repo_root()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(p.resolve())

    data = {
        "db_id": plan.db_id,
        "source_db": str(plan.source_db.resolve()),
        "split_type": plan.split_type,
        "num_agents": plan.num_agents,
        "seed": plan.seed,
        "agents": [
            {
                "agent_id": a.agent_id,
                "db_path": _rel(a.db_path),
                "tables": a.tables,
                "description": a.description,
                "split_type": a.split_type,
                "metadata": a.metadata,
            }
            for a in plan.agents
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_bird_databases(db_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in db_root.iterdir()
        if p.is_dir() and (p / f"{p.name}.sqlite").exists()
    )


def build_distributed_bird(
    *,
    db_root: Path,
    output_root: Path,
    num_agents: int = 4,
    seed: int = 42,
    topologies: tuple[SplitType, ...] = TOPOLOGIES,
    db_ids: list[str] | None = None,
    skip_existing: bool = True,
    query_parquet: Path | None = None,
    query_list: Path | None = None,
) -> dict[str, list[str]]:
    """Build Distributed-BIRD for all topologies. Returns {topology: [db_id, ...]}."""
    random.seed(seed)
    built: dict[str, list[str]] = {t: [] for t in topologies}
    targets = db_ids or list_bird_databases(db_root)

    cooccurrence_by_db: dict[str, Counter] = {}
    if query_parquet and query_parquet.exists():
        try:
            from collabsql.data.query_column_stats import build_db_cooccurrence

            cooccurrence_by_db = build_db_cooccurrence(
                parquet_path=query_parquet,
                query_list_path=query_list,
            )
        except ImportError:
            cooccurrence_by_db = {}

    for split_type in topologies:
        topo_dir = output_root / f"{split_type}_n{num_agents}"
        for db_id in targets:
            source_db = db_root / db_id / f"{db_id}.sqlite"
            if not source_db.exists():
                continue
            out_dir = topo_dir / db_id
            plan_path = out_dir / "plan.json"
            if skip_existing and plan_path.exists():
                all_sqlite = all(
                    (out_dir / f"{db_id}_A{i}.sqlite").exists() for i in range(1, num_agents + 1)
                )
                if all_sqlite:
                    built[split_type].append(db_id)
                    continue

            plan = fragment_database(
                source_db=source_db,
                output_dir=out_dir,
                db_id=db_id,
                split_type=split_type,
                num_agents=num_agents,
                seed=seed,
                column_cooccurrence=cooccurrence_by_db.get(db_id) if split_type == "vertical" else None,
            )
            save_plan(plan, plan_path, repo_root=_repo_root())
            built[split_type].append(db_id)

    manifest = {
        "dataset": "Distributed-BIRD",
        "num_agents": num_agents,
        "seed": seed,
        "topologies": list(topologies),
        "built": built,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return built
