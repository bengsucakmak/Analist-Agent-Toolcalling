# tools/schema_intel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sqlite3, json
from langchain_core.tools import tool

_SCHEMA_INTEL = None  # global cache

@dataclass
class TableInfo:
    name: str
    columns: List[Tuple[str, str, int, Optional[str]]]  # (name, type, notnull, default)
    fks: List[Tuple[str, str, str]]  # (from_col, ref_table, ref_col)
    rowcount: Optional[int]
    indexes: List[str]

@dataclass
class SchemaIntel:
    db_path: str
    tables: Dict[str, TableInfo]          # table -> info
    fk_graph: Dict[str, List[str]]        # table -> neighbor tables (FK/PK edges)

def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_schema_intel(db_path: str, with_counts: bool = True) -> None:
    """Build schema intelligence: columns, FKs, rowcounts, FK graph."""
    global _SCHEMA_INTEL
    conn = _connect(db_path)
    try:
        # list tables
        tables = []
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"):
            tables.append(r["name"])

        info_map: Dict[str, TableInfo] = {}
        for t in tables:
            cols = []
            for r in conn.execute(f"PRAGMA table_info({t});"):
                cols.append((r["name"], r["type"], r["notnull"], r["dflt_value"]))

            fks = []
            for r in conn.execute(f"PRAGMA foreign_key_list({t});"):
                # (from_col, ref_table, ref_col)
                fks.append((r["from"], r["table"], r["to"]))

            idxs = []
            for r in conn.execute(f"PRAGMA index_list({t});"):
                idxs.append(r["name"])

            rc = None
            if with_counts:
                try:
                    rc = conn.execute(f"SELECT COUNT(*) AS c FROM {t};").fetchone()["c"]
                except Exception:
                    rc = None

            info_map[t] = TableInfo(name=t, columns=cols, fks=fks, rowcount=rc, indexes=idxs)

        # build FK graph (undirected)
        graph = {t: [] for t in tables}
        for t, inf in info_map.items():
            for (_from, ref_t, _to) in inf.fks:
                if ref_t not in graph[t]:
                    graph[t].append(ref_t)
                if t not in graph[ref_t]:
                    graph[ref_t].append(t)

        _SCHEMA_INTEL = SchemaIntel(db_path=db_path, tables=info_map, fk_graph=graph)
    finally:
        conn.close()

def _require_loaded() -> SchemaIntel:
    if _SCHEMA_INTEL is None:
        raise RuntimeError("SchemaIntel not initialized. Call load_schema_intel(db_path) once at startup.")
    return _SCHEMA_INTEL

@tool
def list_tables(_: str = "") -> str:
    """List all tables with row counts (if available)."""
    si = _require_loaded()
    lines = []
    for t, inf in si.tables.items():
        lines.append(f"- {t} (rows: {inf.rowcount if inf.rowcount is not None else 'unknown'})")
    return "\n".join(lines)

@tool
def describe_table(table: str) -> str:
    """Describe columns, types, indexes, and FKs of a table. Input: table name."""
    si = _require_loaded()
    t = table.strip()
    if t not in si.tables:
        return f"Table '{t}' not found."
    inf = si.tables[t]
    cols = ", ".join([f"{c[0]}:{c[1]}{' NOT NULL' if c[2] else ''}" for c in inf.columns])
    idxs = ", ".join(inf.indexes) if inf.indexes else "—"
    fks = "; ".join([f"{x[0]} -> {x[1]}.{x[2]}" for x in inf.fks]) if inf.fks else "—"
    return f"Table {t}\nColumns: {cols}\nIndexes: {idxs}\nFKs: {fks}\nRows: {inf.rowcount}"

def _shortest_join_path(si: SchemaIntel, targets: List[str]) -> List[str]:
    """BFS on FK graph to connect all target tables; returns a table visitation order."""
    # Simple heuristic: connect sequentially
    from collections import deque
    if not targets:
        return []
    order = [targets[0]]
    for goal in targets[1:]:
        # BFS from any node in 'order' to goal
        visited = set()
        q = deque([(order[-1], [order[-1]])])
        found = None
        while q:
            node, path = q.popleft()
            if node == goal:
                found = path
                break
            if node in visited:
                continue
            visited.add(node)
            for nb in si.fk_graph.get(node, []):
                if nb not in visited:
                    q.append((nb, path + [nb]))
        if found:
            # append path (skip overlap)
            for n in found[1:]:
                order.append(n)
        else:
            order.append(goal)  # disconnected; leave as-is
    # unique keep order
    seen, uniq = set(), []
    for x in order:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

@tool
def find_join_path(tables_csv: str) -> str:
    """
    Find a reasonable join path between tables, based on FK graph.
    Input: comma-separated table names, e.g. "orders, customers, items".
    """
    si = _require_loaded()
    targets = [t.strip() for t in tables_csv.split(",") if t.strip()]
    for t in targets:
        if t not in si.tables:
            return f"Unknown table: {t}"
    path = _shortest_join_path(si, targets)
    return " → ".join(path)

@tool
def profile_column(args_json: str) -> str:
    """
    Profile a column's distribution. Input JSON: {"table":"...", "column":"...", "top_k":5}
    Returns top values and counts (or min/max for numeric).
    """
    si = _require_loaded()
    try:
        args = json.loads(args_json)
        t, c = args["table"], args["column"]
        top_k = int(args.get("top_k", 5))
    except Exception:
        return "Invalid JSON. Use: {\"table\":\"...\",\"column\":\"...\",\"top_k\":5}"

    if t not in si.tables:
        return f"Unknown table: {t}"
    colnames = [x[0] for x in si.tables[t].columns]
    if c not in colnames:
        return f"Unknown column: {t}.{c}"

    conn = _connect(si.db_path)
    try:
        # Numeric min/max?
        try:
            row = conn.execute(f"SELECT MIN({c}) AS mn, MAX({c}) AS mx FROM {t};").fetchone()
            if row and (row["mn"] is not None or row["mx"] is not None):
                # if both min/max are numeric-like, we still return them; LLM will interpret.
                pass
        except Exception:
            row = None

        freq = []
        try:
            for r in conn.execute(f"SELECT {c} AS v, COUNT(*) AS c FROM {t} GROUP BY {c} ORDER BY c DESC LIMIT {top_k};"):
                freq.append((r["v"], r["c"]))
        except Exception:
            freq = []

        report = []
        if row:
            report.append(f"Min/Max: {row['mn']} / {row['mx']}")
        if freq:
            report.append("Top values: " + ", ".join([f"{v} ({cnt})" for v, cnt in freq]))
        return "\n".join(report) if report else "No profile available."
    finally:
        conn.close()
