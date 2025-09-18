# tools/paginator.py
from __future__ import annotations
import json, re, sqlite3
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
import time
from utils import telemetry
_DB_PATH: Optional[str] = None

def init_paginator(db_path: str):
    global _DB_PATH
    _DB_PATH = db_path

def _conn():
    if not _DB_PATH:
        raise RuntimeError("paginator not initialized. Call init_paginator(db_path).")
    c = sqlite3.connect(_DB_PATH); c.row_factory = sqlite3.Row
    return c

def _wrap_select(sql: str, limit: int, offset: int) -> str:
    s = sql.strip().rstrip(";")
    # temel güvenlik: sadece SELECT izin ver
    if not re.match(r"(?is)^\s*select\b", s):
        raise ValueError("Only SELECT queries can be paginated.")
    # SELECT already has LIMIT? yine de dıştan sarmak güvenli yoldur
    # SELECT * FROM ( <original> ) AS sub LIMIT ... OFFSET ...
    return f"SELECT * FROM ({s}) AS sub LIMIT {int(limit)} OFFSET {int(offset)}"

@tool
def paginate_sql(args_json: str) -> str:
    """
    Run a SELECT query with pagination.
    Input JSON: {"query":"...", "page_size":50, "page":1}
    Output JSON: {"columns":[...],"rows":[...],"page":1,"page_size":50,"has_more":true}
    """
    t0 = time.time()
    args = json.loads(args_json)
    q = args["query"]; page_size = int(args.get("page_size", 50)); page = int(args.get("page", 1))
    if page < 1: page = 1
    offset = (page-1) * page_size

    con = _conn()
    try:
        # toplamı kabaca tahmin etmek pahalı; has_more için bir satır fazla çekelim
        wrapped = _wrap_select(q, page_size + 1, offset)
        cur = con.execute(wrapped)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows: List[List[Any]] = []
        for i, r in enumerate(cur):
            if i >= page_size + 1: break
            rows.append([r[c] if isinstance(r, sqlite3.Row) else r[i] for i, c in enumerate(cols)])
        has_more = len(rows) > page_size
        if has_more: rows = rows[:page_size]
        telemetry.step("paginate_sql", ms=int((time.time()-t0)*1000), ok=True, page=page, page_size=page_size, has_more=has_more)
        return json.dumps({"columns": cols, "rows": rows, "page": page, "page_size": page_size, "has_more": has_more})
    finally:
        con.close()
