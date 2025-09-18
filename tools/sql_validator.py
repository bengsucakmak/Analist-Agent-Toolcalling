# tools/sql_validator.py
import sqlite3
from langchain_core.tools import tool
import time
from utils import telemetry

_DB_PATH = None

def init_sql_validator(db_path: str):
    global _DB_PATH
    _DB_PATH = db_path

def _conn():
    if not _DB_PATH:
        raise RuntimeError("sql_validator not initialized. Call init_sql_validator(db_path).")
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

@tool
def validate_sql(query: str) -> str:
    """
    Validate SQL without running it. Uses 'EXPLAIN QUERY PLAN'.
    Returns 'OK' or an error message. Input: raw SQL string.
    """
    t0 = time.time()
    try:
        con = _conn()
        with con:
            con.execute(f"EXPLAIN QUERY PLAN {query}")
        telemetry.step("validate_sql", ms=int((time.time()-t0)*1000), ok=True)
        return "OK"
    except Exception as e:
        msg = f"INVALID: {e}"
        telemetry.step("validate_sql", ms=int((time.time()-t0)*1000), ok=False, error=str(e))
        return msg

@tool
def explain_sql(query: str) -> str:
    """
    Return the SQLite query plan (human-readable).
    """
    try:
        con = _conn()
        rows = list(con.execute(f"EXPLAIN QUERY PLAN {query}").fetchall())
        if not rows:
            return "No plan."
        lines = []
        for r in rows:
            # columns: id, parent, notused, detail
            lines.append(" | ".join([str(r[0]), str(r[1]), str(r[3])]))
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"
