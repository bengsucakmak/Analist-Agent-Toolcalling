# tools/query_cache.py
from __future__ import annotations
import os, json, sqlite3, hashlib, time
from typing import Optional
from langchain_core.tools import tool
from utils import telemetry

_CACHE_DB: Optional[str] = None  # ayrı küçük bir sqlite dosyası

def init_query_cache(path: str = ".cache.sqlite"):
    """Küçük bir SQLite cache dosyası hazırla."""
    global _CACHE_DB
    _CACHE_DB = path
    # klasör yoksa oluştur
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    con = sqlite3.connect(_CACHE_DB)
    try:
        con.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            k TEXT PRIMARY KEY,
            created_at REAL,
            sql TEXT,
            result_json TEXT
        )
        """)
        con.commit()
    finally:
        con.close()

def _key(sql: str) -> str:
    norm = " ".join(sql.split()).strip().lower()
    return hashlib.sha256(norm.encode()).hexdigest()

def _open():
    if not _CACHE_DB:
        raise RuntimeError("query_cache not initialized. Call init_query_cache().")
    return sqlite3.connect(_CACHE_DB)

@tool
def cache_get(args_json: str) -> str:
    """
    Input: {"sql":"...","ttl_sec":3600}
    TTL süresi dolduysa veya kayıt yoksa boş string döner.
    """
    t0 = time.time()
    data = json.loads(args_json) if isinstance(args_json, str) else args_json
    sql = data["sql"]
    ttl = int(data.get("ttl_sec", 0))

    con = _open()
    ret = ""
    try:
        k = _key(sql)
        row = con.execute("SELECT created_at, result_json FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            created_at, result_json = row
            if ttl and (time.time() - float(created_at) > ttl):
                ret = ""  # expired
            else:
                ret = result_json or ""
    finally:
        con.close()

    # telemetri: hit/miss
    telemetry.step("cache_get", ms=int((time.time() - t0) * 1000), ok=bool(ret), hit=bool(ret))
    return ret

@tool
def cache_put(args_json: str) -> str:
    """
    Store result_json for a SQL.
    Input JSON: {"sql":"...","result_json":"..."}
    Returns "OK".
    """
    t0 = time.time()
    data = json.loads(args_json) if isinstance(args_json, str) else args_json
    sql = data["sql"]
    result_json = data["result_json"]

    con = _open()
    try:
        k = _key(sql)
        now = time.time()
        con.execute(
            "REPLACE INTO cache(k, created_at, sql, result_json) VALUES(?, ?, ?, ?)",
            (k, now, sql, result_json),
        )
        con.commit()
    finally:
        con.close()

    telemetry.step("cache_put", ms=int((time.time() - t0) * 1000), ok=True)
    return "OK"
