# tools/safe_sql.py
from __future__ import annotations
import json, sqlite3
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

from .sql_validator import validate_sql, explain_sql  # mevcut dosyadan
from .repair_sql import propose_fix, get_schema_snapshot  # kütüphane fonksiyonlarını doğrudan kullan
import time
from utils import telemetry

_DB_PATH: Optional[str] = None

def init_safe_sql(db_path: str):
    global _DB_PATH
    _DB_PATH = db_path

def _conn():
    if not _DB_PATH:
        raise RuntimeError("safe_sql not initialized. Call init_safe_sql(db_path) first.")
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _validate(query: str) -> str:
    # sql_validator.validate_sql zaten "OK" veya "INVALID: ..." döndürüyor
    return validate_sql.run(query) if hasattr(validate_sql, "run") else validate_sql.invoke({"input": query})

def _explain(query: str) -> str:
    return explain_sql.run(query) if hasattr(explain_sql, "run") else explain_sql.invoke({"input": query})

def _run(query: str, limit: int = 200) -> Dict[str, Any]:
    con = _conn()
    try:
        cur = con.execute(query)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows: List[List[Any]] = []
        count = 0
        for r in cur:
            if count >= limit:
                break
            rows.append([r[c] if isinstance(r, sqlite3.Row) else r[i] for i, c in enumerate(cols)])
            count += 1
        return {"columns": cols, "rows": rows}
    finally:
        con.close()

@tool
def safe_run_sql(args_json: str) -> str:
    """
    SQL'i güvenli şekilde çalıştır:
      1) EXPLAIN ile doğrula
      2) Hata varsa otomatik onarım dene (2 tur)
      3) Geçerse çalıştır ve sonuçları döndür
    Girdi JSON: {"query":"...", "max_attempts":2, "row_limit":200}
    Çıktı JSON:
      {
        "status": "ok"|"error",
        "final_query": "...",
        "plan": "....",
        "result": {"columns":[...], "rows":[...]},
        "attempts": [{"query":"...","reason":"...","validation":"OK|..."}]
      }
    """
    T0 = time.time()
    try:
        args = json.loads(args_json)
        query = args["query"]
        max_attempts = int(args.get("max_attempts", 2))
        row_limit = int(args.get("row_limit", 200))
    except Exception:
        return json.dumps({"status":"error","message":"Invalid JSON. Use {\"query\":\"...\"}."})

    attempts = []

    # 0) İlk doğrulama
    v = _validate(query)
    if v == "OK":
        plan = _explain(query)
        res = _run(query, limit=row_limit)
        return json.dumps({"status":"ok","final_query":query,"plan":plan,"result":res,"attempts":[{"query":query,"reason":"initial","validation":"OK"}]})

    # 1) Otomatik onarım döngüsü
    try:
        tables, cols = get_schema_snapshot()
    except Exception as e:
        return json.dumps({"status":"error","message":f"Schema snapshot failed: {e}","attempts":[{"query":query,"reason":str(v),"validation":"INVALID"}]})

    last_err = v
    cur_q = query
    for i in range(max_attempts):
        fixed, reason = propose_fix(cur_q, last_err, tables, cols)
        if not fixed or fixed == cur_q:
            attempts.append({"query":cur_q, "reason": reason, "validation": str(last_err)})
            break

        cur_q = fixed
        v2 = _validate(cur_q)
        attempts.append({"query":cur_q, "reason":reason, "validation":str(v2)})
        if v2 == "OK":
            plan = _explain(cur_q)
            res = _run(cur_q, limit=row_limit)
            return json.dumps({"status":"ok","final_query":cur_q,"plan":plan,"result":res,"attempts":attempts})
        else:
            last_err = v2
    telemetry.step("safe_run_sql", ms=int((time.time()-T0)*1000), ok=True, attempts=attempts, final_query=cur_q)
    return json.dumps({"status":"error","message":str(last_err),"final_query":cur_q,"attempts":attempts})
