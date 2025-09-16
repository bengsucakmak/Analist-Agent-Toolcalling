#tools/db.py
import sqlite3, time, logging
from typing import List, Dict, Any

log = logging.getLogger("db")

def connect_readonly(path: str, timeout_ms: int=4000, max_instructions: int=200000) -> sqlite3.Connection:
    # SQLite bağlantısını URI ile read-only (mode=ro) açar; böylece yazma/DDL engellenir.
    uri = f"file:{path}?mode=ro"  # read-only
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Satırlara kolon isimleriyle erişmeyi sağlar.

    # Zaman ölçümü: progress handler bu start zamanına göre timeout kontrolü yapacak.
    start = time.time()
    def aborter():
        # progress handler belirli instruction aralığında çağrılır; burada süreye bakarak iptal edebiliriz.
        if (time.time() - start)*1000 > timeout_ms:
            return 1  # 1 döndürmek sorguyu abort eder.
        return 0     # 0 devam anlamına gelir.

    # max_instructions: handler'ın çağrılma sıklığını belirler (her N instruction'da bir).
    conn.set_progress_handler(aborter, max(1, max_instructions))
    # Her çalıştırılan SQL ifadesini debug loglamak için trace callback.
    conn.set_trace_callback(lambda s: log.debug("SQL> %s", s))
    return conn

def list_tables(conn) -> List[str]:
    # Kullanıcı tablolarını (sqlite_% hariç) ada göre sıralı getirir.
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    return [r["name"] for r in cur.fetchall()]

def table_columns(conn, tbl: str) -> List[Dict[str, Any]]:
    # PRAGMA table_info ile tablo kolon meta verilerini çeker.
    cur = conn.execute(f"PRAGMA table_info({tbl});")
    out = []
    for r in cur.fetchall():
        out.append({"cid": r["cid"], "name": r["name"], "type": r["type"], "notnull": r["notnull"], "dflt": r["dflt_value"], "pk": r["pk"]})
    return out

def schema_document(conn) -> str:
    # Basit metinsel şema çıktısı üretir: "Table T — columns: a:type, b:type, ..."
    parts = []
    for t in list_tables(conn):
        cols = table_columns(conn, t)
        cols_s = ", ".join([f"{c['name']}:{c['type']}" for c in cols])
        parts.append(f"Table {t} — columns: {cols_s}")
    return "\n".join(parts)

def explain_query_plan(conn, sql: str) -> str:
    # EXPLAIN QUERY PLAN çalıştırıp satırları dict->string olarak birleştirir.
    cur = conn.execute(f"EXPLAIN QUERY PLAN {sql}")
    rows = cur.fetchall()
    return "\n".join([str(dict(r)) for r in rows])

def execute_preview(conn, sql: str, limit: int=1000, preview_rows: int=50):
    # Sorguyu çalıştırır; ilk 'preview_rows' satırı {kolon: değer} sözlükleri olarak döndürür.
    cur = conn.execute(sql)
    colnames = [d[0] for d in cur.description] if cur.description else []
    rows = []
    for i, r in enumerate(cur.fetchall()):
        if i >= preview_rows: break
        rows.append({colnames[j]: r[j] for j in range(len(colnames))})
    # Dönüş sözlüğü: kolon adları, örnek satırlar ve bu örneklerin sayısı (rowcount)
    return {"columns": colnames, "rows": rows, "rowcount": len(rows)}