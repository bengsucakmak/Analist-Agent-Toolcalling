# tools/repair_sql.py
from __future__ import annotations
import json, re, sqlite3, difflib
from typing import Dict, List, Tuple, Optional, Set
from langchain_core.tools import tool
import time
from utils import telemetry


_DB_PATH: Optional[str] = None

def init_repair_sql(db_path: str):
    global _DB_PATH
    _DB_PATH = db_path

# ──────────────────────────
# Şema yardımcıları
# ──────────────────────────
def _conn():
    if not _DB_PATH:
        raise RuntimeError("repair_sql not initialized. Call init_repair_sql(db_path) first.")
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def get_schema_snapshot() -> Tuple[List[str], Dict[str, Set[str]]]:
    """Tablo listesi ve tablo->kolon seti haritası döndürür."""
    con = _conn()
    try:
        tables = [r["name"] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )]
        cols: Dict[str, Set[str]] = {}
        for t in tables:
            cs = set()
            for r in con.execute(f"PRAGMA table_info({t});"):
                cs.add(r["name"])
            cols[t] = cs
        return tables, cols
    finally:
        con.close()

# ──────────────────────────
# Basit SQL sezgileri
# ──────────────────────────
_WORD = r"[A-Za-z_][A-Za-z0-9_]*"

def _extract_used_tables(sql: str) -> List[str]:
    candidates = []
    for m in re.finditer(r"\bFROM\s+("+_WORD+")", sql, flags=re.IGNORECASE):
        candidates.append(m.group(1))
    for m in re.finditer(r"\bJOIN\s+("+_WORD+")", sql, flags=re.IGNORECASE):
        candidates.append(m.group(1))
    # uniq in order
    seen, out = set(), []
    for t in candidates:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _closest(name: str, pool: List[str], cutoff: float = 0.6) -> Optional[str]:
    m = difflib.get_close_matches(name, pool, n=1, cutoff=cutoff)
    return m[0] if m else None

def _word_boundary_replace(sql: str, bad: str, good: str) -> str:
    return re.sub(rf"\b{re.escape(bad)}\b", good, sql)

def _qualify_column(sql: str, col: str, table: str) -> str:
    # 'col' zaten table.col ise dokunma; aksi halde table.col yap
    pattern = rf"(?<!\.)\b{re.escape(col)}\b"
    return re.sub(pattern, f"{table}.{col}", sql)

# ──────────────────────────
# Düzeltme mantığı
# ──────────────────────────
def propose_fix(query: str, error: str, tables: List[str], cols: Dict[str, Set[str]]) -> Tuple[Optional[str], str]:
    """
    Hata mesajından tek bir mantıklı düzeltme öner.
    Dönen: (fixed_query or None, reason)
    """
    q = query
    used = _extract_used_tables(q)

    # 1) no such table: X
    m = re.search(r"no such table:\s*([A-Za-z_][A-Za-z0-9_]*)", error, flags=re.IGNORECASE)
    if m:
        bad = m.group(1)
        good = _closest(bad, tables)
        if good and good != bad:
            return _word_boundary_replace(q, bad, good), f"Tablo adı düzeltildi: '{bad}' → '{good}'."
        return None, f"Tablo bulunamadı: {bad}."

    # 2) no such column: y
    m = re.search(r"no such column:\s*([A-Za-z_][A-Za-z0-9_]*)", error, flags=re.IGNORECASE)
    if m:
        badc = m.group(1)
        # Tercihen FROM/JOIN içinde kullanılan tablolarda ara
        search_tables = used if used else tables
        # Önce aynen mevcut kolon adını hangi tablolar barındırıyor?
        candidate_tables = [t for t in search_tables if badc in cols.get(t, set())]
        if candidate_tables:
            # sadece niteleme eksikliği ise table.col yap
            return _qualify_column(q, badc, candidate_tables[0]), f"Kolon nitelendi: '{badc}' → '{candidate_tables[0]}.{badc}'."

        # Yakın kolon ismi öner (typo)
        all_columns = sorted({c for s in cols.values() for c in s})
        goodc = _closest(badc, all_columns)
        if goodc and goodc != badc:
            # En makul tabloyu seç (FROM/JOIN içinde goodc olan ilk tablo)
            holder = next((t for t in search_tables if goodc in cols.get(t, set())), None)
            repl = f"{holder}.{goodc}" if holder else goodc
            fixed = _word_boundary_replace(q, badc, repl)
            return fixed, f"Kolon adı düzeltildi: '{badc}' → '{repl}'."
        return None, f"Kolon bulunamadı: {badc}."

    # 3) ambiguous column name: col
    m = re.search(r"ambiguous column name:\s*([A-Za-z_][A-Za-z0-9_]*)", error, flags=re.IGNORECASE)
    if m:
        col = m.group(1)
        # FROM/JOIN'da geçen ve o kolonu barındıran ilk tabloyu seç
        holder = next((t for t in used if col in cols.get(t, set())), None)
        if not holder:
            holder = next((t for t in tables if col in cols.get(t, set())), None)
        if holder:
            return _qualify_column(q, col, holder), f"Belirsiz kolon nitelendi: '{col}' → '{holder}.{col}'."
        return None, f"Belirsiz kolon için tablo bulunamadı: {col}."

    # 4) near '...': syntax error  → çok genel; burada otomatik düzeltme denemiyoruz
    if "syntax error" in error.lower():
        return None, "Sözdizimi hatası: otomatik düzeltme uygulanmadı."

    return None, "Otomatik düzeltme kuralı bulunamadı."

# ──────────────────────────
# Tool sargısı
# ──────────────────────────
@tool
def repair_sql(args_json: str) -> str:
    """
    Hata mesajına göre SQL'i onarmayı dener.
    Girdi JSON: {"query": "...", "error": "..."}
    Çıktı JSON: {"fixed_query": "...", "reason": "...", "applied": true/false}
    """
    t0 = time.time()
    try:
        args = json.loads(args_json)
        query = args["query"]
        error = args.get("error", "")
    except Exception:
        return json.dumps({"applied": False, "reason": "Invalid JSON. Use {\"query\":\"...\",\"error\":\"...\"}."})

    try:
        tables, cols = get_schema_snapshot()
        fixed, reason = propose_fix(query, error, tables, cols)
        return json.dumps({
            "fixed_query": fixed if fixed is not None else query,
            "reason": reason,
            "applied": fixed is not None
        })
    except Exception as e:
        telemetry.step("repair_sql", ms=int((time.time()-t0)*1000), ok=False, error=str(e))
        return json.dumps({"applied": False, "reason": f"repair_sql internal error: {e}"})
