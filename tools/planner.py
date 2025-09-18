# tools/planner.py
from __future__ import annotations
import json, re, sqlite3, difflib
from typing import Optional, Dict, List, Tuple, Set
from langchain_core.tools import tool
import time
from utils import telemetry

_DB_PATH: Optional[str] = None

def init_planner(db_path: str):
    """Graph başlangıcında çağrılır."""
    global _DB_PATH
    _DB_PATH = db_path

def _conn():
    if not _DB_PATH:
        raise RuntimeError("planner not initialized. Call init_planner(db_path).")
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _schema_snapshot() -> Tuple[List[str], Dict[str, List[Tuple[str, str]]], Dict[str, Set[str]], Dict[str, List[Tuple[str,str,str]]]]:
    """
    tables, columns_map (table -> [(col, type), ...]),
    columns_set (table -> {col,...}), fks (table -> [(from_col, ref_table, ref_col), ...])
    """
    con = _conn()
    try:
        tables = [r["name"] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )]
        columns_map: Dict[str, List[Tuple[str, str]]] = {}
        columns_set: Dict[str, Set[str]] = {}
        fks: Dict[str, List[Tuple[str,str,str]]] = {}
        for t in tables:
            cols = []
            colset: Set[str] = set()
            for r in con.execute(f"PRAGMA table_info({t});"):
                cols.append((r["name"], (r["type"] or "").upper()))
                colset.add(r["name"])
            columns_map[t] = cols
            columns_set[t] = colset

            fklist = []
            for r in con.execute(f"PRAGMA foreign_key_list({t});"):
                fklist.append((r["from"], r["table"], r["to"]))
            fks[t] = fklist
        return tables, columns_map, columns_set, fks
    finally:
        con.close()

def _build_fk_graph(tables: List[str], fks: Dict[str, List[Tuple[str,str,str]]]) -> Dict[str, List[str]]:
    g = {t: [] for t in tables}
    for t, lst in fks.items():
        for (_from, ref_t, _to) in lst:
            if ref_t not in g[t]: g[t].append(ref_t)
            if t not in g[ref_t]: g[ref_t].append(t)
    return g

def _shortest_join_path(graph: Dict[str, List[str]], targets: List[str]) -> List[str]:
    if not targets: return []
    from collections import deque
    order = [targets[0]]
    for goal in targets[1:]:
        q = deque([(order[-1], [order[-1]])])
        vis = set(); found = None
        while q:
            node, path = q.popleft()
            if node == goal:
                found = path; break
            if node in vis: continue
            vis.add(node)
            for nb in graph.get(node, []):
                if nb not in vis:
                    q.append((nb, path + [nb]))
        if found:
            for n in found[1:]:
                order.append(n)
        else:
            order.append(goal)
    # uniq keep order
    seen, uniq = set(), []
    for x in order:
        if x not in seen: seen.add(x); uniq.append(x)
    return uniq

def _guess_intent(q: str) -> str:
    s = q.lower()
    if any(k in s for k in ["trend", "zaman", "hafta", "ay", "yıl", "time series"]): return "trend"
    if any(k in s for k in ["en çok", "top ", "ilk ", "en az"]): return "topk"
    if any(k in s for k in ["ortalama", "avg", "mean", "toplam", "sum", "min", "max", "medyan", "count", "kaç"]): return "aggregate"
    return "list"

def _extract_topk(q: str) -> Optional[int]:
    m = re.search(r"(top|ilk)\s*(\d+)", q.lower())
    if m: return int(m.group(2))
    m = re.search(r"en çok\s*(\d+)", q.lower())
    if m: return int(m.group(1))
    return None

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

# basit TR→EN eşlemeleri (ihtiyaca göre genişlet)
_SYNONYMS = {
    "yaş": "age", "yas": "age",
    "kullanıcı": "user", "kullanici": "user",
    "oturum": "chat_session",
    "mesaj": "message",
    "sağlayıcı": "provider", "sağlayici": "provider",
}

def _expand_synonyms(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        out.append(t)
        if t in _SYNONYMS:
            out.append(_SYNONYMS[t])
    return out


def _closest(name: str, pool: List[str], cutoff: float = 0.6) -> Optional[str]:
    m = difflib.get_close_matches(name, pool, n=1, cutoff=cutoff)
    return m[0] if m else None

def _candidate_tables(question: str, tables: List[str], columns_map: Dict[str, List[Tuple[str,str]]]) -> List[str]:
    toks = _expand_synonyms(_tokenize(question))
    scores = {t: 0 for t in tables}

    for t in tables:
        # tablo adı eşleşmesi
        for tok in toks:
            if tok and tok in t.lower(): scores[t] += 2
        # kolon adı eşleşmesi
        for c, _ in columns_map[t]:
            for tok in toks:
                if tok and tok in c.lower(): scores[t] += 1
    ranked = sorted(tables, key=lambda x: scores[x], reverse=True)
    return [t for t in ranked if scores[t] > 0][:4] or ranked[:2]

def _numeric_columns(columns_map: Dict[str, List[Tuple[str,str]]], table: str) -> List[str]:
    nums = []
    for c, typ in columns_map[table]:
        if any(k in typ for k in ["INT", "REAL", "NUM", "DEC"]):
            nums.append(c)
    return nums

def _time_column(columns_map: Dict[str, List[Tuple[str,str]]], table: str) -> Optional[str]:
    prefer = ["created_at", "createdat", "create_time", "timestamp", "ts", "date", "dt", "time"]
    names = [c for c,_ in columns_map[table]]
    for p in prefer:
        for n in names:
            if p == n.lower(): return n
    for n in names:
        if any(k in n.lower() for k in ["date","time","ts","created"]): return n
    return None

@tool
def plan_query(args_json: str) -> str:
    """
    Kullanıcı sorusundan QuerySpec planı üretir.
    Girdi JSON: {"question":"...","default_limit":50}
    Çıktı JSON: {
      "intent": "list|aggregate|topk|trend",
      "tables": [...],
      "metrics": [{"expr":"COUNT(*)","alias":"count"}],
      "dimensions": ["colA", ...],
      "filters": [{"column":"...","op":"=","value":"..."}],
      "time": {"column":"created_at","grain":"day","start":null,"end":null},
      "order_by": [{"expr":"count","desc":true}],
      "limit": 50,
      "join_path": ["t1","t2",...]
    }
    """
    try:
        args = json.loads(args_json)
        question = args["question"]
        default_limit = int(args.get("default_limit", 50))
    except Exception:
        return json.dumps({"error":"Invalid JSON. Use {\"question\":\"...\"}."})

    tables, columns_map, columns_set, fks = _schema_snapshot()
    intent = _guess_intent(question)
    cand = _candidate_tables(question, tables, columns_map)
    fk_graph = _build_fk_graph(tables, fks)
    join_path = _shortest_join_path(fk_graph, cand)

    # metrics & dimensions
    metrics = []
    dimensions: List[str] = []
    order_by = []
    limit = default_limit
    topk = _extract_topk(question)
    if topk: limit = topk

    # choose metric
    if intent in ("aggregate","topk","trend"):
        # pick a numeric column if any, else COUNT(*)
        numcol = None
        for t in cand:
            nums = _numeric_columns(columns_map, t)
            if nums:
                numcol = f"{t}.{nums[0]}"; break
        text = question.lower()
        if "ortalama" in text or "avg" in text or "mean" in text:
            if numcol: metrics = [{"expr": f"AVG({numcol})", "alias": "avg"}]
            else: metrics = [{"expr":"COUNT(*)","alias":"count"}]
        elif "toplam" in text or "sum" in text:
            if numcol: metrics = [{"expr": f"SUM({numcol})", "alias": "sum"}]
            else: metrics = [{"expr":"COUNT(*)","alias":"count"}]
        elif "min" in text:
            if numcol: metrics = [{"expr": f"MIN({numcol})", "alias": "min"}]
            else: metrics = [{"expr":"COUNT(*)","alias":"count"}]
        elif "max" in text or "en büyük" in text:
            if numcol: metrics = [{"expr": f"MAX({numcol})", "alias": "max"}]
            else: metrics = [{"expr":"COUNT(*)","alias":"count"}]
        else:
            if any(k in text for k in ["kaç","count","sayısı","sayı"]):
                metrics = [{"expr":"COUNT(*)","alias":"count"}]
            else:
                if numcol: metrics = [{"expr": f"AVG({numcol})", "alias": "avg"}]
                else: metrics = [{"expr":"COUNT(*)","alias":"count"}]
    else:
        metrics = []

    # dimension: eğer trend ise zaman kolonu, değilse ilk uygun string kolon
    time_spec = {"column": None, "grain": "day", "start": None, "end": None}
        
    if intent == "trend":
        for t in cand:
            tc = _time_column(columns_map, t)
            if tc:
                time_spec["column"] = f"{t}.{tc}"
                dimensions.append(f"{t}.{tc}")
                break

    # Aggregate/trend için dimension ekleme; liste/topk’da gerekirse ekle
    if intent in ("list", "topk") and not dimensions:
        for t in cand:
            for c, typ in columns_map[t]:
                if "CHAR" in typ or "TEXT" in typ or typ == "":
                    dimensions.append(f"{t}.{c}")
                    break
            if dimensions:
                break

    # order_by
    if intent == "topk":
        if metrics:
            order_by = [{"expr": metrics[0]["alias"], "desc": True}]
        elif dimensions:
            order_by = [{"expr": dimensions[0], "desc": False}]

    spec = {
        "intent": intent,
        "tables": cand,
        "metrics": metrics,
        "dimensions": dimensions,
        "filters": [],
        "time": time_spec,
        "order_by": order_by,
        "limit": limit,
        "join_path": join_path
    }
    
    _t0 = time.time()
    out = json.dumps(spec)
    telemetry.step("plan_query", ms=int((time.time()-_t0)*1000), ok=True, intent=intent, tables=cand, join_path=join_path)
    return out