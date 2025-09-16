# nodes/query_generator.py
import logging
from utils.llm import call_llm_text
from utils.types import AgentState
from utils.cost import CostTracker

log = logging.getLogger("qgen")

SYSTEM_PROMPT_TMPL = """
You are an expert SQL generator for an AI Analyst Agent.

DATABASE: SQLite (NOT Postgres/MySQL). Use only SQLite syntax.

HARD RULES:
- Use ONLY the tables listed below. If a name is not here, DO NOT use or invent it.
  Allowed tables: {allowed_tables}
- Single output: one SELECT or WITH..SELECT, no DDL/DML/PRAGMA/ATTACH, no comments, no prose.
- Use SQLite time functions (strftime). Prefer human-readable fields (e.g., unit_name) via JOINs when they EXIST.

REPORTING PRINCIPLES:
- Prefer human-readable fields (e.g., unit_name) over IDs (unit_id). If needed, JOIN the dimension table to get names.
- For date/time: use strftime (e.g., strftime('%w', col) for weekday).
- Generate ONE single SELECT (or WITH...SELECT) statement. No DML/DDL.
- Do NOT include explanations, comments, markdown, or backticks.
- End result must be plain SQL text only (no prose).

LIMITS & SAFETY:
- Include LIMIT (<= {max_limit}) unless it is a pure aggregate with few rows.
"""

def _clean_sql(txt: str) -> str:
    sql = (txt or "").strip()

    # Kod blok çitleri ve inline backtick temizliği
    for fence in ("```sql", "```", "`"):
        if fence in sql:
            # En sondan bölmek bazı durumlarda kalanı verir; daha güvenlisi:
            parts = sql.split(fence)
            sql = " ".join(parts)
            sql = sql.strip()

    # Satır bazlı temizlik
    lines = []
    for line in sql.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("to "):   # "To show the ..." gibi yönergeleri at
            continue
        if s.startswith("--"):            # SQL yorumları
            continue
        lines.append(s)

    sql = " ".join(lines).strip()

    # Tek trailing ';' varsa kaldır (validator/exec tarafı tolere eder)
    if sql.endswith(";"):
        sql = sql[:-1]
    return sql


def _is_pure_aggregate(sql_lower: str) -> bool:
    """
    Çok kaba bir sezgi: select ifadesinde aggregate fonksiyon var ve
    group by/yinelenen satır üretmiyor, ayrıca limit yok → 'pure aggregate' kabul edilebilir.
    (Tam doğruluk için parser gerekir; validator zaten ikinci savunma hattı.)
    """
    has_agg = any(fn in sql_lower for fn in ("count(", "avg(", "sum(", "min(", "max("))
    return has_agg and " group by " not in sql_lower


def run(
    state: AgentState,
    cost: CostTracker,
    llm_service,
    max_limit: int = 1000,
    allowed_tables: list[str] | None = None,
) -> AgentState:
    """
    LLM'den tek bir SELECT (veya WITH...SELECT) üretir, basit normalize eder,
    LIMIT kuralını uygular ve state.candidate_sql'e yazar.
    """
    if not getattr(state, "schema_doc", None):
        state.validation_report = {"ok": False, "reason": "No schema"}
        return state

    allowed_tables = allowed_tables or []  # boşsa yine de formatta boş gösteririz
    allowed_tables_str = ", ".join(allowed_tables)

    system_prompt = SYSTEM_PROMPT_TMPL.format(
        allowed_tables=allowed_tables_str,
        max_limit=max_limit,
    )

    user_prompt = f"""SCHEMA (SQLite):
{state.schema_doc}

QUESTION:
{state.question}

HINTS:
- If the question refers to entities like units or users, join to fetch their names (e.g., unit_name, user.name) instead of IDs.
- For weekday: SELECT strftime('%w', message_date) AS weekday, COUNT(*) FROM chat_session GROUP BY weekday ORDER BY weekday;
"""

    # --- LLM çağrısı ---
    raw = call_llm_text(llm_service, system_prompt, user_prompt, cost=cost)
    sql = _clean_sql(raw)

    # İlkel güvenlik: cümle/yorum dönerse fallback
    if not sql.lower().startswith(("select", "with")):
        sql = "SELECT 1 AS dummy"

    # LIMIT kuralı
    sql_l = sql.lower()
    if not _is_pure_aggregate(sql_l):
        # 'with' ile başlayanlarda dış SELECT'te limit olup olmadığını kaba şekilde kontrol ederiz
        if " limit " not in sql_l:
            sql = f"{sql} LIMIT {max_limit}"

    # normalize (tek trailing ';' kaldırılmıştı)
    state.candidate_sql = [sql.strip()]
    log.info("SQL adayı üretildi.")
    return state