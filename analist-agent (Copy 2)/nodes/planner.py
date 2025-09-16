"""
nodes/planner.py — LLM-aware, tool-aware planner wrapper

İlk düzenleme: raw içerik temel alınmıştır; artık bu dosya, LLM varsa
`tools.PlannerTool` üzerine delege eder, LLM yoksa **güvenli varsayılan**
heuristic planı uygular (eski davranışla uyumlu).

Kullanım:
    from nodes import planner
    state = planner.run(state, llm=llm, cfg=cfg, conn=conn, cost=cost)

Çıktı:
    - state.intent           ("sql_query" | "non_sql" | "exit")
    - state.use_rag          (bool)
    - state.plan             (tool-aware adımlar listesi)
    - opsiyonel: state.answer_text (non_sql açıklaması), state.error
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from utils.types import AgentState

log = logging.getLogger("planner")

# ──────────────────────────────────────────────────────────────────────────────
# Raw sürümdeki sözlükler ve yardımcılar (geriye dönük uyumluluk için tutuldu)
# ──────────────────────────────────────────────────────────────────────────────
DB_KEYWORDS: List[str] = [
    # Sorgu/analitik niyet göstergeleri
    "kaç", "say", "listele", "ortalama", "avg", "sum", "count", "max", "min",
    "group by", "sırala", "sort", "en fazla", "en çok", "hangi", "dağılım", "distribution",
    # Varlık/şema isimleri ve sinonimler
    "unit", "birim", "user", "kullanıcı", "kulanıcı",  # yazım hatası varyantı
    "chat", "session", "oturum", "sohbet", "mesaj", "message", "age", "yaş",
]

NON_SQL_NOISE: List[str] = [
    "merhaba", "selam", "hello", "hi", "poem", "şiir", "story", "hikaye",
    "hava", "weather", "şaka", "joke", "lol", "deneme", "test",
]

NON_SQL_EDU: List[str] = [
    "sql injection nedir", "veritabanı nedir", "database nedir",
    "how to install", "what is sql injection", "how does it work",
]

EXIT_TOKENS = {"q", ":q", ":quit", "quit", "exit", ":exit"}


def _alpha_ratio(s: str) -> float:
    letters = sum(ch.isalpha() for ch in s)
    return letters / max(1, len(s))


def _tokenize(q: str) -> List[str]:
    return re.findall(r"[a-z0-9ğüşöçıİĞÜŞÖÇ]+", q.lower())


def _has_fuzzy_keyword(q: str) -> bool:
    toks = _tokenize(q)
    for t in toks:
        if any(t in kw or kw in t for kw in DB_KEYWORDS):
            return True
        if get_close_matches(t, DB_KEYWORDS, n=1, cutoff=0.8):
            return True
    return False


def _looks_like_noise(q: str) -> bool:
    ql = q.lower().strip()
    if ql in EXIT_TOKENS:
        return True
    if len(ql) < 2:
        return True
    if _alpha_ratio(ql) < 0.3:
        return True
    if any(tok in ql for tok in NON_SQL_NOISE):
        return True
    if any(tok in ql for tok in NON_SQL_EDU):
        return True
    return False


def _mentions_db_semantics(q: str, schema_hint: Optional[str] = None) -> bool:
    ql = q.lower()
    if any(tok in ql for tok in DB_KEYWORDS):
        return True
    if _has_fuzzy_keyword(ql):
        return True
    if schema_hint:
        for line in schema_hint.lower().splitlines():
            m = re.search(r"table\s+([a-z0-9_]+)", line)
            if m and m.group(1) in ql:
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# LLM-aware Planlama (tools.PlannerTool delege)
# ──────────────────────────────────────────────────────────────────────────────

def _llm_plan(question: str, schema_doc: Optional[str], *, llm, cfg, conn, cost, logger=None) -> Dict[str, Any]:
    """tools.PlannerTool'u çağırıp planı JSON olarak döndürür.
    llm yoksa Tool katmanı güvenli varsayılan plan döndürür.
    """
    try:
        from agent_tools import run_tool  # TOOL_REGISTRY["planner"].run(...)
        out = run_tool(
            "planner",
            {"question": question, "schema_doc": schema_doc},
            {"llm": llm, "cfg": cfg, "conn": conn, "cost": cost, "logger": logger or log},
        )
        return out  # {intent, use_rag, plan: [{tool, input}], decision_rationale}
    except Exception as e:
        log.warning("PlannerTool delege başarısız: %s — varsayılan plana düşülüyor", e)
        # Güvenli varsayılan plan
        return {
            "intent": "sql_query",
            "use_rag": False,
            "plan": [
                {"tool": "schema_retriever", "input": {}},
                {"tool": "query_generator", "input": {}},
                {"tool": "query_validator", "input": {}},
                {"tool": "sql_executor", "input": {}},
                {"tool": "postprocessor", "input": {}},
                {"tool": "summarizer", "input": {}},
                {"tool": "guardian", "input": {}},
            ],
            "decision_rationale": "PlannerTool hatası/fallback",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Ana giriş (geriye dönük uyumlu API)
# ──────────────────────────────────────────────────────────────────────────────

def run(
    state: AgentState,
    *,
    llm: Any | None = None,
    cfg: Dict[str, Any] | None = None,
    conn: Any | None = None,
    cost: Any | None = None,
    logger: Any | None = None,
    rag_enabled_default: bool = True,
) -> AgentState:
    """Soru → intent + plan. LLM varsa tool-aware plan; yoksa heuristics.

    Not: Eski imzayla uyum için `rag_enabled_default` korunmuştur.
    """
    q = state.question or ""
    ql = q.strip().lower()

    # 1) LLM yoksa — eski heuristics (raw mantık)
    if llm is None:
        has_db_sem = _mentions_db_semantics(ql, state.schema_doc)
        if not has_db_sem and _looks_like_noise(ql):
            state.intent = "non_sql"
            state.answer_text = "Çıkış komutu algılandı." if ql in EXIT_TOKENS else "Bu soru veritabanıyla ilgili değil."
            state.plan = [
                {"tool": "summarizer", "input": {}},  # planı boşaltmak yerine tek adım bırakıyoruz
            ]
            state.use_rag = False
            return state
        if not has_db_sem:
            state.intent = "non_sql"
            state.answer_text = "Bu soru veritabanıyla ilgili değil."
            state.plan = [{"tool": "summarizer", "input": {}}]
            state.use_rag = False
            return state

        # Varsayılan güvenli SQL planı
        state.intent = "sql_query"
        state.use_rag = bool(rag_enabled_default)
        state.plan = [
            {"tool": "schema_retriever", "input": {}},
            *([{ "tool": "rag", "input": {}}] if state.use_rag else []),
            {"tool": "query_generator", "input": {}},
            {"tool": "query_validator", "input": {}},
            {"tool": "sql_executor", "input": {}},
            {"tool": "postprocessor", "input": {}},
            {"tool": "summarizer", "input": {}},
            {"tool": "guardian", "input": {}},
        ]
        return state

    # 2) LLM varsa — Tool Planner ile karar ver
    out = _llm_plan(q, state.schema_doc, llm=llm, cfg=cfg, conn=conn, cost=cost, logger=logger)
    state.intent = out.get("intent", "sql_query")
    state.use_rag = bool(out.get("use_rag", False))
    state.plan = [
        {"tool": step.get("tool"), "input": step.get("input", {})}
        for step in (out.get("plan") or [])
        if isinstance(step, dict) and step.get("tool")
    ]
    # opsiyonel açıklama
    rationale = out.get("decision_rationale")
    if rationale:
        try:
            state.planner_rationale = rationale  # type: ignore[attr-defined]
        except Exception:
            pass
    return state
