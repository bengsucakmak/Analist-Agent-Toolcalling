"""
Analist-Agent — graph.py (Tool-Calling uyumlu)

Bu sürüm, mevcut node’ları tek-dosya agent_tools.py içindeki Tool sınıfları üzerinden çağırır.
- LangGraph varsa StateGraph ile orkestrasyon
- Yoksa SequentialGraph (basit yedek)

Akış:
  planner → (opsiyonel) schema_retriever → (opsiyonel) rag → query_generator →
  query_validator → sql_executor → postprocessor → summarizer → guardian

Notlar:
- İlk düzenleme raw içeriklere dayanır; sonraki tüm düzenlemeler bu canvas üstünden ilerleyecek.
- Tool çalıştırma: agent_tools.run_tool(tool_name, payload, ctx)
- ctx: {"llm", "cfg", "conn", "cost", "logger", **"state"** ← agent_tools Tool'ları state üzerinden çalışır}
"""
from __future__ import annotations

from typing import Any, Dict
import logging

try:
    # LangGraph varsa kullan
    from langgraph.graph import StateGraph, START, END  # type: ignore
    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover
    _HAS_LANGGRAPH = False

from utils.types import AgentState  # question, plan, use_rag, schema_doc, rag_snippets, candidate_sql, ...
from agent_tools import run_tool

log = logging.getLogger("graph")

# ──────────────────────────────────────────────────────────────────────────────
# Yardımcı: ctx hazırlama
# ──────────────────────────────────────────────────────────────────────────────

def _make_ctx(llm, cfg, conn, cost, logger=None, *, state: AgentState | None = None) -> Dict[str, Any]:
    return {"llm": llm, "cfg": cfg, "conn": conn, "cost": cost, "logger": logger or log, "state": state}


# ──────────────────────────────────────────────────────────────────────────────
# Node implementasyonları (tool çağrıları)
# ──────────────────────────────────────────────────────────────────────────────

def _run_planner(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    out = run_tool("planner", {"question": state.question, "schema_doc": getattr(state, "schema_doc", None)}, ctx)
    state.intent = out.get("intent", "sql_query")
    raw_plan = out.get("plan", []) or []

    # AgentState.plan: List[str]
    out = run_tool("planner", {"question": state.question, "schema_doc": getattr(state, "schema_doc", None)}, ctx)
    state.intent = out.get("intent", "sql_query")
    raw_plan = out.get("plan", []) or []

    # AgentState.plan → List[str]
    state.plan = [(p.get("tool") if isinstance(p, dict) else str(p)) for p in raw_plan]

    # (opsiyonel) ayrıntıyı ayrı alana yaz; UI kullanabilir
    try:
        state.plan_detail = [
            {"tool": (p.get("tool") if isinstance(p, dict) else str(p)),
            "input": (p.get("input", {}) if isinstance(p, dict) else {})}
            for p in raw_plan
        ]
    except Exception:
        pass
    state.use_rag = bool(out.get("use_rag", False))

    # UI/trace için ayrıntı
    try:
        state.plan_detail = [
            {"tool": (p.get("tool") if isinstance(p, dict) else str(p)),
            "input": (p.get("input", {}) if isinstance(p, dict) else {})}
            for p in raw_plan
        ]
    except Exception:
        pass

    state.use_rag = bool(out.get("use_rag", False))
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "planner", "intent": state.intent, "use_rag": state.use_rag, "plan": state.plan})
        except Exception:
            pass
    return state



def _maybe_schema(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    """Plan veya ihtiyaç varsa şemayı çek."""
    needs_schema = (getattr(state, "schema_doc", None) is None) or any(s.get("tool") == "schema_retriever" for s in (state.plan or []))
    if not needs_schema:
        return state
    out = run_tool("schema_retriever", {}, ctx)
    state.schema_doc = out.get("schema_doc")
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "schema_retriever", "len": len(state.schema_doc or "")})
        except Exception:
            pass
    return state


def _maybe_rag(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    if not getattr(state, "use_rag", False):
        return state
    out = run_tool("rag", {"enable": True, "top_k": 4, "min_score": 0.1, "prefer": "hybrid"}, ctx)
    state.rag_snippets = out.get("rag_snippets", [])
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "rag", "k": len(state.rag_snippets)})
        except Exception:
            pass
    return state


def _qgen(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    # allowed_tables/config opsiyonel — tool state üzerinden erişiyor
    out = run_tool("query_generator", {"allowed_tables": [], "max_limit": 1000}, ctx)
    state.candidate_sql = out.get("candidate_sql", [])
    if hasattr(state, "trace"):
        try:
            sql0 = (state.candidate_sql or [""])[0]
            state.trace.append({"node": "query_generator", "sql_preview": sql0[:160]})
        except Exception:
            pass
    return state


def _qval(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    # Validator tool, state içindeki candidate_sql'i okuyup rapor yazar
    out = run_tool("query_validator", {
        "banned_keywords": ["drop","alter","delete","update","insert","attach","pragma","vacuum","create","replace"],
        "enforce_select_only": True,
        "max_limit": 1000,
        "allowed_tables": [],
    }, ctx)
    if not out.get("ok", False):
        fixed = out.get("validated_sql")
        reason = out.get("reason")
        if not fixed:
            state.error = f"SQL doğrulama başarısız: {reason}"
            return state
        state.candidate_sql = [fixed]
    else:
        # validator validated_sql dönderdiyse onu esas al
        if out.get("validated_sql"):
            state.candidate_sql = [out["validated_sql"]]
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "query_validator", "ok": out.get("ok", False)})
        except Exception:
            pass
    return state


def _exec(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    out = run_tool("sql_executor", {"preview_rows": 50}, ctx)
    state.rows_preview = out.get("rows_preview", [])
    state.execution_stats = out.get("execution_stats", {})
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "sql_executor", "rows": len(state.rows_preview)})
        except Exception:
            pass
    return state


def _post(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    out = run_tool("postprocessor", {}, ctx)
    # Postprocessor tool aynı alan isimleriyle güncel state'i dolduruyor; yine de güvenli kopya alalım
    state.rows_preview = out.get("rows_preview", state.rows_preview if hasattr(state, "rows_preview") else [])
    state.execution_stats = out.get("execution_stats", state.execution_stats if hasattr(state, "execution_stats") else {})
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "postprocessor", "preview_rows": len(state.rows_preview or [])})
        except Exception:
            pass
    return state


def _summ(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    out = run_tool("summarizer", {"show_sql": False}, ctx)
    # Tool SummarizerOut(answer_text)
    state.answer_text = out.get("answer_text", "")
    # UI geriye dönük uyum için state.answer alanını da dolduralım
    state.answer = state.answer_text
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "summarizer", "chars": len(state.answer_text or "")})
        except Exception:
            pass
    return state


def _guard(state: AgentState, ctx: Dict[str, Any]) -> AgentState:
    out = run_tool("guardian", {}, ctx)
    if not out.get("ok", True):
        state.error = out.get("reason", "Guard reddetti")
    if hasattr(state, "trace"):
        try:
            state.trace.append({"node": "guardian", "ok": out.get("ok", True)})
        except Exception:
            pass
    return state


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph derlemesi
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(conn, cfg, cost, llm_service, *, logger=None):
    # ctx'ye state, invoke sırasında enjekte edilecek
    def _ctx_for(state: AgentState):
        return _make_ctx(llm_service, cfg, conn, cost, logger, state=state)

    if _HAS_LANGGRAPH:
        g = StateGraph(AgentState)

        # Node ekle
        g.add_node("planner", lambda s: _run_planner(s, _ctx_for(s)))
        g.add_node("schema", lambda s: _maybe_schema(s, _ctx_for(s)))
        g.add_node("rag", lambda s: _maybe_rag(s, _ctx_for(s)))
        g.add_node("qgen", lambda s: _qgen(s, _ctx_for(s)))
        g.add_node("qval", lambda s: _qval(s, _ctx_for(s)))
        g.add_node("exec", lambda s: _exec(s, _ctx_for(s)))
        g.add_node("post", lambda s: _post(s, _ctx_for(s)))
        g.add_node("summ", lambda s: _summ(s, _ctx_for(s)))
        g.add_node("guard", lambda s: _guard(s, _ctx_for(s)))

        # Kenarlar
        g.add_edge(START, "planner")
        g.add_edge("planner", "schema")
        g.add_edge("schema", "rag")
        g.add_edge("rag", "qgen")
        g.add_edge("qgen", "qval")
        g.add_edge("qval", "exec")
        g.add_edge("exec", "post")
        g.add_edge("post", "summ")
        g.add_edge("summ", "guard")
        g.add_edge("guard", END)

        return g.compile()

    # Yedek: basit sıralı çalıştırıcı
    class SequentialGraph:
        def invoke(self, state: AgentState, config: Dict[str, Any] | None = None) -> AgentState:
            ctx = _ctx_for(state)
            for fn in (_run_planner, _maybe_schema, _maybe_rag, _qgen, _qval, _exec, _post, _summ, _guard):
                state = fn(state, ctx)
                if getattr(state, "error", None):
                    break
            return state

    return SequentialGraph()
