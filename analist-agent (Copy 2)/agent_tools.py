"""
Analist-Agent — agent_tools.py

Tek dosyada tüm node'ları Tool sınıfları olarak sarmalayan altyapı.

Tasarımlar:
- ToolBase: name, input_model, output_model, run arayüzü
- Her node için Tool sınıfı (SchemaRetrieverTool, RAGTool, QueryGeneratorTool, QueryValidatorTool,
  SQLExecutorTool, PostprocessorTool, SummarizerTool, GuardianTool, PlannerTool)
- Pydantic I/O şemaları (her tool için açık giriş/çıkış)
- TOOL_REGISTRY: tool adı → örnek
- Context sözlüğü (llm, cfg, conn, cost, logger, vb.)

Ekstra:
- LangChain adapter: TOOL_REGISTRY içindeki tool'ları StructuredTool'a çevirir, bind_tools() ile kullanılabilir.

NOT:
- **Bu canvas agent_tools.py için tek kaynaktır (source of truth).** Başka canvas açmayacağım; tüm düzenlemeler burada yapılacaktır.

"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError

# ──────────────────────────────────────────────────────────────────────────────
# Ortak Yapılar
# ──────────────────────────────────────────────────────────────────────────────

class ToolError(Exception):
    def __init__(self, tool: str, message: str, *, detail: Optional[str] = None):
        super().__init__(f"[{tool}] {message}")
        self.tool = tool
        self.detail = detail


class ToolBase:
    name: str = "tool"

    def input_model(self) -> type[BaseModel]:
        raise NotImplementedError

    def output_model(self) -> type[BaseModel]:
        raise NotImplementedError

    def run(self, inputs: BaseModel, ctx: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError


# Graph/State ile daha kolay konuşmak için minimal bir context veri sınıfı
@dataclass
class RunContext:
    llm: Any | None = None
    cfg: Dict[str, Any] | None = None
    conn: Any | None = None  # DB connection (tools/db.py)
    cost: Any | None = None  # utils/cost.py
    logger: Any | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Planner (LLM-aware, beyin)
# ──────────────────────────────────────────────────────────────────────────────

PlannerIntent = Literal["sql_query", "non_sql", "exit"]

class PlannerIn(BaseModel):
    question: str = Field(..., description="Kullanıcı sorusu (ham metin)")
    schema_doc: Optional[str] = Field(None, description="Şema özeti / metadata")
    history: List[Dict[str, str]] = Field(default_factory=list, description="Konuşma geçmişi (opsiyonel)")

class PlannerStep(BaseModel):
    tool: Literal[
        "schema_retriever",
        "rag",
        "query_generator",
        "query_validator",
        "sql_executor",
        "postprocessor",
        "summarizer",
        "guardian",
    ]
    input: Dict[str, Any] = Field(default_factory=dict)

class PlannerOut(BaseModel):
    intent: PlannerIntent
    use_rag: bool = False
    plan: List[PlannerStep] = Field(default_factory=list)
    decision_rationale: Optional[str] = None


class PlannerTool(ToolBase):
    name = "planner"

    def input_model(self) -> type[BaseModel]:
        return PlannerIn

    def output_model(self) -> type[BaseModel]:
        return PlannerOut

    def _system_prompt(self, allowed_tools: List[str]) -> str:
        return (
            "SEN: Analist-Agent planlayıcısısın. Amacın, verilen soruyu çözecek en kısa ve güvenli tool sırasını seçmek.\n"
            "KURALLAR: sadece SELECT, LIMIT 1000, banned keywords yok, RAG tetikleyicileri vs.\n"
            f"KULLANABİLECEĞİN TOOL'LAR: {', '.join(allowed_tools)}\n"
            "ÇIKTI ŞEMASI: intent, use_rag, plan[{tool,input}], decision_rationale."
        )

    def run(self, inputs: PlannerIn, ctx: Dict[str, Any]) -> PlannerOut:
        try:
            llm = (ctx or {}).get("llm")
            if llm is None:
                return PlannerOut(
                    intent="sql_query",
                    use_rag=False,
                    plan=[PlannerStep(tool="schema_retriever"),
                          PlannerStep(tool="query_generator"),
                          PlannerStep(tool="query_validator"),
                          PlannerStep(tool="sql_executor"),
                          PlannerStep(tool="postprocessor"),
                          PlannerStep(tool="summarizer"),
                          PlannerStep(tool="guardian")],
                    decision_rationale="LLM yok: varsayılan güvenli plan"
                )
            allowed_tools = ["schema_retriever","rag","query_generator","query_validator","sql_executor","postprocessor","summarizer","guardian"]
            prompt = self._system_prompt(allowed_tools)
            user = f"Soru: {inputs.question}\nSchema: {inputs.schema_doc or '—'}"
            raw = llm.get_text(prompt, user)
            import json
            try:
                return PlannerOut(**json.loads(raw))
            except Exception:
                return PlannerOut(intent="sql_query", use_rag=False, plan=[PlannerStep(tool="schema_retriever"),PlannerStep(tool="query_generator"),PlannerStep(tool="query_validator"),PlannerStep(tool="sql_executor"),PlannerStep(tool="postprocessor"),PlannerStep(tool="summarizer"),PlannerStep(tool="guardian")], decision_rationale="LLM çıktısı parse edilemedi")
        except Exception as e:
            raise ToolError(self.name, "Planner çalıştırma hatası", detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Gerçek Tool Sınıfları (I/O şemaları + run implementasyonları)
# ──────────────────────────────────────────────────────────────────────────────

# Ortak boş şema
class _Empty(BaseModel):
    pass

# --- Schema Retriever ---------------------------------------------------------
class SchemaRetrieverIn(_Empty):
    pass

class SchemaRetrieverOut(BaseModel):
    schema_doc: str

class SchemaRetrieverTool(ToolBase):
    name = "schema_retriever"

    def input_model(self):
        return SchemaRetrieverIn

    def output_model(self):
        return SchemaRetrieverOut

    def run(self, inputs: SchemaRetrieverIn, ctx: Dict[str, Any]) -> SchemaRetrieverOut:
        try:
            from nodes import schema_retriever as n_schema
            state = ctx.get("state")
            conn = ctx.get("conn")
            if state is None or conn is None:
                raise ToolError(self.name, "state/conn eksik")
            state = n_schema.run(conn, state)
            return SchemaRetrieverOut(schema_doc=state.schema_doc or "")
        except Exception as e:
            raise ToolError(self.name, "Schema retriever hatası", detail=str(e))

# --- RAG ----------------------------------------------------------------------
class RAGIn(BaseModel):
    enable: bool = True
    top_k: int = 5
    min_score: float = 0.1
    prefer: Literal["hybrid", "tfidf"] = "hybrid"

class RAGOut(BaseModel):
    rag_snippets: List[str]

class RAGTool(ToolBase):
    name = "rag"

    def input_model(self):
        return RAGIn

    def output_model(self):
        return RAGOut

    def run(self, inputs: RAGIn, ctx: Dict[str, Any]) -> RAGOut:
        try:
            from tools.rag import get_rag
            state = ctx.get("state")
            if state is None:
                raise ToolError(self.name, "state eksik")
            if not inputs.enable:
                state.rag_snippets = []
                return RAGOut(rag_snippets=[])
            schema = state.schema_doc or ""
            docs = [ln for ln in schema.splitlines() if ln.strip()]
            rag = get_rag(docs, prefer=inputs.prefer)
            results = rag.query(state.question or "", top_k=inputs.top_k, min_score=inputs.min_score)
            snippets = [txt for score, txt in results]
            state.rag_snippets = snippets
            return RAGOut(rag_snippets=snippets)
        except Exception as e:
            raise ToolError(self.name, "RAG çalıştırma hatası", detail=str(e))

# --- Query Generator ----------------------------------------------------------
class QueryGeneratorIn(BaseModel):
    allowed_tables: List[str] = Field(default_factory=list)
    max_limit: int = 1000

class QueryGeneratorOut(BaseModel):
    candidate_sql: List[str]

class QueryGeneratorTool(ToolBase):
    name = "query_generator"

    def input_model(self):
        return QueryGeneratorIn

    def output_model(self):
        return QueryGeneratorOut

    def run(self, inputs: QueryGeneratorIn, ctx: Dict[str, Any]) -> QueryGeneratorOut:
        try:
            from nodes import query_generator as n_qgen
            state = ctx.get("state")
            cost = ctx.get("cost")
            llm = ctx.get("llm")
            if state is None or llm is None:
                raise ToolError(self.name, "state/llm eksik")
            cfg = (ctx or {}).get("cfg") or {}
            allowed = inputs.allowed_tables or ((cfg.get("security") or {}).get("allowed_tables") or [])

            state = n_qgen.run(
                state,
                cost,
                llm,
                max_limit=inputs.max_limit,
                allowed_tables=allowed,
            )

            return QueryGeneratorOut(candidate_sql=state.candidate_sql or [])
        except Exception as e:
            raise ToolError(self.name, "Query generator hatası", detail=str(e))

# --- Query Validator ----------------------------------------------------------
class QueryValidatorIn(BaseModel):
    banned_keywords: List[str] = Field(default_factory=lambda: [
        "drop", "alter", "delete", "update", "insert", "attach", "pragma", "vacuum", "create", "replace"
    ])
    enforce_select_only: bool = True
    allow_multiple: bool = False  # geriye dönük imza
    max_limit: int = 1000         # geriye dönük imza
    allowed_tables: List[str] = Field(default_factory=list)

class QueryValidatorOut(BaseModel):
    ok: bool
    reason: Optional[str] = None
    validated_sql: Optional[str] = None

class QueryValidatorTool(ToolBase):
    name = "query_validator"

    def input_model(self):
        return QueryValidatorIn

    def output_model(self):
        return QueryValidatorOut

    def run(self, inputs: QueryValidatorIn, ctx: Dict[str, Any]) -> QueryValidatorOut:
        try:
            from nodes import query_validator as n_val
            state = ctx.get("state")
            conn = ctx.get("conn")
            llm = ctx.get("llm")
            cost = ctx.get("cost")
            if state is None or conn is None:
                raise ToolError(self.name, "state/conn eksik")
            state = n_val.run(
                conn,
                state,
                banned_keywords=inputs.banned_keywords,
                enforce_select_only=inputs.enforce_select_only,
                allow_multiple=inputs.allow_multiple,
                max_limit=inputs.max_limit,
                llm_service=llm,
                cost=cost,
                allowed_tables=inputs.allowed_tables,
            )
            rep = state.validation_report or {}
            return QueryValidatorOut(ok=bool(rep.get("ok")), reason=rep.get("reason"), validated_sql=state.validated_sql)
        except Exception as e:
            raise ToolError(self.name, "Query validator hatası", detail=str(e))

# --- SQL Executor -------------------------------------------------------------
class SQLExecutorIn(BaseModel):
    preview_rows: int = 50

class SQLExecutorOut(BaseModel):
    rows_preview: List[Dict[str, Any]] = Field(default_factory=list)
    execution_stats: Dict[str, Any] = Field(default_factory=dict)

class SQLExecutorTool(ToolBase):
    name = "sql_executor"

    def input_model(self):
        return SQLExecutorIn

    def output_model(self):
        return SQLExecutorOut

    def run(self, inputs: SQLExecutorIn, ctx: Dict[str, Any]) -> SQLExecutorOut:
        try:
            from nodes import sql_executor as n_exec
            state = ctx.get("state")
            conn = ctx.get("conn")
            if state is None or conn is None:
                raise ToolError(self.name, "state/conn eksik")
            state = n_exec.run(conn, state, preview_rows=inputs.preview_rows)
            return SQLExecutorOut(rows_preview=state.rows_preview or [], execution_stats=state.execution_stats or {})
        except Exception as e:
            raise ToolError(self.name, "SQL executor hatası", detail=str(e))

# --- Postprocessor ------------------------------------------------------------
class PostprocessorIn(_Empty):
    pass

class PostprocessorOut(BaseModel):
    rows_preview: List[Dict[str, Any]] = Field(default_factory=list)
    execution_stats: Dict[str, Any] = Field(default_factory=dict)

class PostprocessorTool(ToolBase):
    name = "postprocessor"

    def input_model(self):
        return PostprocessorIn

    def output_model(self):
        return PostprocessorOut

    def run(self, inputs: PostprocessorIn, ctx: Dict[str, Any]) -> PostprocessorOut:
        try:
            from nodes import postprocessor as n_post
            state = ctx.get("state")
            conn = ctx.get("conn")
            if state is None or conn is None:
                raise ToolError(self.name, "state/conn eksik")
            state = n_post.run(state, conn)
            return PostprocessorOut(rows_preview=state.rows_preview or [], execution_stats=state.execution_stats or {})
        except Exception as e:
            raise ToolError(self.name, "Postprocessor hatası", detail=str(e))

# --- Summarizer ---------------------------------------------------------------
class SummarizerIn(BaseModel):
    show_sql: bool = False

class SummarizerOut(BaseModel):
    answer_text: str

class SummarizerTool(ToolBase):
    name = "summarizer"

    def input_model(self):
        return SummarizerIn

    def output_model(self):
        return SummarizerOut

    def run(self, inputs: SummarizerIn, ctx: Dict[str, Any]) -> SummarizerOut:
        try:
            from nodes import summarizer as n_sum
            state = ctx.get("state")
            cost = ctx.get("cost")
            llm = ctx.get("llm")
            if state is None or llm is None:
                raise ToolError(self.name, "state/llm eksik")
            state = n_sum.run(state, cost, inputs.show_sql, llm)
            return SummarizerOut(answer_text=state.answer_text or "")
        except Exception as e:
            raise ToolError(self.name, "Summarizer hatası", detail=str(e))

# --- Guardian (passthrough / basit kontrol) ----------------------------------
class GuardianIn(_Empty):
    pass

class GuardianOut(BaseModel):
    ok: bool = True
    reason: Optional[str] = None

class GuardianTool(ToolBase):
    name = "guardian"

    def input_model(self):
        return GuardianIn

    def output_model(self):
        return GuardianOut

    def run(self, inputs: GuardianIn, ctx: Dict[str, Any]) -> GuardianOut:
        # Basit koruma: validator/exec raporlarına bakarak ok=false döndür
        state = ctx.get("state")
        if state is None:
            return GuardianOut(ok=False, reason="state eksik")
        if state.validation_report and not state.validation_report.get("ok", True):
            return GuardianOut(ok=False, reason=str(state.validation_report.get("reason")))
        if state.execution_stats and not state.execution_stats.get("ok", True):
            return GuardianOut(ok=False, reason=str(state.execution_stats.get("reason")))
        return GuardianOut(ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# TOOL_REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, ToolBase] = {
    "planner": PlannerTool(),
    "schema_retriever": SchemaRetrieverTool(),
    "rag": RAGTool(),
    "query_generator": QueryGeneratorTool(),
    "query_validator": QueryValidatorTool(),
    "sql_executor": SQLExecutorTool(),
    "postprocessor": PostprocessorTool(),
    "summarizer": SummarizerTool(),
    "guardian": GuardianTool(),
}

# ──────────────────────────────────────────────────────────────────────────────
# Yardımcı: tool çalıştır (Graph tarafı için)
# ──────────────────────────────────────────────────────────────────────────────

def run_tool(tool_name: str, payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise ToolError("dispatcher", f"Bilinmeyen tool: {tool_name}")
    tool = TOOL_REGISTRY[tool_name]
    InModel = tool.input_model()
    OutModel = tool.output_model()
    try:
        inputs = InModel(**payload)
    except ValidationError as ve:
        raise ToolError(tool_name, "Girdi doğrulama hatası", detail=str(ve))
    out = tool.run(inputs, ctx)
    if not isinstance(out, OutModel):
        out = OutModel(**out.dict() if hasattr(out, "dict") else dict(out))
    return out.dict()


# ──────────────────────────────────────────────────────────────────────────────
# LangChain Adapter — TOOL_REGISTRY → StructuredTool
# ve @tool (decorator) eşdeğeri için yardımcılar
# ──────────────────────────────────────────────────────────────────────────────

def as_langchain_tools(selected: list[str] | None = None, ctx: Dict[str, Any] | None = None):
    from langchain_core.tools import StructuredTool
    items = []
    for name, tool in TOOL_REGISTRY.items():
        if selected and name not in selected:
            continue
        InModel = tool.input_model()

        def _make_func(t):
            def _fn(**kwargs):
                return t.run(InModel(**kwargs), ctx or {}).dict()
            return _fn

        fn = _make_func(tool)
        fn.__name__ = name
        fn.__doc__ = f"{name} tool"
        items.append(
            StructuredTool.from_function(
                fn,
                name=name,
                description=fn.__doc__ or name,
                args_schema=InModel,
                return_direct=False,
            )
        )
    return items


def as_langchain_decorated_tools(selected: list[str] | None = None, ctx: Dict[str, Any] | None = None):
    from langchain_core.tools import tool as tool_decorator
    wrapped = []
    for name, tool in TOOL_REGISTRY.items():
        if selected and name not in selected:
            continue
        InModel = tool.input_model()
        desc = f"{name} tool"

        def _make_callable(_tool: ToolBase, _InModel: type[BaseModel]):
            def _impl(**kwargs):
                return _tool.run(_InModel(**kwargs), ctx or {}).dict()
            _impl.__name__ = name
            _impl.__doc__ = desc
            return _impl

        call_fn = _make_callable(tool, InModel)
        wrapped.append(tool_decorator(call_fn, args_schema=InModel, name=name, description=desc))
    return wrapped
