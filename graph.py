# graph.py — Analist AI Agent (katmanlı tool-calling akışı, Pylance-dostu opsiyonel importlar)
from __future__ import annotations

import operator
import logging
from typing import Annotated, List, Optional

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

log = logging.getLogger("graph")


# ──────────────────────────────────────────────
# Esnek importlar (repo düzeni farklarına dayanıklı)
# ──────────────────────────────────────────────
def _import_db_module():
    try:
        from tools.db import create_database_tools  # type: ignore
        return create_database_tools
    except Exception:
        from db import create_database_tools  # type: ignore
        return create_database_tools

def _import_rag_module():
    try:
        from tools.rag import schema_search, initialize_rag  # type: ignore
        return schema_search, initialize_rag
    except Exception:
        try:
            from rag import schema_search, initialize_rag  # type: ignore
            return schema_search, initialize_rag
        except Exception:
            return None, None

def _import_llm_service():
    try:
        from utils.llm import LLMService  # type: ignore
        return LLMService
    except Exception:
        from llm import LLMService  # type: ignore
        return LLMService

def _try_import_optional_tools():
    """
    İsteğe bağlı tool seti:
      - schema_intel: load_schema_intel, list_tables, describe_table, find_join_path, profile_column
      - sql_validator: init_sql_validator, validate_sql, explain_sql
      - result_utils: tabulate_result, suggest_chart

    Pylance uyarılarını engellemek için sadece VAR olan modülleri import eder.
    """
    import importlib
    import importlib.util

    def _maybe(module_path: str, names: list[str]):
        """module_path varsa import eder ve istenen isimleri döndürür, yoksa None döndürür."""
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            return {n: None for n in names}
        mod = importlib.import_module(module_path)
        return {n: getattr(mod, n, None) for n in names}

    # İlk tercih: tools.* (paket doğruysa burası çalışır)
    schema_intel = _maybe(
        "tools.schema_intel",
        ["load_schema_intel", "list_tables", "describe_table", "find_join_path", "profile_column"],
    )
    sql_validator = _maybe(
        "tools.sql_validator",
        ["init_sql_validator", "validate_sql", "explain_sql"],
    )
    result_utils = _maybe(
        "tools.result_utils",
        ["tabulate_result", "suggest_chart"],
    )
    planner = _maybe(
    "tools.planner",
    ["init_planner", "plan_query"],
    )
    summarizer = _maybe(
        "tools.summarizer",
        ["summarize_answer"],
    )

    paginator = _maybe(
    "tools.paginator",
    ["init_paginator", "paginate_sql"],
    )
    qcache = _maybe(
        "tools.query_cache",
        ["init_query_cache", "cache_get", "cache_put"],
    )

    safe_sql = _maybe(
    "tools.safe_sql",
    ["init_safe_sql", "safe_run_sql"],
    )
    repair_mod = _maybe(
    "tools.repair_sql",
    ["init_repair_sql", "repair_sql"],  
    )
    # ─── FALLBACK: tools.* bulunamazsa proje kökünden import dene ───
    if all(v is None for v in schema_intel.values()):
        schema_intel = _maybe(
            "schema_intel",
            ["load_schema_intel", "list_tables", "describe_table", "find_join_path", "profile_column"],
        )

    if all(v is None for v in sql_validator.values()):
        sql_validator = _maybe(
            "sql_validator",
            ["init_sql_validator", "validate_sql", "explain_sql"],
        )

    if all(v is None for v in result_utils.values()):
        result_utils = _maybe(
            "result_utils",
            ["tabulate_result", "suggest_chart"],
        )

    if all(v is None for v in safe_sql.values()):
        safe_sql = _maybe(
            "safe_sql",
            ["init_safe_sql", "safe_run_sql"],
        )

    if all(v is None for v in repair_mod.values()):
        repair_mod = _maybe(
            "repair_sql",
            ["init_repair_sql", "repair_sql"],
        )

    if all(v is None for v in paginator.values()):
        paginator = _maybe(
            "paginator",
            ["init_paginator", "paginate_sql"],
        )

    if all(v is None for v in qcache.values()):
        qcache = _maybe(
            "query_cache",
            ["init_query_cache", "cache_get", "cache_put"],
        )

    if all(v is None for v in planner.values()):
        planner = _maybe(
            "planner",
            ["init_planner", "plan_query"],
        )

    if all(v is None for v in summarizer.values()):
        summarizer = _maybe(
            "summarizer",
            ["summarize_answer"],
        )
    # Tek bir sözlük döndür
    return {
        **schema_intel, **sql_validator, **result_utils,
        **safe_sql, **repair_mod,
        **paginator, **qcache,
        **planner, **summarizer,
    }



# ──────────────────────────────────────────────
# Agent State
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    # LangGraph "reduce" operatörü ile mesajları biriktiririz
    messages: Annotated[List[BaseMessage], operator.add]


# ──────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────
def _safe_tool_invoke(tool, payload=""):
    """
    LangChain tool objesini programatik çağırmak için esnek yardımcı:
      - .invoke({...}) varsa onu dener
      - yoksa .run(str) dener
    """
    try:
        if hasattr(tool, "invoke"):
            return tool.invoke(payload if isinstance(payload, dict) else {"input": payload})
        return tool.run(payload)
    except Exception as e:
        log.warning(f"Tool invoke failed ({getattr(tool,'name',tool)}): {e}")
        return None


def _build_system_prompt(cfg: dict) -> str:
    mode = cfg.get("runtime", {}).get("mode", "guarded")
    common = (
        "Sen bir 'Analist AI Agent'sin. Araçları gerektiğinde kullan; "
        "yanlış tablo/kolon kullanma; açıklanabilir, kısa ve doğru özet ver.\n"
    )
    if mode == "free":
        rules = (
            "ALTIN KURAL:\n"
            "• Sadece şemadaki tablo/sütunları kullan.\n"
            "• Gerek gördüğünde plan_query/validate_sql/safe_run_sql/paginate_sql/summarize_answer çağır.\n"
        )
    else:  # guarded
        rules = (
            "ALTIN KURAL:\n"
            "0) Gerekirse önce 'plan_query' ile QuerySpec çıkar.\n"
            "1) Sadece mevcut tablo/sütunları kullan.\n"
            "2) Emin değilsen 'describe_table'/'schema_search' çağır.\n"
            "3) JOIN yolunu 'find_join_path' ile kontrol et.\n"
            "4) Çalıştırmadan 'validate_sql'; hata varsa 'repair_sql'.\n"
            "5) Çalıştırırken öncelik 'safe_run_sql'.\n"
            "6) Büyük sonuçlarda 'paginate_sql'; tekrar eden SQL’de 'cache_get/put'.\n"
            "7) Sunumda 'tabulate_result' + 'summarize_answer'.\n"
        )
    return common + "\n" + rules

# ──────────────────────────────────────────────
# build_graph
# ──────────────────────────────────────────────
def build_graph(cfg: dict):
    """
    cfg beklenen alanlar (esnek):
      cfg['db']['path']: SQLite dosya yolu
      cfg['llm']: { 'provider', 'model_name', 'temperature', 'api_key' (opsiyonel) }
      cfg['rag']['enabled']: bool
    """
    # 0) Esnek importları hazırla
    create_database_tools = _import_db_module()
    schema_search, initialize_rag = _import_rag_module()
    LLMService = _import_llm_service()
    optional = _try_import_optional_tools()

    db_path: Optional[str] = None
    try:
        db_path = cfg.get("db", {}).get("path")
    except Exception:
        pass
    if not db_path:
        raise ValueError("cfg['db']['path'] bulunamadı (SQLite yolu gerekli).")

    # 1) DB araçlarını kur
    db_tools = create_database_tools(db_path)  # [run_sql, get_schema]

    # 2) Ham şemayı çekip (opsiyonel) RAG'i init et
    try:
        get_schema_tool = next((t for t in db_tools if getattr(t, "name", "") == "get_schema"), None)
        raw_schema = None
        if get_schema_tool:
            raw_schema = _safe_tool_invoke(get_schema_tool, {})
        else:
            log.warning("get_schema aracı bulunamadı; RAG devre dışı kalacak.")
        if raw_schema and initialize_rag and cfg.get("rag", {}).get("enabled", True):
            try:
                initialize_rag([str(raw_schema)])
                log.info("RAG initialize edildi (şema beslemesi yapıldı).")
            except Exception as e:
                log.warning(f"RAG initialize başarısız: {e}")
    except Exception as e:
        log.warning(f"Şema/RAG hazırlığında sorun: {e}")

    # 3) Opsiyonel şema zekası + SQL doğrulayıcıyı başlat
    if optional.get("load_schema_intel"):
        try:
            optional["load_schema_intel"](db_path)
            log.info("SchemaIntel yüklendi.")
        except Exception as e:
            log.warning(f"SchemaIntel yüklenemedi: {e}")
    if optional.get("init_sql_validator"):
        try:
            optional["init_sql_validator"](db_path)
            log.info("SQL Validator başlatıldı.")
        except Exception as e:
            log.warning(f"SQL Validator başlatılamadı: {e}")
        # Safe SQL ve Repair init
    # Paginator & Query Cache init
    # Planner init
    if optional.get("init_planner"):
        try:
            optional["init_planner"](db_path)
            log.info("planner initialized.")
        except Exception as e:
            log.warning(f"planner init failed: {e}")

    if optional.get("init_paginator"):
        try:
            optional["init_paginator"](db_path)
            log.info("paginator initialized.")
        except Exception as e:
            log.warning(f"paginator init failed: {e}")
    if optional.get("init_query_cache"):
        try:
            # .cache.sqlite proje kökünde dosya açar
            optional["init_query_cache"](".cache.sqlite")
            log.info("query cache initialized.")
        except Exception as e:
            log.warning(f"query cache init failed: {e}")


    if optional.get("init_repair_sql"):
        try:
            optional["init_repair_sql"](db_path)
            log.info("repair_sql initialized.")
        except Exception as e:
            log.warning(f"repair_sql init failed: {e}")

    if optional.get("init_safe_sql"):
        try:
            optional["init_safe_sql"](db_path)
            log.info("safe_sql initialized.")
        except Exception as e:
            log.warning(f"safe_sql init failed: {e}")

    # 4) Tool setini birleştir
    tools = []
    tools += db_tools
    if schema_search:
        tools += [schema_search]
    for k in ("list_tables", "describe_table", "find_join_path", "profile_column",
          "validate_sql", "explain_sql", "tabulate_result", "suggest_chart",
          "safe_run_sql", "repair_sql",
          "paginate_sql", "cache_get", "cache_put",
          "plan_query", "summarize_answer"):
        if optional.get(k):
            tools.append(optional[k])

    # 5) LLM'i hazırla ve tool'lara bağla
    llm_cfg = cfg.get("llm", {})
    provider_priority = llm_cfg.get("provider_priority") or [llm_cfg.get("provider", "gemini")]
    temperature = float(llm_cfg.get("temperature", 0.1))

    system_prompt = _build_system_prompt(cfg)
    llm_service = LLMService(
        # eski alanlar desteklenmeye devam ediyor
        provider=llm_cfg.get("provider"),
        model_name=llm_cfg.get("model_name"),
        api_key=llm_cfg.get("api_key"),
        temperature=temperature,
        tools=tools,
        # yeni: fallback yapılandırma
        provider_priority=provider_priority,
        provider_configs=llm_cfg,
    )
    agent_llm = llm_service.llm


    # 6) Ajan ve Tool düğümleri
    def agent_node(state: AgentState):
        # Her tur başında sistem mesajını sadece ilk turda eklemek için kontrol:
        msgs: List[BaseMessage] = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=system_prompt)] + msgs
        resp = agent_llm.invoke(msgs)
        return {"messages": [resp]}

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        # LangChain AIMessage.tool_calls → varsa tool çağırmaya devam
        tc = getattr(last, "tool_calls", None)
        if tc:
            return "continue"
        return "end"

    # 7) Grafı ör
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("action", tool_node)

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
    g.add_edge("action", "agent")

    return g.compile()
