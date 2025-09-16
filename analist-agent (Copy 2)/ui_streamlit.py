"""
ui_streamlit.py — Analist-Agent Streamlit UI (tool-calling aware)

Özellikler:
- Tek satır metin alanı + "Çalıştır" düğmesi
- Planner çıktısını (tool-aware plan) timeline olarak gösterir
- Sorgu çalıştıktan sonra: cevap, tablo önizleme, istatistikler
- İzlenebilirlik: trace içeriğini düğüm bazında listeler
- Seçenekler: SQL’i göster, RAG snippet’lerini göster

Not: Bu dosya ui_streamlit.py olarak kaydedilmeli. Tool modülü ayrı olarak agent_tools.py adıyla tutuluyor.
"""
from __future__ import annotations

import os
import yaml
import time
import json
import logging
import streamlit as st
from typing import Any, Dict, List

from utils.types import AgentState
from utils.llm import LLMService
from utils.cost import CostTracker
from tools.db import connect_readonly
from graph import build_graph

# Tool registry ve plan tipleri için import (patch kısmında kullanıyoruz)
try:
    from agent_tools import TOOL_REGISTRY, PlannerOut, PlannerStep
except Exception:  # agent_tools henüz import edilemiyorsa UI yine de çalışsın
    TOOL_REGISTRY, PlannerOut, PlannerStep = {}, None, None

log = logging.getLogger("ui")

# ──────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("ANALIST_AGENT_CONFIG", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner=False)
def _init_services(cfg: Dict[str, Any]):
    # LLMService için bilinen parametreler ve ekstra model_kwargs ayrıştırması
    raw_llm_cfg = dict(cfg.get("llm", {}) or {})

    known_keys = {
        "provider", "model", "model_name", "api_key", "base_url", "temperature", "max_tokens",
        "timeout", "top_p", "top_k", "stream", "retry", "max_retries"
    }

    # 1) Bilinen anahtarları doğrudan al
    llm_init = {k: raw_llm_cfg.pop(k) for k in list(raw_llm_cfg.keys()) if k in known_keys}

    # 2) model_name → model (geriye dönük uyumluluk)
    if "model" not in llm_init and "model_name" in llm_init:
        llm_init["model"] = llm_init.pop("model_name")

    # 3) Kalan anahtarları model_kwargs içine topla ve olası çakışmaları temizle
    if raw_llm_cfg:
        raw_llm_cfg.pop("model", None)
        raw_llm_cfg.pop("model_name", None)
        llm_init["model_kwargs"] = raw_llm_cfg

    # 4) LLMService — utils/llm.py zaten son savunmayı da yapıyor (model/model_name normalize + pop)
    llm = LLMService(**llm_init)

    # connect_readonly beklenmeyen parametreler alabilir; temizle
    db_cfg = dict(cfg.get("db", {}) or {})
    db_cfg.pop("read_only", None)
    conn = connect_readonly(**db_cfg)

    cost = CostTracker()
    return llm, conn, cost


def _build_ctx(llm, cfg, conn, cost) -> Dict[str, Any]:
    return {"llm": llm, "cfg": cfg, "conn": conn, "cost": cost, "logger": log}


# ──────────────────────────────────────────────────────────────────────────────
# UI — Başlık / Ayarlar
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Analist-Agent", page_icon="📊", layout="wide")
st.title("Analist‑Agent · Tool‑Calling UI")

cfg = _load_config()
llm, conn, cost = _init_services(cfg)

# graph.py içinde tool-calling orkestrasyonunu koruyoruz
graph = build_graph(conn, cfg, cost, llm, logger=log)

with st.sidebar:
    st.subheader("Ayarlar")
    show_sql = st.checkbox("SQL’i göster", value=True)
    show_rag = st.checkbox("RAG snippet’lerini göster", value=False)
    st.caption("Config yolu: " + os.environ.get("ANALIST_AGENT_CONFIG", "config.yaml"))

# ──────────────────────────────────────────────────────────────────────────────
# Soru alanı
# ──────────────────────────────────────────────────────────────────────────────

col_q, col_run = st.columns([0.8, 0.2])
with col_q:
    question = st.text_input("Soru", placeholder="Hangi birim en fazla kullanıcıya sahiptir? …")
with col_run:
    run_btn = st.button("Çalıştır", type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# Çalıştırma
# ──────────────────────────────────────────────────────────────────────────────

if run_btn and question.strip():
    state = AgentState(question=question)

    t0 = time.time()
    try:
        out: AgentState = graph.invoke(state)
        invoke_error = None
    except Exception as e:
        out, invoke_error = None, e
    elapsed_ms = (time.time() - t0) * 1000

    # Sonuç kutusu
    if invoke_error:
        st.error(f"Hata: {invoke_error}")
    elif getattr(out, "error", None):
        st.error(f"Hata: {out.error}")
    else:
        st.success("✅ Tamamlandı")

    # Cevap / Özet
    with st.container(border=True):
        st.subheader("📝 Cevap")
        answer = None
        if out is not None:
            # farklı alan adlarını destekle (answer / answer_text)
            answer = getattr(out, "answer", None) or getattr(out, "answer_text", None)
        st.write(answer or "(boş)")
        st.caption(f"⏱ {elapsed_ms:.1f} ms")

    # Planner Plan Timeline
    def _as_dict(step) -> Dict[str, Any]:
        if isinstance(step, dict):
            return step
        # PlannerStep pydantic modeli ise
        try:
            return step.dict()
        except Exception:
            return {"tool": str(step), "input": {}}

    if out is not None and getattr(out, "plan", None):
        plan_detail = getattr(out, "plan_detail", None)
        plan_list   = getattr(out, "plan", None)

        if plan_detail:
            with st.expander("🧭 Plan (Tool-aware)", expanded=True):
                for i, step in enumerate(plan_detail, 1):
                    st.markdown(f"**{i}. {step.get('tool')}**")
                    args = step.get("input") or {}
                    if args:
                        st.code(json.dumps(args, ensure_ascii=False, indent=2), language="json")
        elif plan_list:
            with st.expander("🧭 Plan (Tool-aware)", expanded=True):
                for i, tool in enumerate(plan_list, 1):
                    st.markdown(f"**{i}. {tool}**")
        else:
            st.info("Plan mevcut değil veya planner devre dışı.")

            for i, step in enumerate(out.plan, start=1):
                d = _as_dict(step)
                tool = d.get("tool")
                args = d.get("input", {})
                st.markdown(f"**{i}. {tool}**  ")
                if args:
                    st.code(json.dumps(args, ensure_ascii=False, indent=2), language="json")
    else:
        st.info("Plan mevcut değil veya planner devre dışı.")

    # SQL — Opsiyonel göster
    if show_sql:
        sql = None
        if out is not None:
            try:
                # öncelik: validated_sql → candidate_sql[0]
                sql = getattr(out, "validated_sql", None)
                if not sql and getattr(out, "candidate_sql", None):
                    sql = (out.candidate_sql or [None])[0]
            except Exception:
                sql = None
        with st.expander("🧩 SQL", expanded=bool(sql)):
            st.code(sql or "-- SQL üretilmedi", language="sql")

    # RAG Snippet’leri (opsiyonel)
    if show_rag:
        with st.expander("🔎 RAG Snippet’leri", expanded=False):
            snippets: List[str] = []
            if out is not None:
                snippets = getattr(out, "rag_snippets", []) or []
            if snippets:
                for s in snippets:
                    st.markdown(f"- {s}")
            else:
                st.caption("RAG kullanılmadı veya snippet yok.")

    # Tablo önizleme + basit istatistikler
    preview = []
    stats = {}
    if out is not None:
        # farklı alan adları için tolerans: rows_preview / _table_preview, execution_stats / _stats
        preview = getattr(out, "rows_preview", None) or getattr(out, "_table_preview", []) or []
        stats = getattr(out, "execution_stats", None) or getattr(out, "_stats", {}) or {}
    if preview:
        st.subheader("📊 Tablo Önizleme")
        st.dataframe(preview, hide_index=True, use_container_width=True)
    if stats:
        with st.expander("📈 İstatistikler", expanded=False):
            st.json(stats)

    # Trace / Log
    trace = getattr(out, "trace", None) if out is not None else None
    if isinstance(trace, list) and trace:
        with st.expander("🔬 Çalışma İzleri (Trace)", expanded=False):
            for rec in trace:
                st.write(rec)


# ──────────────────────────────────────────────────────────────────────────────
# Patch: Planner'ı hata toleranslı hale getir (LLM hatasında güvenli plana düş)
# ──────────────────────────────────────────────────────────────────────────────
try:
    if TOOL_REGISTRY and TOOL_REGISTRY.get("planner") is not None and PlannerOut is not None:
        _orig_planner_cls = TOOL_REGISTRY["planner"].__class__
        class _SafePlanner(_orig_planner_cls):  # type: ignore
            def _default_plan(self):
                return PlannerOut(
                    intent="sql_query",
                    use_rag=False,
                    plan=[
                        PlannerStep(tool="schema_retriever"),
                        PlannerStep(tool="query_generator"),
                        PlannerStep(tool="query_validator"),
                        PlannerStep(tool="sql_executor"),
                        PlannerStep(tool="postprocessor"),
                        PlannerStep(tool="summarizer"),
                        PlannerStep(tool="guardian"),
                    ],
                    decision_rationale="LLM yok/hata: varsayılan güvenli plan",
                )

            def _call_llm(self, llm, system: str, user: str) -> str:
                # 1) generate(system, user)
                if hasattr(llm, "generate"):
                    try:
                        resp = llm.generate(system, user)
                        if isinstance(resp, str):
                            return resp
                        if isinstance(resp, dict):
                            for key in ("text", "content", "message", "output"):
                                if isinstance(resp.get(key), str):
                                    return resp[key]
                    except TypeError:
                        try:
                            resp = llm.generate(system=system, user=user)
                            if isinstance(resp, str):
                                return resp
                            if isinstance(resp, dict):
                                for key in ("text", "content", "message", "output"):
                                    if isinstance(resp.get(key), str):
                                        return resp[key]
                        except Exception:
                            pass
                    except Exception:
                        pass
                # 2) chat(messages)
                if hasattr(llm, "chat"):
                    try:
                        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                        resp = llm.chat(messages)
                        if isinstance(resp, str):
                            return resp
                        if isinstance(resp, dict):
                            for key in ("text", "content", "message"):
                                if isinstance(resp.get(key), str):
                                    return resp[key]
                    except Exception:
                        pass
                # 3) invoke(messages)
                if hasattr(llm, "invoke"):
                    try:
                        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                        resp = llm.invoke(messages)
                        if isinstance(resp, str):
                            return resp
                        if isinstance(resp, dict):
                            for key in ("text", "content"):
                                if isinstance(resp.get(key), str):
                                    return resp[key]
                    except Exception:
                        pass
                return "{}"

            def run(self, inputs, ctx):  # type: ignore[override]
                llm = (ctx or {}).get("llm")
                if llm is None:
                    return self._default_plan()
                try:
                    allowed_tools = [
                        "schema_retriever","rag","query_generator","query_validator",
                        "sql_executor","postprocessor","summarizer","guardian",
                    ]
                    prompt = self._system_prompt(allowed_tools)
                    schema_preview = (inputs.schema_doc[:500] + "...") if getattr(inputs, "schema_doc", None) else "—"
                    user = (
                        "Soru: " + inputs.question + "\n" +
                        "Schema kısmi: " + schema_preview + "\n" +
                        "Planı JSON olarak ver."
                    )
                    raw = self._call_llm(llm, prompt, user)
                    try:
                        data = json.loads(raw)
                        return PlannerOut(**data)
                    except Exception:
                        return self._default_plan()
                except Exception:
                    return self._default_plan()
        # Registry'de sınıfı değiştir
        TOOL_REGISTRY["planner"] = _SafePlanner()
except Exception:
    pass
