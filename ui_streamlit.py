# ui_streamlit.py
import os, io, csv, json
import streamlit as st
import yaml
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph import build_graph

# Opsiyonel: paginator ve telemetry mevcutsa kullan
try:
    from tools.paginator import init_paginator, paginate_sql  # type: ignore
    PAGER_AVAILABLE = True
except Exception:
    PAGER_AVAILABLE = False

try:
    from utils import telemetry
    TELEMETRY_AVAILABLE = True
except Exception:
    TELEMETRY_AVAILABLE = False

# ─────────────────────────────────────────────
# Sayfa yapılandırması
# ─────────────────────────────────────────────
st.set_page_config(page_title="Analist AI Agent", layout="wide")
st.title("Analist AI Agent 🕵️‍♂️")

# ─────────────────────────────────────────────
# Agent & config (tek sefer cache)
# ─────────────────────────────────────────────
@st.cache_resource
def load_agent_and_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
            # OpenRouter/HF anahtarlarını ortama bas (boşsa dokunma)
    import os
    or_key = ((cfg.get("llm", {}) or {}).get("openrouter", {}) or {}).get("api_key")
    if or_key and not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = or_key

    hf_key = ((cfg.get("llm", {}) or {}).get("hf", {}) or {}).get("token")
    if hf_key and not os.getenv("HUGGINGFACE_API_TOKEN"):
        os.environ["HUGGINGFACE_API_TOKEN"] = hf_key

    app = build_graph(cfg)
    # paginator'ı doğrudan kullanacaksak init et
    if PAGER_AVAILABLE:
        try:
            db_path = cfg.get("db", {}).get("path")
            if db_path:
                init_paginator(db_path)
        except Exception:
            pass
    return app, cfg

graph_app, app_cfg = load_agent_and_config()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
ss = st.session_state
ss.setdefault("messages", [])          # Human/AI message listesi
ss.setdefault("last_sql", "")          # final SQL
ss.setdefault("last_result", None)     # {"columns":[...], "rows":[...]}
ss.setdefault("last_plan_json", "")    # plan_query çıktısı (string JSON)
ss.setdefault("last_attempts", [])     # safe_run_sql attempts
ss.setdefault("tool_trace", [])        # canlı tool logları (UI için)

# ─────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────
def _safe_json_load(x: str):
    try:
        return True, json.loads(x)
    except Exception:
        try:
            return True, json.loads(x.strip().removeprefix("```json").removesuffix("```").strip())
        except Exception:
            return False, None

def _download_buttons(result_json: dict | None, sql_text: str | None):
    c1, c2, c3 = st.columns(3)
    if sql_text:
        with c1:
            st.markdown("**Kullanılan SQL**")
            st.code(sql_text, language="sql")
            st.caption("Kopyalamak için metni seçip Ctrl/Cmd + C yapabilirsin.")
    if result_json:
        cols = result_json.get("columns", [])
        rows = result_json.get("rows", [])
        # CSV
        csv_buf = io.StringIO()
        w = csv.writer(csv_buf)
        if cols: w.writerow(cols)
        w.writerows(rows)
        with c2:
            st.download_button("⬇️ CSV indir", data=csv_buf.getvalue(),
                               file_name="query_result.csv", mime="text/csv", use_container_width=True)
        # JSON
        with c3:
            st.download_button("⬇️ JSON indir", data=json.dumps(result_json, ensure_ascii=False),
                               file_name="query_result.json", mime="application/json", use_container_width=True)

def _render_plan_attempts(tool_data: dict):
    with st.expander("🧭 Plan & 🔧 Attempts", expanded=False):
        colA, colB = st.columns(2)
        # Plan
        with colA:
            plan = tool_data.get("plan_query")
            if plan:
                st.subheader("Plan (QuerySpec)")
                st.json(plan, expanded=False)
                jp = plan.get("join_path") or []
                dims = plan.get("dimensions") or []
                mets = plan.get("metrics") or []
                if jp:
                    st.caption("Join path: " + " → ".join(jp))
                if dims:
                    st.caption("Dimensions: " + ", ".join(map(str, dims[:5])))
                if mets:
                    st.caption("Metrics: " + ", ".join([m.get("alias") or m.get("expr") for m in mets[:5]]))
            else:
                st.caption("Plan bilgisi yok (model çağırmamış olabilir).")
        # Attempts (safe_run_sql)
        with colB:
            safe = tool_data.get("safe_run_sql")
            if safe and "attempts" in safe:
                st.subheader("Repair Attempts")
                st.json(safe["attempts"], expanded=False)
            else:
                st.caption("Attempt bilgisi yok.")
    with st.expander("🗺 Query Plan (EXPLAIN)", expanded=False):
        plan_text = tool_data.get("explain_sql_text") or (tool_data.get("safe_run_sql") or {}).get("plan")
        if plan_text:
            st.code(str(plan_text))
        else:
            st.caption("EXPLAIN metni yok.")

def _render_pagination(sql_text: str | None):
    if not (PAGER_AVAILABLE and sql_text):
        return
    st.markdown("### 📄 Sayfalama")
    with st.form("pager_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            page = st.number_input("Sayfa", min_value=1, value=1, step=1)
        with c2:
            page_size = st.number_input("Sayfa Boyutu", min_value=5, max_value=500, value=50, step=5)
        submit = st.form_submit_button("Getir")
    if submit:
        try:
            args = {"query": sql_text, "page_size": int(page_size), "page": int(page)}
            payload = json.dumps(args, ensure_ascii=False)
            raw = paginate_sql.run(payload) if hasattr(paginate_sql, "run") else paginate_sql.invoke({"input": payload})
            ok, data = _safe_json_load(raw if isinstance(raw, str) else str(raw))
            if ok and data and data.get("columns"):
                st.success(f"Sayfa {data.get('page')} — has_more={data.get('has_more')}")
                st.dataframe(data, use_container_width=True, hide_index=True)
            else:
                st.warning("Sayfalı sonuç alınamadı.")
        except Exception as e:
            st.error(f"Sayfalama hatası: {e}")

def _render_run_history():
    if not TELEMETRY_AVAILABLE:
        return
    with st.expander("📜 Run History (last 20)", expanded=False):
        path = "logs/runs.jsonl"
        if not os.path.exists(path):
            st.caption("Henüz kayıt yok.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-20:]
        except Exception:
            lines = []
        for ln in reversed(lines):
            try:
                doc = json.loads(ln)
            except Exception:
                continue
            ok_badge = "✅" if all(s.get("ok", True) for s in doc.get("steps", [])) else "⚠️"
            repaired = any(s.get("tool") == "repair_sql" and s.get("ok") for s in doc.get("steps", []))
            st.write(f"{ok_badge} **{doc.get('ts','')}** · `{doc.get('trace_id','')}` · {doc.get('latency_ms','?')} ms · repaired: {repaired}")
            with st.popover("Detay", use_container_width=True):
                st.json(doc, expanded=False)

# ─────────────────────────────────────────────
# Geçmişi çiz
# ─────────────────────────────────────────────
for m in ss.messages:
    role = "human" if isinstance(m, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(m.content)

# ─────────────────────────────────────────────
# Girdi / Çalıştırma
# ─────────────────────────────────────────────
prompt = st.chat_input("Veritabanı hakkında bir soru sorun...")
if prompt:
    if not graph_app:
        st.warning("Agent düzgün yüklenemedi. Konsol loglarını kontrol edin.")
        st.stop()

    # Telemetry trace başlat
    if TELEMETRY_AVAILABLE:
        telemetry.configure("logs/runs.jsonl")
        telemetry.set_trace(None, prompt)

    # Kullanıcı mesajını yaz
    ss.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    # Agent çalıştır
    left, right = st.columns([0.62, 0.38], gap="large")
    tool_data, final_answer = {}, ""

    with left:
        with st.chat_message("ai"):
            status = st.status("Agent çalışıyor…", expanded=False)
            try:
                initial_state = {"messages": [HumanMessage(content=prompt)]}
                # Tool çağrılarını canlı olarak topla
                last_msgs = []
                for event in graph_app.stream(initial_state, config={"recursion_limit": 25}):
                    for node, output in event.items():
                        msgs = output.get("messages", [])
                        last_msgs = msgs or last_msgs
                        for msg in msgs:
                            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                                status.write(f"⚙️ Araç çağrısı: {msg.tool_calls}")
                            elif isinstance(msg, ToolMessage):
                                tname = getattr(msg, "name", "") or "tool"
                                content_text = str(msg.content)
                                short = (content_text[:200] + "…") if len(content_text) > 220 else content_text
                                status.write(f"📦 {tname}: {short}")
                                # Özel tool verileri
                                if tname == "plan_query":
                                    ok, data = _safe_json_load(content_text)
                                    if ok and data:
                                        tool_data["plan_query"] = data
                                        ss.last_plan_json = json.dumps(data, ensure_ascii=False)
                                elif tname == "safe_run_sql":
                                    ok, data = _safe_json_load(content_text)
                                    if ok and data:
                                        tool_data["safe_run_sql"] = data
                                        ss.last_sql = data.get("final_query", "") or ss.last_sql
                                        res = data.get("result")
                                        if res and res.get("columns"):
                                            ss.last_result = res
                                elif tname == "paginate_sql":
                                    ok, data = _safe_json_load(content_text)
                                    if ok and data:
                                        tool_data["paginate_sql"] = data
                                elif tname == "explain_sql":
                                    tool_data["explain_sql_text"] = content_text
                # Son mesaj
                final_answer = (last_msgs[-1].content if last_msgs else "") or "Bir cevap üretilemedi."
                status.update(label="İşlem tamamlandı ✅", state="complete")
                st.markdown(final_answer)
            except Exception as e:
                status.update(label="Hata ⛔", state="error")
                st.exception(e)

    with right:
        # Sonuç tablosu (örnek gösterim)
        if ss.last_result and ss.last_result.get("columns"):
            st.markdown("### 🔎 Sonuç (örnek)")
            st.dataframe(ss.last_result, use_container_width=True, hide_index=True)
        # İndirme butonları + SQL
        _download_buttons(ss.last_result, ss.last_sql)
        # Plan / Attempts / Explain
        _render_plan_attempts(tool_data)
        # Sayfalama formu
        _render_pagination(ss.last_sql)

    # Geçmişe AI cevabını yaz
    ss.messages.append(AIMessage(content=final_answer))

    # Telemetry finalize (emniyet)
    if TELEMETRY_AVAILABLE and telemetry.is_open():
        telemetry.finalize()

# Geçmiş koşular
_render_run_history()
