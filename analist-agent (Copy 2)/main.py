"""
Analist-Agent — main.py

Tek giriş noktası (CLI):
  • ask  : Tek bir soru sor, cevabı yazdır
  • repl : Etkileşimli kabuk (çoklu soru)
  • ui   : Streamlit arayüzünü başlat

İlk düzenleme: raw içerik mantığına sadık; sonraki tüm değişiklikler bu canvas üzerinden.
"""
from __future__ import annotations

import os
import sys
import time
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

# Proje modülleri
from utils.types import AgentState
from utils.cost import CostTracker
from utils.llm import LLMService
from tools.db import connect_readonly
from graph import build_graph

# (Opsiyonel) tool-calling kullanan diğer entegrasyonlarda global ctx lazımsa
try:
    from agent_tools import set_global_ctx  # type: ignore
except Exception:  # tools.set_global_ctx olmayabilir — sorun değil
    def set_global_ctx(_ctx: Dict[str, Any]) -> None:  # type: ignore
        return None

log = logging.getLogger("main")


# ──────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config.yaml bulunamadı: {p!s}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _init_services(cfg: Dict[str, Any]):
    # LLMService için bilinen parametreler ve ekstra model_kwargs ayrıştırması
    raw_llm_cfg = dict(cfg.get("llm", {}) or {})
    known_keys = {
        "provider", "model", "model_name", "api_key", "base_url", "temperature", "max_tokens",
        "timeout", "top_p", "top_k", "stream", "retry", "max_retries"
    }
    llm_init = {k: raw_llm_cfg.pop(k) for k in list(raw_llm_cfg.keys()) if k in known_keys}

    # Normalize: model_name → model (UI/CLI tarafı için tek tip anahtar kullan)
    if "model" not in llm_init and "model_name" in llm_init:
        llm_init["model"] = llm_init.pop("model_name")

    # Geriye kalan parametreleri model_kwargs içine aktar
    if raw_llm_cfg:
        raw_llm_cfg.pop("model", None)
        raw_llm_cfg.pop("model_name", None)
        llm_init["model_kwargs"] = raw_llm_cfg

    llm = LLMService(**llm_init)

    # connect_readonly beklenmeyen parametreler alabilir; temizle
    db_cfg = dict(cfg.get("db", {}) or {})
    db_cfg.pop("read_only", None)
    conn = connect_readonly(**db_cfg)

    cost = CostTracker()
    logger = logging.getLogger("agent")
    return llm, conn, cost, logger


def _make_ctx(llm, cfg, conn, cost, logger) -> Dict[str, Any]:
    ctx = {"llm": llm, "cfg": cfg, "conn": conn, "cost": cost, "logger": logger}
    # Tool-calling adaptörleri için global ctx ayarla (LangChain vs.)
    try:
        set_global_ctx(ctx)
    except Exception:
        pass
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Çalıştırıcılar
# ──────────────────────────────────────────────────────────────────────────────

def _run_once(graph, question: str) -> AgentState:
    state = AgentState(question=question)
    t0 = time.time()
    out = graph.invoke(state)
    elapsed = (time.time() - t0) * 1000

    # Kullanışlı özet
    if getattr(out, "error", None):
        print(f"\n❌ Hata: {out.error}")
    else:
        # Summarizer hem answer_text hem answer doldurabilir
        answer = getattr(out, "answer", None) or getattr(out, "answer_text", None) or ""
        print("\n📝 Cevap:\n" + answer)

    # SQL önizleme (varsa)
    sql = getattr(out, "validated_sql", None)
    if not sql:
        sqls = getattr(out, "candidate_sql", None) or []
        sql = sqls[0] if sqls else None
    if sql:
        print("\n🧩 SQL:\n" + str(sql))

    # Mini telemetri
    print("\n———")
    print(f"⏱ Süre: {elapsed:.1f} ms")
    try:
        from pprint import pprint
        if getattr(out, "trace", None):
            print("📋 Trace (son 5):")
            preview = out.trace if isinstance(out.trace, list) else []
            pprint(preview[-5:])
    except Exception:
        pass
    return out


def cmd_ask(args):
    cfg = _load_config(args.config)
    llm, conn, cost, logger = _init_services(cfg)
    g = build_graph(conn, cfg, cost, llm, logger=logger)
    _make_ctx(llm, cfg, conn, cost, logger)
    _run_once(g, args.q)


def cmd_repl(args):
    cfg = _load_config(args.config)
    llm, conn, cost, logger = _init_services(cfg)
    g = build_graph(conn, cfg, cost, llm, logger=logger)
    _make_ctx(llm, cfg, conn, cost, logger)

    print("Analist-Agent REPL. Çıkış için :q veya :quit")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkış…")
            break
        if q in {":q", ":quit", "exit"}:
            break
        if not q:
            continue
        _run_once(g, q)


def cmd_ui(args):
    # Streamlit arayüzünü başlat — mevcut ui_streamlit.py dosyasını çalıştırır
    ui_path = Path(__file__).parent / "ui_streamlit.py"
    if not ui_path.exists():
        print("ui_streamlit.py bulunamadı.")
        sys.exit(1)
    env = os.environ.copy()
    # Config yolunu UI'ye aktar (opsiyonel)
    env["ANALIST_AGENT_CONFIG"] = str(Path(args.config).resolve())
    print("Streamlit başlatılıyor… (Ctrl+C ile durdurabilirsiniz)")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], env=env, check=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analist-Agent CLI")
    p.add_argument("--config", default="config.yaml", help="Config dosyası yolu")

    sp = p.add_subparsers(dest="cmd", required=True)

    p_ask = sp.add_parser("ask", help="Tek bir soru çalıştır")
    p_ask.add_argument("-q", "--q", required=True, help="Soru")
    p_ask.set_defaults(func=cmd_ask)

    p_repl = sp.add_parser("repl", help="Etkileşimli kabuk")
    p_repl.set_defaults(func=cmd_repl)

    p_ui = sp.add_parser("ui", help="Streamlit arayüzü")
    p_ui.set_defaults(func=cmd_ui)

    return p


def main(argv: list[str] | None = None) -> int:
    _setup_logging(os.environ.get("LOGLEVEL", "INFO"))
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
        return 0
    except Exception as e:
        log.exception("Çalışma hatası: %s", e)
        print(f"Hata: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
