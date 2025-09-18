# main.py
import argparse
import yaml
import logging
import sys
from langchain_core.messages import HumanMessage
from utils.logging import setup_logging
from graph import build_graph

BANNER = """
Analist AI Agent — Google Gemini + LangGraph Tool Calling
Veritabanı ile ilgili sorularınızı sorabilirsiniz.
Komutlar: :q, :quit, :exit -> Çıkış
"""

def main():
    parser = argparse.ArgumentParser(description="Analist AI Agent (Interactive)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logging(cfg.get("runtime", {}).get("log_dir", "logs"))
    logger.info("Uygulama başlıyor…")

    try:
        graph_app = build_graph(cfg)
    except Exception as e:
        logger.exception("Agent grafiği oluşturulurken hata: %s", e)
        sys.exit(1)

    print(BANNER)
    while True:
        try:
            q = input("Soru > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor…")
            break
        if not q: continue
        if q in (":q", ":quit", ":exit"): break

        try:
            initial_state = {"messages": [HumanMessage(content=q)]}
            print("\n--- Agent Çalışıyor... ---")
            for event in graph_app.stream(initial_state, config={"recursion_limit": 25}):
                for key, value in event.items():
                    print(f"Düğüm: '{key}' | Çıktı: {value}")
            final_answer = list(event.values())[0]['messages'][-1].content
            print("\n================= CEVAP =================")
            print(final_answer)
            print("=========================================\n")
        except Exception as e:
            logger.exception("Çalışma sırasında hata: %s", e)
            print(f"[HATA] {e}\n(Detaylar için log dosyasına bakın.)")

if __name__ == "__main__":
    main()