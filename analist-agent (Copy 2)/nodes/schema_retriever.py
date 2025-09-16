"""
Schema Retriever — Analist‑Agent

Bu düğüm, SQLite veritabanından tablo ve kolon bilgilerini toplayıp
LLM'e ipucu olarak verilecek düz (metinsel) bir şema dökümanı (schema_doc)
üretir ve bunu AgentState'e yazar. Açıklamalar genel seviyededir; davranış
aynıdır.
"""

import logging
import sqlite3

from utils.types import AgentState

log = logging.getLogger("schema")

# Manuel kolon açıklamaları sözlüğü (LLM'e ipucu için).
# Not: Buradaki anahtarlar "<tablo>.<kolon>" biçimindedir.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "chat_session.num_of_mess": "number of messages in this chat session",
    "chat_session.message_date": "timestamp (datetime) of the chat session",
    "user.age": "age of the user in years",
    "user.unit_id": "foreign key → unit table",
    "unit.unit_name": "name of the organizational unit",
}


def run(conn: sqlite3.Connection, state: AgentState) -> AgentState:
    """DB şemasını okur ve `state.schema_doc` içine düz metin olarak yazar.

    Adımlar (SQLite):
    1) `sqlite_master` üzerinden tablo adlarını al
    2) Her tablo için `PRAGMA table_info(<tablo>)` ile kolon ad/tiplerini topla
    3) Manuel sözlükten (COLUMN_DESCRIPTIONS) ipuçlarını ekle
    4) Tek bir metin dokümanına dönüştür (schema_doc)
    """

    # DB bağlantısından cursor aç
    cur = conn.cursor()

    # 1) Tabloları listele (sqlite_master: tablo metadata)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # Dahili SQLite tablolarını (sqlite_...) atla
    tables = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]

    schema_lines: list[str] = []

    for t in tables:
        # Tablo başlığı
        schema_lines.append(f"TABLE {t} (")

        # 2) Her tablo için kolon bilgisi al (PRAGMA table_info)
        cur.execute(f"PRAGMA table_info({t})")
        cols = cur.fetchall()
        # PRAGMA table_info dönüşü: (cid, name, type, notnull, dflt_value, pk)
        for cid, name, ctype, notnull, dflt, pk in cols:
            # Kolon adı + tipi (sade format)
            schema_lines.append(f"  {name} {ctype}")

        # Tablo kapanışı
        schema_lines.append(")")

    # 3) Sözlük bölümü: manuel açıklamalar
    schema_lines.append("\nCOLUMN DICTIONARY:")
    for col, desc in COLUMN_DESCRIPTIONS.items():
        schema_lines.append(f"- {col} → {desc}")

    # 4) Şemayı tek string olarak birleştir ve state'e yaz
    schema_doc = "\n".join(schema_lines)
    state.schema_doc = schema_doc

    log.info("Şema dökümanı hazır (%d karakter).", len(schema_doc))
    return state