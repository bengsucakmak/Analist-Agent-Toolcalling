"""
SQL Executor — Analist‑Agent

Doğrulanan SQL'i güvenli şekilde çalıştırıp küçük bir satır önizlemesi döndürür.
Bu dosya, davranışı değiştirmeden *genel seviye* açıklamalar ile zenginleştirilmiştir.

Özet akış
1) `state.validated_sql` kontrol edilir (yoksa çalıştırılmaz)
2) `tools.db.execute_preview` ile *salt-okuma* sorgu yürütülür
3) Süre ve satır sayısı `state.execution_stats` içine yazılır
4) Örnek satırlar `state.rows_preview` içine konur
"""

import logging
import time
from utils.types import AgentState
from tools.db import execute_preview

# Yürütücü düğüm için logger
log = logging.getLogger("exec")


def run(conn, state: AgentState, preview_rows: int = 50) -> AgentState:
    """
    Doğrulanmış SQL'i (state.validated_sql) çalıştırır ve örnek (preview) satırları döndürür.

    Başarı →
      - `state.rows_preview`: tablo önizlemesi için satır listesi
      - `state.execution_stats`: { ok: True, ms: <süre_ms>, rowcount: <toplam> }

    Hata →
      - `state.execution_stats`: { ok: False, reason: <mesaj> }

    Notlar
      - `execute_preview` yalnızca *okuma* amaçlıdır; DDL/DML yürütmez.
      - `preview_rows`, UI'da gösterilecek örnek satır sayısını sınırlar.
    """
    # 1) Önkoşul: validated_sql gelmemişse yürütmeye kalkma
    if not getattr(state, "validated_sql", None):
        state.execution_stats = {"ok": False, "reason": "validated_sql yok"}
        return state

    t0 = time.time()  # süre ölçümü başlangıcı
    try:
        # 2) Güvenli yürütme: execute_preview sonuç sözlüğü döndürür
        out = execute_preview(conn, state.validated_sql, preview_rows=preview_rows)

        # 3) Süreyi hesapla ve state'e yaz
        dt = time.time() - t0
        state.rows_preview = out["rows"]  # önizleme için satır listesi
        state.execution_stats = {
            "ok": True,
            "ms": int(dt * 1000),          # milisaniye cinsinden süre
            "rowcount": out["rowcount"],   # toplam okunan/etkilenen satır sayısı
        }

        # 4) Bilgi logu (operasyonel telemetri)
        log.info("Sorgu çalıştı: %d satır (%.1f ms).", out["rowcount"], dt * 1000)

    except Exception as e:
        # 5) Hata durumunda reason ile raporla ve istisnayı logla
        state.execution_stats = {"ok": False, "reason": str(e)}
        log.exception("SQL yürütme hatası: %s", e)

    return state