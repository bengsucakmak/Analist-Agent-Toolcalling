"""
SQL Utilities — Analist‑Agent

Bu yardımcı fonksiyonlar, LLM'den gelen SQL metnini basit ve hızlı kontrollerden
geçirmek için kullanılır. Davranış korunmuş, yalnızca okunabilirlik ve genel
seviye açıklamalar eklenmiştir.
"""

from __future__ import annotations

import re
from typing import Tuple


# ─────────────────────────────────────
# Basit denetimler
# ─────────────────────────────────────

def is_select_only(sql: str) -> bool:
    """Metin SELECT veya WITH ile mi başlıyor?
    (Yalnızca SELECT/CTE köküne izin verildiği durumlar için.)
    """
    s = sql.strip().strip(";").lower()
    return s.startswith("select") or s.startswith("with")


def has_multiple_statements(sql: str) -> bool:
    """Naif çoklu-statement kontrolü: ';' sayısı > 1 ise True.
    (Sonda tek noktalı virgül tolere edilir.)
    """
    return sql.strip().count(";") > 1


def contains_banned(sql: str, banned_keywords) -> str | None:
    """Yasaklı anahtar kelime var mı? Varsa eşleşen kelimeyi döndür."""
    s = re.sub(r"\s+", " ", sql.lower())
    for kw in banned_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", s):
            return kw
    return None


def ensure_limit(sql: str, max_limit: int) -> str:
    """LIMIT yoksa `max_limit` ekleyip tek bir ';' ile bitirir."""
    s = sql.strip().rstrip(";")
    if re.search(r"\blimit\s+\d+\b", s, flags=re.I):
        return s + ";"
    return f"{s} LIMIT {max_limit};"


def sanitize_sql(sql: str) -> str:
    """Yorumları (--) ve /* ... */ bloklarını temizleyip kırp."""
    sql = re.sub(r"--.*?$", "", sql, flags=re.M)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    return sql.strip()


def static_checks(sql: str, banned_keywords, select_only: bool = True, allow_multi: bool = False) -> Tuple[bool, str]:
    """Hızlı statik kontroller:
    - SELECT-only kuralı
    - Çoklu statement yasağı
    - Yasaklı anahtar kelimeler
    """
    s = sanitize_sql(sql)
    if select_only and not is_select_only(s):
        return False, "SELECT-only kuralı ihlali."
    if not allow_multi and has_multiple_statements(s):
        return False, "Çoklu statement yasak."
    bad = contains_banned(s, banned_keywords)
    if bad:
        return False, f"Yasaklı anahtar kelime tespit edildi: {bad}"
    return True, "OK"


# ─────────────────────────────────────
# Metinden tek bir SELECT çıkarımı
# ─────────────────────────────────────
SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_single_select(sql_text: str) -> str | None:
    """Serbest metinden **TEK** bir SELECT/WITH statement çıkarır.

    - Varsa önce ```sql ... ``` kod bloğunu tercih eder
    - Yoksa tüm metinde (WITH|SELECT) ile başlayan ilk cümleyi alır
    - İlk `;` karakterine kadar keser ve trailing `;` ekler
    - Güvenlik için yorumları temizler
    """
    txt = sql_text.strip()

    # Kod bloklarını öncele
    m = SQL_BLOCK_RE.search(txt)
    cand = m.group(1).strip() if m else txt

    # İlk SELECT/WITH'i yakala (mümkünse ';' ile biten)
    m2 = re.search(r"(?is)\b(with\b.*?;|\bselect\b.*?;)", cand)
    if not m2:
        # Noktalı virgül yoksa satır sonuna kadar al
        m3 = re.search(r"(?is)\b(with\b.*|\bselect\b.*)", cand)
        if not m3:
            return None
        cand = m3.group(0).strip()
    else:
        cand = m2.group(1).strip()

    # Fazla cümle ayrımı (ilk ';'e kadar) ve sonuna ';'
    if ";" in cand:
        cand = cand.split(";", 1)[0] + ";"

    # Yorum temizliği
    cand = sanitize_sql(cand)
    return cand