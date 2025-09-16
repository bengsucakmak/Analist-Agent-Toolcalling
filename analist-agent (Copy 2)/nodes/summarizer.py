# nodes/summarizer.py
import logging
import json
import re
from typing import List, Dict, Any, Optional

from utils.types import AgentState
from utils.cost import CostTracker
from utils.llm import call_llm_text

log = logging.getLogger("sum")

# ---------------------------
# Yardımcılar
# ---------------------------
def _find_group_and_metric_cols(rows: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """
    İki kolonlu (veya daha fazla) tablolarda grup kolonu (string) ve metrik kolonu (sayısal) seç.
    Örn: ['unit_name', 'SUM(num_of_mess)'] -> ('unit_name','SUM(num_of_mess)')
    """
    if not rows:
        return None, None
    keys = list(rows[0].keys())
    if len(keys) == 1:
        return None, keys[0]

    def is_num(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    numeric_scores = {k: 0 for k in keys}
    text_scores = {k: 0 for k in keys}
    for r in rows:
        for k, v in r.items():
            if v is None:
                continue
            (numeric_scores if is_num(v) else text_scores)[k] += 1

    metric = max(numeric_scores, key=numeric_scores.get, default=None)
    group = max(text_scores, key=text_scores.get, default=None)

    # fallback: 2 kolonluysa ilkini grup, ikincisini metrik
    if not group or not metric:
        if len(keys) == 2:
            group, metric = keys[0], keys[1]
    return group, metric


def _mk_markdown_table(rows: List[Dict[str, Any]], top_n: int | None = None) -> str:
    """Satırları Markdown tabloya çevirir."""
    if not rows:
        return ""
    keys = list(rows[0].keys())
    data = rows[:top_n] if top_n else rows
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in data)
    return "\n".join([header, sep, body])


def _format_sections(text: str) -> str:
    """
    LLM'in döndürdüğü serbest metni başlıklara göre bölüp
    daha temiz Markdown formatına sokar.
    """
    sections = {
        "Kısa Yanıt": [],
        "Öne Çıkan Metrikler": [],
        "Basit Eğilimler / Desenler": [],
        "Uyarılar": [],
        "Devam/Öneri Adımları": []
    }

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    current = None
    for l in lines:
        moved = False
        for key in sections:
            if l.lower().startswith(key.lower()):
                current = key
                if ":" in l:
                    parts = l.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        sections[key].append(parts[1].strip())
                moved = True
                break
        if not moved and current:
            sections[current].append(l)

    out = []
    if sections["Kısa Yanıt"]:
        out.append(f"### 📝 Kısa Yanıt\n{sections['Kısa Yanıt'][0]}")
    if sections["Öne Çıkan Metrikler"]:
        out.append("### 📊 Öne Çıkan Metrikler")
        for m in sections["Öne Çıkan Metrikler"]:
            out.append(m if m.startswith("*") or m.startswith("-") else f"- {m}")
    if sections["Basit Eğilimler / Desenler"]:
        out.append("### 📈 Basit Eğilimler / Desenler")
        out += [x if x.startswith("-") else f"- {x}" for x in sections["Basit Eğilimler / Desenler"]]
    if sections["Uyarılar"]:
        out.append("### ⚠️ Uyarılar")
        out += [x if x.startswith("-") else f"- {x}" for x in sections["Uyarılar"]]
    if sections["Devam/Öneri Adımları"]:
        out.append("### 🔮 Devam / Öneri Adımları")
        out += [x if x.startswith("-") else f"- {x}" for x in sections["Devam/Öneri Adımları"]]

    return "\n".join(out)


def _prettify_singleton_table(rows, question: str | None):
    """
    Tek satır & tek kolonluk sonuçları daha okunaklı hale getirir.
    Döndürür: (kisa_yanit_str, markdown_table_str)
    """
    if not rows or len(rows) != 1:
        return None, None
    cols = list(rows[0].keys())
    vals = list(rows[0].values())
    if len(cols) != 1:
        return None, None

    raw_col = cols[0]
    val = vals[0]
    q = (question or "").lower()

    # Kolon adını sadeleştir (COUNT/AVG/SUM/DISTINCT vb.)
    col_clean = raw_col
    m = re.search(r"(?:count|avg|sum|min|max)\s*\(\s*(?:distinct\s+)?(.+?)\s*\)", raw_col, flags=re.I)
    if m:
        col_clean = m.group(1)
    if "." in col_clean:
        col_clean = col_clean.split(".")[-1]
    col_clean = col_clean.strip().strip('"').strip("'")

    # Soruya göre başlık
    if "unit" in q or "organizasyon" in q:
        header = "Unit Sayısı"
        short = f"Toplam unit sayısı = {val}"
    elif "user" in q or "kullanıcı" in q:
        header = "Kullanıcı Sayısı"
        short = f"Toplam kullanıcı sayısı = {val}"
    else:
        header = col_clean or raw_col
        short = f"{header} = {val}"

    table = f"| {header} |\n|---|\n| {val} |"
    return short, table


def _extract_metric_candidates(rows: List[Dict[str, Any]]) -> list[str]:
    """Öne çıkarılabilecek metrik/kolon isimleri (LLM'e ipucu)."""
    if not rows:
        return []
    keys = set().union(*(r.keys() for r in rows))
    prefs = ["avg", "average", "mean", "sum", "count", "total", "rate", "ratio", "share", "min", "max"]
    human = [k for k in keys if "name" in k.lower() or "title" in k.lower()]
    metrics = [k for k in keys if any(p in k.lower() for p in prefs)]
    return (human[:3] + metrics[:3])[:6]


def _detect_user_instruction(question: str) -> str:
    """
    Çıktı modu:
      - 'yorumla', 'analiz et'          -> commentary
      - 'tek cümle', 'kısaca'           -> one_liner
      - 'sadece tablo', 'tablo olarak'  -> table_only
      - 'sadece metrik', 'madde madde'  -> bullets_only
      - aksi halde                      -> default
    """
    q = (question or "").lower()
    if re.search(r"\byorumla\b|\banaliz et\b", q):
        return "commentary"
    if re.search(r"\btek cümle\b|\bkısaca\b|\bone[- ]line\b", q):
        return "one_liner"
    if re.search(r"\bsadece tablo\b|\btablo olarak\b|\btable\b", q):
        return "table_only"
    if re.search(r"\bsadece metrik\b|\bmadde madde\b|\bbullet\b", q):
        return "bullets_only"
    return "default"


def _is_listing_intent(question: str | None, sql: str | None) -> bool:
    """
    'listele', 'kimler', 'list', 'who', 'which users' gibi sorgular
    ve/veya SELECT'in açık kolon seçmesi (SELECT usr.name, ...) -> listeleme niyeti.
    """
    q = (question or "").lower()
    if any(tok in q for tok in [
        "listele", "kimler", "hangileri", "list", "who", "which users",
        "kullanan kişileri", "kullanan kullanıcıları", "kullananlar"
    ]):
        return True

    s = (sql or "").lower()
    if "select" in s and " from " in s:
        if any(k in s for k in [" name", "surname", "title", "unit_name", "email"]):
            return True

    return False


def _is_data_insufficient(rows: List[Dict[str, Any]] | None, question: Optional[str] = None) -> tuple[bool, str]:
    """
    Basit veri yeterlilik kontrolü:
      - 0 satır -> yetersiz
      - 1 satır:
          * Soru 'en fazla/en çok/max' gibi süperlatif içeriyorsa: yeterli (tek satır beklenir)
          * Satırda tek kolon varsa ve aggregate kolonsa: yeterli; değilse yetersiz
      - Aksi halde yeterli
    """
    if not rows:
        return True, "Hiç satır dönmedi."

    if len(rows) == 1:
        q = (question or "").lower()
        keys = list(rows[0].keys())

        if any(tok in q for tok in ["en fazla", "en çok", "max"]):
            return False, ""

        if len(keys) == 1:
            col = keys[0].lower()
            if any(agg in col for agg in ["count", "avg", "average", "sum", "min", "max"]):
                return False, ""
            return True, "Yalnızca 1 satır ve 1 kolon var."
    return False, ""


# ---------------------------
# Ana fonksiyon
# ---------------------------

def run(state: AgentState, cost: CostTracker, show_sql: bool, llm_service) -> AgentState:
    rows = state.rows_preview or []
    listing_intent = _is_listing_intent(state.question, state.validated_sql)

    insufficient, reason = _is_data_insufficient(rows, state.question)

    # ÖZEL DURUM: Listeleme niyeti varsa ve hiç satır yoksa → "boş sonuç"
    if not rows and listing_intent:
        answer = (
            "EŞLEŞME BULUNAMADI\n"
            "- Sorgu koşullarına uyan kayıt bulunamadı. "
            "Filtre koşullarını kontrol etmeyi veya aralığı genişletmeyi deneyin."
        )
        if show_sql and state.validated_sql:
            answer += "\n\nKullanılan SQL:\n" + state.validated_sql
        state.answer_text = answer
        log.info("Özet: listeleme niyeti, boş sonuç.")
        return state

    # Genel kural: yetersizse 'YETERSİZ KANIT'
    if insufficient:
        answer = (
            "YETERSİZ KANIT\n"
            f"- Gerekçe: {reason}\n"
            "- Öneri: Soru kapsamını netleştirin ya da tarih/filtre aralığını genişletin. "
            "Gerekirse alternatif bir toplulaştırma (ör. haftalık/aylık) deneyebiliriz."
        )
        if show_sql and state.validated_sql:
            answer += "\n\nKullanılan SQL:\n" + state.validated_sql
        state.answer_text = answer
        log.info("Özet: yetersiz veri nedeniyle cevap verilmedi.")
        return state

    # LLM'e küçük bir kesit ver (ilk 50 satır)
    data_excerpt = json.dumps(rows[:50], ensure_ascii=False)

    # Kullanıcı talimatından çıktı modunu belirle
    # summarizer.run içinde:
    mode = _detect_user_instruction(state.question or "")
    pref = getattr(state, "output_pref", None)
    if pref == "analyst" and mode == "default":
        mode = "default"
    elif pref == "table_only":
        mode = "table_only"
    elif pref == "bullets_only":
        mode = "bullets_only"


    log.info("Summarizer mode: %s", mode)

    # Mod'a göre system prompt
    if mode == "commentary":
        system_prompt = (
            "Sen kıdemli bir veri analistisın (Turkish only).\n"
            "Kurallar:\n"
            "1) SADECE verilen veri excerpt’ündeki sayıları/alanları kullan; uydurma/tahmin YAPMA.\n"
            "2) ID yerine mümkünse insan-okur alanlar (ör. unit_name) kullan.\n"
            "3) ÇIKTI: Serbest yorumlayıcı bir paragraf yaz; başlık/şablon KULLANMA.\n"
            "4) Sadece gözlenen değerlere dayan; istatistiksel güven ifadesi kurma.\n"
            "5) Gerekirse son cümlede kısa bağlam uyarısı ekle."
        )
    elif mode == "one_liner":
        system_prompt = (
            "Sen kıdemli bir veri analistisın (Turkish only).\n"
            "SADECE verilen veri excerpt’ünü kullan. ÇIKTI: TEK cümlelik kısa yanıt.\n"
            "Uydurma yapma; ID yerine insan-okur alanları tercih et; istatistiksel güven iddiası kurma."
        )
    elif mode == "table_only":
        system_prompt = (
            "Sen kıdemli bir veri analistisın (Turkish only).\n"
            "SADECE verilen veri excerpt’ünü kullan.\n"
            "ÇIKTI: Markdown tablo olarak yaz. Tablo düzgün formatlı olsun:\n"
            " - Başlık satırı (| col1 | col2 |)\n"
            " - Altına ayraç (|---|---|)\n"
            " - Altına veriler\n"
            "Ekstra yorum ekleme, sadece tablo yaz."
        )
    elif mode == "bullets_only":
        system_prompt = (
            "Sen kıdemli bir veri analistisın (Turkish only).\n"
            "SADECE verilen veri excerpt’ünü kullan. ÇIKTI: Sadece madde madde metrikler yaz.\n"
            "Her maddede alan adı ve değerleri ver; yorum/başlık ekleme."
        )
    else:
        system_prompt = (
            "Sen kıdemli bir veri analistisın (Turkish only). "
            "Aşağıdaki kurallara KESİNLİKLE uy:\n"
            "1) SADECE verilen tablo/veri excerpt’ündeki sayıları ve alanları kullan. Uydurma/tahmin/varsayım YAPMA.\n"
            "2) ID alanları yerine mümkünse insan-okur alanları (ör. unit_name) üzerinde anlatım yap.\n"
            "3) Çıktıyı şu yapıda, kısa ve öz yaz:\n"
            "   - Kısa Yanıt (1–2 cümle)\n"
            "   - Öne Çıkan Metrikler (madde madde)\n"
            "   - Basit Eğilimler / Desenler\n"
            "   - Uyarılar (ör. az satır, eksik veri, aykırı değer)\n"
            "   - Devam/Öneri Adımları (en fazla 2 madde)\n"
            "4) Sadece tabloda gözüken değerleri kullan; istatistiksel güven iddiası kurma.\n"
            "5) Sayıları gerekiyorsa 2 ondalıkla ver; binlik ayırıcı kullanma.\n"
            "6) Gerekirse tek cümlelik bağlam uyarısı ekle (örn. küçük örneklem).\n"
            "7) Tablo boş değilse her zaman kısa bir özet ver (ör: toplam kaç grup, en yüksek ve en düşük değer)."
        )

    # Ek bilgi (opsiyonel metrik adayları)
    extras_lines = []
    candidates = _extract_metric_candidates(rows)
    if candidates:
        extras_lines.append("OLASI_METRIK_ALANLARI: " + ", ".join(candidates))
    extras = "\n".join(extras_lines) if extras_lines else "(yok)"

    # User prompt
    user_prompt = (
        f"SORU:\n{state.question}\n\n"
        f"VERI_EXCERPT (ilk 50 satır):\n{data_excerpt}\n\n"
        f"EK_BILGI:\n{extras}\n\n"
        f"CIKTI_MODU: {mode}"
    )

    # LLM çağrısı
    txt = call_llm_text(llm_service, system_prompt, user_prompt, cost=cost)

    # ---- Deterministic istatistikler: genel toplam, ortalama, max/min ----
    try:
        if rows:
            gcol, mcol = _find_group_and_metric_cols(rows)
            if mcol:
                vals = []
                for r in rows:
                    try:
                        if r.get(mcol) is not None:
                            vals.append(float(r[mcol]))
                    except Exception:
                        pass
                if vals:
                    total = sum(vals)
                    avg = total / len(vals)
                    if gcol:
                        sorted_rows = sorted(rows, key=lambda r: float(r.get(mcol) or 0), reverse=True)
                        top_g, top_v = sorted_rows[0][gcol], sorted_rows[0][mcol]
                        low_g, low_v = sorted_rows[-1][gcol], sorted_rows[-1][mcol]
                        stats_text = (
                        "### 📊 Hesaplanan İstatistikler\n"
                        f"- Toplam: **{int(total) if float(total).is_integer() else round(total,2)}**\n"
                        f"- Ortalama (grup başına): **{round(avg,2)}**\n"
                        f"- En yüksek: **{top_g} ({top_v})**\n"
                        f"- En düşük: **{low_g} ({low_v})**"
)   
                    else:
                        stats_text = (
                            f"\n\n**Hesaplanan İstatistikler**\n"
                            f"- Toplam: **{int(total) if float(total).is_integer() else round(total, 2)}**\n"
                            f"- Ortalama: **{round(avg, 2)}**"
                        )
                    # Her zaman 2 satır boşlukla ayır, Markdown başlığı gibi hizala
                    txt = (txt or "").strip()
                    if txt:
                        txt += "\n\n---\n\n"   # ayırıcı çizgi
                    txt += stats_text

    except Exception as _e:
        log.warning("Özet postprocess istatistikleri atlandı: %s", _e)

    # --- Postprocess: başlıklara göre yeniden formatla ---
    txt = _format_sections(txt or "")

    # --- Postprocess: Tek kolon & tek satır sonucu güzelleştir ---
    short2, table2 = _prettify_singleton_table(rows, state.question)
    if short2 and table2:
        # Kısa yanıtı ve tabloyu net biçimde yaz; LLM metnini ez.
        txt = f"### 📝 \n{short2}\n\n### 🗂️ Tablo\n{table2}"

    # Ek güvenlik: tek satır-tek kolon durumu yakalanmadıysa da düzelt
    try:
        if rows and len(rows) == 1:
            cols = list(rows[0].keys())
            vals = list(rows[0].values())
            if len(cols) == 1:
                col = cols[0]
                val = vals[0]
                pretty_table = f"| {col} |\n|---|\n| {val} |"
                txt = f"Kısa Yanıt: {col} = {val}\n\n{pretty_table}"
    except Exception as e:
        log.warning("Tablo postprocess atlandı: %s", e)

    # Boş/çok kısa cevap güvenliği
    if not txt or len(txt.strip()) < 8:
        if rows:
            gcol, mcol = _find_group_and_metric_cols(rows)
            if gcol and mcol:
                vals = [float(r[mcol]) for r in rows if r.get(mcol) is not None]
                total = sum(vals) if vals else 0.0
                avg = (total / len(vals)) if vals else 0.0
                srt = sorted(rows, key=lambda r: float(r.get(mcol) or 0), reverse=True)
                auto = []
                auto.append(f"### 📝{mcol} kolonuna göre **{len(rows)}** grup bulundu.")
                auto.append("### 📊 Öne Çıkan Metrikler")
                for r in srt[:3]:
                    auto.append(f"- {r[gcol]}: {r[mcol]}")
                auto.append("### 📈 Toplam / Ortalama")
                auto.append(f"- Toplam: **{int(total) if float(total).is_integer() else round(total, 2)}**")
                auto.append(f"- Ortalama: **{round(avg, 2)}**")
                txt = "\n\n".join(auto)
            else:
                txt = "### 📝 Kısa Yanıt\nVeriden kısa bir özet çıkardım; detaylar tabloda.\n\n### 🗂️ Tablo (ilk 20)\n" + _mk_markdown_table(rows, top_n=20)

        else:
            txt = (
                "YETERSİZ KANIT\n"
                "- Gerekçe: Modelden anlamlı bir özet alınamadı.\n"
                "- Öneri: Soru kapsamını netleştirelim veya daha geniş bir veri aralığı deneyelim."
            )

    # İstenirse SQL'i ekle
    if show_sql and state.validated_sql:
        txt += "\n\nKullanılan SQL:\n" + state.validated_sql

    state.answer_text = txt
    log.info("Özet hazır (mode=%s).", mode)
    return state