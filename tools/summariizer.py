# tools/summarizer.py
from __future__ import annotations
import json, math
from typing import List, Any, Dict
from langchain_core.tools import tool

def _parse_result(payload: str):
    try:
        data = json.loads(payload)
        cols = data.get("columns", [])
        rows = data.get("rows", [])
        return cols, rows
    except Exception:
        return [], []

def _is_number(x: Any) -> bool:
    try:
        float(x); return True
    except Exception:
        return False

@tool
def summarize_answer(args_json: str) -> str:
    """
    Sonucu doğal dille özetler ve açıklanabilirlik bilgisi ekler.
    Girdi JSON:
      {
        "question":"...",
        "result_json":"...",
        "final_query":"...",
        "plan_json":"...",         # plan_query çıktısı (opsiyonel)
        "attempts": [],            # safe_run_sql denemeleri (opsiyonel)
        "limit_info": {"row_limit":200,"paged":false}   # opsiyonel
      }
    Çıktı: Markdown metin
    """
    try:
        args = json.loads(args_json)
        question = args.get("question","")
        result_json = args.get("result_json","")
        final_query = args.get("final_query","")
        plan_json = args.get("plan_json")
        attempts = args.get("attempts", [])
        limit_info = args.get("limit_info", {})
    except Exception:
        return "Özet üretilemedi (geçersiz JSON)."

    cols, rows = _parse_result(result_json)
    if not cols:
        return "YETERSİZ KANIT — sorgu veri döndürmedi veya sonuç boş."

    # Basit sezgisel özet
    summary_lines: List[str] = []
    summary_lines.append(f"**Soru:** {question}")

    if len(cols) == 1 and rows:
        v = rows[0][0]
        summary_lines.append(f"**Cevap:** `{cols[0]}` = **{v}**")
    elif len(cols) == 2 and all(_is_number(r[1]) for r in rows[:5] if len(r) > 1):
        # label, value → top-n gibi özet
        topn = min(5, len(rows))
        bullet = "\n".join([f"- {rows[i][0]} → **{rows[i][1]}**" for i in range(topn)])
        summary_lines.append("**Özet (ilk satırlar):**\n" + bullet)
    else:
        summary_lines.append(f"**Şekil:** {len(rows)} satır × {len(cols)} kolon döndü (örnek ilk 5 satır gösterilebilir).")

    # Kısıtlama bilgisi
    if isinstance(limit_info, dict):
        if limit_info.get("paged"): summary_lines.append("_Not: Sonuç sayfalanmıştır._")
        elif limit_info.get("row_limit"):
            summary_lines.append(f"_Not: En fazla {limit_info['row_limit']} satır döndürüldü._")

    # Plan bilgisi
    plan_snip = ""
    if plan_json:
        try:
            p = json.loads(plan_json)
            intent = p.get("intent")
            dims = p.get("dimensions", [])[:3]
            mets = [m.get("alias") or m.get("expr") for m in p.get("metrics", [])][:3]
            jp = " → ".join(p.get("join_path", [])[:5])
            plan_snip = f"_Plan:_ intent=`{intent}`, dim={dims}, metric={mets}, join={jp}"
        except Exception:
            pass
    if plan_snip:
        summary_lines.append(plan_snip)

    # Son SQL (kısa)
    if final_query:
        code = final_query.strip().replace("```","`")
        if len(code) > 500: code = code[:500] + " …"
        summary_lines.append("\n**Kullanılan SQL (özet):**\n```sql\n" + code + "\n```")

    return "\n\n".join(summary_lines)
