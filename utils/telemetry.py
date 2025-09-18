# utils/telemetry.py
from __future__ import annotations
import os, json, time, uuid, threading
from typing import Any, Dict, List, Optional

_LOG_PATH = "logs/runs.jsonl"
_LOCK = threading.Lock()

class _RunCtx:
    def __init__(self):
        self.trace_id: Optional[str] = None
        self.question: str = ""
        self.start_ts: float = 0.0
        self.steps: List[Dict[str, Any]] = []
        self.finalized: bool = False

_ctx = _RunCtx()

def configure(log_path: str = "logs/runs.jsonl"):
    global _LOG_PATH
    _LOG_PATH = log_path
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)

def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]

def set_trace(trace_id: Optional[str], question: str):
    """Her run başında çağır: trace_id vermezsen üretilir."""
    _ctx.trace_id = trace_id or new_trace_id()
    _ctx.question = question or ""
    _ctx.start_ts = time.time()
    _ctx.steps = []
    _ctx.finalized = False

def step(tool: str, ms: int, ok: bool, **kw):
    """Her tool çağrısından sonra (veya önemli aşama sonrası) çağır."""
    if not _ctx.trace_id:
        return
    ev = {"tool": tool, "ms": int(ms), "ok": bool(ok)}
    ev.update(kw or {})
    _ctx.steps.append(ev)

def is_open() -> bool:
    return bool(_ctx.trace_id) and not _ctx.finalized

def finalize(**final_fields):
    """Run bittiğinde tek satır JSONL olarak kaydet."""
    if not _ctx.trace_id or _ctx.finalized:
        return
    total_ms = int((time.time() - _ctx.start_ts) * 1000)
    doc = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "trace_id": _ctx.trace_id,
        "question": _ctx.question,
        "latency_ms": total_ms,
        "steps": _ctx.steps,
    }
    doc.update(final_fields or {})
    line = json.dumps(doc, ensure_ascii=False)
    with _LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    _ctx.finalized = True

def get_current_trace_id() -> Optional[str]:
    return _ctx.trace_id
