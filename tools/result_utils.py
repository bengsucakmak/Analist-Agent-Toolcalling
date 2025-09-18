# tools/result_utils.py
import json
from langchain_core.tools import tool

def _parse_result(json_payload: str):
    """
    Expect json like: {"columns":["c1","c2"], "rows":[["v11","v12"], ...]}
    (db.run_sql aracı bu yapıyı döndürüyor olmalı.)
    """
    try:
        data = json.loads(json_payload)
        cols = data.get("columns", [])
        rows = data.get("rows", [])
        return cols, rows
    except Exception:
        return [], []

@tool
def tabulate_result(args_json: str) -> str:
    """
    Render JSON query result as Markdown table.
    Input JSON: {"result_json":"...", "max_rows": 20}
    """
    try:
        args = json.loads(args_json)
        payload = args["result_json"]
        max_rows = int(args.get("max_rows", 20))
    except Exception:
        return "Invalid JSON. Use: {\"result_json\":\"...\", \"max_rows\":20}"
    cols, rows = _parse_result(payload)
    if not cols:
        return "No data."
    rows = rows[:max_rows]
    # Markdown table
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join(["| " + " | ".join([str(x) for x in r]) + " |" for r in rows])
    return "\n".join([head, sep, body])

@tool
def suggest_chart(args_json: str) -> str:
    """
    Heuristically suggest a chart type for the result.
    Input JSON: {"result_json":"..."}; Returns a short suggestion string.
    """
    try:
        payload = json.loads(args_json)["result_json"]
    except Exception:
        return "Invalid JSON."
    cols, rows = _parse_result(payload)
    if len(cols) == 2 and rows:
        # (label, value) → bar chart
        return "Bar chart (x: first column, y: second column)."
    if any("date" in c.lower() or "time" in c.lower() for c in cols):
        return "Line chart over time (x: time, y: a numeric column)."
    return "Table is sufficient."
