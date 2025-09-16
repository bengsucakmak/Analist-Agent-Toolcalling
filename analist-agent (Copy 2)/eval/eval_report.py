import json, subprocess
from tabulate import tabulate

EVAL_FILE = "data/eval_questions.jsonl"
REPORT_FILE = "eval_report.csv"

def classify_error(expected, predicted):
    if not predicted:
        return "Empty SQL"
    if "error" in predicted.lower():
        return "Syntax Error"
    if "join" in expected.lower() and "join" not in predicted.lower():
        return "Join Error"
    if "group by" in expected.lower() and "group by" not in predicted.lower():
        return "Aggregation Error"
    if "count" in expected.lower() and "count" not in predicted.lower():
        return "Aggregation Error"
    return "Other/Wrong"

def run_eval():
    rows = []
    y_true, y_pred = [], []

    with open(EVAL_FILE, "r") as f:
        for line in f:
            sample = json.loads(line)
            q, exp_sql = sample["question"], sample["expected_sql"].lower()

            result = subprocess.run(
                ["python", "main.py", "-q", q],
                capture_output=True, text=True
            )
            out = result.stdout.lower()

            pred_sql = None
            if "select" in out:
                pred_sql = "select " + out.split("select", 1)[1].split("\n", 1)[0]

            ok = pred_sql and exp_sql.split()[0] in pred_sql
            y_true.append(1)
            y_pred.append(1 if ok else 0)

            status = "+" if ok else "-"
            error_type = "-" if ok else classify_error(exp_sql, pred_sql or "")
            rows.append([sample["id"], q, exp_sql[:45]+"...", (pred_sql or "N/A")[:45]+"...", status, error_type])

    # metrikler
    total = len(y_true)
    correct = sum(y_pred)
    accuracy = correct / total if total else 0.0
    precision = correct / (sum(y_pred) if sum(y_pred) > 0 else 1)
    recall = correct / (sum(y_true) if sum(y_true) > 0 else 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0.0

    metrics_text = f"""
Evaluation Metrics:
- Accuracy : {accuracy:.2f}
- Precision: {precision:.2f}
- Recall   : {recall:.2f}
- F1 Score : {f1:.2f}
"""

    table_text = tabulate(
        rows,
        headers=["ID","Question","Expected","Predicted","Result","Error Type"],
        tablefmt="github"
    )

    full_report = table_text + "\n" + metrics_text

    # ekrana bas
    print(full_report)

    # dosyaya yaz
    with open(REPORT_FILE, "w") as f:
        f.write(full_report)

    print(f"\n[i] Rapor kaydedildi: {REPORT_FILE}")

if __name__ == "__main__":
    run_eval()
