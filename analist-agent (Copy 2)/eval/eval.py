import json, subprocess

EVAL_FILE = "/home/nagme/Desktop/analist-agent/data/eval_questions.jsonl"

def run_eval():
    total, correct = 0, 0
    with open(EVAL_FILE, "r") as f:
        for line in f:
            sample = json.loads(line)
            q = sample["question"]
            exp_sql = sample["expected_sql"].strip().lower()

            # pipeline'ı CLI ile çalıştır
            result = subprocess.run(
                ["python", "main.py", "-q", q],
                capture_output=True, text=True
            )

            output = result.stdout.lower()
            ok = exp_sql.split()[0] in output and "select" in output
            total += 1
            if ok:
                correct += 1
                status = "+"
            else:
                status = "-"
            print(f"[{status}] {q}\n  Expected: {exp_sql}\n")

    print(f"\nToplam {correct}/{total} doğru ({100*correct/total:.1f}%).")

if __name__ == "__main__":
    run_eval()
