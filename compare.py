"""Join test accuracy and laptop latency for the same three checkpoints."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    results = ROOT / "results"
    teacher = read_json(results / "teacher-test.json")
    students = read_json(results / "student_test.json")
    benchmark = read_json(results / "final/summary.json")
    pilot = read_json(results / "pilot/summary.json")
    for key in ("cpu", "torch", "dtype", "batch", "intraop_threads", "manifest_sha256"):
        if pilot[key] != benchmark[key]:
            raise ValueError(f"Pilot and final benchmark disagree on {key}")

    samples = teacher["samples"]
    hashes = {"teacher": teacher["checkpoint_sha256"], **students["checkpoints_sha256"]}
    accuracy = {"teacher": dict(n=samples, top1_correct=teacher["top1_correct"],
                                top3_correct=teacher["top3_correct"]), **students["models"]}
    comparison = {}
    for role, counts in accuracy.items():
        timing = benchmark["models"][role]
        if hashes[role] != timing["weights_sha256"] or counts["n"] != samples:
            raise ValueError(f"Accuracy and latency records disagree for {role}")
        comparison[role] = dict(
            test_n=samples, top1_pct=100 * counts["top1_correct"] / samples,
            top3_pct=100 * counts["top3_correct"] / samples, parameters=timing["parameters"],
            median_forward_ms=timing["stages"]["forward"]["median"],
            p95_end_to_end_ms=timing["stages"]["end_to_end"]["p95"],
        )

    with (results / "student_test_predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != samples or len({r["id"] for r in rows}) != samples:
        raise ValueError("Expected one paired prediction per official test photo")
    changes = {}
    for k in (1, 3):
        correct = {}
        for role in students["models"]:
            hits = {
                r["id"] for r in rows
                if any(r["label"] == r[f"{role}_top{i}"] for i in range(1, k + 1))
            }
            if len(hits) != accuracy[role][f"top{k}_correct"]:
                raise ValueError(f"Per-photo top-{k} outcomes disagree with the {role} test metrics")
            correct[role] = hits
        gains = len(correct["distilled"] - correct["plain"])
        losses = len(correct["plain"] - correct["distilled"])
        changes[f"top{k}"] = dict(gained=gains, lost=losses, delta_pp=100 * (gains - losses) / samples)
    print(json.dumps({"models": comparison, "distilled_vs_plain": changes}, indent=2))


if __name__ == "__main__":
    main()
