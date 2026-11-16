"""Collect three paired FP32/INT8 runs from the installed Food-101 PTQ APK."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


APP = "org.fooddiary.benchmark.ptq"
ACTIVITY = APP + "/org.fooddiary.benchmark.MainActivity"
ROLES = ("resnet18_fp32", "resnet18_int8")
HERE = Path(__file__).resolve().parent


def percentile(values, p):
    values = sorted(values)
    index = p * (len(values) - 1)
    left = int(index)
    return values[left] + (index - left) * (values[min(left + 1, len(values) - 1)] - values[left])


def validate(report, reference, hashes):
    if report["debugger_attached"] or report["threads"] != 4 or report["batch_size"] != 1:
        raise ValueError("Unexpected debugger or CPU settings")
    if report["reference_sha256"] != hashes["reference"]:
        raise ValueError("The phone and host used different reference files")
    for role in ROLES:
        rows = [r for r in report["rows"] if r["model"] == role]
        stats = report["models"][role]
        if len(rows) != 101 or {r["id"] for r in rows} != set(reference):
            raise ValueError(f"Unexpected JPEG sample for {role}")
        if stats["pte_sha256"] != hashes[role] or stats["backends"] != ["XnnpackBackend"]:
            raise ValueError(f"Wrong model or backend for {role}")
        for r in rows:
            if r["python_top3"] != reference[r["id"]][role] or not 0 < r["forward_ms"] <= r["jpeg_to_top3_ms"]:
                raise ValueError(f"Invalid timing or reference for {r['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", help="USB serial; defaults to the sole USB device")
    parser.add_argument("--out", type=Path, required=True, help="New directory for these phone runs")
    args = parser.parse_args()
    target = ["-s", args.serial] if args.serial else ["-d"]

    def adb(*command, check=True):
        result = subprocess.run([args.adb, *target, *command], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        if check and result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    assets = HERE / "app/src/precision/assets"
    ref = assets / "reference.json"
    reference = {r["id"]: r for r in json.loads(ref.read_text(encoding="utf-8"))["photos"]}
    hashes = {role: hashlib.sha256((assets / "models" / (role + ".pte")).read_bytes()).hexdigest()
              for role in ROLES}
    hashes["reference"] = hashlib.sha256(ref.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for run in range(1, 4):
        old = adb("exec-out", "run-as", APP, "cat", "files/benchmark.json", check=False)
        previous = json.loads(old)["finish_unix_ms"] if old.startswith("{") else 0
        adb("shell", "am", "start", "-S", "-W", "-n", ACTIVITY, "--ez", "benchmark", "true")
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            time.sleep(3)
            raw = adb("exec-out", "run-as", APP, "cat", "files/benchmark.json", check=False)
            if raw.startswith("{"):
                report = json.loads(raw)
                if report["start_unix_ms"] > previous:
                    break
        else:
            raise RuntimeError("No fresh report. Check the phone screen and Food101Bench logcat.")
        validate(report, reference, hashes)
        (args.out / f"run-{run}.json").write_text(raw + "\n", encoding="utf-8")
        reports.append(report)
        print(f"Run {run}: " + ", ".join(f"{k} p95 {report['models'][k]['p95_jpeg_to_top3_ms']:.2f} ms"
                                        for k in ROLES), flush=True)
        if run < 3:
            time.sleep(20)

    summary = {"runs": 3, "photos_per_run": 101, "cooldown_seconds": 20, "models": {}}
    for role in ROLES:
        rows = [r for report in reports for r in report["rows"] if r["model"] == role]
        summary["models"][role] = {
            "pte_sha256": hashes[role],
            "median_forward_ms": percentile([r["forward_ms"] for r in rows], 0.5),
            "p95_jpeg_to_top3_ms": percentile([r["jpeg_to_top3_ms"] for r in rows], 0.95),
            "ordered_python_matches": sum(r["top3"] == r["python_top3"] for r in rows),
            "requests": len(rows),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
