"""Export the CE student and prepare verified reference labels for paired Android series C."""
import csv
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import torch

from export_models import ROOT, export_one, sha256


def main():
    flatc = Path(sys.executable).with_name("flatc.exe")
    if os.name == "nt" and flatc.is_file():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assets = ROOT / "android/app/src/main/assets"
    control = ROOT / "android/app/src/control/assets"
    reference = json.loads((assets / "reference.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "results/student_test.json").read_text(encoding="utf-8"))
    predictions = ROOT / "results/student_test_predictions.csv"
    if sha256(predictions) != test["predictions_sha256"]:
        raise ValueError("Full-test prediction file differs from training record")
    with predictions.open(encoding="utf-8", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    reference["checkpoint_sha256"] = test["checkpoints_sha256"]
    reference["ce_prediction_source"] = "Full-test PyTorch FP32 prediction CSV, filtered to paired JPEG IDs"
    reference["ce_predictions_sha256"] = test["predictions_sha256"]
    for photo in reference["photos"]:
        row = rows[photo["id"]]
        if int(row["label"]) != photo["label"] or sha256(assets / photo["asset"]) != photo["sha256"]:
            raise ValueError(f"JPEG or reference class mismatch for {photo['id']}")
        photo["plain"] = [int(row[f"plain_top{k}"]) for k in (1, 2, 3)]
        del photo["teacher"]
    base = json.loads((ROOT / "results/android/export.json").read_text(encoding="utf-8"))
    dist = base["models"]["distilled"]
    if sha256(assets / "models/distilled.pte") != dist["pte_sha256"] or dist["checkpoint_sha256"] != test["checkpoints_sha256"]["distilled"]:
        raise ValueError("Distilled paired control differs from series A")
    plain = export_one("plain", ROOT / "checkpoints/plain.pt", assets, reference, model_dir=control / "models")
    reference["pte_sha256"] = {"plain": plain["pte_sha256"], "distilled": dist["pte_sha256"]}
    (control / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    report = dict(series="C", executorch=version("executorch"), torch=str(torch.__version__),
                  input_shape=[1, 3, 224, 224], dtype="FP32", models={"plain": plain, "distilled": dist})
    folder = ROOT / "results/android_ce"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
