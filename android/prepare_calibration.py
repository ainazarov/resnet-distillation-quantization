"""Select five fit-set JPEGs per class, checking against the official split manifests."""

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(dataset, output, manifest):
    classes = (ROOT / "assets/classes_food101.txt").read_text().splitlines()
    official_train = set((dataset / "meta/train.txt").read_text().splitlines())
    official_test = set((dataset / "meta/test.txt").read_text().splitlines())
    if len(classes) != 101 or len(official_train) != 75750 or len(official_test) != 25250:
        raise ValueError("Unexpected official Food-101 split or class count")
    groups = defaultdict(list)
    seen = set()
    with (ROOT / "results/split.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            image_id, label = row["id"], int(row["label"])
            if image_id in seen or image_id.split("/")[0] != classes[label]:
                raise ValueError(f"Invalid split row {row}")
            seen.add(image_id)
            if row["split"] == "fit":
                groups[label].append(image_id)
            elif row["split"] != "validation":
                raise ValueError(f"Unknown split in {row}")
    if seen != official_train or seen & official_test:
        raise ValueError("Fit and validation must partition only the official training split")

    rows = []
    for label in range(len(classes)):
        if len(groups[label]) != 700:
            raise ValueError(f"Expected 700 fit photos for {classes[label]}")
        ordered = sorted(groups[label], key=lambda s: hashlib.sha256(("food101-ptq-42/" + s).encode()).digest())
        for image_id in ordered[:5]:
            name = image_id + ".jpg"
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dataset / "images" / name, destination)
            rows.append([image_id, label, hashlib.sha256(destination.read_bytes()).hexdigest()])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "sha256"])
        w.writerows(rows)
    print(f"Verified and copied {len(rows)} fit JPEGs from {len(groups)} classes")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data/food-101")
    p.add_argument("--output", type=Path, default=ROOT / "data/calibration")
    p.add_argument("--manifest", type=Path, default=ROOT / "results/android_ptq/calibration.csv")
    args = p.parse_args()
    prepare(args.dataset, args.output, args.manifest)
