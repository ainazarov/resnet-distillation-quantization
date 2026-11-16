"""Measure an exported PTE's accuracy on each original official Food-101 test JPEG."""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def init_worker(model, images):
    global forward, program, transform, image_root
    import torch
    from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool
    from executorch.runtime import Runtime
    from data_models import eval_transform

    # Models have a fixed batch of one. Independent workers handle different photos.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _unsafe_reset_threadpool(1)
    image_root = images
    transform = eval_transform()
    program = Runtime.get().load_program(model)
    forward = program.load_method("forward")


def score_chunk(items):
    import torch
    from PIL import Image

    rows = []
    with torch.inference_mode():
        for image_id, label in items:
            with Image.open(image_root / (image_id + ".jpg")) as photo:
                x = transform(photo.convert("RGB")).unsqueeze(0)
            y = forward.execute((x,))[0]
            if y.shape != (1, 101) or not torch.isfinite(y).all():
                raise ValueError(f"Invalid output for {image_id}")
            ranked = y[0].argsort(descending=True, stable=True)[:4].tolist()
            boundary_tie = bool(y[0, ranked[2]] == y[0, ranked[3]])
            rows.append((image_id, label, *ranked[:3], boundary_tie))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--data", default=ROOT / "data/food-101", type=Path)
    parser.add_argument("--results", default=ROOT / "results/android_ptq", type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    classes = (ROOT / "assets/classes_food101.txt").read_text().splitlines()
    class_index = {c: i for i, c in enumerate(classes)}
    meta = args.data / "meta"
    manifest = meta / "test.txt"
    ids = manifest.read_text().splitlines()
    test_json = json.loads((meta / "test.json").read_text())
    if classes != sorted(test_json) or len(set(ids)) != 25250 or len(ids) != 25250:
        raise ValueError("Unexpected test size, duplicate IDs, or class order")
    if set(ids) != {x for values in test_json.values() for x in values}:
        raise ValueError("Official test manifests disagree")
    items = [(s, class_index[s.split("/")[0]]) for s in ids]
    if Counter(n for _, n in items) != {i: 250 for i in range(101)}:
        raise ValueError("Expected 250 test photos per class")
    with (args.results / "calibration.csv").open(newline="") as f:
        calibration = {row["id"] for row in csv.DictReader(f)}
    if calibration & set(ids) or len(calibration) != 505:
        raise ValueError("Expected 505 calibration IDs disjoint from test")

    tic = time.monotonic()
    work = [items[i:i+128] for i in range(0, len(items), 128)]
    predictions = []
    # A fresh interpreter initializes its own ExecuTorch XNNPACK runtime.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(args.model, args.data / "images")) as pool:
        for future in as_completed([pool.submit(score_chunk, items_chunk) for items_chunk in work]):
            predictions.extend(future.result())
            if len(predictions) % 2048 == 0:
                print(f"{len(predictions)}/{len(items)} at {time.monotonic()-tic:.0f}s", flush=True)
    by_id = {row[0]: row for row in predictions}
    if len(predictions) != len(items) or set(by_id) != set(ids):
        raise ValueError("Incomplete or duplicated evaluation results")
    rows = [by_id[s] for s in ids]
    hits1 = sum(row[1] == row[2] for row in rows)
    hits3 = sum(row[1] in row[2:5] for row in rows)
    summary = dict(model=args.model.name, model_sha256=sha256(args.model), samples=len(ids),
                   test_manifest_sha256=sha256(manifest), calibration_overlap=0,
                   top1_correct=hits1, top3_correct=hits3, top1_percent=100*hits1/len(ids),
                   top3_percent=100*hits3/len(ids), executorch=version("executorch"),
                   torch=version("torch"), torchvision=version("torchvision"), pillow=version("pillow"),
                   top3_boundary_ties=sum(row[5] for row in rows),
                   tie_break="ascending class index", device="CPU", elapsed_seconds=time.monotonic()-tic)
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / (args.model.stem + "_test.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "top1", "top2", "top3", "top3_boundary_tie"])
        w.writerows(rows)
    summary["predictions_sha256"] = sha256(path)
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
