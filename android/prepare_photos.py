"""Package a fixed, class-balanced subset of the laptop benchmark and score it in PyTorch."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_models import build_student, eval_transform, load_teacher


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def prepare(images, assets):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = ROOT / "benchmark_jpegs.csv"
    classes_path = ROOT / "assets/classes_food101.txt"
    labels = classes_path.read_text(encoding="utf-8").splitlines()
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # Choose without looking at either model's predictions.
    selected = []
    for idx, category in enumerate(labels):
        choices = [row for row in rows if int(row["label"]) == idx]
        if len(choices) != 5 or any(r["id"].split("/")[0] != category for r in choices):
            raise ValueError(f"Unexpected benchmark manifest for {category}")
        selected.append(min(choices, key=lambda r: r["sha256"]))
    random.Random(42).shuffle(selected)

    teacher_path = ROOT / "assets/resnet50_food101_best.pt"
    distilled_path = ROOT / "checkpoints/distilled.pt"
    teacher = load_teacher(teacher_path)
    distilled = build_student()
    distilled.load_state_dict(torch.load(distilled_path, map_location="cpu", weights_only=True))
    distilled.eval()
    transform = eval_transform()
    (assets / "photos").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(classes_path, assets / classes_path.name)
    photos = []
    for row in selected:
        idx = int(row["label"])
        source = images / (row["id"] + ".jpg")
        if sha(source) != row["sha256"]:
            raise ValueError(f"Unexpected bytes for {source}")
        asset = f"photos/{idx:03d}.jpg"
        shutil.copyfile(source, assets / asset)
        with Image.open(source) as image:
            if image.format != "JPEG" or image.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Unexpected image dimensions or format for {source}")
            batch = transform(image.convert("RGB")).unsqueeze(0)
        photos.append(dict(id=row["id"], asset=asset, label=idx, sha256=row["sha256"],
                           width=int(row["width"]), height=int(row["height"]),
                           teacher=teacher(batch)[0].topk(3).indices.tolist(),
                           distilled=distilled(batch)[0].topk(3).indices.tolist()))
    reference = {
        "format_version": 1, "runtime": {"torch": str(torch.__version__),
                                        "torchvision": str(torchvision.__version__), "pillow": pillow_version},
        "source_manifest_sha256": sha(manifest), "classes_sha256": sha(classes_path),
        "checkpoint_sha256": {"teacher": sha(teacher_path), "distilled": sha(distilled_path)},
        "selection": "Lowest JPEG SHA-256 per class in the verified 505-image laptop manifest, shuffled with seed 42.",
        "photos": photos,
    }
    path = assets / "reference.json"
    path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"Scored {len(photos)} paired JPEGs; wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT.parent / "dam_proj_v5/data/photos")
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parent / "app/src/main/assets")
    args = parser.parse_args()
    prepare(args.images, args.assets)
