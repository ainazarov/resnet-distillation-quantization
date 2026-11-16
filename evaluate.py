"""Score a Food-101 checkpoint on all 25,250 official test images."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from data_models import build_student, eval_transform, load_teacher, verify_classes


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(args):
    meta_dir = args.data / "food-101" / "meta"
    data = datasets.Food101(args.data, split="test", transform=eval_transform())
    classes = verify_classes(args.classes, data.classes)
    if data.class_to_idx != {name: i for i, name in enumerate(classes)}:
        raise ValueError("Food101 class_to_idx differs from the teacher logits")
    raw_classes = (meta_dir / "classes.txt").read_text().splitlines()
    if sorted(raw_classes) != sorted(classes):
        raise ValueError("Archive class inventory differs from the teacher")
    # Food101 sorts the JSON class keys, while the archive text swaps two rows.
    raw_order_differences = [
        dict(index=i, archive=raw, dataset=name)
        for i, (raw, name) in enumerate(zip(raw_classes, classes)) if raw != name
    ]
    split_ids = (meta_dir / "test.txt").read_text().splitlines()
    if len(set(split_ids)) != 25250 or len(data) != 25250:
        raise ValueError("Expected 25,250 distinct official test images")
    test_json = json.loads((meta_dir / "test.json").read_text())
    if Counter(split_ids) != Counter(i for ids in test_json.values() for i in ids):
        raise ValueError("Test manifests disagree")
    if any(len(test_json[c]) != 250 or any(i.split("/")[0] != c for i in test_json[c]) for c in classes):
        raise ValueError("Test set must contain 250 photos for every class")

    model_path = args.teacher or args.student
    if args.teacher:
        model = load_teacher(model_path)
    else:
        model = build_student()
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    device = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model.to(device=device, memory_format=torch.channels_last).eval()
    loader = DataLoader(data, batch_size=args.batch, num_workers=args.workers,
                        shuffle=False)

    started = time.monotonic()
    hits1 = hits3 = seen = 0
    for images, labels in loader:
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        pred = model(images).topk(3, dim=1).indices.cpu()
        correct = pred.eq(labels[:, None])
        hits1 += correct[:, 0].sum().item()
        hits3 += correct.any(dim=1).sum().item()
        seen += len(labels)
        if seen % (args.batch * 50) == 0:
            print(f"Scored {seen}/{len(data)}", flush=True)
    if seen != len(data):
        raise RuntimeError(f"Scored only {seen} of {len(data)} test images")
    result = dict(model=model_path.name, checkpoint_sha256=file_sha256(model_path),
                  classes_match_dataset=True, class_count=len(classes),
                  archive_text_order_differences=raw_order_differences,
                  classes_sha256=file_sha256(args.classes),
                  test_manifest_sha256=file_sha256(meta_dir / "test.txt"),
                  samples=seen, top1_correct=hits1, top3_correct=hits3,
                  top1_percent=100 * hits1 / seen, top3_percent=100 * hits3 / seen,
                  parameters=sum(p.numel() for p in model.parameters()),
                  device=str(device), dtype="float32", tf32=False,
                  seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    kind = p.add_mutually_exclusive_group(required=True)
    kind.add_argument("--teacher", type=Path)
    kind.add_argument("--student", type=Path)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--classes", type=Path, default=Path("assets/classes_food101.txt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    evaluate(p.parse_args())
