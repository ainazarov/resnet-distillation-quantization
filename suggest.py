"""Print the three most likely Food-101 dishes for one photo."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from data_models import build_student, eval_transform


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/distilled.pt")
    parser.add_argument("--classes", type=Path, default=ROOT / "assets/classes_food101.txt")
    args = parser.parse_args()

    classes = args.classes.read_text(encoding="utf-8").splitlines()
    if len(classes) != 101 or len(set(classes)) != 101:
        parser.error("Expected 101 distinct Food-101 labels in checkpoint order.")
    model = build_student(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with Image.open(args.photo) as photo:
        batch = eval_transform()(photo.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        scores, indices = model(batch).softmax(dim=1)[0].topk(3)
    for score, idx in zip(scores.tolist(), indices.tolist()):
        print(f"{classes[idx].replace('_', ' ')}\t{score:.1%}")


if __name__ == "__main__":
    main()
