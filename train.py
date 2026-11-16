"""Matched cross-entropy and distillation runs for the Food-101 diary."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import Food101


ROOT = Path(__file__).resolve().parent
IMAGENET_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    else:
        torch.save(obj, tmp)
    tmp.replace(path)


def initialize(out, assets, seed):
    torch.manual_seed(seed)
    imagenet = assets / "resnet18-f37072fd.pth"
    if imagenet.is_file():
        if digest(imagenet) != IMAGENET_SHA:
            raise ValueError(f"Unexpected ImageNet checkpoint hash for {imagenet}")
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(imagenet, weights_only=True, map_location="cpu"))
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 101)
    path = out / "checkpoints/student_init.pt"
    if path.exists():
        existing = torch.load(path, weights_only=True, map_location="cpu")
        if not all(torch.equal(existing[k], v) for k, v in model.state_dict().items()):
            raise ValueError(f"Existing initialization at {path} differs for seed {seed}")
    else:
        save(path, model.state_dict())
    print(f"Shared initialization: {path} sha256={digest(path)}", flush=True)
    return path


def image_id(dataset, index):
    return "/".join(dataset._image_files[index].with_suffix("").parts[-2:])


def training_split(data, classes_file, seed, results):
    from data_models import eval_transform, verify_classes

    # Keep stochastic preprocessing separate from the deterministic validation transform.
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    dataset = Food101(data, split="train", transform=aug)
    validation = Food101(data, split="train", transform=eval_transform())
    verify_classes(classes_file, dataset.classes)
    by_class = [[] for _ in dataset.classes]
    for i, label in enumerate(dataset._labels):
        by_class[label].append(i)
    train_indices, val_indices = [], []
    for group in by_class:
        if len(group) != 750:
            raise ValueError("Expected 750 training images for each Food-101 class")
        group.sort(key=lambda i: hashlib.sha256(f"{seed}/{image_id(dataset, i)}".encode()).digest())
        val_indices.extend(group[:50])
        train_indices.extend(group[50:])
    manifest = results / "split.csv"
    results.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "id", "label"])
        for name, indices in [("fit", train_indices), ("validation", val_indices)]:
            writer.writerows((name, image_id(dataset, i), dataset._labels[i]) for i in indices)
    if manifest.exists() and digest(manifest) != digest(tmp):
        raise ValueError(f"Existing run has a different training split at {manifest}")
    tmp.replace(manifest)
    return Subset(dataset, train_indices), Subset(validation, val_indices), digest(manifest)


def loader(dataset, args, seed=0, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                      generator=torch.Generator().manual_seed(seed))


def to_device(model, device):
    return model.to(device=device, memory_format=torch.channels_last)


@torch.inference_mode()
def score(models_by_role, dataset, args, keep_predictions=False):
    counts = {name: torch.zeros(2, device=args.device, dtype=torch.int64) for name in models_by_role}
    top3 = {name: [] for name in models_by_role}
    for model in models_by_role.values():
        model.eval()
    for images, labels in loader(dataset, args):
        images = images.to(args.device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(args.device, non_blocking=True)
        for name, model in models_by_role.items():
            ranked = model(images).topk(3).indices
            hits = ranked.eq(labels[:, None])
            counts[name] += torch.stack((hits[:, 0].sum(), hits.any(dim=1).sum()))
            if keep_predictions:
                top3[name].append(ranked.cpu())
    metrics = {}
    for name, count in counts.items():
        c1, c3 = count.tolist()
        metrics[name] = dict(n=len(dataset), top1_correct=c1, top3_correct=c3,
                             top1=c1 / len(dataset), top3=c3 / len(dataset))
    return metrics, {k: torch.cat(v).tolist() for k, v in top3.items() if v}


def student_from(path, device):
    from data_models import build_student

    model = build_student(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return to_device(model, device)


def train(args, role, fit, val, init, split_sha):
    from data_models import load_teacher

    distilled = role == "distilled"
    report_path = args.output / f"results/training/{role}.json"
    ckpt_dir = args.output / "checkpoints"
    best_path = ckpt_dir / f"{role}.pt"
    resume_path = ckpt_dir / f"{role}.last.pt"
    teacher_path = args.assets / "resnet50_food101_best.pt"
    config = dict(role=role, epochs=args.epochs, seed=args.seed, workers=args.workers,
                  batch=args.batch, lr=args.lr, weight_decay=0.01, optimizer="AdamW",
                  schedule="0.15 epoch linear warmup, cosine decay to 2% of peak",
                  selection="highest validation top3, then top1, then earliest epoch",
                  augment="RandomResizedCrop(224, scale=(0.65,1)), RandomHorizontalFlip(0.5)",
                  fit_n=len(fit), validation_n=len(val), split_sha256=split_sha,
                  initialization_sha256=digest(init), teacher_sha256=digest(teacher_path) if distilled else None,
                  loss="0.5 CE + 0.5 T^2 KL(teacher || student), T=2" if distilled else "CE",
                  training_dtype="bfloat16 autocast" if args.amp else "float32",
                  torch=str(torch.__version__), device=str(args.device))
    if report_path.is_file():
        record = json.loads(report_path.read_text())
        if record["config"] != config:
            raise ValueError(f"Existing configuration differs in {report_path}")
        if record["status"] == "complete" and digest(best_path) == record["checkpoint_sha256"]:
            print(f"Reusing completed {role} run", flush=True)
            return
    model = student_from(init, args.device)
    teacher = to_device(load_teacher(teacher_path), args.device).requires_grad_(False).eval() if distilled else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                  fused=args.device.type == "cuda")
    history, best_weights = [], None
    best_score = (-1, -1)
    selected_epoch = 0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError(f"Resume settings differ at {resume_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best_weights = saved["history"], saved["best_weights"]
        best_score, selected_epoch = tuple(saved["best_score"]), saved["selected_epoch"]
        del saved
    n_batches = math.ceil(len(fit) / args.batch)
    total_steps = args.epochs * n_batches
    warmup = max(1, round(0.15 * n_batches))
    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)

    for epoch in range(len(history) + 1, args.epochs + 1):
        model.train()
        epoch_seed = args.seed + epoch
        torch.manual_seed(epoch_seed)
        epoch_start = time.monotonic()
        total_loss = torch.zeros((), device=args.device)
        for batch_id, (images, labels) in enumerate(loader(fit, args, epoch_seed, shuffle=True)):
            if batch_id == 0:
                fingerprint = hashlib.sha256(images.numpy().tobytes())
                fingerprint.update(labels.numpy().tobytes())
                first_batch_sha = fingerprint.hexdigest()
                plain_report = args.output / "results/training/plain.json"
                if distilled and plain_report.exists():
                    plain_history = json.loads(plain_report.read_text())["history"]
                    if plain_history[epoch - 1]["first_batch_sha256"] != first_batch_sha:
                        raise ValueError(f"Student minibatches differ in epoch {epoch}")
            images = images.to(device=args.device, dtype=torch.float32, non_blocking=True,
                               memory_format=torch.channels_last).div_(255).sub_(mean).div_(std)
            labels = labels.to(args.device, non_blocking=True)
            step = (epoch - 1) * n_batches + batch_id
            progress = (step - warmup) / max(1, total_steps - 1 - warmup)
            factor = ((step + 1) / warmup if step < warmup else
                      0.02 + 0.98 * (1 + math.cos(math.pi * progress)) / 2)
            optimizer.param_groups[0]["lr"] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(args.device.type, dtype=torch.bfloat16, enabled=args.amp):
                if teacher is not None:
                    with torch.no_grad():
                        targets = F.softmax(teacher(images).float() / 2, dim=1)
                logits = model(images)
                loss = F.cross_entropy(logits.float(), labels)
                if teacher is not None:
                    kl = F.kl_div(F.log_softmax(logits.float() / 2, dim=1), targets, reduction="batchmean")
                    loss = 0.5 * loss + 0.5 * 4 * kl
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * labels.numel()
            done = batch_id + 1
            if done == 50 or done % 150 == 0:
                if args.device.type == "cuda":
                    torch.cuda.synchronize()
                seconds_per_step = (time.monotonic() - epoch_start) / done
                eta = seconds_per_step * (total_steps - step - 1)
                print(f"{role} epoch {epoch}/{args.epochs} step {done}/{n_batches}, "
                      f"{args.batch / seconds_per_step:.0f} img/s, "
                      f"remaining training batches ~{eta / 60:.1f} min (plus validation)", flush=True)
        mean_loss = float(total_loss / len(fit))
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"Non-finite {role} loss in epoch {epoch}")
        evaluated, _ = score({role: model}, val, args)
        metrics = evaluated[role]
        rank = (metrics["top3_correct"], metrics["top1_correct"])
        if rank > best_score:
            best_score, selected_epoch = rank, epoch
            best_weights = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch, loss=mean_loss, validation=metrics,
                            first_batch_sha256=first_batch_sha, seconds=time.monotonic() - epoch_start))
        save(resume_path, dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                               history=history, best_weights=best_weights, selected_epoch=selected_epoch,
                               best_score=best_score))
        save(report_path, dict(status="training", config=config, history=history, selected_epoch=selected_epoch))
        print(f"{role} epoch {epoch}: validation top1={metrics['top1']:.4f} top3={metrics['top3']:.4f}, "
              f"loss={mean_loss:.4f} ({history[-1]['seconds']:.1f}s)", flush=True)
    save(best_path, best_weights)
    save(report_path, dict(status="complete", config=config, history=history, selected_epoch=selected_epoch,
                           checkpoint_sha256=digest(best_path)))
    resume_path.unlink(missing_ok=True)


def test_students(args):
    from data_models import eval_transform, verify_classes

    test = Food101(args.data, split="test", transform=eval_transform())
    verify_classes(args.assets / "classes_food101.txt", test.classes)
    if len(test) != 25250:
        raise ValueError("Expected 25,250 official test images")
    selected = {}
    hashes = {}
    for role in ("plain", "distilled"):
        ckpt = args.output / f"checkpoints/{role}.pt"
        report = json.loads((args.output / f"results/training/{role}.json").read_text())
        hashes[role] = digest(ckpt)
        if report["status"] != "complete" or report["checkpoint_sha256"] != hashes[role]:
            raise ValueError(f"{role} checkpoint does not match training record")
        selected[role] = student_from(ckpt, args.device)
    metrics, ranked = score(selected, test, args, keep_predictions=True)
    results = args.output / "results"
    csv_path = results / "student_test_predictions.csv"
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", *[f"{role}_top{k}" for role in selected for k in (1, 2, 3)]])
        for i, label in enumerate(test._labels):
            writer.writerow([image_id(test, i), label, *ranked["plain"][i], *ranked["distilled"][i]])
    tmp.replace(csv_path)
    save(results / "student_test.json", dict(models=metrics, checkpoints_sha256=hashes, precision="FP32",
                                             predictions_sha256=digest(csv_path)))
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("init", "plain", "distilled", "test", "all"), default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch size and lr must be positive and workers must be nonnegative")
    args.device = torch.device(args.device)
    args.amp = args.device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.role != "test":
        init = initialize(args.output, args.assets, args.seed)
        if args.role == "init":
            return
        fit, val, split_sha = training_split(args.data, args.assets / "classes_food101.txt", args.seed,
                                             args.output / "results")
        for role in ("plain", "distilled") if args.role == "all" else (args.role,):
            train(args, role, fit, val, init, split_sha)
    if args.role in ("test", "all"):
        test_students(args)


if __name__ == "__main__":
    main()
