"""Time warm FP32 Food-101 inference on the same original JPEGs for each model."""
import argparse
import csv
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
import torchvision
from PIL import Image, __version__ as pillow_version
from torch.profiler import ProfilerActivity, profile, record_function

from data_models import build_student, eval_transform, load_teacher

ROOT = Path(__file__).resolve().parent
STAGES = ("decode", "preprocess", "forward", "top3", "end_to_end")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(images, manifest, classes):
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Benchmark IDs must be unique")
    if Counter(int(row["label"]) for row in rows) != {i: 5 for i in range(101)}:
        raise ValueError("The benchmark expects five original test JPEGs per class")
    result = []
    for row in rows:
        image_id = row["id"]
        path = images / (image_id + ".jpg")
        if image_id.split("/")[0] != classes[int(row["label"])] or digest(path) != row["sha256"]:
            raise ValueError(f"Image or class mismatch for {image_id}")
        result.append((image_id, path))
    random.Random(42).shuffle(result)
    return result


def decode(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def measure(net, path, transform, classes):
    t0 = perf_counter_ns()
    rgb = decode(path)
    t1 = perf_counter_ns()
    x = transform(rgb).unsqueeze(0)
    t2 = perf_counter_ns()
    logits = net(x)
    t3 = perf_counter_ns()
    indices = logits[0].topk(3).indices.tolist()
    names = [classes[i] for i in indices]
    t4 = perf_counter_ns()
    intervals = (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)
    return dict(zip(STAGES, (ns / 1e6 for ns in intervals))), names


def profile_forward(name, net, tensors, output):
    # Keep profiling out of the paired wall-clock measurements.
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        for x in tensors:
            with record_function("model_inference"):
                net(x)
    events = prof.key_averages()
    (output / f"{name}-operators.txt").write_text(
        events.table(sort_by="cpu_time_total", row_limit=40) + "\n\n"
        + events.table(sort_by="self_cpu_time_total", row_limit=40) + "\n", encoding="utf-8"
    )
    prof.export_chrome_trace(str(output / f"{name}-trace.json.gz"))
    operators = [e for e in events if e.key.startswith("aten::")]
    denominator = sum(e.self_cpu_time_total for e in events)
    return [
        {"operator": e.key, "calls": e.count, "self_ms": e.self_cpu_time_total / 1000,
         "percent_of_profile_self_time": 100 * e.self_cpu_time_total / denominator}
        for e in sorted(operators, key=lambda e: -e.self_cpu_time_total)[:10]
    ]


def cpu_name():
    if platform.system() == "Windows":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor()


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.out or ROOT / "results" / args.phase
    output.mkdir(parents=True, exist_ok=True)
    classes = (args.assets / "classes_food101.txt").read_text(encoding="utf-8").splitlines()
    files = samples(args.images, args.manifest, classes)
    transform = eval_transform()
    paths = {"teacher": args.assets / "resnet50_food101_best.pt"}
    for name in (("student_init",) if args.phase == "pilot" else ("plain", "distilled")):
        paths[name] = args.checkpoints / (name + ".pt")
    models = {}
    for name, path in paths.items():
        if name == "teacher":
            model = load_teacher(path)
        else:
            model = build_student(pretrained=False)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        models[name] = model.cpu().float().eval()

    for _, path in files[:args.warmup]:
        for model in models.values():
            measure(model, path, transform, classes)

    rows = []
    order = list(models)
    for i, (image_id, path) in enumerate(files):
        # Cycle first/last position to balance cache and clock effects.
        offset = i % len(order)
        for position, name in enumerate(order[offset:] + order[:offset]):
            elapsed, names = measure(models[name], path, transform, classes)
            rows.append({"image": image_id, "model": name, "position": position,
                         **elapsed, "suggestions": "|".join(names)})
    with (output / "timings_ms.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "phase": args.phase, "images": len(files), "warmup_per_model": args.warmup,
        "manifest_sha256": digest(args.manifest), "cpu": cpu_name(), "os": platform.platform(),
        "batch": 1, "dtype": "FP32", "intraop_threads": args.threads, "interop_threads": 1,
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "pillow": pillow_version,
        "latency_unit": "ms", "file_cache": "warm (every file hash verified before timing)",
        "end_to_end": "local file open through three class names; excludes model loading and network",
        "profiled_forwards_per_model": 3 if args.phase == "pilot" else 0, "models": {},
    }
    for name, model in models.items():
        observations = [r for r in rows if r["model"] == name]
        total = sum(r["end_to_end"] for r in observations)
        result = {"parameters": sum(p.numel() for p in model.parameters()),
                  "weights_sha256": digest(paths[name]), "stages": {}}
        for stage in STAGES:
            values = [r[stage] for r in observations]
            p50, p95 = np.percentile(values, (50, 95)).tolist()
            result["stages"][stage] = {"median": p50, "p95": p95, "percent_of_elapsed": 100 * sum(values) / total}
        summary["models"][name] = result
    if args.phase == "pilot":
        tensors = [transform(decode(path)).unsqueeze(0) for _, path in files[:3]]
        for name, model in models.items():
            summary["models"][name]["profile"] = profile_forward(name, model, tensors, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output / "summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("pilot", "final"))
    p.add_argument("--images", type=Path, default=ROOT.parent / "dam_proj_v5/data/photos",
                   help="Directory of original JPEGs organized by class, e.g. food-101/images")
    p.add_argument("--manifest", type=Path, default=ROOT / "benchmark_jpegs.csv")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--out", type=Path)
    p.add_argument("--warmup", type=int, default=20, choices=range(1, 506), metavar="1..505")
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())
