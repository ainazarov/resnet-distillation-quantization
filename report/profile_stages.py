"""Measure inclusive ResNet layer1–layer4 CPU latency on the paired v5 inputs."""

import csv
import gzip
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch
import torchvision

from profile_layers import (
    ASSETS, ROOT, build_student, cpu_name, decode, digest, eval_transform,
    load_teacher, paired_models,
)


STAGES = ("layer1", "layer2", "layer3", "layer4")
REFERENCE = ROOT / "results/pilot/layers/summary.json"
OUTPUT = ROOT / "results/pilot/stages"


def timed_stage(name, forward, intervals, x):
    start = perf_counter_ns()
    output = forward(x)
    end = perf_counter_ns()
    intervals.append((name, start, end))
    return output


@torch.inference_mode()
def measure(models, inputs, tensors, warmups):
    for i in range(warmups):
        for role in paired_models(models, i):
            models[role](tensors[i % len(tensors)])
    rows, events, intervals = [], [], []
    origin = perf_counter_ns()

    def record(name, start, end, args):
        events.append({"name": name, "ph": "X", "cat": "boundary_timer", "pid": 1, "tid": 1,
                       "ts": (start - origin) / 1000, "dur": (end - start) / 1000, "args": args})

    with ExitStack() as wrappers:
        for model in models.values():
            for name in STAGES:
                stage = getattr(model, name)
                assert isinstance(stage, torch.nn.Sequential)
                wrappers.enter_context(patch.object(stage, "forward",
                                                    partial(timed_stage, name, stage.forward, intervals)))
        for i, (photo, x) in enumerate(zip(inputs, tensors, strict=True)):
            for order, role in enumerate(paired_models(models, i), 1):
                intervals.clear()
                start = perf_counter_ns()
                models[role](x)
                end = perf_counter_ns()
                assert tuple(name for name, _, _ in intervals) == STAGES
                context = {"pass_index": i + 1, "image_id": photo["id"], "model": role, "order": order}
                record("forward:" + role, start, end, context)
                previous = start
                for name, begin, finish in intervals:
                    assert previous <= begin < finish <= end
                    previous = finish
                    record(name, begin, finish, context)
                    rows.append({**context, "stage": name, "duration_ms": (finish - begin) / 1e6})
    return rows, events


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_paired_inputs():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    paths = {"resnet50": ROOT / "assets/resnet50_food101_best.pt",
             "resnet18": ROOT / "checkpoints/distilled.pt"}
    hashes = {role: digest(path) for role, path in paths.items()}
    assert hashes == ref["checkpoint_sha256"]
    models = {"resnet50": load_teacher(paths["resnet50"]), "resnet18": build_student(False)}
    models["resnet18"].load_state_dict(torch.load(paths["resnet18"], map_location="cpu", weights_only=True))
    for model in models.values():
        model.cpu().float().eval()

    assets = json.loads((ASSETS / "reference.json").read_text(encoding="utf-8-sig"))
    photos = {p["id"]: p for p in assets["photos"]}
    transform = eval_transform()
    tensors = []
    for photo in ref["inputs"]:
        path = ASSETS / photos[photo["id"]]["asset"]
        assert digest(path) == photo["jpeg_sha256"]
        x = transform(decode(path)).unsqueeze(0).contiguous()
        assert tuple(x.shape) == (1, 3, 224, 224) and x.dtype == torch.float32
        tensors.append(x)
    n = len(tensors)
    assert n == ref["profiled_forwards_per_model"] and n % 2 == 0
    return models, tensors, ref, hashes


def main():
    models, tensors, ref, hashes = load_paired_inputs()
    n = len(tensors)
    rows, events = measure(models, ref["inputs"], tensors, ref["warmup_per_model"])
    stages = []
    for stage in STAGES:
        times = {}
        for role in models:
            values = [r["duration_ms"] for r in rows if r["stage"] == stage and r["model"] == role]
            assert len(values) == n
            times[role] = mean(values)
        delta = times["resnet18"] - times["resnet50"]
        stages.append({"stage": stage, "resnet50_ms": times["resnet50"], "resnet18_ms": times["resnet18"],
                       "delta_ms": delta, "delta_percent": 100 * delta / times["resnet50"]})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT) as work:
        tmp = Path(work)
        write_csv(tmp / "stages.csv", stages)
        write_csv(tmp / "passes.csv", rows)
        with gzip.open(tmp / "paired-trace.json.gz", "wt", encoding="utf-8") as trace:
            json.dump({"displayTimeUnit": "ms", "traceEvents": events}, trace, separators=(",", ":"))
        filenames = ["stages.csv", "passes.csv", "paired-trace.json.gz"]
        summary = {"cpu": cpu_name(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "batch": 1, "dtype": "FP32", "shape": list(tensors[0].shape), "eval": True,
                   "inference_mode": True, "intraop_threads": torch.get_num_threads(),
                   "interop_threads": torch.get_num_interop_threads(), "warmup_per_model": ref["warmup_per_model"],
                   "measured_forwards_per_model": n, "order": ref["order"],
                   "timing": "perf_counter_ns immediately before and after the synchronous CPU forward of each complete layer1–layer4 Sequential. No per-operator hooks. Means, deltas and percentages use unrounded times.",
                   "checkpoint_sha256": hashes, "inputs": ref["inputs"], "stages": stages,
                   "file_sha256": {name: digest(tmp / name) for name in filenames}}
        (tmp / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        for name in filenames + ["summary.json"]:
            (tmp / name).replace(OUTPUT / name)
    print(json.dumps(stages, indent=2))


if __name__ == "__main__":
    main()
