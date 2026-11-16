"""Time the consecutive parts of a torchvision ResNet on the paired Intel CPU pilot inputs."""
import json
import platform
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import torch
import torchvision

from profile_stages import ROOT, cpu_name, digest, load_paired_inputs, paired_models, write_csv


OUTPUT = ROOT / "results/pilot/components"
PARTS = ("stem", "maxpool", "layer1", "layer2", "layer3", "layer4", "gap", "flatten", "classifier")


def timed_forward(model, x):
    # Same operation order as torchvision.models.resnet.ResNet._forward_impl.
    # Read the clock at shared boundaries, with no per-operator hooks.
    start = perf_counter_ns()
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    stem = perf_counter_ns()
    x = model.maxpool(x)
    pool = perf_counter_ns()
    x = model.layer1(x)
    stage1 = perf_counter_ns()
    x = model.layer2(x)
    stage2 = perf_counter_ns()
    x = model.layer3(x)
    stage3 = perf_counter_ns()
    x = model.layer4(x)
    stage4 = perf_counter_ns()
    x = model.avgpool(x)
    gap = perf_counter_ns()
    x = torch.flatten(x, 1)
    flatten = perf_counter_ns()
    x = model.fc(x)
    finish = perf_counter_ns()
    stamps = (start, stem, pool, stage1, stage2, stage3, stage4, gap, flatten, finish)
    durations = {name: b - a for name, a, b in zip(PARTS, stamps[:-1], stamps[1:], strict=True)}
    assert sum(durations.values()) == finish - start and min(durations.values()) > 0
    durations["head"] = finish - stage4
    durations["forward"] = finish - start
    return x, durations


@torch.inference_mode()
def main():
    models, tensors, ref, hashes = load_paired_inputs()
    images = ref["inputs"]
    for i in range(ref["warmup_per_model"]):
        for role in paired_models(models, i):
            timed_forward(models[role], tensors[i % len(tensors)])
    rows, logits = [], {}
    for i, (image, x) in enumerate(zip(images, tensors, strict=True)):
        for order, role in enumerate(paired_models(models, i), 1):
            output, durations = timed_forward(models[role], x)
            logits[i, role] = output
            for part, ns in durations.items():
                rows.append({"pass_index": i + 1, "image_id": image["id"], "model": role, "order": order,
                             "component": part, "duration_ns": ns})
    for i, x in enumerate(tensors):
        for role, model in models.items():
            direct = model(x)
            assert direct.shape == (1, 101) and torch.equal(logits[i, role], direct), (i, role)

    summary_rows = []
    for part in (*PARTS, "head", "forward"):
        times = {}
        for role in models:
            samples = [r["duration_ns"] for r in rows if r["component"] == part and r["model"] == role]
            assert len(samples) == len(tensors)
            times[role] = mean(samples) / 1e6
        teacher, student = times["resnet50"], times["resnet18"]
        delta = student - teacher
        summary_rows.append({"component": part, "resnet50_ms": teacher, "resnet18_ms": student,
                             "delta_ms": delta, "delta_percent": 100 * delta / teacher})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT) as work:
        tmp = Path(work)
        write_csv(tmp / "passes.csv", rows)
        write_csv(tmp / "components.csv", summary_rows)
        summary = {
            "cpu": cpu_name(), "os": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "dtype": "FP32",
            "batch": 1, "shape": list(tensors[0].shape), "eval": True, "inference_mode": True,
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "warmup_per_model": ref["warmup_per_model"], "measured_forwards_per_model": len(tensors),
            "order": ref["order"], "checkpoint_sha256": hashes, "inputs": images,
            "verification": "For all 16 input IDs, all 101 logits in each timed forward equal those from model(x) bit for bit.",
            "timing": "perf_counter_ns reads at consecutive module boundaries in the synchronous torchvision forward sequence. Inputs are prepared before timing. Reported times are arithmetic means of 16 paired individual passes.",
            "components": {
                "stem": "conv1, bn1, relu", "maxpool": "maxpool",
                "layer1–layer4": "Four complete residual stages including both branches.",
                "gap": "avgpool", "flatten": "torch.flatten(x, 1)", "classifier": "fc; teacher includes dropout in eval mode",
                "head": "Aggregate: gap + flatten + classifier",
                "forward": "Aggregate: prepared RGB tensor to logits (all nine non-overlapping components)."
            },
            "file_sha256": {name: digest(tmp / name) for name in ("components.csv", "passes.csv")},
        }
        (tmp / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        for name in ("components.csv", "passes.csv", "summary.json"):
            (tmp / name).replace(OUTPUT / name)
    print((OUTPUT / "components.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
