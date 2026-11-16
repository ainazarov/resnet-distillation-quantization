"""Repeat the paired Intel pilot with named Conv2d scopes on the trained Food-101 models."""
import csv
import gzip
import json
import platform
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torchvision
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import cpu_name, decode, digest
from data_models import build_student, eval_transform, load_teacher

ASSETS = ROOT / "android/app/src/main/assets"
OUTPUT = ROOT / "results/pilot/layers"
TRACE = "paired-trace.json.gz"
PASSES = 16
WARMUPS = 20
GROUPS = ("main_1x1", "shortcut_1x1", "block_3x3", "stem_7x7")
EXPECTED_COUNTS = {"resnet50": (32, 4, 16, 1), "resnet18": (0, 3, 16, 1)}
BACKEND = "aten::mkldnn_convolution"


def describe_layer(name, module):
    height, width = module.kernel_size
    stage = "stem" if name == "conv1" else name.split(".")[0]
    if (name, height, width) == ("conv1", 7, 7):
        group = "stem_7x7"
    elif height == width == 1:
        group = "shortcut_1x1" if ".downsample." in name else "main_1x1"
    elif height == width == 3 and ".downsample." not in name:
        group = "block_3x3"
    else:
        raise ValueError(f"Unexpected convolution {name}: {module.kernel_size}")
    return {"stage": stage, "group": group, "kernel": f"{height}x{width}"}


def labeled_forward(label, forward, x):
    with record_function(label):
        return forward(x)


def paired_models(models, index):
    roles = list(models)
    return roles[::-1] if index % 2 else roles


@torch.inference_mode()
def capture(models, catalog, tensors, path):
    for i in range(WARMUPS):
        for role in paired_models(models, i):
            models[role](tensors[i % len(tensors)])
    print(f"Warmed both models, recording {len(tensors)} paired forwards.", flush=True)
    with ExitStack() as scopes:
        for role, model in models.items():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    catalog[role][name] = describe_layer(name, module)
                    wrapped = partial(labeled_forward, "Conv2d:" + name, module.forward)
                    scopes.enter_context(patch.object(module, "forward", wrapped))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            for i, x in enumerate(tensors):
                for role in paired_models(models, i):
                    with record_function(f"forward:{role}:{i}"):
                        models[role](x)
    prof.export_chrome_trace(str(path))


def within(inner, outer):
    return (inner["pid"] == outer["pid"] and inner["tid"] == outer["tid"]
            and outer["ts"] <= inner["ts"]
            and inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"])


def summarize(path, catalog):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    forwards = [e for e in events if e["name"].startswith("forward:")]
    labels = [e for e in events if e["name"].startswith("Conv2d:")]
    kernels = [e for e in events if e["name"] == BACKEND]
    expected_order = [f"forward:{role}:{i}" for i in range(PASSES) for role in paired_models(catalog, i)]
    assert [e["name"] for e in sorted(forwards, key=lambda e: e["ts"])] == expected_order
    durations = defaultdict(list)
    used_kernels = set()
    for span in forwards:
        role = span["name"].split(":")[1]
        scopes = [e for e in labels if within(e, span)]
        ops = [e for e in kernels if within(e, span)]
        names = [e["name"].removeprefix("Conv2d:") for e in scopes]
        assert Counter(names) == Counter(catalog[role].keys())
        for name, scope in zip(names, scopes):
            # Labels identify the layer. Only the enclosed backend event supplies time.
            matches = [e for e in ops if within(e, scope)]
            assert len(matches) == 1, (span["name"], name, len(matches))
            event = matches[0]
            assert id(event) not in used_kernels
            used_kernels.add(id(event))
            weight = event["args"]["Input Dims"][1]
            assert catalog[role][name]["kernel"] == f"{weight[-2]}x{weight[-1]}"
            durations[role, name].append(event["dur"])
    assert len(used_kernels) == len(kernels) == len(labels)
    rows, groups = [], {}
    for role, layers in catalog.items():
        groups[role] = {g: {"calls_per_forward": 0, "mean_ms_per_forward": 0.0} for g in GROUPS}
        for name, info in layers.items():
            times = durations[role, name]
            assert len(times) == PASSES
            ms = sum(times) / (1000 * PASSES)
            rows.append({"model": role, "layer": name, **info, "calls": len(times),
                         "calls_per_forward": 1, "mean_ms_per_forward": f"{ms:.9f}"})
            group = groups[role][info["group"]]
            group["calls_per_forward"] += 1
            group["mean_ms_per_forward"] += ms
        assert tuple(groups[role][g]["calls_per_forward"] for g in GROUPS) == EXPECTED_COUNTS[role]
    return rows, groups


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    paths = {"resnet50": ROOT / "assets/resnet50_food101_best.pt",
             "resnet18": ROOT / "checkpoints/distilled.pt"}
    hashes = {role: digest(path) for role, path in paths.items()}
    refs = json.loads((ASSETS / "reference.json").read_text(encoding="utf-8-sig"))
    assert hashes == {"resnet50": refs["checkpoint_sha256"]["teacher"],
                      "resnet18": refs["checkpoint_sha256"]["distilled"]}
    models = {"resnet50": load_teacher(paths["resnet50"]), "resnet18": build_student(False)}
    models["resnet18"].load_state_dict(torch.load(paths["resnet18"], weights_only=True, map_location="cpu"))
    for model in models.values():
        model.cpu().float().eval()
    tensors, inputs = [], []
    transform = eval_transform()
    for photo in refs["photos"][:PASSES]:
        path = ASSETS / photo["asset"]
        assert digest(path) == photo["sha256"]
        x = transform(decode(path)).unsqueeze(0).contiguous()
        assert x.shape == (1, 3, 224, 224) and x.dtype == torch.float32
        tensors.append(x)
        inputs.append({"id": photo["id"], "jpeg_sha256": photo["sha256"]})
    assert len({image["id"] for image in inputs}) == PASSES
    catalog = {role: {} for role in models}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT) as tmp:
        tmp = Path(tmp)
        capture(models, catalog, tensors, tmp / TRACE)
        rows, groups = summarize(tmp / TRACE, catalog)
        with (tmp / "layers.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "cpu": cpu_name(), "os": platform.platform(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "batch": 1, "shape": list(tensors[0].shape),
            "dtype": "FP32", "eval": True, "inference_mode": True,
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "warmup_per_model": WARMUPS, "profiled_forwards_per_model": PASSES,
            "order": "teacher first on even input indices, distilled student first on odd indices",
            "timing": "sum of inclusive durations of the unique aten::mkldnn_convolution nested in each named Conv2d scope, divided by forwards and converted from us to ms",
            "checkpoint_sha256": hashes, "inputs": inputs, "groups": groups,
            "trace_sha256": digest(tmp / TRACE), "layers_csv_sha256": digest(tmp / "layers.csv"),
        }
        (tmp / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in [TRACE, "layers.csv", "summary.json"]:
            (tmp / name).replace(OUTPUT / name)
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
