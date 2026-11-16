"""Calibrate the selected Food-101 student and export a matched FP32/static-INT8 pair."""

import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from export_models import ROOT, build_student, eval_transform, sha256
from inspect_xnnpack import audit_int8

ASSETS = ROOT / "android/app/src/main/assets"
PRECISION = ROOT / "android/app/src/precision/assets"
RESULTS = ROOT / "results/android_ptq"
FP32, INT8 = "resnet18_fp32", "resnet18_int8"


def write_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def calibration_set():
    with (RESULTS / "calibration.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with (ROOT / "results/split.csv").open(encoding="utf-8-sig", newline="") as f:
        fit = {r["id"]: r["label"] for r in csv.DictReader(f) if r["split"] == "fit"}
    if (len({r["id"] for r in rows}) != 505
            or Counter(int(r["label"]) for r in rows) != {i: 5 for i in range(101)}):
        raise ValueError("Expected 505 unique fit photos, five from each class")
    paths = []
    for r in rows:
        path = ROOT / "data/calibration" / (r["id"] + ".jpg")
        if fit.get(r["id"]) != r["label"] or sha256(path) != r["sha256"]:
            raise ValueError(f"Calibration membership or hash mismatch: {r['id']}")
        paths.append(path)
    return paths


def input_tensor(path, transform):
    with Image.open(path) as src:
        return transform(src.convert("RGB")).unsqueeze(0)


def make_reference(reference, quantized, transform, paths):
    runtime = Runtime.get()
    programs = {name: runtime.load_program(path) for name, path in paths.items()}
    methods = {name: p.load_method("forward") for name, p in programs.items()}
    eager_comparison = []
    for i, photo in enumerate(reference["photos"]):
        jpeg = ASSETS / photo["asset"]
        if sha256(jpeg) != photo["sha256"]:
            raise ValueError(f"Mobile JPEG mismatch: {photo['id']}")
        x = input_tensor(jpeg, transform)
        for name, method in methods.items():
            logits = method.execute((x,))[0]
            if logits.shape != (1, 101) or not logits.isfinite().all():
                raise ValueError(f"Invalid output for {name} on {photo['id']}")
            pred = logits[0].argsort(descending=True, stable=True)[:3].tolist()
            photo[name] = pred
            if name == FP32 and pred != photo["distilled"]:
                raise ValueError(f"FP32 disagrees with previous Python reference on {photo['id']}")
            if name == INT8 and i < 5:
                eager = quantized(x)
                eager_top3 = eager[0].argsort(descending=True, stable=True)[:3].tolist()
                eager_comparison.append(dict(id=photo["id"], eager_top3=eager_top3, pte_top3=pred,
                                             max_abs_logit_difference=(eager-logits).abs().max().item()))
        del photo["teacher"], photo["distilled"]
    reference["pte_sha256"] = {name: sha256(path) for name, path in paths.items()}
    ckpt_hash = reference["checkpoint_sha256"]["distilled"]
    reference["checkpoint_sha256"] = {name: ckpt_hash for name in paths}
    reference["runtime"] = dict(executorch=version("executorch"), torch=torch.__version__,
                                torchvision=version("torchvision"), pillow=version("pillow"))
    reference["ranking"] = "Desktop PTE logits, descending stable sort (lower class index breaks ties)"
    write_json(PRECISION / "reference.json", reference)
    return eager_comparison


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    flatc = Path(sys.executable).with_name("flatc.exe")
    if os.name == "nt" and flatc.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc))
    reference = json.loads((ASSETS / "reference.json").read_text(encoding="utf-8"))
    prev = json.loads((ROOT / "results/android/export.json").read_text(encoding="utf-8"))["models"]["distilled"]
    checkpoint = ROOT / "checkpoints/distilled.pt"
    baseline = ASSETS / "models/distilled.pte"
    if (sha256(checkpoint) != prev["checkpoint_sha256"] or sha256(baseline) != prev["pte_sha256"]
            or reference["checkpoint_sha256"]["distilled"] != prev["checkpoint_sha256"]):
        raise ValueError("Selected checkpoint or its existing verified FP32 export has changed")
    model_dir = PRECISION / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: model_dir / (name + ".pte") for name in (FP32, INT8)}
    shutil.copyfile(baseline, paths[FP32])
    calibration = calibration_set()
    transform = eval_transform()
    student = build_student()
    student.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    student.float().requires_grad_(False).eval()
    example = (torch.zeros(1, 3, 224, 224),)
    config = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False, is_qat=False)
    quantizer = XNNPACKQuantizer().set_global(config)
    captured = torch.export.export(student, example, strict=False).module()
    prepared = prepare_pt2e(captured, quantizer)
    start = time.monotonic()
    with torch.no_grad():
        for i, path in enumerate(calibration, start=1):
            prepared(input_tensor(path, transform))
            if i % 50 == 0 or i == len(calibration):
                print(f"Calibrated {i}/{len(calibration)} in {time.monotonic()-start:.1f}s", flush=True)
    quantized = convert_pt2e(prepared)
    exported = torch.export.export(quantized, example, strict=False)
    edge = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()], transform_passes=get_transform_passes(),
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
    serialized = edge.to_executorch()
    plan = serialized.executorch_program.execution_plan[0]
    staged = paths[INT8].with_suffix(".pte.tmp")
    with staged.open("wb") as dst:
        serialized.write_to_file(dst)
    staged.replace(paths[INT8])
    print(f"Exported {paths[INT8]}", flush=True)
    with torch.no_grad():
        checks = make_reference(reference, quantized, transform, paths)
    report = dict(executorch=version("executorch"), torch=torch.__version__, torchao=version("torchao"),
                  checkpoint_sha256=sha256(checkpoint), calibration_count=len(calibration),
                  calibration_manifest_sha256=sha256(RESULTS / "calibration.csv"),
                  precision_reference_sha256=sha256(PRECISION / "reference.json"),
                  calibration_transform=str(transform), activation_observer="HistogramObserver",
                  quantization="static per-tensor affine signed INT8 activations, symmetric per-channel INT8 weights",
                  models={name:dict(bytes=path.stat().st_size, sha256=sha256(path)) for name,path in paths.items()},
                  delegates=[d.id for d in plan.delegates],
                  portable_operators=[op.name for op in plan.operators], converted_pt2e_vs_int8_pte=checks,
                  int8_delegate_graph=audit_int8(paths[INT8], len(plan.delegates)))
    write_json(RESULTS / "export.json", report)
    print(RESULTS / "export.json", flush=True)


if __name__ == "__main__":
    main()
