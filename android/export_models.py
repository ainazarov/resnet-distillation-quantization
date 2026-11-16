"""Export the two selected Food-101 checkpoints and verify on packaged JPEGs."""

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_models import build_student, eval_transform, load_teacher


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_one(role, checkpoint, assets, reference, *, model_dir=None):
    if sha256(checkpoint) != reference["checkpoint_sha256"][role]:
        raise ValueError(f"Python reference and {role} checkpoint hashes differ")
    if role == "teacher":
        model = load_teacher(checkpoint)
    else:
        model = build_student()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.float().requires_grad_(False).eval()
    example = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    exported = torch.export.export(model, (example,), strict=False)
    edge = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()], transform_passes=get_transform_passes(),
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
    serialized = edge.to_executorch()
    plan = serialized.executorch_program.execution_plan[0]
    delegates = [delegate.id for delegate in plan.delegates]
    if not delegates or set(delegates) != {"XnnpackBackend"}:
        raise ValueError(f"Unexpected {role} delegates: {delegates}")
    destination = (model_dir or assets / "models") / f"{role}.pte"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_suffix(".pte.tmp")
    with staged.open("wb") as output:
        serialized.write_to_file(output)
    program = Runtime.get().load_program(staged)
    forward = program.load_method("forward")
    transform = eval_transform()
    checks = []
    with torch.inference_mode():
        for photo in reference["photos"][:5]:
            path = assets / photo["asset"]
            if sha256(path) != photo["sha256"]:
                raise ValueError(f"JPEG hash differs for {photo['id']}")
            with Image.open(path) as jpeg:
                batch = transform(jpeg.convert("RGB")).unsqueeze(0)
            eager = model(batch)
            actual = forward.execute((batch,))[0]
            torch.testing.assert_close(actual, eager, rtol=1e-4, atol=1e-4)
            indices = actual[0].topk(3).indices.tolist()
            if indices != photo[role]:
                raise ValueError(f"Export changed {role} top three on {photo['id']}")
            checks.append(dict(id=photo["id"], top3=indices,
                               max_abs_logit_error=(actual - eager).abs().max().item()))
    # Release the runtime mapping before replacing the staged file on Windows.
    del forward, program
    staged.replace(destination)
    print(f"Verified {destination}, {len(checks)} photos, delegates {delegates}", flush=True)
    return dict(checkpoint_sha256=sha256(checkpoint), pte_sha256=sha256(destination),
                bytes=destination.stat().st_size, delegates=delegates,
                portable_operators=[op.name for op in plan.operators], parity=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "android/app/src/main/assets")
    parser.add_argument("--report", type=Path, default=ROOT / "results/android/export.json")
    args = parser.parse_args()
    # The Windows wheel ships flatc.exe alongside Python. Support invoking
    # the venv interpreter directly, without relying on shell activation.
    bundled_flatc = Path(sys.executable).with_name("flatc.exe")
    if os.name == "nt" and bundled_flatc.is_file():
        os.environ.setdefault("FLATC_EXECUTABLE", str(bundled_flatc))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reference = json.loads((args.assets / "reference.json").read_text(encoding="utf-8"))
    metadata = dict(executorch=version("executorch"), torch=str(torch.__version__),
                    torchvision=version("torchvision"), input_shape=[1, 3, 224, 224],
                    dtype="float32", models={})
    paths = {"teacher": ROOT / "assets/resnet50_food101_best.pt", "distilled": ROOT / "checkpoints/distilled.pt"}
    for role, path in paths.items():
        metadata["models"][role] = export_one(role, path, args.assets, reference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
