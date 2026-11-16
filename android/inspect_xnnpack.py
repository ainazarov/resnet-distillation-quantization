"""Inspect the actual serialized XNNPACK nodes and quantization data types."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from importlib import resources
from pathlib import Path

from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import XNNHeader


def delegate_graphs(pte):
    data = pte.read_bytes()
    schema = resources.files("executorch.backends.xnnpack.serialization") / "schema.fbs"
    compiler = os.environ.get("FLATC_EXECUTABLE") or shutil.which("flatc")
    if compiler is None:
        raise RuntimeError("Set FLATC_EXECUTABLE to the flatc bundled with ExecuTorch")
    with tempfile.TemporaryDirectory() as tmp, resources.as_file(schema) as schema_path:
        payload = Path(tmp) / "delegate.bin"
        # A delegate header points directly to the schema FlatBuffer; validate both magics.
        for match in re.finditer(b"XH00", data):
            start = match.start() - 4
            try:
                header = XNNHeader.from_bytes(data[start:start + XNNHeader.EXPECTED_LENGTH])
            except ValueError:
                continue
            pos = start + header.flatbuffer_offset
            if not header.is_valid() or data[pos + 4:pos + 8] != b"XN01":
                continue
            payload.write_bytes(data[pos:pos + header.flatbuffer_size])
            subprocess.run([compiler, "--json", "--strict-json", "--defaults-json", "-o", tmp,
                            str(schema_path), "--", str(payload)], check=True)
            raw = payload.with_suffix(".json").read_text(encoding="utf-8")
            # FlatBuffers renders unbounded clamp limits as bare 'inf'/'-inf'.
            yield json.loads(re.sub(r"(?<=: )(-?)inf\b", r"\1Infinity", raw))


def audit_int8(pte, expected_delegates):
    ops, layers, dtype_hist = Counter(), Counter(), Counter()
    channels, quantized_inputs, quantized_outputs = [], [], []
    n_delegates = 0
    for graph in delegate_graphs(pte):
        n_delegates += 1
        values = {}
        for value in graph["xvalues"]:
            record = value["xvalue_union"]
            tensor = record.get("tensor_value", record)
            values[tensor["id_out"]] = (record, tensor)
            dtype_hist[tensor["datatype"]] += 1
        for entry in graph["xnodes"]:
            kind = entry["xnode_union_type"]
            node = entry["xnode_union"]
            ops[kind] += 1
            if kind not in ("XNNConv2d", "XNNFullyConnected"):
                continue
            weight, tensor = values[node["filter_id"]]
            input_record, input_tensor = values[node["input1_id"]]
            output_record, output_tensor = values[node["output_id"]]
            q = weight.get("quant_params", {})
            if (tensor["datatype"] != "xnn_datatype_qcint8" or weight["quant_params_type"] != "PerChannelQuant"
                    or q["channel_dim"] != 0 or (q.get("num_scales") or len(q.get("scale", []))) != tensor["dims"][0]):
                raise ValueError(f"{kind} lacks per-output-channel signed INT8 weight scales")
            for record, t in [(input_record, input_tensor), (output_record, output_tensor)]:
                if (t["datatype"] != "xnn_datatype_qint8" or record["quant_params_type"] != "PerTensorQuant"
                        or record["quant_params"]["scale"] <= 0):
                    raise ValueError(f"{kind} lacks static signed INT8 activation parameters")
            channels.append(tensor["dims"][0])
            quantized_inputs.append(input_record["quant_params"])
            quantized_outputs.append(output_record["quant_params"])
            layers[kind] += 1
    if n_delegates != expected_delegates or layers != {"XNNConv2d": 20, "XNNFullyConnected": 1}:
        raise ValueError(f"Unexpected delegate count or quantized layer coverage: {n_delegates}, {layers}")
    return dict(node_counts=dict(ops), value_dtypes=dict(dtype_hist), verified_static_int8_layers=dict(layers),
                weight_channel_axis=0, scales_per_weight_tensor=channels,
                input_activation_qparams=quantized_inputs, output_activation_qparams=quantized_outputs)
