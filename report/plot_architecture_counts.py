"""Plot exact Food-101 parameter and MAC counts from one recorded forward per model."""
import gzip
import json
from math import prod
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "results/pilot/layers/paired-trace.json.gz"
OUT = ROOT / "report/figures/parameter-profile-v5.pdf"
PARTS = {
    "stem": "Входная свёртка и BN", "layer1": "Стадия 1", "layer2": "Стадия 2",
    "layer3": "Стадия 3", "layer4": "Стадия 4", "fc": "Классификатор",
}
INK, SUB, GRID = "#162C3C", "#435B65", "#CEDBD9"
BLUE, TEAL = "#245D8B", "#098172"


def enclosed(event, scope):
    return (event["pid"] == scope["pid"] and event["tid"] == scope["tid"]
            and scope["ts"] <= event["ts"]
            and event["ts"] + event["dur"] <= scope["ts"] + scope["dur"])


def count_by_part(events, name):
    forward = next(e for e in events if e["name"] == f"forward:{name}:0")
    ops = sorted((e for e in events if enclosed(e, forward)), key=lambda e: e["ts"])
    labels = [e for e in ops if e["name"].startswith("Conv2d:")]
    kernels = [e for e in ops if e["name"] == "aten::mkldnn_convolution"]
    norms = [e for e in ops if e["name"] == "aten::batch_norm"]
    counts = {part: [0, 0] for part in PARTS}
    # In these ResNets, every convolution is followed by an affine BatchNorm.
    # Its input gives the actual output spatial dimensions of that convolution.
    for label, conv, bn in zip(labels, kernels, norms, strict=True):
        assert enclosed(conv, label) and conv["ts"] + conv["dur"] <= bn["ts"]
        weight, bias = conv["args"]["Input Dims"][1:3]
        output, scale, offset = bn["args"]["Input Dims"][:3]
        assert output[:2] == [1, weight[0]] and scale == offset == [weight[0]]
        part = label["name"].removeprefix("Conv2d:").split(".")[0]
        row = counts["stem" if part == "conv1" else part]
        row[0] += prod(weight) + (prod(bias) if bias else 0) + prod(scale) + prod(offset)
        row[1] += prod(weight) * prod(output[2:])
    (linear,) = [e for e in ops if e["name"] == "aten::linear"]
    features, weight, bias = linear["args"]["Input Dims"]
    assert features == [1, weight[1]] and weight[0] == 101 and bias == [101]
    counts["fc"] = [prod(weight) + prod(bias), prod(weight)]
    counts["total"] = [sum(column) for column in zip(*counts.values())]
    return counts


def plot():
    with gzip.open(TRACE, "rt", encoding="utf-8") as source:
        events = [e for e in json.load(source)["traceEvents"] if e.get("ph") == "X"]
    teacher, student = [count_by_part(events, name) for name in ("resnet50", "resnet18")]
    reference = json.loads((ROOT / "results/final/summary.json").read_text())["models"]
    assert teacher["total"][0] == reference["teacher"]["parameters"]
    assert student["total"][0] == reference["distilled"]["parameters"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.05, 5.65), facecolor="white")
    fig.text(.005, .96, "Число параметров по частям сети", weight="bold", size=11.2, color=INK)
    ax = fig.add_axes([.21, .58, .76, .33])
    for values, shift, colour, label in ((teacher, -.18, BLUE, "ResNet-50"),
                                         (student, .18, TEAL, "ResNet-18")):
        xs = [values[part][0] / 1e6 for part in PARTS]
        ys = [i + shift for i in range(len(xs))]
        ax.barh(ys, xs, height=.31, color=colour, label=label)
        for x, y in zip(xs, ys):
            ax.text(x + .10, y, f"{x:.2f}".replace(".", ","), va="center", size=7.8, color=colour)
    ax.set_yticks(range(len(PARTS)), ["Вход (7 × 7)", *list(PARTS.values())[1:]])
    ax.invert_yaxis()
    ax.set_xlim(0, 16.6)
    ax.set_xticks(range(0, 17, 4))
    ax.set_xlabel("Млн параметров", color=SUB, fontsize=8.5, labelpad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=.55, color=GRID)
    ax.tick_params(axis="both", length=0, labelsize=8.6, colors=INK)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.035), ncol=2, frameon=False,
              fontsize=8.5, handlelength=1.15, columnspacing=1.3, borderaxespad=0)

    fig.text(.005, .473, "Вклад в общее число параметров и операций", size=11.2,
             weight="bold", color=INK)
    # Pair each model's two columns under one heading. Align all counts on the right.
    edges = (.225, .396, .592, .785, .99)
    for left, right, label, colour in ((edges[0], edges[2], "ResNet-50", BLUE),
                                       (edges[2]+.014, edges[4], "ResNet-18", TEAL)):
        fig.text((left+right)/2, .433, label, color=colour, size=9, weight="bold", ha="center")
        fig.add_artist(Line2D([left, right], [.422, .422], transform=fig.transFigure,
                              linewidth=.85, color=colour))
    fig.text(.015, .393, "Часть сети", size=8.6, weight="bold", color=INK)
    for x, label in zip(edges[1:], ("Параметры", "MAC", "Параметры", "MAC")):
        fig.text(x-.008, .393, label, size=8.6, weight="bold", ha="right", color=INK)
    row_h = .043
    for index, (part, label) in enumerate({**PARTS, "total": "Всего"}.items()):
        y = .357 - index*row_h
        total = part == "total"
        if total or index % 2 == 0:
            fig.add_artist(Rectangle((.003, y-.011), .99, row_h-.001, transform=fig.transFigure,
                                     facecolor="#E4F1EF" if total else "#F3F6F7",
                                     edgecolor="none", zorder=-1))
        weight = "bold" if total else "normal"
        fig.text(.015, y, label, size=8.6, weight=weight, color=INK)
        for x, value in zip(edges[1:], [*teacher[part], *student[part]]):
            fig.text(x-.008, y, f"{value:,}".replace(",", " "), size=8.6, weight=weight,
                     color=INK, ha="right")
    fig.text(.015, .050, "Параметры включают веса и смещения, в том числе в BatchNorm.", size=8.2, color=SUB)
    fig.text(.015, .020, "Один MAC — умножение с накоплением. Учтены свёртки и линейный слой при входе 224 × 224.",
             size=8.2, color=SUB)
    fig.savefig(OUT)
    plt.close(fig)


if __name__ == "__main__":
    plot()
