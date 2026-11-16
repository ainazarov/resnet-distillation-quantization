"""Render the Food-101 ResNet architecture figure used in the HW1 report."""
import argparse
import csv
from math import isclose
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "figures"
COMPONENT_CSV = OUT.parent.parent / "results/pilot/components/components.csv"
WIDTH, HEIGHT = 174, 114  # millimetres
INK, MUTED = "#172D40", "#465E70"
BLUE, SKY = "#245D8B", "#EAF3FA"
TEAL, MINT = "#096B60", "#E5F4F0"
RULE = "#BCCDD4"


def measured_components():
    with COMPONENT_CSV.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    timings = {}
    for row in rows:
        name = row["component"]
        if name in timings:
            raise ValueError(f"Duplicate measurement for {name}")
        teacher, student = float(row["resnet50_ms"]), float(row["resnet18_ms"])
        if not (teacher > 0 and student > 0):
            raise ValueError(f"Nonpositive component mean in {name}")
        delta = student - teacher
        percent = 100 * delta / teacher
        for calculated, key in ((delta, "delta_ms"), (percent, "delta_percent")):
            if not isclose(calculated, float(row[key]), abs_tol=1e-6):
                raise ValueError(f"Inconsistent {key} for {name}")
        timings[name] = (teacher, student, delta, percent)
    return timings


def duration(ms):
    return (f"{ms:.3f}" if ms < 1 else f"{ms:.2f}").replace(".", ",") + " мс"


def draw(with_timings=False):
    components = measured_components() if with_timings else {}
    timings = [components[f"layer{i}"] for i in range(1, 5)] if with_timings else []
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "path",
        "axes.unicode_minus": False, "figure.facecolor": "white",
    })
    top = -5 if with_timings else 0
    fig = plt.figure(figsize=(WIDTH / 25.4, (HEIGHT-top) / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, top)
    ax.axis("off")

    def text(x, y, words, size=8.0, colour=INK, bold=False, align="left"):
        ax.text(x, y, words, fontsize=size, color=colour, ha=align, va="center",
                weight="bold" if bold else "normal", linespacing=1.24)

    def box(x, y, w, h, fill, stroke="none"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.35",
                                    linewidth=0.7, edgecolor=stroke, facecolor=fill))

    def arrow(x1, y1, x2, y2, colour=MUTED, size=6.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     shrinkA=0, shrinkB=0, mutation_scale=size,
                                     linewidth=0.9, color=colour))

    if with_timings:
        text(0.7, -2.4, "Входной блок и MaxPool имеют одинаковую структуру у обеих сетей. Для них показано время учителя.",
             7.6, MUTED)

    def flow_node(x, w, title, detail, component=None):
        box(x, 1, w, 13.8, "#F0F4F6")
        text(x + w / 2, 5.6, title, 8.1, bold=True, align="center")
        if with_timings and component:
            detail += "   ·   Учитель " + duration(components[component][0])
        text(x + w / 2, 10.9, detail, 7.1, MUTED, align="center")

    flow_node(0.5, 30, "Вход RGB", "3 × 224 × 224")
    arrow(31.5, 8, 36.0, 8)
    flow_node(37, 70, "Свёртка 7 × 7, шаг 2 → BN → ReLU", "64 × 112 × 112", "stem")
    arrow(108, 8, 112.5, 8)
    flow_node(113.5, 60, "MaxPool 3 × 3, шаг 2", "64 × 56 × 56", "maxpool")

    for i, w in enumerate((56, 28, 14, 7)):
        center = 38.9 + 23.4 * i
        text(center, 18.4 if with_timings else 22.3, f"Стадия {i + 1}", 8, bold=True, align="center")
        text(center, 22.7 if with_timings else 27.0, f"{w} × {w} px", 7.1, MUTED, align="center")
    head_title = "GAP → flatten → классификатор" if with_timings else "GAP → классификатор"
    text(147.0, 18.4 if with_timings else 22.3, head_title, 8, bold=True, align="center")
    text(147.0, 22.7 if with_timings else 27.0, "101 логит", 7.1, MUTED, align="center")

    def stages(y, model, role, count, channels, head, fill, stroke, model_index):
        height = 15.8 if with_timings else 13
        mid = y + height / 2
        text(0.7, mid - 2.7, model, 9.2, stroke, True)
        text(0.7, mid + 2.9, role, 7.2, MUTED)
        for i, (n, width) in enumerate(zip(count, channels)):
            left = 28.4 + 23.4 * i
            box(left, y, 21, height, fill)
            text(left + 10.5, y + (3.5 if with_timings else 4.4), f"× {n}", 9.4, stroke, True, "center")
            text(left + 10.5, y + (8.1 if with_timings else 9.6), f"{width} кан.", 7.6, INK, align="center")
            if with_timings:
                label = duration(timings[i][model_index])
                text(left + 10.5, y + 12.7, label, 8.0, stroke, True, "center")
            arrow(left + 21.2, mid, left + 23.1, mid, stroke, 3.7)
        box(122, y, 51.5, height, fill)
        text(147.75, y + 5.7 if with_timings else mid, head, 8, INK, align="center")
        if with_timings:
            text(147.75, y + 12.8, "Весь блок · " + duration(components["head"][model_index]),
                 8.0, stroke, True, "center")

    stages(26 if with_timings else 31, "ResNet-50", "Учитель", [3, 4, 6, 3], [256, 512, 1024, 2048],
           "Dropout 0,3 (обучение)\nLinear 2048 → 101", SKY, BLUE, 0)
    stages(43.2 if with_timings else 46, "ResNet-18", "Ученик", [2, 2, 2, 2], [64, 128, 256, 512],
           "Linear 512 → 101", MINT, TEAL, 1)
    if with_timings:
        text(0.7, 65, "Изменение\nвремени", 8.1, INK, True)
        for i, (_, _, delta, percent) in enumerate(timings):
            left = 28.4 + 23.4 * i
            box(left, 60.5, 21, 10.2, "#F3F5F6")
            change = f"{delta:+.2f} мс".replace(".", ",").replace("-", "−")
            fraction = f"{percent:+.1f}%".replace(".", ",").replace("-", "−")
            text(left + 10.5, 63.4, change, 8.0, INK, True, "center")
            text(left + 10.5, 67.9, fraction, 7.8, MUTED, align="center")
    text(0.7, 74.1 if with_timings else 63.4, "× n — число блоков. На стадиях 2–4 первый блок уменьшает высоту и ширину вдвое.",
         7.6, MUTED)
    divider_y = 77.3 if with_timings else 68.2
    ax.plot([0.5, 173.5], [divider_y, divider_y], lw=0.65, color=RULE)

    def residual(x, title, ops, fill, stroke):
        text(x + 0.6, 79.9 if with_timings else 74.0, title, 9, stroke, True)
        cy = 101.4 if with_timings else 92.0
        skip_y = 89.0 if with_timings else 81.7
        start, dot = x + 1.5, x + 4.0
        merge, end = x + 66.4, x + 85.1
        # Only branch-specific operations lie inside these frames. The input
        # fork, shared addition and final activation stay outside both frames.
        if with_timings:
            left, width = x + 7.0, 55.5
            shortcut_fill, shortcut_stroke = "#FFF9ED", "#886018"
            ax.add_patch(Rectangle((left, 82.6), width, 8.9, facecolor=shortcut_fill,
                                   edgecolor=shortcut_stroke, linewidth=.8, linestyle=(0, (4, 2)),
                                   zorder=0))
            ax.add_patch(Rectangle((left, 93.0), width, 13.6, facecolor="#F6F9FA",
                                   edgecolor=stroke, linewidth=.85, zorder=0))
            text(left + 1.5, 85.0, "Остаточная ветвь", 8.0, shortcut_stroke, True)
            text(left + 1.5, 95.2, "Основная ветвь", 8.0, stroke, True)
        ax.plot([start, dot], [cy, cy], lw=1.0, color=stroke)
        ax.add_patch(Circle((dot, cy), 0.7, facecolor=stroke, edgecolor="none"))
        ax.plot([dot, dot, merge], [cy, skip_y, skip_y], lw=0.8, color=stroke)
        if with_timings:
            # This compact label describes the two possibilities on the bypass.
            ax.text(x + 34.75, skip_y, "вход напрямую или 1×1 + BN", fontsize=7.4,
                    ha="center", va="center", color=INK, zorder=4,
                    bbox={"facecolor": shortcut_fill, "edgecolor": "none", "pad": 1.3})
        arrow(merge, skip_y, merge, cy - 3.15, stroke)
        w = 15 if len(ops) == 3 else 24.3
        slot = 18.3 if len(ops) == 3 else 28.2
        previous = dot
        for i, label in enumerate(ops):
            left = x + 9.4 + slot * i
            arrow(previous, cy, left - 0.2, cy, stroke, 4.2)
            half_height = 3.55 if with_timings else 4.3
            box(left, cy - half_height, w, 2 * half_height, fill, stroke)
            text(left + w / 2, cy, label, 8.25, stroke, bold=True, align="center")
            previous = left + w
        arrow(previous, cy, merge - 3.15, cy, stroke)
        ax.add_patch(Circle((merge, cy), 3.0, color=stroke))
        text(merge, cy - 0.1, "+", 10.7, "white", True, "center")
        arrow(merge + 3.1, cy, end - 10.9, cy, stroke, 5.3)
        text(end - 4.8, cy, "ReLU", 7.4, stroke, True, "center")

    residual(0.5, "Bottleneck · блок ResNet-50", ["1 × 1", "3 × 3", "1 × 1"], SKY, BLUE)
    residual(88.3, "BasicBlock · блок ResNet-18", ["3 × 3", "3 × 3"], MINT, TEAL)
    if with_timings:
        text(0.7, 109.3, "В основной ветви после каждой свёртки — BatchNorm (BN), между свёртками — ReLU.", 7.4, MUTED)
        text(0.7, 112.8, "Проекция 1×1 + BN — в первом блоке стадий 1–4 у ResNet-50 и 2–4 у ResNet-18.", 7.4, MUTED)
    else:
        text(0.7, 103.2, "После каждой свёртки — BatchNorm (BN), после промежуточных свёрток — ReLU.",
             7.5, MUTED)
        text(0.7, 109.0, "Верхняя ветвь передаёт вход. При смене формы используется проекция 1 × 1 с BN.",
             7.5, MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    if with_timings:
        fig.savefig(OUT / "architecture-stages-v5.pdf")
    else:
        for ext in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"architecture.{ext}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", action="store_true", help="use the measured inclusive stage times")
    draw(parser.parse_args().timings)
