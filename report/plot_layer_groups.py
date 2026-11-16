"""Render the Intel pilot pie and the measured convolution groups for HW1 v5."""
import csv
from collections import Counter
from math import ceil

from profile_figures import (
    GREY, NAVY, RESULTS, TEAL, axes_format, np, plt, read_json, ru, save, style,
)


GROUPS = {
    "main_1x1": "1×1 · основная ветвь",
    "shortcut_1x1": "1×1 · остаточная ветвь",
    "block_3x3": "3×3 · в блоках",
    "stem_7x7": "7×7 · входная",
}


def group_times(path):
    times = {"resnet50": Counter(), "resnet18": Counter()}
    counts = {"resnet50": Counter(), "resnet18": Counter()}
    with path.open(encoding="utf-8-sig", newline="") as data:
        for row in csv.DictReader(data):
            model, group = row["model"], row["group"]
            if group not in GROUPS:
                raise ValueError(f"Unexpected group {group}")
            times[model][group] += float(row["mean_ms_per_forward"])
            counts[model][group] += float(row["calls_per_forward"])
    for model, expected in (("resnet50", [32, 4, 16, 1]), ("resnet18", [0, 3, 16, 1])):
        observed = [counts[model][group] for group in GROUPS]
        if observed != expected:
            raise ValueError(f"{model}: observed calls {observed}, expected {expected}")
    return times


def plot_jpeg_stages(pilot_summary):
    stages = pilot_summary["models"]["teacher"]["stages"]
    keys = ("forward", "decode", "preprocess", "top3")
    shares = [stages[k]["percent_of_elapsed"] for k in keys]
    labels = ("Прямой проход", "Декодирование JPEG", "Подготовка входа", "Выбор top-3")
    colors = (NAVY, TEAL, "#96BED3", "#B57737")

    fig = plt.figure(figsize=(7.05, 2.50))
    fig.text(.02, .915, "Этапы обработки JPEG", fontsize=12, weight="bold")
    fig.text(.97, .915, f"ResNet-50 · {pilot_summary['images']} тестовых JPEG",
             fontsize=9.2, color=GREY, ha="right")
    ax = fig.add_axes([.055, .015, .33, .805])
    ax.pie(shares, colors=colors, startangle=90, counterclock=True, radius=1.15,
           wedgeprops={"edgecolor": "none"})
    for y, label, share, color in zip((.695, .518, .341, .164), labels, shares, colors):
        fig.text(.45, y, "■", fontsize=13, color=color, va="center")
        fig.text(.493, y, label, fontsize=10.5, va="center")
        fig.text(.95, y, ru(share, 2) + " %", fontsize=10.5, weight="bold",
                 ha="right", va="center")
    save(fig, "jpeg-stages-v5")


def plot_convolution_groups(times):
    fig = plt.figure(figsize=(7.05, 1.85))
    fig.text(.02, .921, "Время свёрток по группам", fontsize=11.4, weight="bold")
    ax = fig.add_axes([.247, .210, .726, .635])
    y = np.arange(len(GROUPS))[::-1]
    for model, name, color, offset in [
        ("resnet50", "ResNet-50", NAVY, .192),
        ("resnet18", "ResNet-18", "#489C8D", -.192),
    ]:
        values = [times[model][group] for group in GROUPS]
        ax.barh(y + offset, values, height=.32, label=name, color=color)
        for value, ypos in zip(values, y + offset):
            ax.annotate(ru(value, 2), (value, ypos), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.6,
                        color=NAVY if model == "resnet50" else "#20695C")

    largest = max(max(model.values()) for model in times.values())
    tick_step = max(1, ceil(largest / 6 / 5) * 5)
    ax.set_xlim(0, ceil(largest * 1.18 / tick_step) * tick_step)
    ax.set_xticks(np.arange(0, ax.get_xlim()[1]+.01, tick_step))
    ax.set_yticks(y, GROUPS.values(), fontsize=8.5)
    ax.set_ylim(-.5, len(GROUPS)-.5)
    axes_format(ax, "Время за прямой проход, мс")
    fig.legend(*ax.get_legend_handles_labels(), loc="center right", bbox_to_anchor=(.974, .946),
               ncol=2, frameon=False, borderaxespad=0, borderpad=0, columnspacing=1.6,
               handlelength=1.15, handletextpad=.45, fontsize=8.5)
    save(fig, "conv-groups-v5")


def main():
    style()
    pilot = RESULTS / "pilot"
    plot_jpeg_stages(read_json(pilot / "summary.json"))
    plot_convolution_groups(group_times(pilot / "layers" / "layers.csv"))


if __name__ == "__main__":
    main()
