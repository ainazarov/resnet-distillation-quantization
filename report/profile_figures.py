"""Regenerate the report's CPU and Android profiling figures from per-request timings."""
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "figures"
NAVY, TEAL, INK, GREY, PALE = "#285779", "#00796B", "#20303A", "#536671", "#D6E0E5"
VARIANTS = ("resnet18_fp32", "resnet18_int8")
WIDTH, HEIGHT = 7.25, 2.78


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ru(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.8,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
        "text.color": INK, "axes.labelcolor": GREY,
        "xtick.color": GREY, "ytick.color": INK,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.spines.top": False, "axes.edgecolor": PALE,
        "axes.axisbelow": True, "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.7,
    })


def axes_format(ax, xlabel, grid="x"):
    ax.grid(axis=grid, color="#E4EAEC", linewidth=0.65)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_xlabel(xlabel, labelpad=5, fontsize=8.1)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)


def plot_stage_medians(fig, samples):
    ax = fig.add_axes([.153, .215, .322, .56])
    med = np.median(samples, axis=0)
    y = np.arange(4)[::-1]
    ax.plot(med, y, marker="o", ls="none", markerfacecolor=NAVY,
            markeredgecolor="white", markeredgewidth=.5, markersize=6, zorder=3)
    for num, ypos in zip(med, y):
        ax.annotate(ru(num, 3 if num < .1 else 2), (num, ypos), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.6, color=NAVY)
    ax.set_xscale("log")
    ax.set_xlim(.027, 480)
    ax.set_xticks([.1, 1, 10, 100], ["0,1", "1", "10", "100"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([3, 2, 1, 0], ["Декодирование", "Подготовка", "Прямой проход", "Выбор top-3"])
    ax.set_ylim(-.46, 3.46)
    axes_format(ax, "Медиана, мс · логарифмическая шкала")


def plot_stage_shares(fig, shares):
    labels = ["Декодирование", "Подготовка", "Прямой проход", "Выбор top-3"]
    colors = [TEAL, "#96BED3", NAVY, "#B57737"]
    ax = fig.add_axes([.002, .175, .226, .59])
    ax.pie(shares, colors=colors, startangle=90, counterclock=False,
           radius=1.15, wedgeprops={"edgecolor": "none"})
    for y, index in zip([.66, .53, .40, .27], [2, 0, 1, 3]):
        fig.text(.235, y, "■", fontsize=8.8, color=colors[index], va="center")
        fig.text(.26, y, labels[index], fontsize=8.2, va="center")
        fig.text(.497, y, ru(shares[index], 2) + " %", fontsize=8.2,
                 va="center", ha="right", weight="bold")


def kernel_times(path, expected_forwards):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        events = json.load(f)["traceEvents"]
    forwards = [e for e in events if e.get("ph") == "X" and e.get("name") == "model_inference"]
    convs = [e for e in events if e.get("ph") == "X" and e.get("name") == "aten::mkldnn_convolution"]
    assert len(forwards) == expected_forwards
    totals = {1: 0.0, 3: 0.0, 7: 0.0}
    counts = []
    for forward in forwards:
        calls = [e for e in convs
                 if (e["pid"], e["tid"]) == (forward["pid"], forward["tid"])
                 and forward["ts"] <= e["ts"]
                 and e["ts"] + e["dur"] <= forward["ts"] + forward["dur"]]
        counts.append(len(calls))
        for event in calls:
            kh, kw = event["args"]["Input Dims"][1][-2:]
            assert kh == kw
            # Chrome trace durations are microseconds. Select the oneDNN
            # operator alone so its enclosing convolution wrappers do not count.
            totals[kh] += event["dur"]
    assert min(counts) > 0 and len(set(counts)) == 1 and sum(counts) == len(convs)
    return [total / (1000 * len(forwards)) for total in totals.values()]


def plot_kernel_times(fig, folder, expected_forwards):
    fig.text(.555, .935, "Свёртки (oneDNN) по размеру ядра", fontsize=9.8, weight="bold")
    ax = fig.add_axes([.619, .215, .355, .545])
    y = np.array([2, 1, 0])
    for role, model, color, offset in [
        ("teacher", "ResNet-50", NAVY, .19), ("student_init", "ResNet-18", "#489C8D", -.19),
    ]:
        times = kernel_times(folder / f"{role}-trace.json.gz", expected_forwards)
        ax.barh(y + offset, times, height=.30, color=color, label=model)
        for duration, ypos in zip(times, y + offset):
            ax.annotate(ru(duration, 2), (duration, ypos), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.6)
    ax.set_yticks(y, ["1×1", "3×3", "7×7"])
    ax.set_ylim(-.44, 2.44)
    ax.set_xlim(0, 51)
    ax.set_xticks(np.arange(0, 51, 10))
    axes_format(ax, "Время за прямой проход, мс")
    fig.legend(*ax.get_legend_handles_labels(), loc="center left", bbox_to_anchor=(.555, .855),
               ncol=2, frameon=False, borderaxespad=0, borderpad=0, columnspacing=1.6,
               handlelength=1.15, handletextpad=.45, fontsize=8.2)


def plot_operator_shares(fig, folder, summary):
    ops = ("aten::mkldnn_convolution", "aten::native_batch_norm", "aten::max_pool2d_with_indices")
    ox = fig.add_axes([.661, .215, .312, .56])
    fig.text(.573, .935, "Операторы внутри сети", fontsize=10.8, weight="bold")
    fig.text(.573, .847, "torch.profiler · три прямых прохода", fontsize=8.3, color=GREY)
    profile = {p["operator"]: p for p in summary["models"]["teacher"]["profile"]}
    with gzip.open(folder / "teacher-trace.json.gz", "rt", encoding="utf-8") as f:
        events = json.load(f)["traceEvents"]
    conv_calls = sum(e.get("ph") == "X" and e.get("name") == ops[0] for e in events)
    assert conv_calls == profile[ops[0]]["calls"]
    share = [profile[op]["percent_of_profile_self_time"] for op in ops]
    y = np.arange(3)[::-1]
    ox.barh(y, share, height=.4, color=NAVY)
    for pct, ypos in zip(share, y):
        ox.annotate(ru(pct), (pct, ypos), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.6, color=NAVY)
    ox.set_xlim(0, 89)
    ox.set_xticks([0, 25, 50, 75])
    ox.set_yticks([2, 1, 0], ["Свёртки\n(oneDNN)", "BatchNorm", "MaxPool"])
    ox.set_ylim(-.4, 2.4)
    axes_format(ox, "Доля собственного времени CPU, %")


def pilot_figure(*, pie=False):
    folder = RESULTS / "pilot"
    summary = read_json(folder / "summary.json")
    with (folder / "timings_ms.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stages = ("decode", "preprocess", "forward", "top3")
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    fig.text(.015, .935, "Этапы обработки JPEG", fontsize=10.8, weight="bold")
    fig.text(.015, .847, "ResNet-50 · 505 тестовых JPEG", fontsize=8.3, color=GREY)

    selected = [r for r in rows if r["model"] == "teacher"]
    assert len(selected) == len({r["image"] for r in selected}) == 505
    samples = np.array([[float(r[key]) for key in stages] for r in selected])
    totals = np.array([float(r["end_to_end"]) for r in selected])
    np.testing.assert_allclose(samples.sum(1), totals, rtol=1e-9)
    stage_summary = summary["models"]["teacher"]["stages"]
    if pie:
        shares = 100 * samples.sum(axis=0) / totals.sum()
        np.testing.assert_allclose(shares, [stage_summary[k]["percent_of_elapsed"] for k in stages])
        plot_stage_shares(fig, shares)
        plot_kernel_times(fig, folder, summary["profiled_forwards_per_model"])
    else:
        np.testing.assert_allclose(np.median(samples, axis=0), [stage_summary[k]["median"] for k in stages])
        plot_stage_medians(fig, samples)
        plot_operator_shares(fig, folder, summary)
    save(fig, "pilot-profile-v3" if pie else "pilot-profile")


def plot_phone_stages(fig, ax):
    folder = RESULTS / "android_ptq/stages"
    summary = read_json(folder / "summary.json")
    measured = read_json(folder / "precision.json")
    stages = ("decode_rgb_ms", "resize_crop_norm_tensor_ms", "forward_ms", "top3_readback_labels_ms")
    colors = ("#4F9687", "#D8BC83", NAVY, "#A04425")
    times = []
    image_sets = []
    for role in VARIANTS:
        rows = [row for row in measured["rows"] if row["model"] == role]
        ids = {row["id"] for row in rows}
        assert len(ids) == len(rows) == summary["n_per_model"] == 101
        image_sets.append(ids)
        intervals = np.array([[row[key] for key in stages] for row in rows])
        np.testing.assert_allclose(intervals.sum(1), [row["jpeg_to_top3_ms"] for row in rows])
        mean = intervals.mean(0)
        expected = summary["models"][role]["mean_ms"]
        np.testing.assert_allclose(mean, [expected[key] for key in stages])
        times.append(mean)
    assert image_sets[0] == image_sets[1]

    # Top-3 is sub-pixel at the shared time scale. Give its value in the key.
    top3_labels = [ru(t[-1], 3) for t in times]
    top3 = top3_labels[0] if top3_labels[0] == top3_labels[1] else " / ".join(top3_labels)
    labels = ("Декодирование JPEG", "Подготовка тензора", "Прямой проход", f"Выбор top-3 · {top3} мс")
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = .015 if i % 2 == 0 else .235
        y = .77 if i < 2 else .695
        fig.text(x, y, "■", fontsize=8.1, va="center", color=color)
        fig.text(x+.022, y, label, fontsize=7.6, va="center")
    for y, means in zip((1, 0), times):
        left = 0.0
        for key, color, value in zip(stages, colors, means):
            ax.barh(y, value, height=.42, left=left, color=color)
            if key == "forward_ms":
                ax.text(left + value / 2, y, ru(value, 2), va="center", ha="center",
                        color="white", fontsize=7.5, weight="bold")
            left += value


def phone_figure(*, instrumented=False):
    folder = RESULTS / "android_ptq/galaxy-s24-plus"
    runs = [read_json(folder / f"run-{i}.json") for i in (1, 2, 3)]
    summary = read_json(folder / "summary.json")
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([.082, .215, .375, .42] if instrumented else [.082, .215, .355, .56])
    ecdf = fig.add_axes([.588, .215, .374, .56])
    fig.text(.015, .935, "Из чего состоит запрос", fontsize=10.8, weight="bold")
    fig.text(.535, .935, "Распределение задержки", fontsize=10.8, weight="bold")
    stage_note = "По 101 фото на вариант · замер с таймерами" if instrumented else "Серым выделено время вне модели"
    fig.text(.015, .848, stage_note, color=GREY, fontsize=8.1)
    fig.text(.535, .848, "Доля завершённых запросов, %", color=GREY, fontsize=8.1)
    ecdf.axhline(95, ls=(0, (3, 2)), color="#70838E", lw=.8)
    if instrumented:
        plot_phone_stages(fig, ax)
    for i, (role, color, name) in enumerate(zip(VARIANTS, (NAVY, TEAL), ("FP32", "INT8"))):
        rows = []
        for run in runs:
            subset = [r for r in run["rows"] if r["model"] == role]
            assert len(subset) == len({r["id"] for r in subset}) == 101
            rows.extend(subset)
        forward = np.array([r["forward_ms"] for r in rows])
        elapsed = np.array([r["jpeg_to_top3_ms"] for r in rows])
        remaining = elapsed - forward
        assert (remaining > 0).all()
        observed = summary["models"][role]
        p95 = np.percentile(elapsed, 95)
        np.testing.assert_allclose([p95, np.median(forward)],
                                   [observed["p95_jpeg_to_top3_ms"], observed["median_forward_ms"]])
        if not instrumented:
            f_mean, other_mean = forward.mean(), remaining.mean()
            y = 1 - i
            ax.barh(y, f_mean, height=.43, color=color)
            ax.barh(y, other_mean, left=f_mean, height=.43, color=PALE)
            for val, pos, labelcolor in [(f_mean, f_mean / 2, "white"),
                                         (other_mean, f_mean + other_mean / 2, INK)]:
                ax.text(pos, y, ru(val, 1), color=labelcolor, ha="center", va="center",
                        fontsize=7.7, weight="bold")
        sorted_elapsed = np.sort(elapsed)
        ecdf.step(sorted_elapsed, np.arange(1, len(rows)+1)*100/len(rows), where="post",
                  color=color, label=name, lw=1.65)
        ecdf.plot(p95, 95, marker="o", ms=4.1, mec="white", mew=.4, color=color)
        ecdf.annotate("p95 " + ru(p95, 2), (p95, 95), xytext=(8, -12),
                      textcoords="offset points", fontsize=7.7, weight="bold", color=color,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=.6))
    ax.set_yticks([1, 0], ["FP32", "INT8"])
    ax.set_ylim(-.55, 1.55)
    ax.set_xlim(0, 28)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    axes_format(ax, "Среднее время на запрос, мс")
    ecdf.set_xlim(0, 39)
    ecdf.set_ylim(0, 104)
    ecdf.set_yticks([0, 50, 95])
    ecdf.set_xticks([0, 10, 20, 30])
    axes_format(ecdf, "JPEG → три названия блюд, мс")
    ecdf.legend(loc="lower right", fontsize=7.6, frameon=False, handlelength=1.5, labelspacing=.3)
    save(fig, "phone-profile-v5" if instrumented else "phone-profile")


if __name__ == "__main__":
    style()
    pilot_figure()
    pilot_figure(pie=True)
    phone_figure()
    phone_figure(instrumented=True)
