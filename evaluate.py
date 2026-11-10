"""
Evaluation figures for the RF and UNet++ runs (run after both training scripts).

Outputs in <task folder>/evaluation/ (four_class, binary/snow, binary/river,
binary/combined), one per dataset:
  compare_<date>.png     true colour | hand labels | RF | UNet++ (whole box)
  zoom_<date>.png        same four panels, zoomed on the labelled area
  scores.png             per-class F1 / precision / recall, RF vs UNet++
  confusion.png          cross-validated confusion matrices (row-normalised)
  summary.csv            per-class scores for both models, all dates and per date

Scores are read from <task folder>/<model>/metrics.txt, so they are the
cross-validated ones (test rows never seen in training), not scores on the
training pixels.

Run:
    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/evaluate.py --task four --dataset exclude_daksh
    (--task combined after combine.py)
"""

import re, csv, argparse
import numpy as np
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import data as D

MODELS  = D.MODEL_NAMES
TASK_DIR = OUT = COLOURS = CMAP = NAMES = None              # set in main()


def parse_metrics(path):
    """metrics.txt -> {section title: (per-class rows, confusion matrix)}."""
    out, title, rows, cm = {}, None, {}, []
    for line in path.read_text().splitlines():
        if line.startswith("--- ") or (" — " in line and not out and title is None):
            if title:
                out[title] = (rows, np.array(cm))
            title, rows, cm = (line[4:].strip() if line.startswith("--- ") else "all dates"), {}, []
            continue
        p = line.split()
        if not p or p[0] not in NAMES:
            continue
        if len(p) == 5 and "." in p[2]:
            rows[p[0]] = dict(n=int(p[1].replace(",", "")), precision=float(p[2]),
                              recall=float(p[3]), f1=float(p[4]))
        elif len(p) == 1 + len(NAMES) and all(re.match(r"[\d,]+$", v) for v in p[1:]):
            cm.append([int(v.replace(",", "")) for v in p[1:]])
    if title:
        out[title] = (rows, np.array(cm))
    return out


def true_colour(x):
    rgb = np.stack([x[2], x[1], x[0]], -1)                   # red, green, blue
    return np.nan_to_num(np.clip(rgb / 0.4, 0, 1))


def show_map(ax, a, title, aspect):
    m = np.ma.masked_where((a < 1) | (a > len(NAMES)), a)
    ax.imshow(m, cmap=CMAP, vmin=0.5, vmax=len(NAMES) + 0.5, interpolation="nearest", aspect=aspect)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def panels(s, preds, aspect, window, path):
    r0, r1, c0, c1 = window
    lab = np.where(s["y"] == D.IGNORE, 0, s["y"] + 1)
    fig, axes = plt.subplots(1, 2 + len(preds), figsize=(3.2 * (2 + len(preds)), 6.2))
    axes[0].imshow(true_colour(s["x"][:, r0:r1, c0:c1]),
                   aspect=aspect, interpolation="nearest")
    axes[0].set_title(f"{s['date']} true colour", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    show_map(axes[1], lab[r0:r1, c0:c1], "hand labels", aspect)
    for ax, (name, p) in zip(axes[2:], preds.items()):
        show_map(ax, p[r0:r1, c0:c1], MODELS[name], aspect)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLOURS, NAMES)],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130); plt.close(fig)


def main():
    global TASK_DIR, OUT, COLOURS, CMAP, NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(D.TASKS) + ["combined"], default="four")
    ap.add_argument("--dataset", choices=D.DATASETS, default="exclude_daksh")
    args = ap.parse_args()
    D.configure("four" if args.task == "combined" else args.task, args.dataset)
    TASK_DIR = D.task_dir(args.task, args.dataset)
    OUT = TASK_DIR / "evaluation"
    COLOURS = [D.COLOURS[i] for i in D.CLASSES]
    CMAP = ListedColormap(COLOURS)
    NAMES = list(D.CLASSES.values())
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = D.load_all()
    have = [m for m in MODELS if (TASK_DIR / D.MODEL_DIRS[m] / "metrics.txt").exists()]
    print("models:", ", ".join(MODELS[m] for m in have))

    # 1. maps
    for s in scenes:
        preds = {}
        for m in have:
            f = TASK_DIR / D.MODEL_DIRS[m] / f"pred_{s['date']}.tif"
            if f.exists():
                preds[m] = gdal.Open(str(f)).ReadAsArray()
        if not preds:
            continue
        lat = s["ref"].GetGeoTransform()[3]
        aspect = 1 / np.cos(np.radians(lat - 0.22))          # square pixels on the ground
        H, W = s["y"].shape
        panels(s, preds, aspect, (0, H, 0, W), OUT / f"compare_{s['date']}.png")
        rr, cc = np.nonzero(s["y"] != D.IGNORE)
        if len(rr):
            pad = 40
            win = (max(0, rr.min() - pad), min(H, rr.max() + pad),
                   max(0, cc.min() - pad), min(W, cc.max() + pad))
            panels(s, preds, aspect, win, OUT / f"zoom_{s['date']}.png")
        print("  maps", s["date"])

    # 2. scores
    res = {m: parse_metrics(TASK_DIR / D.MODEL_DIRS[m] / "metrics.txt") for m in have}
    rows = []
    for m in have:
        for sec, (cls, _) in res[m].items():
            for n in NAMES:
                if n in cls:
                    rows.append([MODELS[m], sec, n, cls[n]["n"], cls[n]["precision"],
                                 cls[n]["recall"], cls[n]["f1"]])
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dates", "class", "n_test_px", "precision", "recall", "f1"])
        w.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    xs = np.arange(len(NAMES)); bw = 0.8 / max(len(have), 1)
    for ax, metric in zip(axes, ["f1", "precision", "recall"]):
        for i, m in enumerate(have):
            cls = res[m]["all dates"][0]
            v = [cls.get(n, {}).get(metric, 0) for n in NAMES]
            b = ax.bar(xs + (i - (len(have) - 1) / 2) * bw, v, bw, label=MODELS[m],
                       color=["#4c72b0", "#dd8452"][i])
            ax.bar_label(b, fmt="%.2f", fontsize=8)
        ax.set_xticks(xs, NAMES); ax.set_title(metric if metric != "f1" else "F1")
        ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle("External test: trained on the patch dates, scored on the labelled daksh date"
                 if D.DATASET == "external_daksh" else "Cross-validated scores, all labelled dates")
    fig.tight_layout(); fig.savefig(OUT / "scores.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(have), figsize=(5 * len(have), 4.4))
    for ax, m in zip(np.atleast_1d(axes), have):
        cm = res[m]["all dates"][1]
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(NAMES)):
            for j in range(len(NAMES)):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(NAMES)), NAMES); ax.set_yticks(range(len(NAMES)), NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(MODELS[m])
    fig.tight_layout(); fig.savefig(OUT / "confusion.png", dpi=130); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
