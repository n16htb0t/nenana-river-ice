"""
Test figures for the Word report, in report/figures/:

  gt_<dataset>_<date>.png   labelled dates, zoomed on the labels:
                            true colour | ground truth | RF and UNet++ on the labelled
                            pixels (cross-validated / external test) | RF and UNet++ maps
  new_<date>.png            unlabelled dates, whole box:
                            true colour | RF | UNet++ | RF state | UNet++ state

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/report/test_panels.py
"""

import sys
from pathlib import Path
import numpy as np
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data as D
import scene_mask as S
import images as I

FIG = HERE / "figures"
ASPECT = 1 / np.cos(np.radians(64.31))           # square pixels on the ground
NEW = {"2024-04-20": None, "2024-04-29": None, "2026-03-26": "new", "2023-05-05": "new"}


def overlay(img, mask, colours, alpha=0.65):
    out = img.astype(np.float32) / 255
    for k, c in colours.items():
        m = mask == k
        out[m] = (1 - alpha) * out[m] + alpha * np.array(c) / 255
    return out


def show(ax, im, title):
    ax.imshow(im, aspect=ASPECT, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def oof_map(path, date, shape):
    z = np.load(path)
    m = z["date"] == date
    out = np.zeros(shape, np.uint8)
    out[z["row"][m], z["col"][m]] = z["p"][m] + 1
    acc = (z["p"][m] == z["y"][m]).mean() if m.any() else float("nan")
    return out, acc, int(m.sum())


WIN = (120, 276)                                     # rows x cols: about 3.6 x 3.6 km


def windows(lab, n=2):
    """Up to n non-overlapping WIN windows holding the most labelled pixels."""
    h, w = WIN
    S_ = np.pad(lab.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    tot = S_[h:, w:] - S_[:-h, w:] - S_[h:, :-w] + S_[:-h, :-w]
    out = []
    for _ in range(n):
        r, c = np.unravel_index(tot.argmax(), tot.shape)
        if tot[r, c] <= 0:
            break
        out.append((int(r), int(c)))
        tot[max(0, r - h):r + h, max(0, c - w):c + w] = -1
    return out


def ground_truth(dataset):
    D.configure("four", dataset)
    tdir = D.task_dir("four", dataset)
    cols = I.rgb(D.COLOURS)
    legend = [Patch(color=D.COLOURS[k], label=n) for k, n in D.CLASSES.items()]
    test = "external test" if dataset == "external_daksh" else "cross-validated"
    for s in D.load_all():
        lab = s["y"] != D.IGNORE
        if not lab.any() or (dataset == "external_daksh" and s["group"] != "daksh"):
            continue
        d = s["date"]
        img = I.image_for(d)
        gt = np.where(lab, s["y"] + 1, 0)
        cv = {m: oof_map(tdir / D.MODEL_DIRS[m] / "oof.npz", d, lab.shape) for m in D.MODEL_NAMES}
        maps = {m: gdal.Open(str(tdir / D.MODEL_DIRS[m] / f"pred_{d}.tif")).ReadAsArray() for m in D.MODEL_NAMES}
        for k, (r0, c0) in enumerate(windows(lab)):
            r1, c1 = r0 + WIN[0], c0 + WIN[1]
            n_here = int(lab[r0:r1, c0:c1].sum())
            panels = [(img, "true colour"), (overlay(img, gt, cols), "ground truth (hand labels)")]
            for m, name in D.MODEL_NAMES.items():
                p, acc, n = cv[m]
                panels.append((overlay(img, p, cols), f"{name}, labelled pixels\n{test}: accuracy {acc:.2f} on {n:,} px"))
            panels += [(overlay(img, maps[m], cols), f"{name} map") for m, name in D.MODEL_NAMES.items()]
            order = [0, 2, 3, 1, 4, 5]               # row 1: image, RF test, UNet++ test; row 2: truth, RF map, UNet++ map
            fig, axes = plt.subplots(2, 3, figsize=(15, 11.8))
            for ax, i in zip(axes.ravel(), order):
                show(ax, panels[i][0][r0:r1, c0:c1], panels[i][1])
            fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=11)
            fig.suptitle(f"{d} · {D.DATASET_DIRS[dataset]} · four classes · area {k + 1} "
                         f"(rows {r0}-{r1}, cols {c0}-{c1}, {n_here:,} labelled px here)", fontsize=12)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.07, wspace=0.03, hspace=0.2)
            out = FIG / f"gt_{dataset}_{d}_area{k + 1}.png"
            fig.savefig(out, dpi=100); plt.close(fig)
            print(out.name)


def unlabelled(dataset="with_daksh"):
    D.configure("four", dataset)
    tdir = D.task_dir("four", dataset)
    cols, scols = I.rgb(D.COLOURS), I.rgb(S.COLOURS)
    legend = [Patch(color=D.COLOURS[k], label=n) for k, n in D.CLASSES.items()]
    legend += [Patch(color=S.COLOURS[k], label=n.replace("_", " ")) for k, n in S.STATES.items()]
    for d, kind in NEW.items():
        img = I.image_for(d)
        panels = [(img, "true colour")]
        for what in ("pred", "state"):
            for m, name in D.MODEL_NAMES.items():
                md = tdir / D.MODEL_DIRS[m]
                if kind == "new":
                    f = tdir / "new_dates" / d / f"{m}_{'pred' if what == 'pred' else 'scene_mask'}.tif"
                else:
                    f = md / (f"pred_{d}.tif" if what == "pred" else f"scene_mask_{d}.tif")
                a = gdal.Open(str(f)).ReadAsArray()
                panels.append((overlay(img, a, cols if what == "pred" else scols, 0.6),
                               f"{name} {'classes' if what == 'pred' else 'river/land state'}"))
        fig, axes = plt.subplots(1, 5, figsize=(17, 9.4))
        for ax, (im, t) in zip(axes, panels):
            show(ax, im, t)
        fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)
        fig.suptitle(f"{d} (no labels) · {D.DATASET_DIRS[dataset]} · four classes", fontsize=12)
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        out = FIG / f"new_{d}.png"
        fig.savefig(out, dpi=100); plt.close(fig)
        print(out.name)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    ground_truth("with_daksh")
    ground_truth("external_daksh")
    unlabelled()  # noqa
