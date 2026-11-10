"""
JPEG overlays of every prediction map on its true-colour Landsat image.

For every task folder of a dataset (four_class, binary/*, independent/*) and
every model in it, writes to <task folder>/images/:
  <model>_<date>.jpg           image | image + class mask   (labelled dates)
  <model>_<date>_state.jpg     image | image + river/land state mask (four-class / combined)
  new_<model>_<date>.jpg       the same for the unlabelled dates in new_dates/

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/images.py --dataset with_daksh
"""

import argparse
from pathlib import Path
import numpy as np
from osgeo import gdal
from PIL import Image, ImageDraw
import data as D
import scene_mask as S

ALPHA  = 0.55
MODELS = D.MODEL_NAMES
NEW    = Path.home() / "Desktop/Patches"


def rgb(hexes):
    return {k: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for k, h in hexes.items()}


def image_for(date):
    """True-colour uint8 image of a date (patch, daksh or new-date folder)."""
    s = date.replace("-", "")
    for folder in (D.PATCHES / f"nenana_{s}", D.CROPS / s, NEW / f"nenana_{s}"):
        if (folder / f"{date}_sr.tif").exists() or any(folder.glob("landsat_*_box.tif")):
            x, _ = D.load_features(folder, date)
            return (np.nan_to_num(np.clip(np.stack([x[2], x[1], x[0]], -1) / 0.4, 0, 1)) * 255).astype(np.uint8)
    return None


def panel(img, mask, colours, names, title, path):
    h, w = img.shape[:2]
    W = int(round(w * np.cos(np.radians(64.31))))           # square pixels on the ground
    over = img.astype(np.float32).copy()
    for k, c in colours.items():
        m = mask == k
        over[m] = (1 - ALPHA) * over[m] + ALPHA * np.array(c)
    a = Image.fromarray(img).resize((W, h), Image.LANCZOS)
    b = Image.fromarray(over.astype(np.uint8)).resize((W, h), Image.NEAREST)
    canvas = Image.new("RGB", (2 * W + 30 + 330, h + 50), "white")
    canvas.paste(a, (0, 50)); canvas.paste(b, (W + 30, 50))
    dr = ImageDraw.Draw(canvas)
    dr.text((10, 15), title, fill=(0, 0, 0))
    for i, (k, n) in enumerate(names.items()):
        dr.rectangle([2 * W + 50, 60 + i * 22, 2 * W + 66, 76 + i * 22], fill=colours[k])
        dr.text((2 * W + 74, 62 + i * 22), n, fill=(0, 0, 0))
    canvas.save(path, quality=90)


def task_dirs(dataset):
    tasks = ["four", "snow", "river", "combined", "ind_river", "ind_ice", "ind_snow", "ind_land"]
    out = [(t, D.task_dir(t, dataset)) for t in tasks]
    return [(t, d) for t, d in out if d.is_dir()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=D.DATASETS, default="exclude_daksh")
    args = ap.parse_args()
    cache = {}
    for task, folder in task_dirs(args.dataset):
        D.configure("four" if task == "combined" else task, args.dataset)
        cls, cols = D.CLASSES, rgb(D.COLOURS)
        st_cols = rgb(S.COLOURS)
        out = folder / "images"
        out.mkdir(exist_ok=True)
        jobs = []
        for m in MODELS:
            for f in sorted((folder / D.MODEL_DIRS[m]).glob("pred_*.tif")):
                d = f.stem[5:]
                jobs.append((m, d, f, folder / D.MODEL_DIRS[m] / f"scene_mask_{d}.tif", f"{m}_{d}"))
            for f in sorted(folder.glob(f"new_dates/*/{m}_pred.tif")):
                d = f.parent.name
                jobs.append((m, d, f, f.parent / f"{m}_scene_mask.tif", f"new_{m}_{d}"))
        for m, d, f, sm, stem in jobs:
            if d not in cache:
                cache[d] = image_for(d)
            img = cache[d]
            if img is None:
                continue
            title = f"{d}   {MODELS[m]}   {folder.relative_to(D.OUTPUT)}"
            panel(img, gdal.Open(str(f)).ReadAsArray(), cols, cls, title, out / f"{stem}.jpg")
            if sm.exists():
                panel(img, gdal.Open(str(sm)).ReadAsArray(), st_cols, S.STATES,
                      title + "   river/land state", out / f"{stem}_state.jpg")
        print(f"{folder.relative_to(D.OUTPUT)}: {len(jobs)} map(s) -> images/")


if __name__ == "__main__":
    main()
