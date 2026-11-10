"""
Prediction images v2 for the river runs (19 by default): same panels as viirs_river_predict.py, redrawn from the
saved maps (no retraining), with

  - state 3 (ice-covered river, snow-free land) in pink instead of brown
  - river ice shown as ice: a pixel whose water fraction (open water + ice) is > 10 % and whose state is
    ice-covered (3 or 4) is drawn pink, whatever its land snow. Each panel uses its own water fraction:
    Landsat-derived panel -> run 16-derived %, RF / UNet++ / 2D CNN -> that model's % predicted from VIIRS
  - no totals box and no state-% box

Outputs: <run>/predict/images_v2/pred_<date>.jpg (rule50), pred_<date>_rule90.jpg

    RIVER_BUFFER=1 ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/make_pred_images_v2.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from osgeo import gdal

import rf_viirs_unmix as U
import viirs_river_predict as P
import viirs_river_states as R

gdal.UseExceptions()
OUT = R.OUT / "predict" / "images_v2"
PCT_LODO = (R.OUT / "pct_models") if R.GAUSS else U.HERE / "output" / "18_viirs_states"
TRUTH_PCT = (R.OUT / "pct_models" / "truth") if R.GAUSS else U.HERE / "output" / "18_viirs_states" / "truth"
WATER_MIN = 0.10
PINK = "#ff00ff"                                    # the run 16 ice colour
COLOURS = ["#2b83ba", "#00b4b4", PINK, "#e6f5ff"]   # states 1-4 (3 was brown)
STATE_CMAP = ListedColormap(["#9a9a9a"] + COLOURS)
STATES = ["1 ice-free river, snow-free land", "2 ice-free river, snow-covered land",
          "3 ice-covered river, snow-free land / river ice", "4 ice-covered river, snow-covered land"]
MODELS = (("rf", "RF"), ("unetpp", "UNet++"), ("cnn2d", "2D CNN"))


def read(path):
    return gdal.Open(str(path)).ReadAsArray().astype(np.float32) if path.exists() else None


def sources(date, labelled, rule):
    """[(title, state map, water-fraction source)] for the Landsat-derived map (if labelled) and each model."""
    out = []
    if labelled:
        out.append(("Landsat-derived (run 16 map)", read(R.OUT / "truth" / rule / f"class_{date}.tif"),
                    read(TRUTH_PCT / f"pct_{date}.tif")))
    for m, name in MODELS:
        if labelled:
            st = read(R.OUT / m / rule / f"class_{date}.tif")
            pct = read((R.OUT / "pct_models" / m if m == "cnn2d" else PCT_LODO / m) / f"pct_{date}.tif")
        else:
            st = read(R.OUT / "predict" / m / rule / f"class_{date}.tif")
            pct = read(R.OUT / "predict" / m / f"pct_{date}.tif")
        out.append((f"{name} from VIIRS", st, pct))
    return out


def display(state, pct):
    """State map for display: ice-covered pixels with > 10 % water drawn as river ice (pink)."""
    s = state.copy()
    if pct is not None:
        water = (np.nan_to_num(pct[0]) + np.nan_to_num(pct[1])) / 100.0
        s[(water > WATER_MIN) & np.isin(state, (3, 4))] = 3
    return s


def image(date, labelled, rule):
    ds = gdal.Open(str(U.PATCHES / f"nenana_{date.replace('-', '')}" / f"{date}_viirs_375m.tif"))
    gt = ds.GetGeoTransform(); hv, wv = ds.RasterYSize, ds.RasterXSize
    ext = [gt[0], gt[0] + wv * gt[1], gt[3] + hv * gt[5], gt[3]]
    occ = R.crop_mask(gt, (hv, wv))
    river, buf = R.river_mask(occ)
    area = river | buf
    day = {"shape": (hv, wv), "occ": occ}
    src = [s for s in sources(date, labelled, rule) if s[1] is not None]
    fig, ax = plt.subplots(1, 3 + len(src), figsize=(4.0 * (3 + len(src)), 8.0), constrained_layout=True)
    info = P.patch_info(date)
    ax[0].imshow(R.viirs_rgb(date), interpolation="nearest", extent=ext)
    P.draw_mask(ax[0], day, ext)
    ax[0].set_title(f"VIIRS 375 m (I2/I2/I1)\n{info['viirs']}", fontsize=10)
    ls = P.landsat_rgb(date)
    if ls is not None:
        ax[1].imshow(ls[0], interpolation="nearest", extent=ls[1])
        P.draw_mask(ax[1], day, ext)
        ax[1].set_xlim(ext[0], ext[1]); ax[1].set_ylim(ext[2], ext[3])
    ax[1].set_title(f"Landsat 30 m (B6/B5/B4)\n{info['landsat']}", fontsize=10)
    m16 = P.run16_map(date)
    if m16 is not None:
        ax[2].imshow(m16[0], cmap=ListedColormap(["#ffffff"] + P.L16_COLOURS), vmin=-0.5, vmax=6.5,
                     interpolation="nearest", extent=m16[1])
        P.draw_mask(ax[2], day, ext)
        ax[2].set_xlim(ext[0], ext[1]); ax[2].set_ylim(ext[2], ext[3])
    ax[2].set_title("Run 16 Landsat 30 m map\n" + ("(the label source)" if labelled else "(display only)"), fontsize=10)
    for a in ax[:3]:
        a.set_aspect("equal")
    for k, (title, st, pct) in enumerate(src):
        a = ax[3 + k]
        a.imshow(np.where(area, display(st, pct), 0), cmap=STATE_CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest")
        a.set_title(f"{title}: 4 states ({rule})", fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    keys = [Patch(fc=c, ec="k", lw=0.3, label=n) for c, n in zip(COLOURS, STATES)]
    keys += [Patch(color="#9a9a9a", label=f"not predicted (outside {R.AREA})"),
             Line2D([], [], color="yellow", lw=1.5, label="mask: river (occ >= 0.05)")]
    if R.BUFFER_CELLS:
        keys.append(Line2D([], [], color="orange", lw=1.5, ls="--", label="mask: outer edge of 1 px buffer"))
    keys += [Patch(color=c, label=f"run 16: {n}") for c, n in zip(P.L16_COLOURS, P.L16_NAMES)]
    fig.legend(handles=keys, loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    kind = "left-out test day" if labelled else "no run 16 label day: final models"
    fig.suptitle(f"Run {R.RUN} VIIRS prediction {date} ({kind}), {R.AREA}\n"
                 f"pink = river ice: ice-covered states on pixels with > {int(100 * WATER_MIN)} % water are shown as ice",
                 fontsize=12)
    f = OUT / (f"pred_{date}.jpg" if rule == "rule50" else f"pred_{date}_{rule}.jpg")
    fig.savefig(f, dpi=115, pil_kwargs={"quality": 90}); plt.close(fig)
    return f


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    labelled = [d for d in U.viirs_dates()
                if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    for d in U.viirs_dates():
        for rule in R.RULES:
            print(image(d, d in labelled, rule))
