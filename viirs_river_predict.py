"""
Run 19 prediction images: one map per model with all four states at once on river + buffer VIIRS
375 m cells (RF and UNet++), and inside each map the split of the area:
  overall  water (river: open water + ice) % vs land (snow + snow-free land) %
  water    open water % vs ice % (shares of the water)
  land     snow % vs snow-free land % (shares of the land)
given for the river + buffer area and for the whole VIIRS image.

States
  1 ice_free_river_snow_free_land
  2 ice_free_river_snow_covered_land
  3 ice_covered_river_snow_free_land
  4 ice_covered_river_snow_covered_land
Main image uses rule50 (majority); a rule90 copy is written as pred_<date>_rule90.jpg.

Labelled days (the 6 run 16 dates): states = run 19 leave-one-date-out maps, part % = run 18
leave-one-date-out % maps (2D CNN: trained here, also leave-one-date-out), truth from the run 16 Landsat map: every number is for a day the model
did not see.
VIIRS-only days (VIIRS file, no run 16 date): final models trained on all 6 labelled days
  states  run 19 RF / UNet++ classifiers (one per rule)
  part %  run 18 RF / UNet++ % models, and the same % model as a 2D CNN (whole image)

Outputs: output/19_viirs_river_states/predict/
  <model>/<rule>/class_<date>.tif, <model>/pct_<date>.tif (VIIRS-only days), totals.json
  images/pred_<date>.jpg (rule50), images/pred_<date>_rule90.jpg

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/viirs_river_predict.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from osgeo import gdal

import rf_viirs_unmix as U
import viirs_states as V
import viirs_river_states as R

gdal.UseExceptions()
OUT = R.OUT / "predict"
RUN18 = U.HERE / "output" / "18_viirs_states"
MODELS = (("rf", "RF"), ("unetpp", "UNet++"), ("cnn2d", "2D CNN"))
NAMES = {"truth": "Landsat truth", "rf": "RF", "unetpp": "UNet++", "cnn2d": "2D CNN"}
AREAS = (("river", R.AREA), ("box", "whole VIIRS image"))


def split(pct, use):
    """Area split over the used cells; pct is (n, 4) fractions open, ice, snow, snow-free (NaN allowed)."""
    q = pct[use]
    q = q[np.isfinite(q).all(1)]
    if not len(q):
        return None
    o, i, s, f = (float(q[:, k].mean()) for k in range(4))
    water, land = o + i, s + f
    return {"water_pct": 100 * water, "land_pct": 100 * land,
            "open_water_pct_of_water": 100 * o / max(water, 1e-9), "ice_pct_of_water": 100 * i / max(water, 1e-9),
            "snow_pct_of_land": 100 * s / max(land, 1e-9), "snow_free_pct_of_land": 100 * f / max(land, 1e-9),
            "open_water_pct": 100 * o, "ice_pct": 100 * i, "snow_pct": 100 * s, "snow_free_land_pct": 100 * f}


def split_text(sp, title):
    return (f"{title}\n"
            f" water {sp['water_pct']:5.1f}%  land {sp['land_pct']:5.1f}%\n"
            f" water = open {sp['open_water_pct_of_water']:3.0f}% + ice {sp['ice_pct_of_water']:3.0f}%\n"
            f" land  = snow {sp['snow_pct_of_land']:3.0f}% + no snow {sp['snow_free_pct_of_land']:3.0f}%")


def read_pct(path):
    a = gdal.Open(str(path)).ReadAsArray().astype(np.float64)
    return (a.reshape(4, -1) / 100).T


def patch_info(date):
    """Landsat acquisition and VIIRS pass lines from the Patches README."""
    txt = (U.PATCHES / f"nenana_{date.replace('-', '')}" / "README.txt")
    out = {"landsat": "", "viirs": ""}
    if txt.exists():
        for line in txt.read_text().splitlines():
            if "acquired" in line and not out["landsat"]:
                out["landsat"] = line.split("acquired", 1)[1].split(",")[0].strip()
            if line.strip().startswith("Pass") and not out["viirs"]:
                out["viirs"] = line.split(":", 1)[1].split(",", 2)[-1].strip()
    return out


def landsat_rgb(date):
    """Landsat SR B6/B5/B4 (SWIR1/NIR/red, the bands of VIIRS I3/I2/I1), 2-98 % stretch, and its extent."""
    f = U.PATCHES / f"nenana_{date.replace('-', '')}" / f"{date}_sr.tif"
    if not f.exists():
        return None
    ds = gdal.Open(str(f))
    a = ds.ReadAsArray().astype(np.float32)              # B2, B3, B4, B5, B6, B7
    rgb = np.stack([a[4], a[3], a[2]], -1)
    ok = np.isfinite(rgb).all(-1)
    lo, hi = np.nanpercentile(rgb[ok], [2, 98], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb[~ok] = 0.6
    gt = ds.GetGeoTransform()
    return rgb, [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]


L16_COLOURS = ["#0066ff", "#ff00ff", "#00ffff", "#1a7f37", "#b5d96b", "#c8a165"]
L16_NAMES = ["river", "ice", "snow", "forest", "shrub/grass/wetland", "bare/developed"]


def run16_map(date):
    """Run 16 with_daksh map: the label maps for the 6 run 16 dates, the run 16 model applied in run 17 otherwise."""
    for p, src in ((U.RUN16 / f"pred_{date}.tif", "run 16 map, used as truth"),
                   (U.RUN16 / "new_dates" / date / "rf_pred.tif", "run 16 map, used as truth"),
                   (U.OUT / "landsat_maps" / f"pred_{date}.tif", "run 16 model on this Landsat; display only")):
        if p.exists():
            ds = gdal.Open(str(p))
            gt = ds.GetGeoTransform()
            return (ds.ReadAsArray(), [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]],
                    src)
    return None


def draw_mask(a, day, ext):
    """River outline (yellow) and, with a buffer, the outer buffer edge (orange dashed)."""
    hv, wv = day["shape"]
    river, buf = R.river_mask(day["occ"])
    xs = ext[0] + (np.arange(wv) + 0.5) * (ext[1] - ext[0]) / wv
    ys = ext[3] - (np.arange(hv) + 0.5) * (ext[3] - ext[2]) / hv
    a.contour(xs, ys, river.astype(float), levels=[0.5], colors="yellow", linewidths=0.7)
    if buf.any():
        a.contour(xs, ys, (river | buf).astype(float), levels=[0.5], colors="orange", linewidths=0.7, linestyles="--")


def image(day, states, sp, labelled, rule):
    """One map per source with all four states; sp[src][area] = split dict. VIIRS and Landsat side by side first."""
    hv, wv = day["shape"]
    g = day["use"]
    cols = (["truth"] if labelled else []) + [m for m, _ in MODELS]
    fig, ax = plt.subplots(1, len(cols) + 3, figsize=(4.0 * (len(cols) + 3), 8.6), constrained_layout=True)
    gt = day["gt_v"]
    ext = [gt[0], gt[0] + wv * gt[1], gt[3] + hv * gt[5], gt[3]]
    info = patch_info(day["date"])
    ax[0].imshow(R.viirs_rgb(day["date"]), interpolation="nearest", extent=ext)
    draw_mask(ax[0], day, ext)
    ax[0].set_title(f"VIIRS 375 m (I3/I2/I1)\n{info['viirs']}", fontsize=10)
    ls = landsat_rgb(day["date"])
    if ls is not None:
        ax[1].imshow(ls[0], interpolation="nearest", extent=ls[1])
        draw_mask(ax[1], day, ext)
        ax[1].set_xlim(ext[0], ext[1]); ax[1].set_ylim(ext[2], ext[3])
        ax[1].set_title(f"Landsat 30 m (B6/B5/B4 = same bands)\n{info['landsat']}", fontsize=10)
    else:
        ax[1].set_title("no Landsat for this date", fontsize=10)
    m16 = run16_map(day["date"])
    if m16 is not None:
        cls, mext, source = m16
        ax[2].imshow(cls, cmap=ListedColormap(["#ffffff"] + L16_COLOURS), vmin=-0.5, vmax=6.5,
                     interpolation="nearest", extent=mext)
        draw_mask(ax[2], day, ext)
        ax[2].set_xlim(ext[0], ext[1]); ax[2].set_ylim(ext[2], ext[3])
        ax[2].set_title(f"Run 16 Landsat 30 m map\n({source})", fontsize=10)
    else:
        ax[2].set_title("no run 16 map", fontsize=10)
    for a in ax[:3]:
        a.set_aspect("equal")                          # VIIRS pixels are square in degrees, as in the map panels
    for c, src in enumerate(cols):
        a = ax[c + 3]
        s = states[src][rule]
        a.imshow(np.where(g, s, 0).reshape(hv, wv), cmap=ListedColormap([R.GREY] + V.COLOURS),
                 vmin=-0.5, vmax=4.5, interpolation="nearest")
        a.set_title(f"{NAMES[src]}: 4 states ({rule})", fontsize=11)
        a.text(0.02, 0.02, f"states, % of {R.AREA} px\n" + "\n".join(
            f" {i + 1} {n.replace('_', ' '):<36}{v:5.1f}" for i, (n, v) in enumerate(zip(V.STATES, R.shares(s, g)))),
            transform=a.transAxes, family="monospace", fontsize=6.6, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.92))
        box = "\n\n".join(split_text(sp[src][k], t) for k, t in AREAS if sp[src].get(k))
        a.text(0.98, 0.98, box, transform=a.transAxes, family="monospace", fontsize=7, va="top", ha="right",
               bbox=dict(boxstyle="round", fc="#fffbe6", ec="#c9b458", alpha=0.95))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    mask_keys = [Line2D([], [], color="yellow", lw=1.5, label="mask: river (occ >= 0.05)")]
    if R.BUFFER_CELLS:
        mask_keys.append(Line2D([], [], color="orange", lw=1.5, ls="--", label=f"mask: outer edge of {R.BUFFER_CELLS} px buffer"))
    l16 = [Patch(color=c, label=f"run 16: {n}") for c, n in zip(L16_COLOURS, L16_NAMES)]
    fig.legend(handles=R.state_legend() + [Patch(color=R.GREY, label=f"not predicted (outside {R.AREA})")] + mask_keys + l16,
               loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    kind = "left-out test day, truth from run 16 Landsat" if labelled else "no run 16 map: final models"
    fig.suptitle(f"VIIRS prediction {day['date']} ({kind}), {int(g.sum())} {R.AREA} px\n"
                 "yellow box: water vs land %, and the open water / ice and snow / no-snow split", fontsize=12)
    name = f"pred_{day['date']}.jpg" if rule == "rule50" else f"pred_{day['date']}_{rule}.jpg"
    fig.savefig(OUT / "images" / name, dpi=115, pil_kwargs={"quality": 90})
    plt.close(fig)


def record(d, states, sp, kind):
    return {"kind": kind, "river_buffer_cells": int(d["use"].sum()),
            "area_split": {NAMES[k]: {AREAS[0][1] if a == "river" else AREAS[1][1]: v for a, v in x.items()}
                           for k, x in sp.items()},
            "state_pct_of_river_buffer": {NAMES[src]: {rule: R.shares(states[src][rule], d["use"]) for rule in R.RULES}
                                          for src in states}}


if __name__ == "__main__":
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "images").glob("pred_*.jpg"):
        old.unlink()
    shifts = {r["date"]: r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text())}
    cache, days = {}, {}
    labelled = [d for d in U.viirs_dates()
                if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    viirs_only = [d for d in U.viirs_dates() if d not in labelled]
    print("labelled", labelled, "\nVIIRS only", viirs_only, flush=True)

    occ = river = buf = None
    for date in labelled + viirs_only:
        d = days[date] = V.load_day(date, shifts.get(date, [0, 0]), cache)
        if occ is None:
            occ = R.crop_mask(d["gt_v"], d["shape"])
            river, buf = R.river_mask(occ)
        d["occ"] = np.nan_to_num(occ).astype(np.float32)
        area = (river | buf).reshape(-1)
        d["finite"] = np.isfinite(d["img"].reshape(len(d["img"]), -1)).all(0)
        if date in labelled:
            d["state"] = {rule: V.states(d["pct"], d["good"], rule) for rule in R.RULES}
            d["use"] = d["good"] & area
        else:                                         # no Landsat truth: any cell with VIIRS inputs
            d["use"] = d["finite"] & area

    report = {}
    for date in labelled:                             # leave-one-date-out maps from runs 18 / 19
        d = days[date]
        states = {"truth": d["state"]}
        sp = {"truth": {"river": split(d["pct"], d["use"]), "box": split(d["pct"], d["good"])}}
        for m, _ in MODELS:
            states[m] = {rule: gdal.Open(str(R.OUT / m / rule / f"class_{date}.tif")).ReadAsArray().reshape(-1)
                         for rule in R.RULES}
            if m == "cnn2d":                          # no run 18 CNN: leave-one-date-out % model here
                p = V.predict_unet(V.fit_unet(days, [x for x in labelled if x != date], arch="cnn2d"), d)
                p = np.where(d["good"][:, None], p, np.nan)
            else:
                p = read_pct(RUN18 / m / f"pct_{date}.tif")
            sp[m] = {"river": split(p, d["use"]), "box": split(p, d["good"])}
        for rule in R.RULES:
            image(d, states, sp, True, rule)
        report[date] = record(d, states, sp, "left-out test day")
        print(date, "done", flush=True)

    if viirs_only:                                    # final models on all labelled days
        cls_models = {("rf", rule): R.fit_rf(days, labelled, rule) for rule in R.RULES}
        for arch in ("unetpp", "cnn2d"):
            cls_models.update({(arch, rule): R.fit_unet(days, labelled, rule, arch=arch) for rule in R.RULES})
        pct_models = {"rf": V.fit_rf(days, labelled), "unetpp": V.fit_unet(days, labelled),
                      "cnn2d": V.fit_unet(days, labelled, arch="cnn2d")}
        print("final models trained", flush=True)
        for date in viirs_only:
            d = days[date]
            d_box = {**d, "good": d["finite"]}       # % over the whole image
            states, sp = {}, {}
            for m, _ in MODELS:
                states[m] = {}
                for rule in R.RULES:
                    s = (R.predict_rf(cls_models[(m, rule)], d) if m == "rf"
                         else R.predict_unet(cls_models[(m, rule)], d))
                    states[m][rule] = s
                    V.write_class(OUT / m / rule, d, s)
                p = V.predict_rf(pct_models[m], d_box) if m == "rf" else V.predict_unet(pct_models[m], d_box)
                p = np.where(d["finite"][:, None], p, np.nan)
                V.write_pct(OUT / m, d_box, p)
                sp[m] = {"river": split(p, d["use"]), "box": split(p, d["finite"])}
            for rule in R.RULES:
                image(d, states, sp, False, rule)
            report[date] = record(d, states, sp, "VIIRS only, final models")
            print(date, "done", flush=True)

    (OUT / "totals.json").write_text(json.dumps(report, indent=1))
    for date, v in report.items():
        print(date, v["kind"])
        for src, areas in v["area_split"].items():
            for a, x in areas.items():
                print(f"   {src:<14}{a:<19} water {x['water_pct']:5.1f} (open {x['open_water_pct_of_water']:3.0f} / "
                      f"ice {x['ice_pct_of_water']:3.0f})  land {x['land_pct']:5.1f} (snow {x['snow_pct_of_land']:3.0f} / "
                      f"no snow {x['snow_free_pct_of_land']:3.0f})")
        for src, r in v["state_pct_of_river_buffer"].items():
            print(f"   {src:<14}states " + "  ".join(f"{rule} " + "/".join(f"{x:.0f}" for x in s) for rule, s in r.items()))
    print("wrote", OUT, flush=True)
