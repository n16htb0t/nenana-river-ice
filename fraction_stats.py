"""
Fraction statistics, no rules: the four class fractions per VIIRS 375 m pixel predicted from VIIRS (RF, UNet++,
2D CNN; left-out days) against the Landsat class fractions (run 16 map aggregated to 375 m, as in the aggregation
overview). Run 19: box fractions; run 22 (RIVER_LABELS=psf): Gaussian fractions.

Per class (open water, ice, snow, snow-free land), model and area (river + buffer, whole image), over all 6 left-out
days and per day: n, MAE and RMSE (percentage points), bias (VIIRS - Landsat), R^2, Pearson r.

Outputs: <run>/fraction_stats/stats.json, stats.csv, scatter.png

    RIVER_BUFFER=1 [RIVER_LABELS=psf] ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/fraction_stats.py
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

import rf_viirs_unmix as U
import viirs_river_states as R

gdal.UseExceptions()
OUT = R.OUT / "fraction_stats"
PCT_LODO = (R.OUT / "pct_models") if R.GAUSS else U.HERE / "output" / "18_viirs_states"
TRUTH = (R.OUT / "pct_models" / "truth") if R.GAUSS else U.HERE / "output" / "18_viirs_states" / "truth"
PARTS = ["open water", "ice", "snow", "snow-free land"]
MODELS = (("rf", "RF"), ("unetpp", "UNet++"), ("cnn2d", "2D CNN"))


def read(path):
    return gdal.Open(str(path)).ReadAsArray().astype(np.float64)


def stats(t, p):
    e = p - t
    ss = ((t - t.mean()) ** 2).sum()
    return {"n": int(len(t)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"),
            "r": float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else float("nan"),
            "landsat_mean": float(t.mean()), "viirs_mean": float(p.mean())}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    labelled = [d for d in U.viirs_dates()
                if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    pairs = {m: {a: ([], []) for a in ("river + buffer", "whole image")} for m, _ in MODELS}
    per_day = {m: {} for m, _ in MODELS}
    for d in labelled:
        t = read(TRUTH / f"pct_{d}.tif")
        ds = gdal.Open(str(TRUTH / f"pct_{d}.tif"))
        river, buf = R.river_mask(R.crop_mask(ds.GetGeoTransform(), (ds.RasterYSize, ds.RasterXSize)))
        for m, _ in MODELS:
            p = read((R.OUT / "pct_models" / m if m == "cnn2d" else PCT_LODO / m) / f"pct_{d}.tif")
            ok = np.isfinite(t).all(0) & np.isfinite(p).all(0)
            for area, mask in (("river + buffer", ok & (river | buf)), ("whole image", ok)):
                pairs[m][area][0].append(t[:, mask]); pairs[m][area][1].append(p[:, mask])
            mk = ok & (river | buf)
            per_day[m][d] = {PARTS[k]: stats(t[k][mk], p[k][mk]) for k in range(4)}
    res = {"run": R.RUN, "labels": "Gaussian" if R.GAUSS else "box", "dates": labelled, "all": {}, "per_day": per_day}
    rows = []
    for m, name in MODELS:
        res["all"][m] = {}
        for area, (ts, ps) in pairs[m].items():
            t, p = np.concatenate(ts, 1), np.concatenate(ps, 1)
            res["all"][m][area] = {PARTS[k]: stats(t[k], p[k]) for k in range(4)}
            res["all"][m][area]["mean of 4 classes"] = {
                k2: float(np.mean([res["all"][m][area][c][k2] for c in PARTS])) for k2 in ("mae", "rmse", "r2", "r")}
            for c in PARTS:
                s = res["all"][m][area][c]
                rows.append([R.RUN, name, area, c, s["n"], s["landsat_mean"], s["viirs_mean"], s["mae"], s["rmse"],
                             s["bias"], s["r2"], s["r"]])
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))
    with open(OUT / "stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "area", "class", "n", "landsat_mean_pct", "viirs_mean_pct", "mae_pp", "rmse_pp",
                    "bias_pp", "r2", "r"])
        w.writerows([[x if isinstance(x, str) else (round(x, 4) if isinstance(x, float) else x) for x in r] for r in rows])

    # scatter: Landsat vs VIIRS fraction per pixel, river + buffer, all left-out days
    fig, ax = plt.subplots(3, 4, figsize=(17, 13), constrained_layout=True)
    for i, (m, name) in enumerate(MODELS):
        t, p = (np.concatenate(x, 1) for x in pairs[m]["river + buffer"])
        for k in range(4):
            a = ax[i, k]
            a.hist2d(t[k], p[k], bins=50, range=[[0, 100], [0, 100]], cmap="viridis", cmin=1)
            a.plot([0, 100], [0, 100], "w--", lw=1)
            s = res["all"][m]["river + buffer"][PARTS[k]]
            a.set_title(f"{name}: {PARTS[k]}\nMAE {s['mae']:.1f} pp, R² {s['r2']:.2f}, r {s['r']:.2f}, bias {s['bias']:+.1f}",
                        fontsize=10)
            a.set_xlabel("Landsat fraction % (run 16 aggregated)"); a.set_ylabel("VIIRS fraction % (predicted)")
            a.set_xlim(0, 100); a.set_ylim(0, 100)
    fig.suptitle(f"Run {R.RUN}: class fractions per VIIRS pixel, VIIRS-predicted vs Landsat-derived "
                 f"({res['labels']} fractions), river + buffer, 6 left-out days, no rules", fontsize=13)
    fig.savefig(OUT / "scatter.png", dpi=110); plt.close(fig)

    for area in ("river + buffer", "whole image"):
        print(f"\n== run {R.RUN} ({res['labels']}), {area}: MAE pp / R2 / r")
        for m, name in MODELS:
            a = res["all"][m][area]
            print(f"{name:<7}" + "  ".join(f"{c} {a[c]['mae']:.1f}/{a[c]['r2']:.2f}/{a[c]['r']:.2f}" for c in PARTS)
                  + f"  | mean MAE {a['mean of 4 classes']['mae']:.1f}")
    print("wrote", OUT)
