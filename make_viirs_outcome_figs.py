"""
Figures of the VIIRS outcome of runs 19 and 22: the four parts (open water / ice / snow / snow-free land %) and
the four states per VIIRS 375 m pixel as PREDICTED FROM VIIRS, on the run's mask. No Landsat map is shown, except
as the run 16 reference bars in the zoom.

  predicted %      RF and UNet++ % models (run 18 design), left-out day: run 19 -> output/18_viirs_states/<m>/,
                   run 22 -> output/22_viirs_river_gauss/pct_models/<m>/; VIIRS-only days: <run>/predict/<m>/
  predicted state  RF, UNet++, 2D CNN state models of the run (rule50), left-out day: <run>/<m>/rule50/,
                   VIIRS-only days: <run>/predict/<m>/rule50/

Outputs: <run output>/images/viirs_outcome_<date>_overview.png, _zoom.png

    RIVER_BUFFER=1 [RIVER_LABELS=psf] ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/make_viirs_outcome_figs.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from osgeo import gdal

import make_aggregation_figs as G
import rf_viirs_unmix as U
import viirs_river_states as R
import viirs_states as V

gdal.UseExceptions()
OUT = R.OUT / "images"
PCT_LODO = (R.OUT / "pct_models") if R.GAUSS else U.HERE / "output" / "18_viirs_states"
MODELS = (("rf", "RF"), ("unetpp", "UNet++"), ("cnn2d", "2D CNN"))
PART_CMAPS = ["Blues", "RdPu", "GnBu", "YlGn"]
STATE_CMAP = ListedColormap(["#f2f2f2"] + V.COLOURS)
SHORT_STATES = ["1 open river, no-snow land", "2 open river, snow land", "3 ice river, no-snow land",
                "4 ice river, snow land"]


def labelled_dates():
    return [d for d in U.viirs_dates()
            if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]


def read(path):
    return gdal.Open(str(path)).ReadAsArray().astype(np.float32) if path.exists() else None


def predicted(date, labelled):
    """{model: (pct (4, hv, wv) in % or None, state (hv, wv) or None)} predicted from VIIRS."""
    out = {}
    for m, _ in MODELS:
        if labelled:
            pct = read((R.OUT / "pct_models" / m if m == "cnn2d" else PCT_LODO / m) / f"pct_{date}.tif")
            st = read(R.OUT / m / "rule50" / f"class_{date}.tif")
        else:
            pct = read(R.OUT / "predict" / m / f"pct_{date}.tif")
            st = read(R.OUT / "predict" / m / "rule50" / f"class_{date}.tif")
        out[m] = (pct, st)
    return out


def overview(date, labelled, area, river, buf):
    pred = predicted(date, labelled)
    rows = [m for m, _ in MODELS if pred[m][0] is not None or pred[m][1] is not None]
    names = dict(MODELS)
    fig, ax = plt.subplots(len(rows), 6, figsize=(22, 6.6 * len(rows)), constrained_layout=True, squeeze=False)
    for i, m in enumerate(rows):
        pct, st = pred[m]
        a0 = ax[i, 0]
        a0.imshow(R.viirs_rgb(date), interpolation="nearest")
        a0.contour(river.astype(float), levels=[0.5], colors="yellow", linewidths=0.6)
        if buf.any():
            a0.contour((river | buf).astype(float), levels=[0.5], colors="orange", linewidths=0.6, linestyles="--")
        a0.set_title(f"VIIRS 375 m (I2/I2/I1) - model input\n{names[m]} row", fontsize=10)
        for k in range(4):
            a = ax[i, k + 1]
            if pct is None:
                a.text(0.5, 0.5, f"no {names[m]} % model\n(states only)", ha="center", va="center", transform=a.transAxes)
                a.set_title(f"{names[m]}: {G.PARTS[k]} %", fontsize=10)
                continue
            img = np.where(area, pct[k], np.nan)
            cmap = plt.get_cmap(PART_CMAPS[k]).copy(); cmap.set_bad("#f2f2f2")
            im = a.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
            a.contour(river.astype(float), levels=[0.5], colors="black", linewidths=0.4)
            a.set_title(f"{names[m]} from VIIRS: {G.PARTS[k]} %\nmean on {R.AREA} {np.nanmean(img):.1f} %", fontsize=10)
            fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.04, pad=0.02)
        a = ax[i, 5]
        if st is not None:
            s = np.where(area, st, 0)
            a.imshow(s, cmap=STATE_CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest")
            n = max(int((s > 0).sum()), 1)
            a.set_title(f"{names[m]} from VIIRS: predicted state (rule50)\n% 1/2/3/4 = "
                        + " / ".join(f"{100 * (s == j).sum() / n:.0f}" for j in range(1, 5)), fontsize=10)
        for x in ax[i]:
            x.set_xticks([]); x.set_yticks([])
    fig.legend(handles=[Patch(fc=c, ec="k", lw=0.3, label=n) for c, n in zip(V.COLOURS, SHORT_STATES)]
               + [Patch(color="#f2f2f2", label=f"outside {R.AREA}")],
               loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    kind = "left-out test day: models trained on the other 5 days" if labelled else "VIIRS-only day: final models"
    fig.suptitle(f"Run {R.RUN} VIIRS outcome, {date} ({kind}): the 4 parts and the 4 states PREDICTED FROM VIIRS per "
                 f"375 m pixel", fontsize=13)
    f = OUT / f"viirs_outcome_{date}_overview.png"
    fig.savefig(f, dpi=100); plt.close(fig)
    return f


def zoom(date, labelled, area, river, buf, size=5):
    pred = predicted(date, labelled)
    pct_rf, st_rf = pred["rf"]
    if pct_rf is None or st_rf is None:
        return None
    hv, wv = st_rf.shape
    p = np.moveaxis(pct_rf, 0, -1).reshape(-1, 4) / 100
    ok = np.isfinite(p).all(1) & area.ravel()
    rows = np.repeat(np.arange(hv), wv); cols = np.tile(np.arange(wv), hv)
    inner = (rows >= size) & (rows < hv - size) & (cols >= size) & (cols < wv - size)
    cand = np.flatnonzero(ok & river.ravel() & inner)
    c0 = cand[np.argmax(np.sort(p[cand], 1)[:, -2])]                 # most mixed predicted river pixel
    r0, q0 = divmod(int(c0), wv)
    r0 = int(np.clip(r0 - size // 2, 0, hv - size)); q0 = int(np.clip(q0 - size // 2, 0, wv - size))
    rgb = R.viirs_rgb(date)[r0:r0 + size, q0:q0 + size]
    ref = None
    if labelled:                                                    # run 16 reference fractions for the bars
        if R.GAUSS:
            import make_aggregation_figs_17_18 as G2
            ref = G2.parts4(G2.gauss17(date)["w_psf"])
        else:
            ref = G.aggregate(date)["parts"]

    fig = plt.figure(figsize=(21, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb, interpolation="nearest")
    ax0.set_xticks(np.arange(-0.5, size, 1), minor=True); ax0.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax0.grid(which="minor", color="white", lw=1.5)
    ax0.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax0.set_title(f"VIIRS 375 m (I2/I2/I1), {size} x {size} pixels: the model input", fontsize=10)
    ax1 = fig.add_subplot(gs[0, 1])
    sub = np.where(area[r0:r0 + size, q0:q0 + size], st_rf[r0:r0 + size, q0:q0 + size], 0)
    ax1.imshow(sub, cmap=STATE_CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax1.set_xticks(np.arange(-0.5, size, 1), minor=True); ax1.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax1.grid(which="minor", color="black", lw=1.5)
    ax1.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for i in range(size):
        for j in range(size):
            c = (r0 + i) * wv + (q0 + j)
            tag = "R" if river[r0 + i, q0 + j] else ("B" if buf[r0 + i, q0 + j] else "-")
            if not ok[c]:
                txt = f"{tag}\nnot\npredicted"
            else:
                txt = (f"{tag}  state {int(st_rf[r0 + i, q0 + j])}\nW {100 * p[c, 0]:.0f} I {100 * p[c, 1]:.0f}\n"
                       f"S {100 * p[c, 2]:.0f} L {100 * p[c, 3]:.0f}")
            ax1.text(j, i, txt, ha="center", va="center", fontsize=7.5, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", alpha=1.0))
    ax1.set_title("RF from VIIRS: predicted state (colour) and predicted % per pixel\n"
                  "R = river mask, B = buffer; W/I/S/L = open water / ice / snow / snow-free land %", fontsize=10)
    axb = fig.add_subplot(gs[0, 2])
    ex = [c for c in [(r0 + i) * wv + (q0 + j) for i in range(size) for j in range(size)] if ok[c] and river.ravel()[c]]
    ex = sorted(ex, key=lambda c: -np.sort(p[c])[-2])[:3]
    bars = []
    for c in ex:
        bars.append((f"pixel {divmod(c, wv)}\nRF (VIIRS)", p[c]))
        if pred["unetpp"][0] is not None:
            bars.append(("UNet++ (VIIRS)", np.moveaxis(pred["unetpp"][0], 0, -1).reshape(-1, 4)[c] / 100))
        if ref is not None:
            bars.append(("run 16 Landsat (reference)", ref[c]))
    for k, (lab, v) in enumerate(bars):
        left = 0
        for i in range(4):
            axb.barh(k, 100 * v[i], left=left, color=G.PART_COLOURS[i], edgecolor="k", lw=0.3,
                     hatch="//" if "run 16" in lab else None)
            if v[i] > 0.07:
                axb.text(left + 50 * v[i], k, f"{100 * v[i]:.0f}", ha="center", va="center", fontsize=8)
            left += 100 * v[i]
    axb.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    axb.set_xlim(0, 100); axb.invert_yaxis(); axb.set_xlabel("% of the VIIRS pixel")
    axb.set_title("4 parts of 3 river pixels: predicted from VIIRS"
                  + (" vs run 16 reference (hatched)" if ref is not None else ""), fontsize=10)
    axb.legend(handles=[Patch(color=c, label=n) for c, n in zip(G.PART_COLOURS, G.PARTS)], ncol=4, fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.legend(handles=[Patch(fc=c, ec="k", lw=0.3, label=n) for c, n in zip(V.COLOURS, SHORT_STATES)],
               loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Run {R.RUN} VIIRS outcome, {date}: 4 parts and 4 states predicted from VIIRS per 375 m pixel",
                 fontsize=13)
    f = OUT / f"viirs_outcome_{date}_zoom.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lab = labelled_dates()
    for d in U.viirs_dates():
        ds = gdal.Open(str(U.PATCHES / f"nenana_{d.replace('-', '')}" / f"{d}_viirs_375m.tif"))
        river, buf = R.river_mask(R.crop_mask(ds.GetGeoTransform(), (ds.RasterYSize, ds.RasterXSize)))
        area = river | buf
        print(overview(d, d in lab, area, river, buf)); print(zoom(d, d in lab, area, river, buf))
