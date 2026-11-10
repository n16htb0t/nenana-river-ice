"""
Figures of the label step of runs 17 and 18: how the run 16 Landsat 30 m map becomes class shares per VIIRS
375 m pixel.

Run 18 (box shares, whole image): the Landsat pixels inside each VIIRS pixel are counted (after the run 17 shift),
pixels >= 98 % mapped are used, classes -> open water / ice / snow / snow-free land %. Same counting as runs 19-20,
without the river mask.
Run 17 (Gaussian shares, six classes): each VIIRS pixel weighs the Landsat pixels around its centre with a Gaussian
(sigma = 375 m / 2.8 = 134 m, cut at 2 sigma; Oleson et al. 1995), so the shares are of the six run 16 classes and
neighbouring Landsat pixels outside the box count a little.

Outputs: output/18_viirs_states/images/aggregation_<date>_{overview,zoom}.png
         output/17_viirs_unmix/images/aggregation_<date>_{overview,zoom}.png

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/make_aggregation_figs_17_18.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch

import make_aggregation_figs as G
import rf_viirs_unmix as U

OUT17 = U.HERE / "output" / "17_viirs_unmix" / "images"
OUT18 = U.HERE / "output" / "18_viirs_states" / "images"
MIN_VALID = U.MIN_VALID
CMAP6 = ListedColormap(["#ffffff"] + G.COLOURS)
SHORT = ["river", "ice", "snow", "forest", "shrub", "bare"]


# ----------------------------------------------------------------------------- run 18

def overview18(a, date):
    hv, wv = a["hv"], a["wv"]
    used = a["share"] >= MIN_VALID
    fig, ax = plt.subplots(1, 6, figsize=(24, 9), constrained_layout=True)
    ax[0].imshow(a["cls"], cmap=CMAP6, vmin=-0.5, vmax=6.5, interpolation="nearest")
    ax[0].set_title(f"Run 16 Landsat 30 m map, {date}\n(the classified labels, before aggregation)", fontsize=10)
    ax[1].imshow(used.reshape(hv, wv), cmap=ListedColormap(["#c0392b", "#1f4e9c"]), vmin=0, vmax=1,
                 interpolation="nearest")
    ax[1].set_title(f"VIIRS 375 m pixels >= 98 % mapped\nused {int(used.sum()):,} of {hv * wv:,}", fontsize=10)
    ax[1].legend(handles=[Patch(color="#1f4e9c", label="used"), Patch(color="#c0392b", label="dropped (< 98 %)")],
                 loc="lower left", fontsize=8)
    for k in range(4):
        img = np.where(used, 100 * a["parts"][:, k], np.nan).reshape(hv, wv)
        cmap = plt.get_cmap(["Blues", "RdPu", "GnBu", "YlGn"][k]).copy(); cmap.set_bad("#f2f2f2")
        im = ax[k + 2].imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
        ax[k + 2].set_title(f"{G.PARTS[k]} % per VIIRS pixel\nmean over the image {np.nanmean(img):.1f} %", fontsize=10)
        fig.colorbar(im, ax=ax[k + 2], orientation="horizontal", fraction=0.04, pad=0.02, label="% of the VIIRS pixel")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(G.COLOURS, G.NAMES)],
               loc="outside lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"Run 18 labels, {date}: run 16 30 m map -> counts per VIIRS pixel -> 4 part % over the whole image  "
                 f"(open water = river, snow-free land = forest + shrub + bare; run 17 shift {a['shift']} Landsat px)",
                 fontsize=12)
    f = OUT18 / f"aggregation_{date}_overview.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


# ----------------------------------------------------------------------------- run 17

def gauss17(date):
    cls, gt_f, v, gt_v, ds_v, _, _ = U.load_date(date, {})
    shift = next(r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text()) if r["date"] == date)
    hv, wv = v.shape[1:]
    bl, sig = U.blur(cls, gt_f)
    w_psf, mapped_psf = U.weights(cls, gt_f, gt_v, (hv, wv), tuple(shift), "psf", bl)
    w_box, _ = U.weights(cls, gt_f, gt_v, (hv, wv), tuple(shift), "box")
    lat_c = gt_v[3] + (np.arange(hv) + 0.5) * gt_v[5]
    lon_c = gt_v[0] + (np.arange(wv) + 0.5) * gt_v[1]
    fr = np.round((lat_c - gt_f[3]) / gt_f[5] - 0.5 - shift[0]).astype(int)       # same centres as U.weights
    fc = np.round((lon_c - gt_f[0]) / gt_f[1] - 0.5 - shift[1]).astype(int)
    return dict(cls=cls, hv=hv, wv=wv, w_psf=w_psf, mapped=mapped_psf, w_box=w_box, sig=sig, fr=fr, fc=fc,
                shift=shift, gt_f=gt_f, gt_v=gt_v)


def overview17(g, date):
    hv, wv = g["hv"], g["wv"]
    used = g["mapped"] >= MIN_VALID
    fig, ax = plt.subplots(2, 4, figsize=(19, 15), constrained_layout=True)
    ax = ax.ravel()
    ax[0].imshow(g["cls"], cmap=CMAP6, vmin=-0.5, vmax=6.5, interpolation="nearest")
    ax[0].set_title(f"Run 16 Landsat 30 m map, {date}", fontsize=10)
    ax[1].imshow(used.reshape(hv, wv), cmap=ListedColormap(["#c0392b", "#1f4e9c"]), vmin=0, vmax=1,
                 interpolation="nearest")
    ax[1].set_title(f"VIIRS pixels with >= 98 % Gaussian weight mapped\nused {int(used.sum()):,} of {hv * wv:,}",
                    fontsize=10)
    for k in range(6):
        img = np.where(used, 100 * g["w_psf"][:, k], np.nan).reshape(hv, wv)
        cmap = plt.get_cmap(["Blues", "RdPu", "GnBu", "Greens", "YlGn", "Oranges"][k]).copy(); cmap.set_bad("#f2f2f2")
        vmax = 30 if k in (0, 1) else 100
        im = ax[k + 2].imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax[k + 2].set_title(f"{G.NAMES[k]}: Gaussian share per VIIRS pixel\nmean {np.nanmean(img):.1f} %", fontsize=10)
        fig.colorbar(im, ax=ax[k + 2], orientation="horizontal", fraction=0.04, pad=0.02, label="% (weighted)")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(G.COLOURS, G.NAMES)],
               loc="outside lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"Run 17 class shares, {date}: Gaussian footprint (sigma 134 m, cut at 2 sigma) over the run 16 map, "
                 f"six classes kept; run 17 shift {g['shift']} Landsat px", fontsize=12)
    f = OUT17 / f"aggregation_{date}_overview.png"
    fig.savefig(f, dpi=105); plt.close(fig)
    return f


def zoom17(g, date):
    hv, wv = g["hv"], g["wv"]
    used = g["mapped"] >= MIN_VALID
    rows = np.repeat(np.arange(hv), wv); cols = np.tile(np.arange(wv), hv)
    inner = (rows > 3) & (rows < hv - 4) & (cols > 3) & (cols < wv - 4)
    cand = np.flatnonzero(used & inner)
    c = int(cand[np.argmax(np.sort(g["w_psf"][cand], 1)[:, -2])])          # most mixed pixel
    r, q = divmod(c, wv)
    cr, cc = int(g["fr"][r]), int(g["fc"][q])
    sr, sc = g["sig"]
    hr, hc = int(np.ceil(2 * sr)) + 3, int(np.ceil(2 * sc)) + 3
    H, Wd = g["cls"].shape
    r0, r1 = max(cr - hr, 0), min(cr + hr + 1, H); c0, c1 = max(cc - hc, 0), min(cc + hc + 1, Wd)
    win = g["cls"][r0:r1, c0:c1]
    yy, xx = np.mgrid[r0:r1, c0:c1]
    d2 = ((yy - cr) / sr) ** 2 + ((xx - cc) / sc) ** 2
    wgt = np.where(d2 <= 4, np.exp(-0.5 * d2), 0.0)                       # cut at 2 sigma
    box_h, box_w = abs(g["gt_v"][5] / g["gt_f"][5]), g["gt_v"][1] / g["gt_f"][1]

    fig, ax = plt.subplots(1, 3, figsize=(21, 8.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    ax[0].imshow(win, cmap=CMAP6, vmin=-0.5, vmax=6.5, interpolation="nearest")
    for k, ls in ((1, "-"), (2, "--")):
        ax[0].add_patch(Ellipse((cc - c0, cr - r0), 2 * k * sc, 2 * k * sr, fill=False, ec="black", lw=1.5, ls=ls))
    ax[0].add_patch(plt.Rectangle((cc - c0 - box_w / 2, cr - r0 - box_h / 2), box_w, box_h, fill=False, ec="red", lw=2))
    ax[0].set_title(f"Run 16 map around VIIRS pixel (row {r}, col {q})\nred = the VIIRS pixel (12.5 x 12.5 Landsat px, "
                    "used for box shares)\nblack = Gaussian footprint, 1 sigma (solid) and 2 sigma = cut (dashed)",
                    fontsize=10)
    im = ax[1].imshow(wgt / wgt.max(), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax[1].add_patch(plt.Rectangle((cc - c0 - box_w / 2, cr - r0 - box_h / 2), box_w, box_h, fill=False, ec="cyan", lw=2))
    fig.colorbar(im, ax=ax[1], fraction=0.04, label="relative weight of each Landsat pixel")
    ax[1].set_title(f"Gaussian weight per Landsat pixel\nsigma = 134 m = {sr:.1f} rows x {sc:.1f} columns of Landsat px "
                    "(a column is ~13 m wide here)", fontsize=10)
    for x in ax[:2]:
        x.set_xticks([]); x.set_yticks([])
    x = np.arange(6); w = 0.38
    ax[2].bar(x - w / 2, 100 * g["w_psf"][c], w, color=G.COLOURS, edgecolor="k", label="Gaussian share (run 17)")
    ax[2].bar(x + w / 2, 100 * g["w_box"][c], w, color=G.COLOURS, edgecolor="k", hatch="//", alpha=0.6,
              label="box share (runs 18-20)")
    for i in range(6):
        ax[2].text(i - w / 2, 100 * g["w_psf"][c][i] + 1, f"{100 * g['w_psf'][c][i]:.0f}", ha="center", fontsize=9)
        ax[2].text(i + w / 2, 100 * g["w_box"][c][i] + 1, f"{100 * g['w_box'][c][i]:.0f}", ha="center", fontsize=9)
    ax[2].set_xticks(x, SHORT); ax[2].set_ylabel("% of the VIIRS pixel"); ax[2].set_ylim(0, 105)
    ax[2].legend(fontsize=9)
    ax[2].set_title("Class shares of this pixel: Gaussian (run 17) vs box (runs 18-20)\n"
                    "run 17 keeps the six classes; runs 18-20 merge forest + shrub + bare", fontsize=10)
    fig.legend(handles=[Patch(color=cl, label=n) for cl, n in zip(G.COLOURS, G.NAMES)],
               loc="outside lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"Run 17 class shares, {date}: one VIIRS pixel weighs the Landsat 30 m pixels around its centre "
                 "(Oleson et al. 1995)", fontsize=13)
    f = OUT17 / f"aggregation_{date}_zoom.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


if __name__ == "__main__":
    OUT17.mkdir(parents=True, exist_ok=True); OUT18.mkdir(parents=True, exist_ok=True)
    run18 = [d for d in U.viirs_dates()
             if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    for d in run18:
        a = G.aggregate(d)
        print(overview18(a, d)); print(G.zoom(a, d, river_only=False, run_label="Run 18", out=OUT18))
    for d in U.viirs_dates():                     # run 17 unmixed every VIIRS date
        g = gauss17(d)
        print(overview17(g, d)); print(zoom17(g, d))
