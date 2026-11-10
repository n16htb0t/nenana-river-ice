"""Run 18 figures as JPG: per-date panels (truth / RF / UNet++ part % and state maps) and a summary chart.

Reads output/18_viirs_states/{truth,rf,unetpp}/ and writes output/18_viirs_states/images/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from osgeo import gdal

gdal.UseExceptions()
HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "18_viirs_states"
IMG = OUT / "images"
RUN16 = HERE / "output" / "16_rf_water_zone" / "2_with_daksh" / "six_class" / "random_forest"
PATCHES = Path.home() / "Desktop" / "Patches"

SOURCES = [("truth", "Landsat truth"), ("rf", "RF"), ("unetpp", "UNet++")]
PARTS = ["open_river", "ice", "snow", "snow_free_land"]
PART_CMAPS = ["Blues", "RdPu", "GnBu", "YlGn"]
STATES = ["1 ice-free river, snow-free land", "2 ice-free river, snow land",
          "3 ice river, snow-free land", "4 ice river, snow land"]
STATE_COLOURS = ["#2b83ba", "#00b4b4", "#a05a2c", "#e6f5ff"]
L16_COLOURS = ["#0066ff", "#ff00ff", "#00ffff", "#1a7f37", "#b5d96b", "#c8a165"]
L16_NAMES = ["river", "ice", "snow", "forest", "shrub/grass/wetland", "bare/developed"]
NODATA_GREY = "#9a9a9a"


def read(path):
    return gdal.Open(str(path)).ReadAsArray().astype(np.float32)


def landsat_map(date):
    for p in (RUN16 / f"pred_{date}.tif", RUN16 / "new_dates" / date / "rf_pred.tif",
              HERE / "output" / "17_viirs_unmix" / "landsat_maps" / f"pred_{date}.tif"):
        if p.exists():
            return read(p)
    return None


def viirs_rgb(date):
    v = read(PATCHES / f"nenana_{date.replace('-', '')}" / f"{date}_viirs_375m.tif")
    rgb = np.stack([v[2], v[1], v[0]], -1)                 # I3, I2, I1 false colour
    ok = np.isfinite(rgb).all(-1)
    lo, hi = np.nanpercentile(rgb[ok], [2, 98], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb[~ok] = 0.6
    return rgb


def state_cmap():
    return ListedColormap([NODATA_GREY] + STATE_COLOURS)


def shares(s):
    n = max(int((s > 0).sum()), 1)
    return [100 * (s == i).sum() / n for i in range(1, 5)]


def date_panel(date):
    fig, ax = plt.subplots(3, 7, figsize=(19, 12.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax[0, 0].imshow(viirs_rgb(date), interpolation="nearest")
    ax[0, 0].set_title("VIIRS 375 m (I3/I2/I1)", fontsize=10)
    lm = landsat_map(date)
    if lm is not None:
        ax[1, 0].imshow(lm, cmap=ListedColormap([NODATA_GREY] + L16_COLOURS), vmin=-0.5, vmax=6.5,
                        interpolation="nearest")
    ax[1, 0].set_title("Run 16 Landsat 30 m", fontsize=10)
    ax[2, 0].axis("off")
    ax[2, 0].legend(handles=[Patch(color=c, label=n) for c, n in zip(L16_COLOURS, L16_NAMES)],
                    loc="upper left", fontsize=8, title="Run 16 classes", frameon=False)

    ims = []
    for r, (src, label) in enumerate(SOURCES):
        pct = read(OUT / src / f"pct_{date}.tif")
        for k, part in enumerate(PARTS):
            cmap = plt.get_cmap(PART_CMAPS[k]).copy()
            cmap.set_bad(NODATA_GREY)
            vmax = 20 if part in ("open_river", "ice") else 100
            im = ax[r, k + 1].imshow(np.ma.masked_invalid(pct[k]), cmap=cmap, vmin=0, vmax=vmax,
                                     interpolation="nearest")
            ax[r, k + 1].set_title(f"{label}: {part} %  (mean {np.nanmean(pct[k]):.1f})", fontsize=9)
            if r == 2:
                ims.append((im, ax[:, k + 1]))
        for j, rule in enumerate(("rule50", "rule90")):
            s = read(OUT / src / rule / f"class_{date}.tif")
            ax[r, 5 + j].imshow(s, cmap=state_cmap(), vmin=-0.5, vmax=4.5, interpolation="nearest")
            ax[r, 5 + j].set_title(f"{label}: {rule} states\n% 1/2/3/4 = "
                                   + " ".join(f"{v:.0f}" for v in shares(s)), fontsize=9)
    for im, col in ims:
        fig.colorbar(im, ax=col, orientation="horizontal", fraction=0.025, pad=0.01, label="% of VIIRS pixel")
    fig.legend(handles=[Patch(fc=c, label=n, ec="k", lw=0.3) for c, n in zip(STATE_COLOURS, STATES)]
               + [Patch(color=NODATA_GREY, label="no data")],
               loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Run 18 VIIRS 375 m, {date} (left-out test day for RF and UNet++)", fontsize=14)
    p = IMG / f"run18_{date}.jpg"
    fig.savefig(p, dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)
    return p


def summary_chart(dates):
    fig, ax = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    w = 0.27
    x = np.arange(len(dates))
    for c, rule in enumerate(("rule50", "rule90")):
        for r_i, (src, _) in enumerate(SOURCES):
            sh = np.array([shares(read(OUT / src / rule / f"class_{d}.tif")) for d in dates])
            bottom = np.zeros(len(dates))
            for i in range(4):
                ax[0, c].bar(x + (r_i - 1) * w, sh[:, i], w * 0.95, bottom=bottom, color=STATE_COLOURS[i],
                             edgecolor="k", lw=0.3)
                bottom += sh[:, i]
        ax[0, c].set_xticks(x, [f"{d}\nT   RF   U++" for d in dates], fontsize=8)
        ax[0, c].set_ylabel("% of VIIRS pixels")
        ax[0, c].set_title(f"States per date, {rule} (bars: truth / RF / UNet++)")
    fig.legend(handles=[Patch(fc=cl, label=n, ec="k", lw=0.3) for cl, n in zip(STATE_COLOURS, STATES)],
               loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    for k, part in enumerate(PARTS):
        a = ax[1, k // 2]
        for r_i, (src, label) in enumerate(SOURCES):
            m = [float(np.nanmean(read(OUT / src / f"pct_{d}.tif")[k])) for d in dates]
            a.plot(x, m, marker="os^"[r_i], ls=["-", "--", ":"][r_i],
                   color=plt.get_cmap(PART_CMAPS[k])(0.5 + 0.2 * r_i), label=f"{part} {label}")
        a.set_xticks(x, dates, fontsize=8)
        a.set_ylabel("mean % over box")
        a.legend(fontsize=8, ncol=2)
    ax[1, 0].set_title("Water parts: open river, ice (mean % per date)")
    ax[1, 1].set_title("Land parts: snow, snow-free land (mean % per date)")
    fig.suptitle("Run 18 summary, leave-one-date-out", fontsize=13)
    p = IMG / "run18_summary.jpg"
    fig.savefig(p, dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)
    return p


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    dates = sorted(f.stem[4:] for f in (OUT / "truth").glob("pct_*.tif"))
    for d in dates:
        print(date_panel(d), flush=True)
    print(summary_chart(dates), flush=True)


if __name__ == "__main__":
    main()
