"""
Figures of the label step of runs 19-20 for one date: how the run 16 Landsat 30 m map becomes the
four part % per VIIRS 375 m pixel.

  1. the ~156 Landsat pixels (12-13 x 12-13) that fall in each VIIRS pixel (after the run 17 shift)
  2. a VIIRS pixel is used only if >= 98 % of them are mapped
  3. class counts -> open water % (river), ice %, snow %, snow-free land % (forest + shrub + bare)

Outputs: output/19_viirs_river_states/images/aggregation_<date>_overview.png, _zoom.png

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/make_aggregation_figs.py [--date 2023-05-05]
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import rf_viirs_unmix as U
import viirs_river_states as R
import viirs_states as V

OUT = U.HERE / "output" / "19_viirs_river_states" / "images"
COLOURS = ["#0066ff", "#ff00ff", "#00ffff", "#1a7f37", "#b5d96b", "#c8a165"]
NAMES = ["river", "ice", "snow", "forest", "shrub_grass_wetland", "bare_developed"]
PARTS = ["open water", "ice", "snow", "snow-free land"]
PART_COLOURS = ["#0066ff", "#ff00ff", "#00ffff", "#4a8f3a"]
MIN_VALID = U.MIN_VALID


def aggregate(date):
    cls, gt_f, v, gt_v, ds_v, _, _ = U.load_date(date, {})
    shift = next(r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text()) if r["date"] == date)
    hv, wv = v.shape[1:]
    idx = V.cell_index(cls.shape, gt_f, gt_v, (hv, wv), tuple(shift))
    ok = idx >= 0
    n = hv * wv
    total = np.bincount(idx[ok], minlength=n)
    mapped = np.bincount(idx[ok], weights=(cls[ok] > 0), minlength=n)
    counts = np.stack([np.bincount(idx[ok], weights=(cls[ok] == k + 1), minlength=n) for k in range(6)], 1)
    smallest = np.floor(gt_v[5] / gt_f[5]) * np.floor(gt_v[1] / gt_f[1])
    share = mapped / np.maximum(total, smallest)
    parts = np.stack([counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3:].sum(1)], 1) / np.maximum(mapped, 1)[:, None]
    occ = R.crop_mask(gt_v, (hv, wv))
    river, buf = R.river_mask(occ)
    return dict(cls=cls, idx=idx, total=total, mapped=mapped, counts=counts, share=share, parts=parts,
                hv=hv, wv=wv, river=river, buf=buf, shift=shift)


def cell_edges(idx):
    """Boundary pixels between VIIRS cells, on the 30 m grid."""
    e = np.zeros(idx.shape, bool)
    e[:, 1:] |= idx[:, 1:] != idx[:, :-1]
    e[1:, :] |= idx[1:, :] != idx[:-1, :]
    return e


def on_30m(idx, cells):
    m = np.zeros(idx.shape, bool)
    ok = idx >= 0
    m[ok] = cells.ravel()[idx[ok]]
    return m


def overview(a, date):
    hv, wv = a["hv"], a["wv"]
    river, buf = a["river"], a["buf"]
    area19 = (river | buf).ravel()
    used = a["share"] >= MIN_VALID
    fig, ax = plt.subplots(1, 6, figsize=(24, 9), constrained_layout=True)
    ax[0].imshow(a["cls"], cmap=ListedColormap(["#ffffff"] + COLOURS), vmin=-0.5, vmax=6.5, interpolation="nearest")
    ax[0].contour(on_30m(a["idx"], river).astype(float), levels=[0.5], colors="yellow", linewidths=0.8)
    ax[0].contour(on_30m(a["idx"], river | buf).astype(float), levels=[0.5], colors="orange", linewidths=0.8,
                  linestyles="--")
    ax[0].set_title(f"Run 16 Landsat 30 m map, {date}\n(the classified labels, before aggregation)", fontsize=10)
    cat = np.zeros(hv * wv, np.uint8)
    cat[area19 & used] = 1; cat[area19 & ~used] = 2
    cat[river.ravel() & used] = 3; cat[river.ravel() & ~used] = 4
    ax[1].imshow(cat.reshape(hv, wv), cmap=ListedColormap(["#f2f2f2", "#8cc8ff", "#ff9b9b", "#1f4e9c", "#c0392b"]),
                 vmin=-0.5, vmax=4.5, interpolation="nearest")
    n19, n20 = int((area19 & used).sum()), int((river.ravel() & used).sum())
    ax[1].set_title(f"VIIRS 375 m pixels: >= 98 % mapped?\nrun 19 uses {n19}, run 20 uses {n20}", fontsize=10)
    ax[1].legend(handles=[Patch(color="#1f4e9c", label="river, used (runs 19 + 20)"),
                          Patch(color="#8cc8ff", label="buffer, used (run 19 only)"),
                          Patch(color="#c0392b", label="river, dropped (< 98 %)"),
                          Patch(color="#ff9b9b", label="buffer, dropped (< 98 %)"),
                          Patch(fc="#f2f2f2", ec="k", lw=0.3, label="outside the mask")], loc="lower left", fontsize=7.5)
    for k in range(4):
        img = np.where(area19 & used, 100 * a["parts"][:, k], np.nan).reshape(hv, wv)
        cmap = plt.get_cmap(["Blues", "RdPu", "GnBu", "YlGn"][k]).copy(); cmap.set_bad("#f2f2f2")
        im = ax[k + 2].imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
        ax[k + 2].contour(river.astype(float), levels=[0.5], colors="black", linewidths=0.5)
        m20 = np.nanmean(np.where(river.ravel() & used, 100 * a["parts"][:, k], np.nan))
        ax[k + 2].set_title(f"{PARTS[k]} % per VIIRS pixel\nmean: run 19 {np.nanmean(img):.1f} %, run 20 {m20:.1f} %",
                            fontsize=10)
        fig.colorbar(im, ax=ax[k + 2], orientation="horizontal", fraction=0.04, pad=0.02, label="% of the VIIRS pixel")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLOURS, NAMES)]
               + [Line2D([], [], color="yellow", lw=1.5, label="river mask (occ >= 0.05): run 20"),
                  Line2D([], [], color="orange", lw=1.5, ls="--", label="+ 1 px buffer: run 19"),
                  Line2D([], [], color="black", lw=1, label="river mask outline on the % maps")],
               loc="outside lower center", ncol=9, fontsize=9, frameon=False)
    fig.suptitle(f"Runs 19-20 labels, {date}: run 16 30 m map -> counts per VIIRS pixel -> 4 part %  "
                 f"(open water = river, snow-free land = forest + shrub + bare; Landsat moved by the run 17 shift "
                 f"{a['shift']} Landsat px)", fontsize=12)
    f = OUT / f"aggregation_{date}_overview.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def zoom(a, date, size=5, river_only=True, run_label="Runs 19-20", out=None):
    """size x size VIIRS pixels around a mixed river pixel, with the 30 m pixels and the counts."""
    hv, wv = a["hv"], a["wv"]
    used = a["share"] >= MIN_VALID
    p = a["parts"]
    rows = np.repeat(np.arange(hv), wv); cols = np.tile(np.arange(wv), hv)
    inner = (rows >= size) & (rows < hv - size) & (cols >= size) & (cols < wv - size)
    base = a["river"].ravel() if river_only else np.ones(hv * wv, bool)
    cand = np.flatnonzero(base & used & inner)
    c0 = cand[np.argmax(np.sort(p[cand], 1)[:, -2])]          # most mixed river pixel: largest second part
    r0, q0 = divmod(int(c0), wv)
    r0 = int(np.clip(r0 - size // 2, 0, hv - size)); q0 = int(np.clip(q0 - size // 2, 0, wv - size))
    cells = [(r, q) for r in range(r0, r0 + size) for q in range(q0, q0 + size)]
    cid = np.array([r * wv + q for r, q in cells])
    rr, cc = np.nonzero(np.isin(a["idx"], cid))
    sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    sub, subidx = a["cls"][sl], a["idx"][sl]

    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.15])
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(np.where(np.isin(subidx, cid), sub, 0), cmap=ListedColormap(["#ffffff"] + COLOURS), vmin=-0.5, vmax=6.5,
               interpolation="nearest")
    e = cell_edges(subidx)
    ax0.imshow(np.ma.masked_where(~e, e), cmap=ListedColormap(["black"]), interpolation="nearest", alpha=0.9)
    for (r, q), c in zip(cells, cid):
        yy, xx = np.nonzero(subidx == c)
        if not len(yy):
            continue
        tag = "R" if a["river"][r, q] else ("B" if a["buf"][r, q] else "-")
        txt = (f"{tag}  {int(a['total'][c])} px\nW {100 * p[c, 0]:.0f} I {100 * p[c, 1]:.0f}\n"
               f"S {100 * p[c, 2]:.0f} L {100 * p[c, 3]:.0f}")
        ok = a["share"][c] >= MIN_VALID
        ax0.text(xx.mean(), yy.mean(), txt, ha="center", va="center", fontsize=7.2, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white" if ok else "#ffd6d6", ec="0.4", alpha=1.0))
    ax0.set_title(f"{size} x {size} VIIRS pixels (black grid) over the 30 m run 16 map\n"
                  "R = river mask, B = buffer, - = outside; W/I/S/L = open water / ice / snow / snow-free land %",
                  fontsize=10)
    ax0.set_xticks([]); ax0.set_yticks([])

    ex = [c for c in cid if base[c] and a["share"][c] >= MIN_VALID]
    ex = sorted(ex, key=lambda c: -np.sort(p[c])[-2])[:3] or list(cid[:3])
    slots = [gs[0, 1], gs[1, 1], gs[1, 2]]
    for k, c in enumerate(ex):
        r, q = divmod(int(c), wv)
        axp = fig.add_subplot(slots[k])
        yy, xx = np.nonzero(a["idx"] == c)
        tile = np.where(a["idx"][yy.min():yy.max() + 1, xx.min():xx.max() + 1] == c,
                        a["cls"][yy.min():yy.max() + 1, xx.min():xx.max() + 1], 0)
        axp.imshow(tile, cmap=ListedColormap(["#ffffff"] + COLOURS), vmin=-0.5, vmax=6.5, interpolation="nearest")
        axp.set_xticks(np.arange(-0.5, tile.shape[1], 1), minor=True); axp.set_yticks(np.arange(-0.5, tile.shape[0], 1), minor=True)
        axp.grid(which="minor", color="white", lw=0.4)
        axp.add_patch(Rectangle((-0.5, -0.5), tile.shape[1], tile.shape[0], fill=False, ec="black", lw=2))
        cnt = a["counts"][c].astype(int)
        lines = [f"VIIRS pixel (row {r}, col {q}): {int(a['total'][c])} Landsat px, mapped {100 * a['share'][c]:.1f} %",
                 ", ".join(f"{NAMES[i].split('_')[0]} {cnt[i]}" for i in range(6)),
                 f"open water {cnt[0]}/{int(a['mapped'][c])} = {100 * p[c, 0]:.1f} %   ice {cnt[1]} = {100 * p[c, 1]:.1f} %",
                 f"snow {cnt[2]} = {100 * p[c, 2]:.1f} %   snow-free land {cnt[3]}+{cnt[4]}+{cnt[5]} = {100 * p[c, 3]:.1f} %"]
        axp.set_title("\n".join(lines), fontsize=8.5, family="monospace")
        axp.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    axb = fig.add_subplot(gs[0, 2])
    for k, c in enumerate(ex):
        left = 0
        for i in range(4):
            axb.barh(k, 100 * p[c, i], left=left, color=PART_COLOURS[i], edgecolor="k", lw=0.3)
            if p[c, i] > 0.06:
                axb.text(left + 50 * p[c, i], k, f"{100 * p[c, i]:.0f}", ha="center", va="center", fontsize=9)
            left += 100 * p[c, i]
    axb.set_yticks(range(len(ex)), [f"pixel {divmod(int(c), wv)}" for c in ex])
    axb.set_xlim(0, 100); axb.set_xlabel("% of the VIIRS pixel"); axb.invert_yaxis()
    axb.set_title(f"The four parts of the example pixels ({run_label} label)", fontsize=10)
    axb.legend(handles=[Patch(color=c, label=n) for c, n in zip(PART_COLOURS, PARTS)], ncol=4, fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLOURS, NAMES)]
               + [Patch(fc="#ffd6d6", ec="0.4", label="pink box: < 98 % mapped, not used")],
               loc="outside lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle(f"{run_label} labels, {date}: the ~156 Landsat 30 m pixels inside each VIIRS 375 m pixel are counted",
                 fontsize=13)
    f = (out or OUT) / f"aggregation_{date}_zoom.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2023-05-05")
    d = ap.parse_args().date
    OUT.mkdir(parents=True, exist_ok=True)
    a = aggregate(d)
    print(overview(a, d)); print(zoom(a, d))
