"""
Run 19: the four river / land states predicted directly (one 4-class model) on VIIRS 375 m cells of
the river only: water mask + a small buffer.

Water mask: ~/Desktop/nenana_crops/alaska_occ_375m.tif (JRC occurrence 0-1 on the VIIRS 375 m grid,
same pixel size and aligned with the Patches VIIRS files). River cells = occurrence >= 0.05 (the
run 16 5 % water-zone rule), buffer = 1 VIIRS cell (375 m, 3 x 3 dilation).

States (scene_mask.py numbering), labels from the run 16 Landsat 30 m map of the day exactly as in
run 18 (box % per cell with the run 17 shift, rule50 and rule90):
  1 ice_free_river_snow_free_land
  2 ice_free_river_snow_covered_land
  3 ice_covered_river_snow_free_land
  4 ice_covered_river_snow_covered_land

Inputs: run 18 VIIRS inputs (I1-I5, NDVI_I, NDSI_I, I4-I5, solar / view zenith, JRC mean and zone
share) + the water-mask occurrence; RF also gets the 3 x 3 mean / std of I1, I2, I3, I5.

Models, one per rule, all four states at once:
  RF      RandomForestClassifier with the run 16 settings: depth 16, min_samples_leaf 20, sqrt,
          300 trees, class_weight balanced_subsample (no top-25 selection: only 21 inputs)
  2D CNN  4 conv 3x3 layers (32-64-64-64, BN, ReLU), no down-sampling, same training as UNet++
  UNet++  smp resnet18, 4-class cross-entropy on mask cells only, whole day images padded to
          160 x 96, flips, AdamW 1e-3, 1500 steps, flip TTA
Training and prediction use mask cells only. Leave-one-date-out over the 6 run 16 dates with VIIRS.

Outputs: output/19_viirs_river_states/
  mask/water_mask_occ.tif, mask/river_mask.tif (1 river, 2 buffer)
  truth|rf|unetpp/<rule>/class_<date>.tif (+ .qml), <model>/metrics.txt, metrics_all.json, summary.txt
  images/water_mask.jpg, images/run19_<date>.jpg, images/run19_summary.jpg

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/viirs_river_states.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from osgeo import gdal
from scipy.ndimage import binary_dilation
from sklearn.ensemble import RandomForestClassifier

import rf_viirs_unmix as U
import viirs_states as V

gdal.UseExceptions()

MASK_FILE = Path.home() / "Desktop" / "nenana_crops" / "alaska_occ_375m.tif"
RIVER_CUT = 0.05
BUFFER_CELLS = int(os.environ.get("RIVER_BUFFER", "1"))     # 1 = run 19 (river + 1 px buffer), 0 = run 20 (river only)
RUN = "19" if BUFFER_CELLS else "20"
AREA = "river + buffer" if BUFFER_CELLS else "river only"
OUT = U.HERE / "output" / ("19_viirs_river_states" if BUFFER_CELLS else "20_viirs_river_only")
RULES = list(V.RULES)
SEED = 0
GREY = "#9a9a9a"
SOURCES = (("truth", "Landsat truth"), ("rf", "RF"), ("unetpp", "UNet++"), ("cnn2d", "2D CNN"))
MODELS = ("rf", "unetpp", "cnn2d")
LONG = {"rf": "Random Forest", "unetpp": "UNet++", "cnn2d": "2D CNN"}


# ----------------------------------------------------------------------------- mask

def crop_mask(gt_v, shape_v):
    m = gdal.Open(str(MASK_FILE))
    gt = m.GetGeoTransform()
    assert abs(gt[1] - gt_v[1]) < 1e-9 and abs(gt[5] - gt_v[5]) < 1e-9, "mask grid differs from VIIRS grid"
    c0 = int(round((gt_v[0] - gt[0]) / gt[1]))
    r0 = int(round((gt_v[3] - gt[3]) / gt[5]))
    a = m.ReadAsArray(c0, r0, shape_v[1], shape_v[0]).astype(np.float32)
    return np.where(a == m.GetRasterBand(1).GetNoDataValue(), np.nan, a)


def river_mask(occ):
    river = np.nan_to_num(occ) >= RIVER_CUT
    if BUFFER_CELLS == 0:                 # scipy treats iterations=0 as 'until no change', so skip it
        return river, np.zeros_like(river)
    buf = binary_dilation(river, np.ones((3, 3), bool), iterations=BUFFER_CELLS) & ~river
    return river, buf


def write_raster(path, arr, gt_v, ds_v, dtype, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    o = gdal.GetDriverByName("GTiff").Create(str(path), arr.shape[1], arr.shape[0], 1, dtype,
                                             options=["COMPRESS=DEFLATE"])
    o.SetGeoTransform(gt_v); o.SetProjection(ds_v.GetProjection())
    b = o.GetRasterBand(1); b.WriteArray(arr); b.SetNoDataValue(nodata); o = None


# ----------------------------------------------------------------------------- models

def rf_rows(day):
    x, names = V.rf_table(day)
    return np.concatenate([x, day["occ"].reshape(-1, 1)], 1), names + ["water_mask_occ"]


def fit_rf(days, train, rule):
    xs, ys = [], []
    for d in train:
        g = days[d]["use"]
        xs.append(rf_rows(days[d])[0][g]); ys.append(days[d]["state"][rule][g])
    rf = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=20, max_features="sqrt",
                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    return rf.fit(np.concatenate(xs), np.concatenate(ys))


def predict_rf(rf, day):
    g = day["use"]
    s = np.zeros(len(g), np.uint8)
    s[g] = rf.predict(rf_rows(day)[0][g])
    return s


def unet_input(day):
    return np.concatenate([day["img"], day["occ"][None]])


def fit_unet(days, train, rule, steps=1500, arch="unetpp"):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = {d: unet_input(days[d]) for d in train}
    stack = np.concatenate([imgs[d].reshape(len(imgs[d]), -1)[:, days[d]["use"]] for d in train], 1)
    mean, std = stack.mean(1), stack.std(1) + 1e-6
    X, T, M = [], [], []
    for d in train:
        hv, wv = days[d]["shape"]
        x = np.zeros((len(mean), *V.PAD), np.float32)
        x[:, :hv, :wv] = np.nan_to_num((imgs[d] - mean[:, None, None]) / std[:, None, None])
        t = np.zeros(V.PAD, np.int64)
        t[:hv, :wv] = np.maximum(days[d]["state"][rule].reshape(hv, wv).astype(np.int64) - 1, 0)
        m = np.zeros(V.PAD, np.float32); m[:hv, :wv] = days[d]["use"].reshape(hv, wv)
        X.append(x); T.append(t); M.append(m)
    X, T, M = (torch.from_numpy(np.stack(a)).to(dev) for a in (X, T, M))
    net = V.make_net(arch, X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    net.train()
    for _ in range(steps):
        x, t, m = X, T, M
        if np.random.rand() < 0.5:
            x, t, m = x.flip(-1), t.flip(-1), m.flip(-1)
        if np.random.rand() < 0.5:
            x, t, m = x.flip(-2), t.flip(-2), m.flip(-2)
        loss = F.cross_entropy(net(x), t, reduction="none")
        loss = (loss * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net, mean, std, dev


def predict_unet(model, day):
    import torch
    net, mean, std, dev = model
    hv, wv = day["shape"]
    x = np.zeros((len(mean), *V.PAD), np.float32)
    x[:, :hv, :wv] = np.nan_to_num((unet_input(day) - mean[:, None, None]) / std[:, None, None])
    with torch.no_grad():
        xb = torch.from_numpy(x)[None].to(dev)
        p = torch.softmax(net(xb), 1)
        p = p + torch.softmax(net(xb.flip(-1)), 1).flip(-1)
        p = p + torch.softmax(net(xb.flip(-2)), 1).flip(-2)
        s = (p.argmax(1)[0, :hv, :wv].cpu().numpy() + 1).reshape(-1).astype(np.uint8)
    return np.where(day["use"], s, 0).astype(np.uint8)


# ----------------------------------------------------------------------------- images

def viirs_rgb(date):
    v = gdal.Open(str(U.PATCHES / f"nenana_{date.replace('-', '')}" / f"{date}_viirs_375m.tif")).ReadAsArray()
    rgb = np.stack([v[2], v[1], v[0]], -1).astype(np.float32)
    ok = np.isfinite(rgb).all(-1)
    lo, hi = np.nanpercentile(rgb[ok], [2, 98], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb[~ok] = 0.6
    return rgb


def shares(s, g):
    n = max(int(g.sum()), 1)
    return [100 * ((s == i) & g).sum() / n for i in range(1, 5)]


def state_legend():
    return [Patch(fc=c, ec="k", lw=0.3, label=f"{i + 1} {s}") for i, (s, c) in enumerate(zip(V.STATES, V.COLOURS))]


def mask_image(date, occ, river, buf):
    fig, ax = plt.subplots(1, 3, figsize=(13, 7.5), constrained_layout=True)
    im = ax[0].imshow(np.ma.masked_invalid(occ), cmap="Blues", vmin=0, vmax=0.5, interpolation="nearest")
    fig.colorbar(im, ax=ax[0], fraction=0.05, label="water occurrence (0-1)")
    ax[0].set_title("alaska_occ_375m.tif cropped to the box")
    lab = np.zeros(occ.shape, np.uint8); lab[buf] = 2; lab[river] = 1
    ax[1].imshow(lab, cmap=ListedColormap(["#f2f2f2", "#1f4e9c", "#8cc8ff"]), vmin=-0.5, vmax=2.5,
                 interpolation="nearest")
    ax[1].set_title(f"River (occ >= {RIVER_CUT}): {int(river.sum())} px\n"
                    f"buffer {BUFFER_CELLS} px: {int(buf.sum())} px")
    ax[1].legend(handles=[Patch(color="#1f4e9c", label="river"), Patch(color="#8cc8ff", label="buffer"),
                          Patch(fc="#f2f2f2", ec="k", lw=0.3, label="not predicted")], loc="lower left", fontsize=8)
    ov = viirs_rgb(date)
    ov[river] = 0.4 * ov[river] + 0.6 * np.array([0.12, 0.3, 0.6])
    ov[buf] = 0.6 * ov[buf] + 0.4 * np.array([0.55, 0.78, 1.0])
    ax[2].imshow(ov, interpolation="nearest")
    ax[2].set_title(f"Mask over VIIRS {date} (I3/I2/I1)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Run {RUN} water mask on the VIIRS 375 m grid (80 x 132 px)", fontsize=13)
    fig.savefig(OUT / "images" / "water_mask.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)


def date_image(day, preds, scores):
    hv, wv = day["shape"]
    g = day["use"]
    fig, ax = plt.subplots(2, 5, figsize=(16, 12), constrained_layout=True)
    ax[0, 0].imshow(viirs_rgb(day["date"]), interpolation="nearest")
    ax[0, 0].set_title("VIIRS 375 m (I3/I2/I1)", fontsize=10)
    ax[1, 0].imshow(g.reshape(hv, wv), cmap=ListedColormap(["#f2f2f2", "#1f4e9c"]), vmin=0, vmax=1,
                    interpolation="nearest")
    ax[1, 0].set_title(f"Predicted area: {AREA}\n{int(g.sum())} VIIRS px", fontsize=10)
    for r, rule in enumerate(RULES):
        for c, (src, label) in enumerate(SOURCES):
            s = day["state"][rule] if src == "truth" else preds[src][rule]
            ax[r, c + 1].imshow(np.where(g, s, 0).reshape(hv, wv), cmap=ListedColormap([GREY] + V.COLOURS),
                                vmin=-0.5, vmax=4.5, interpolation="nearest")
            t = f"{label}: {rule}\n% 1/2/3/4 = " + " ".join(f"{v:.0f}" for v in shares(s, g))
            if src != "truth":
                t += f"   acc {scores[src][rule]['accuracy']:.2f}"
            ax[r, c + 1].set_title(t, fontsize=9)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=state_legend() + [Patch(color=GREY, label="not predicted (outside mask)")],
               loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Run {RUN} states on {AREA}, {day['date']} (left-out test day)", fontsize=13)
    fig.savefig(OUT / "images" / f"run{RUN}_{day['date']}.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)


def summary_image(days, preds_all, results):
    dates = list(days)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    w, x = 0.21, np.arange(len(dates))
    for c, rule in enumerate(RULES):
        for r_i, (src, _) in enumerate(SOURCES):
            sh = np.array([shares(days[d]["state"][rule] if src == "truth" else preds_all[d][src][rule],
                                  days[d]["use"]) for d in dates])
            bottom = np.zeros(len(dates))
            for i in range(4):
                ax[c].bar(x + (r_i - 1.5) * w, sh[:, i], w * 0.95, bottom=bottom, color=V.COLOURS[i],
                          edgecolor="k", lw=0.3)
                bottom += sh[:, i]
        ax[c].set_xticks(x, [f"{d}\nT  RF  U++  CNN" for d in dates], fontsize=8)
        ax[c].set_ylabel(f"% of {AREA} VIIRS pixels")
        ax[c].set_title(f"{rule} acc / F1: " + ",  ".join(f"{dict(SOURCES)[m]} {results[m][rule]['accuracy']:.2f} / {results[m][rule]['macro_f1']:.2f}" for m in MODELS))
    fig.legend(handles=state_legend(), loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Run {RUN}: states on {AREA}, leave-one-date-out (bars: truth / RF / UNet++ / 2D CNN)", fontsize=13)
    fig.savefig(OUT / "images" / f"run{RUN}_summary.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def text_block(title, res):
    L = [f"=== {title}: leave-one-date-out, {res[RULES[0]]['cells']} {AREA} cells"]
    for rule in RULES:
        s = res[rule]
        L += [f"--- {rule}: accuracy {s['accuracy']:.3f}   macro F1 {s['macro_f1']:.3f}",
              f"{'state':<38}{'F1':>6}{'prec':>6}{'rec':>6}{'true':>7}{'pred':>7}"]
        for k, v in s["per_state"].items():
            L.append(f"{k:<38}{v['f1']:>6.2f}{v['precision']:>6.2f}{v['recall']:>6.2f}{v['true_cells']:>7}{v['pred_cells']:>7}")
        L.append("confusion (rows true 1-4, columns predicted 1-4): " + str(s["confusion"]))
    L.append(f"{'date':<12}{'cells':>6}" + "".join(f"{r + ' acc':>12}" for r in RULES))
    for d, v in res["per_date"].items():
        L.append(f"{d:<12}{v['cells']:>6}" + "".join(f"{v[r]['accuracy']:>12.3f}" for r in RULES))
    return "\n".join(L)


if __name__ == "__main__":
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    shifts = {r["date"]: r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text())}
    cache, days = {}, {}
    run16 = [d for d in U.viirs_dates()
             if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    occ = river = buf = None
    for date in run16:
        d = days[date] = V.load_day(date, shifts.get(date, [0, 0]), cache)
        if occ is None:
            occ = crop_mask(d["gt_v"], d["shape"])
            river, buf = river_mask(occ)
            write_raster(OUT / "mask" / "water_mask_occ.tif", np.nan_to_num(occ, nan=-9999), d["gt_v"], d["ds_v"],
                         gdal.GDT_Float32, -9999)
            lab = np.zeros(occ.shape, np.uint8); lab[buf] = 2; lab[river] = 1
            write_raster(OUT / "mask" / "river_mask.tif", lab, d["gt_v"], d["ds_v"], gdal.GDT_Byte, 0)
            mask_image(date, occ, river, buf)
            print(f"mask: river {int(river.sum())} px, buffer {int(buf.sum())} px", flush=True)
        d["occ"] = np.nan_to_num(occ).astype(np.float32)
        d["state"] = {rule: V.states(d["pct"], d["good"], rule) for rule in RULES}
        d["use"] = d["good"] & (river | buf).reshape(-1)
        for rule in RULES:
            V.write_class(OUT / "truth" / rule, d, np.where(d["use"], d["state"][rule], 0).astype(np.uint8))
        print(date, f"cells {int(d['use'].sum())}",
              {rule: [int((d["state"][rule][d["use"]] == s).sum()) for s in range(1, 5)] for rule in RULES}, flush=True)

    dates = list(days)
    preds_all = {d: {} for d in dates}
    results, blocks = {}, []
    for model in MODELS:
        per_date, st_all = {}, {r: ([], []) for r in RULES}
        for test in dates:
            train = [d for d in dates if d != test]
            day = days[test]; g = day["use"]
            preds_all[test][model] = {}
            per_date[test] = {"cells": int(g.sum())}
            for rule in RULES:
                s = (predict_rf(fit_rf(days, train, rule), day) if model == "rf"
                     else predict_unet(fit_unet(days, train, rule, arch=model), day))
                preds_all[test][model][rule] = s
                V.write_class(OUT / model / rule, day, s)
                per_date[test][rule] = V.state_scores(day["state"][rule][g], s[g])
                st_all[rule][0].append(day["state"][rule][g]); st_all[rule][1].append(s[g])
            print(f"{model:<7}{test}  " + "  ".join(f"{r} acc {per_date[test][r]['accuracy']:.3f}" for r in RULES),
                  flush=True)
        res = {"per_date": per_date}
        for rule in RULES:
            res[rule] = V.state_scores(np.concatenate(st_all[rule][0]), np.concatenate(st_all[rule][1]))
        results[model] = res
        blocks.append(text_block(LONG[model], res))
        (OUT / model / "metrics.txt").write_text(blocks[-1] + "\n")
        print(blocks[-1], flush=True)
    (OUT / "metrics_all.json").write_text(json.dumps(results, indent=1))
    head = [f"Run {RUN}: four states predicted at once on {AREA} VIIRS cells only.",
            f"Mask: {MASK_FILE.name}, river = occurrence >= {RIVER_CUT} ({int(river.sum())} px), "
            f"buffer {BUFFER_CELLS} px ({int(buf.sum())} px). Labels: run 16 Landsat map (as run 18).",
            "Leave-one-date-out over " + ", ".join(dates) + ".", ""]
    (OUT / "summary.txt").write_text("\n".join(head) + "\n" + "\n\n".join(blocks) + "\n")
    for d in dates:
        date_image(days[d], preds_all[d], {m: results[m]["per_date"][d] for m in MODELS})
    summary_image(days, preds_all, results)
    print("wrote", OUT, flush=True)
