"""
Run 21: run 16 repeated on VIIRS upsampled to 30 m.

Same as run 16 (rf_water_zone.py) in everything except the image:
  classes      river, ice, snow, forest, shrub_grass_wetland, bare_developed
  water mask   30 m water zone = JRC occurrence >= 5 % (run 16 water_zone.tif): only river or ice;
               land: only snow / forest / shrub / bare
  labels       the hand labels, moved onto the rule ("rule labels"), exactly as run 16
  model        rf_water_zone.ZoneModel: one forest per zone, top 25 inputs ranked on the training fold,
               depth 16, min_samples_leaf 20, sqrt, 300 trees, balanced_subsample
  evaluation   5-fold spatial CV (data.fold_masks row blocks), scores against rule and original labels

Image: VIIRS I1..I5 bilinear-upsampled to the Landsat 30 m grid and moved by the run 17 shift, instead of
Landsat. Input pool = run 16's pool built from the VIIRS bands (red = I1, nir = I2, swir1 = I3, I4,
thermal = I5): bands, all nd_ / diff_ pairs, SAVI, MSAVI, nir + swir1, I4 - I5, NDVI / NDSI / nir / red
textures (3 x 3 and 7 x 7), terrain illumination with the VIIRS sun angle, NLCD. VIIRS has no blue or
green I-band, so the run 16 inputs that need them (EVI, BSI, NDWI, AWEI, tasselled cap) do not exist here.

Dates: run 16 dates that have VIIRS. Labelled: 2023-07-24, 2023-10-04, 2024-04-21, 2024-06-22 (with_daksh);
2024-03-12, 2024-04-20, 2024-04-29 have no VIIRS and are left out. Maps for all 10 VIIRS dates.

Outputs: output/21_viirs_run16/<dataset folder>/random_forest/
  metrics.txt, oof.npz, pred_<date>.tif (+ .qml), images/pred_<date>.jpg, images/conf.png
output/21_viirs_run16/metrics_all.json

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/viirs_run16.py [--datasets with_daksh exclude_daksh]
"""

import argparse
import itertools
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from osgeo import gdal

import data as D
import rf_all_bands as RA
import rf_angles as A
import rf_six_classes as S6
import rf_viirs_unmix as U
import rf_water_zone as W
import viirs_30m as M30

gdal.UseExceptions()
OUT = U.HERE / "output" / "21_viirs_run16"
NAMES, COLOURS = M30.NAMES, M30.COLOURS
ZONE = W.ZONE                                    # the 30 m water mask (JRC >= 5 %)
EPS = 1e-6
DATASETS = {"with_daksh": "2_with_daksh", "exclude_daksh": "1_exclude_daksh"}
RUN16_OOF = {ds: U.HERE / "output" / "16_rf_water_zone" / f / "six_class" / "random_forest" / "oof.npz"
             for ds, f in DATASETS.items()}


def viirs_sun(date):
    """Mean VIIRS solar zenith and azimuth over the image (bands 6 and 7 of the VIIRS file)."""
    ds = gdal.Open(str(U.PATCHES / f"nenana_{date.replace('-', '')}" / f"{date}_viirs_375m.tif"))
    return float(np.nanmean(ds.GetRasterBand(6).ReadAsArray())), float(np.nanmean(ds.GetRasterBand(7).ReadAsArray()))


def viirs_features(day30):
    """(F, H, W) run-16-style inputs from the upsampled VIIRS of one date."""
    x, names = day30["x"], day30["names"]
    i = {n: k for k, n in enumerate(names)}
    b = {"red": x[i["I1_up"]], "nir": x[i["I2_up"]], "swir1": x[i["I3_up"]]}
    mwir, thermal = x[i["I4_up"]], x[i["I5_up"]]
    f, fn = [], []

    def add(n, a):
        f.append(a.astype(np.float32)); fn.append(n)

    for n in ("red", "nir", "swir1"):
        add(n, b[n])
    add("I4", mwir); add("thermal", thermal); add("I4_minus_I5", mwir - thermal)
    for a, c in itertools.combinations(("red", "nir", "swir1"), 2):
        add(f"nd_{a}_{c}", (b[a] - b[c]) / (b[a] + b[c] + EPS))
        add(f"diff_{a}_{c}", b[a] - b[c])
    R, N, S1 = b["red"], b["nir"], b["swir1"]
    add("SAVI", 1.5 * (N - R) / (N + R + 0.5))
    add("MSAVI", (2 * N + 1 - np.sqrt(np.maximum((2 * N + 1) ** 2 - 8 * (N - R), 0))) / 2)
    add("nir_plus_swir1", N + S1)
    for n, a in {"NDVI": (N - R) / (N + R + EPS), "NDSI": (R - S1) / (R + S1 + EPS), "nir": N, "red": R}.items():
        for size in (3, 7):
            m, sd = RA.texture(a, size)
            add(f"{n}_mean{size}", m); add(f"{n}_std{size}", sd)
    sza, saa = viirs_sun(day30["date"])          # terrain illumination with the VIIRS sun; date constants left out
    key = day30["date"] + "_viirs"
    A.SUN[key] = {"sun_elevation": 90.0 - sza, "sun_azimuth": saa}
    for n, a in A.angle_inputs(None, key):
        if n not in ("sun_elevation_deg", "sun_azimuth_deg"):
            add(n, a)
    for n, a in S6.nlcd_inputs():
        add(n, a)
    return np.stack(f), fn


def labels(dataset):
    """{date: (H, W) six-class hand labels} of the label set, VIIRS dates only."""
    D.configure("four", dataset)
    out, have = {}, set(U.viirs_dates())
    for group, date, folder, lab in D.scene_list():
        if date not in have:
            continue
        boxes = list(folder.glob("landsat_*_box.tif"))
        y = D.load_labels(lab, gdal.Open(str(boxes[0] if boxes else folder / f"{date}_sr.tif")))
        if (y != D.IGNORE).any():
            s = {"y": y}; S6.prepare([s])
            out[date] = s["y"]
    return out


def run16_reference(dataset, dates):
    """Run 16's own CV predictions on the same dates, for comparison."""
    z = np.load(RUN16_OOF[dataset])
    m = np.isin(z["date"], dates)
    y, p, r, c = z["y"][m], z["p"][m], z["row"][m], z["col"][m]
    yr = W.rule_labels(y, ZONE[r, c]); k = yr != D.IGNORE
    return {"rule": M30.summarise(M30.confusion(yr[k], p[k])), "original": M30.summarise(M30.confusion(y, p))}


def run(dataset, feats, days30, names):
    out = OUT / DATASETS[dataset] / "random_forest"
    (out / "images").mkdir(parents=True, exist_ok=True)
    ylab = labels(dataset)
    px = []
    for date, y in ylab.items():
        fx = feats[date][0]
        m = (y != D.IGNORE) & np.isfinite(fx).all(0)
        rr, cc = np.nonzero(m)
        px.append(dict(date=date, x=fx[:, m].T, y=y[m], r=rr, c=cc))
    print(f"\n=== {dataset}: {len(names)} inputs, top {W.K}, dates {sorted(ylab)}, "
          f"{sum(len(p['y']) for p in px):,} labelled px", flush=True)
    x, y, r, c, _ = W.stack(px)
    ys, ps, ds, rs, cs = [], [], [], [], []
    for f in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(ZONE.shape[0], f)
        model = W.ZoneModel(f).fit(*W.stack(px, train_rows)[:4])
        xt, yt, rt, ct, dt = W.stack(px, test_rows)
        ps.append(model.predict(xt, ZONE[rt, ct])); ys.append(yt); ds.append(dt); rs.append(rt); cs.append(ct)
        print(f"  fold {f + 1}/{D.N_FOLDS} done", flush=True)
    yt, pt, dt, rt, ct = map(np.concatenate, (ys, ps, ds, rs, cs))
    zt = ZONE[rt, ct]
    D.save_oof(out / "oof.npz", dt, rt, ct, yt, pt)
    yr = W.rule_labels(yt, zt); k = yr != D.IGNORE
    res = {"dates": sorted(ylab), "labelled_px": int(len(yt)),
           "rule": M30.summarise(M30.confusion(yr[k], pt[k])), "original": M30.summarise(M30.confusion(yt, pt)),
           "per_date": {}, "run16_same_dates": run16_reference(dataset, sorted(ylab))}
    blocks = [W.change_table(y, ZONE[r, c])]
    blocks += W.scores(yt, pt, zt, "VIIRS 30 m water-zone RF - 5-fold spatial CV, all labelled dates")
    for date in sorted(ylab):
        m = dt == date
        yrd = W.rule_labels(yt[m], zt[m]); kd = yrd != D.IGNORE
        res["per_date"][date] = {"rule": M30.summarise(M30.confusion(yrd[kd], pt[m][kd])),
                                 "original": M30.summarise(M30.confusion(yt[m], pt[m]))}
        blocks.append(D.report(yrd[kd], pt[m][kd], f"--- {date} (rule labels)")[0])
    model = W.ZoneModel().fit(x, y, r, c)
    res["inputs"] = {n: [names[i] for i in keep] for n, (_, keep) in model.parts.items()}
    for n, keep in res["inputs"].items():
        blocks.append(f"{n} RF inputs ({W.K}): " + ", ".join(keep))
    r16 = res["run16_same_dates"]
    blocks.append(f"run 16 (Landsat) on the same dates: rule acc {r16['rule']['accuracy']:.3f} "
                  f"F1 {r16['rule']['macro_f1']:.3f}; original acc {r16['original']['accuracy']:.3f} "
                  f"F1 {r16['original']['macro_f1']:.3f}")
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print(f"rule labels: acc {res['rule']['accuracy']:.3f} F1 {res['rule']['macro_f1']:.3f} | original: acc "
          f"{res['original']['accuracy']:.3f} F1 {res['original']['macro_f1']:.3f}\n" + blocks[-1], flush=True)
    for date, (fx, _) in feats.items():          # maps for every VIIRS date
        flat = fx.reshape(len(fx), -1)
        idx = np.flatnonzero(np.isfinite(flat).all(0))
        pred = np.zeros(flat.shape[1], np.uint8)
        for i in range(0, len(idx), 300_000):
            j = idx[i:i + 300_000]
            pred[j] = model.predict(flat[:, j].T, ZONE.flat[j]) + 1
        D.write_prediction(out / f"pred_{date}.tif", pred.reshape(fx.shape[1:]), days30[date]["ref"])
        shutil.copy(M30.QML, out / f"pred_{date}.qml")
    res["conf_image"] = str(conf_image(out, res["rule"], f"Run 21 RF ({dataset}): 5-fold spatial CV, rule labels "
                                                         f"(acc {res['rule']['accuracy']:.3f})"))
    return res


def conf_image(out, s, title):
    c = np.asarray(s["confusion"], float)
    rowp = np.divide(c, c.sum(1, keepdims=True), out=np.zeros_like(c), where=c.sum(1, keepdims=True) > 0)
    fig, a = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    a.imshow(rowp, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            a.text(j, i, f"{int(c[i, j]):,}\n{100 * rowp[i, j]:.0f}%", ha="center", va="center", fontsize=7.5,
                   color="white" if rowp[i, j] > 0.55 else "black")
    short = ["river", "ice", "snow", "forest", "shrub", "bare"]
    a.set_xticks(range(6), short); a.set_yticks(range(6), short)
    a.set_xlabel("predicted"); a.set_ylabel("true (hand labels, rule)"); a.set_title(title, fontsize=10)
    f = out / "images" / "conf.png"
    fig.savefig(f, dpi=140); plt.close(fig)
    return f


def date_image(out, day30, hand, res, dataset):
    date = day30["date"]
    ls = M30.landsat_rgb(date)
    yh = hand.get(date)
    cmap = ListedColormap(["#ffffff"] + COLOURS)
    panels = [("VIIRS upsampled to 30 m (I3/I2/I1)", "rgb", M30.viirs_up_rgb(day30)),
              ("Landsat 30 m (B6/B5/B4)", "rgb", ls)]
    if yh is not None and ls is not None:
        panels.append((f"Ground truth: hand labels on Landsat\n{int((yh != D.IGNORE).sum()):,} labelled px", "rgb",
                       M30.hand_overlay(ls, yh)))
    else:
        panels.append(("Ground truth: no hand labels this day", "rgb", None if ls is None else 0.45 * ls + 0.5))
    p16 = M30.run16_map(date)
    panels.append(("Run 16 (Landsat) map", "cls", gdal.Open(str(p16)).ReadAsArray() if p16 else None))
    t = "Run 21 RF from VIIRS at 30 m"
    if date in res["per_date"]:
        t += f"\nCV acc vs hand labels {res['per_date'][date]['rule']['accuracy']:.3f} (rule)"
    panels.append((t, "cls", gdal.Open(str(out / f"pred_{date}.tif")).ReadAsArray()))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 8.4), constrained_layout=True)
    for a, (title, kind, img) in zip(ax, panels):
        if img is None:
            a.set_title(title + "\n(missing)", fontsize=10); a.axis("off"); continue
        if kind == "rgb":
            a.imshow(img, interpolation="nearest")
        else:
            a.imshow(img, cmap=cmap, vmin=-0.5, vmax=6.5, interpolation="nearest")
        a.contour(ZONE.astype(float), levels=[0.5], colors="yellow", linewidths=0.5)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLOURS, NAMES)]
               + [Line2D([], [], color="yellow", lw=1.5, label="water mask (JRC >= 5 %): river / ice only")],
               loc="outside lower center", ncol=7, fontsize=9, frameon=False)
    kind = "hand-labelled" if date in res["per_date"] else "no hand labels"
    fig.suptitle(f"Run 21 = run 16 on VIIRS upsampled to 30 m ({dataset}), {date} ({kind})", fontsize=13)
    fig.savefig(out / "images" / f"pred_{date}.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["with_daksh"], choices=list(DATASETS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shifts = {r["date"]: r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text())}
    cache, days30, feats, names = {}, {}, {}, None
    for date in U.viirs_dates():
        days30[date] = M30.load_day(date, shifts.get(date, [0, 0]), cache)
        feats[date] = viirs_features(days30[date]); names = feats[date][1]
        print(date, "inputs", len(names), flush=True)
    mpath = OUT / "metrics_all.json"
    results = json.loads(mpath.read_text()) if mpath.exists() else {}
    for dataset in args.datasets:
        results[dataset] = run(dataset, feats, days30, names)
        mpath.write_text(json.dumps(results, indent=1))
        out = OUT / DATASETS[dataset] / "random_forest"
        hand = labels(dataset)
        for date in days30:
            date_image(out, days30[date], hand, results[dataset], dataset)
    print("wrote", OUT, flush=True)
