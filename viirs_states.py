"""
Run 18: four river / land states per VIIRS 375 m cell, with the % mix, predicted from VIIRS alone
by a Random Forest and by UNet++. Built on run 17 (Oleson et al. 1995 unmixing set-up): the run 16
Landsat 30 m map gives the % of each VIIRS cell, on the same cells and with the same VIIRS-Landsat
shift that run 17 found.

Four parts of a cell (area %, sum 100):
  open_river       run 16 river
  ice              run 16 ice
  snow             run 16 snow
  snow_free_land   run 16 forest + shrub_grass_wetland + bare_developed

Four states (same numbering as scene_mask.py):
  1 ice_free_river_snow_free_land
  2 ice_free_river_snow_covered_land
  3 ice_covered_river_snow_free_land
  4 ice_covered_river_snow_covered_land
Two rules on the same %:
  rule50  river ice-free if open_river / (open_river + ice) > 0.5,
          land snow-covered if snow / (snow + snow_free_land) > 0.5
  rule90  river ice-free only if open_river / (open_river + ice) > 0.9,
          land snow-free only if snow / (snow + snow_free_land) < 0.1 (mixed cells count as covered)
  A cell with < 1 % river (or < 1 % land) takes the ratio of the whole date (all cells together),
  so every cell gets a state; the river state of a land cell is the river state of that day.

Inputs per cell (from the VIIRS grid file of the day + JRC):
  I1, I2, I3 (TOA reflectance), I4, I5 (brightness temperature K), NDVI_I = (I2-I1)/(I2+I1),
  NDSI_I = (I1-I3)/(I1+I3), I4-I5, solar zenith, view zenith, JRC mean occurrence of the cell,
  JRC water-zone share (occurrence >= 5 %)
  RF also gets 3x3 neighbourhood mean / std of I1, I2, I3, I5 (UNet++ sees context by itself).

Models (both predict the four %; the states come from the predicted % by the rules above):
  RF      RandomForestRegressor, 4 outputs, depth 12, min_samples_leaf 20, sqrt, 300 trees
  UNet++  smp UnetPlusPlus, resnet18 encoder, softmax over the four parts, soft cross-entropy on
          mapped cells, whole 132 x 80 day images padded to 160 x 96, flips, AdamW 1e-3, 1500 steps

Dates: only the run 16 Landsat dates that have a VIIRS grid file (2023-05-05, 2023-07-24, 2023-10-04,
2024-04-21, 2024-06-22, 2026-03-26). Run 16 dates without VIIRS (2024-03-12, 2024-04-20, 2024-04-29)
are left out, and so are VIIRS dates without a run 16 map.

Evaluation: leave-one-date-out over those 6 dates (train on 5 days, predict the 6th), so every
map and score is for a day the model never saw.

Outputs: output/18_viirs_states/
  truth/pct_<date>.tif, truth/<rule>/class_<date>.tif      targets from run 16
  rf/, unetpp/   pct_<date>.tif (4 bands %), <rule>/class_<date>.tif (1-4), metrics.txt, oof.npz
  metrics_all.json, summary.txt

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/viirs_states.py
"""

import json
import os
import numpy as np
from osgeo import gdal
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestRegressor
import rf_viirs_unmix as U

gdal.UseExceptions()

OUT = U.HERE / "output" / ("22_viirs_river_gauss/pct_models" if os.environ.get("RIVER_LABELS", "box") == "psf"
                           else "18_viirs_states")          # run 22: % models on Gaussian labels
PARTS = ["open_river", "ice", "snow", "snow_free_land"]
STATES = ["ice_free_river_snow_free_land", "ice_free_river_snow_covered_land",
          "ice_covered_river_snow_free_land", "ice_covered_river_snow_covered_land"]
COLOURS = ["#2b83ba", "#00b4b4", "#a05a2c", "#e6f5ff"]
RULES = {"rule50": (0.5, 0.5), "rule90": (0.9, 0.1)}   # (open share for ice-free, snow share for snow-covered)
MIN_PART = 0.01
SEED = 0


# ----------------------------------------------------------------------------- data

def cell_index(shape_f, gt_f, gt_v, shape_v, shift):
    H, W = shape_f
    hv, wv = shape_v
    lat = gt_f[3] + (np.arange(H) + 0.5 + shift[0]) * gt_f[5]
    lon = gt_f[0] + (np.arange(W) + 0.5 + shift[1]) * gt_f[1]
    rows = np.floor((lat - gt_v[3]) / gt_v[5]).astype(int)
    cols = np.floor((lon - gt_v[0]) / gt_v[1]).astype(int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    ok = (rr >= 0) & (rr < hv) & (cc >= 0) & (cc < wv)
    return np.where(ok, rr * wv + cc, -1)


def cell_mean(a, idx, n):
    m = (idx >= 0) & np.isfinite(a)
    s = np.bincount(idx[m], weights=a[m], minlength=n)
    c = np.bincount(idx[m], minlength=n)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def load_day(date, shift, cache):
    cls, gt_f, v, gt_v, ds_v, _, source = U.load_date(date, cache)
    hv, wv = v.shape[1:]
    n = hv * wv
    if os.environ.get("RIVER_LABELS", "box") == "psf":      # run 22: run 17's Gaussian footprint shares
        if ("blur", date) not in cache:
            cache[("blur", date)] = U.blur(cls, gt_f)[0]
        w, mapped = U.weights(cls, gt_f, gt_v, (hv, wv), tuple(shift), "psf", cache[("blur", date)])
    else:                                                  # runs 18-20: box counts
        w, mapped = U.weights(cls, gt_f, gt_v, (hv, wv), tuple(shift), "box")
    pct = np.stack([w[:, 0], w[:, 1], w[:, 2], w[:, 3:].sum(1)], 1)          # (n, 4) fractions
    idx = cell_index(cls.shape, gt_f, gt_v, (hv, wv), tuple(shift))
    if "jrc" not in cache:
        import data as D
        cache["jrc"] = D._water_occurrence().astype(np.float64)
    jrc = cache["jrc"]
    jrc_mean = cell_mean(jrc, idx, n)
    jrc_zone = cell_mean((jrc >= 5.0).astype(np.float64), idx, n)
    I1, I2, I3, I4, I5 = v[:5]
    eps = 1e-6
    chans = [I1, I2, I3, I4, I5, (I2 - I1) / (I2 + I1 + eps), (I1 - I3) / (I1 + I3 + eps), I4 - I5,
             v[5], v[7], jrc_mean.reshape(hv, wv), jrc_zone.reshape(hv, wv)]
    names = ["I1", "I2", "I3", "I4", "I5", "NDVI_I", "NDSI_I", "I4_minus_I5", "solar_zenith",
             "view_zenith", "jrc_mean", "jrc_zone_share"]
    img = np.stack(chans).astype(np.float32)                                   # (C, hv, wv)
    good = (mapped >= U.MIN_VALID) & np.isfinite(img.reshape(len(img), -1)).all(0)
    return {"date": date, "img": img, "names": names, "pct": pct, "good": good, "shape": (hv, wv),
            "gt_v": gt_v, "ds_v": ds_v, "source": source}


def rf_table(day):
    img = day["img"]
    extra, names = [], []
    for b in ("I1", "I2", "I3", "I5"):
        a = np.nan_to_num(img[day["names"].index(b)].astype(np.float64))
        m = uniform_filter(a, 3, mode="nearest")
        sd = np.sqrt(np.maximum(uniform_filter(a * a, 3, mode="nearest") - m * m, 0))
        extra += [m, sd]; names += [f"{b}_mean3", f"{b}_std3"]
    x = np.concatenate([img, np.stack(extra).astype(np.float32)])
    return x.reshape(len(x), -1).T, day["names"] + names


# ----------------------------------------------------------------------------- states

def states(p, good, rule):
    """State 1..4 per cell from the four fractions (0 where the cell is not used)."""
    open_cut, snow_cut = RULES[rule]
    q = p[good]
    r_day = q[:, 0].sum() / max(q[:, 0].sum() + q[:, 1].sum(), 1e-9)
    s_day = q[:, 2].sum() / max(q[:, 2].sum() + q[:, 3].sum(), 1e-9)
    river, land = p[:, 0] + p[:, 1], p[:, 2] + p[:, 3]
    open_ratio = np.where(river >= MIN_PART, p[:, 0] / np.maximum(river, 1e-9), r_day)
    snow_ratio = np.where(land >= MIN_PART, p[:, 2] / np.maximum(land, 1e-9), s_day)
    ice_free = open_ratio > open_cut
    snowy = snow_ratio > snow_cut if rule == "rule50" else snow_ratio >= snow_cut
    s = 1 + 2 * (~ice_free) + snowy                                       # 1..4 as in STATES
    return np.where(good, s, 0).astype(np.uint8)


# ----------------------------------------------------------------------------- models

def fit_rf(days, train):
    xs, ys = [], []
    for d in train:
        x, _ = rf_table(days[d]); g = days[d]["good"]
        xs.append(x[g]); ys.append(days[d]["pct"][g])
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=20, max_features="sqrt",
                               n_jobs=-1, random_state=SEED)
    return rf.fit(np.concatenate(xs), np.concatenate(ys))


def predict_rf(rf, day):
    x, _ = rf_table(day)
    p = np.zeros((len(x), 4))
    g = day["good"]
    p[g] = rf.predict(x[g])
    p = np.clip(p, 0, None)
    return p / np.maximum(p.sum(1, keepdims=True), 1e-9)


PAD = (160, 96)


def to_tensor(day, mean, std):
    import torch
    x = (day["img"] - mean[:, None, None]) / std[:, None, None]
    x = np.nan_to_num(x, nan=0.0)
    hv, wv = day["shape"]
    out = np.zeros((len(x), *PAD), np.float32); out[:, :hv, :wv] = x
    t = np.zeros((4, *PAD), np.float32); t[:, :hv, :wv] = day["pct"].T.reshape(4, hv, wv)
    m = np.zeros(PAD, np.float32); m[:hv, :wv] = day["good"].reshape(hv, wv)
    return torch.from_numpy(out), torch.from_numpy(t), torch.from_numpy(m)


def make_net(arch, in_ch, classes=4):
    """unetpp: smp UNet++ resnet18; cnn2d: plain 2D CNN, 4 conv 3x3 layers, no down-sampling (9 x 9 cells seen)."""
    import torch.nn as nn
    if arch == "cnn2d":
        layers, c = [], in_ch
        for w in (32, 64, 64, 64):
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        return nn.Sequential(*layers, nn.Dropout2d(0.1), nn.Conv2d(c, classes, 1))
    import segmentation_models_pytorch as smp
    try:
        return smp.UnetPlusPlus("resnet18", encoder_weights="imagenet", in_channels=in_ch, classes=classes)
    except Exception:
        return smp.UnetPlusPlus("resnet18", encoder_weights=None, in_channels=in_ch, classes=classes)


def fit_unet(days, train, steps=1500, arch="unetpp"):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stack = np.concatenate([days[d]["img"].reshape(len(days[d]["img"]), -1)[:, days[d]["good"]] for d in train], 1)
    mean, std = stack.mean(1), stack.std(1) + 1e-6
    data = [to_tensor(days[d], mean, std) for d in train]
    X = torch.stack([a for a, _, _ in data]).to(dev)
    T = torch.stack([b for _, b, _ in data]).to(dev)
    M = torch.stack([c for _, _, c in data]).to(dev)
    net = make_net(arch, X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    net.train()
    for _ in range(steps):
        x, t, m = X, T, M
        if np.random.rand() < 0.5:
            x, t, m = x.flip(-1), t.flip(-1), m.flip(-1)
        if np.random.rand() < 0.5:
            x, t, m = x.flip(-2), t.flip(-2), m.flip(-2)
        loss = -(t * F.log_softmax(net(x), 1)).sum(1)
        loss = (loss * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net, mean, std, dev


def predict_unet(model, day):
    import torch
    net, mean, std, dev = model
    x, _, _ = to_tensor(day, mean, std)
    hv, wv = day["shape"]
    with torch.no_grad():
        xb = x[None].to(dev)
        p = torch.softmax(net(xb), 1)
        p = p + torch.softmax(net(xb.flip(-1)), 1).flip(-1)
        p = p + torch.softmax(net(xb.flip(-2)), 1).flip(-2)
        p = (p / 3)[0, :, :hv, :wv].cpu().numpy()
    return p.reshape(4, -1).T


# ----------------------------------------------------------------------------- scoring / writing

def part_scores(tp, pp):
    out = {"mae_pp": {}, "r2": {}}
    for k, n in enumerate(PARTS):
        e = pp[:, k] - tp[:, k]
        out["mae_pp"][n] = float(100 * np.abs(e).mean())
        out["r2"][n] = float(1 - (e ** 2).sum() / max(((tp[:, k] - tp[:, k].mean()) ** 2).sum(), 1e-12))
    return out


def state_scores(ts, ps):
    out = {"cells": int(len(ts)), "accuracy": float((ts == ps).mean()), "per_state": {}}
    for s in range(1, 5):
        t, q = ts == s, ps == s
        if t.sum() or q.sum():
            tpos = (t & q).sum(); pr = tpos / max(q.sum(), 1); rc = tpos / max(t.sum(), 1)
            out["per_state"][STATES[s - 1]] = {"f1": float(2 * pr * rc / max(pr + rc, 1e-9)), "precision": float(pr),
                                               "recall": float(rc), "true_cells": int(t.sum()), "pred_cells": int(q.sum())}
    present = [v["f1"] for v in out["per_state"].values() if v["true_cells"] > 0]
    out["macro_f1"] = float(np.mean(present)) if present else float("nan")
    out["confusion"] = [[int(((ts == a) & (ps == b)).sum()) for b in range(1, 5)] for a in range(1, 5)]
    return out


def write_pct(folder, day, pct):
    folder.mkdir(parents=True, exist_ok=True)
    hv, wv = day["shape"]
    U.write(folder / f"pct_{day['date']}.tif", np.where(day["good"], 100 * pct.T, np.nan).reshape(4, hv, wv),
            day["gt_v"], day["ds_v"], [f"{p}_pct" for p in PARTS])


def write_class(folder, day, st):
    folder.mkdir(parents=True, exist_ok=True)
    hv, wv = day["shape"]
    o = gdal.GetDriverByName("GTiff").Create(str(folder / f"class_{day['date']}.tif"), wv, hv, 1, gdal.GDT_Byte,
                                             options=["COMPRESS=DEFLATE"])
    o.SetGeoTransform(day["gt_v"]); o.SetProjection(day["ds_v"].GetProjection())
    b = o.GetRasterBand(1); b.WriteArray(st.reshape(hv, wv)); b.SetNoDataValue(0); o = None
    entries = "\n".join(f'      <paletteEntry value="{i + 1}" color="{c}" alpha="255" label="{i + 1} {s}"/>'
                        for i, (s, c) in enumerate(zip(STATES, COLOURS)))
    (folder / f"class_{day['date']}.qml").write_text(
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n<qgis version=\"3.34\">\n"
        "  <pipe>\n    <rasterrenderer type=\"paletted\" band=\"1\" opacity=\"1\" alphaBand=\"-1\">\n"
        f"      <colorPalette>\n{entries}\n      </colorPalette>\n    </rasterrenderer>\n  </pipe>\n</qgis>\n")


def text_block(title, res):
    L = [f"=== {title}: leave-one-date-out, {res['parts']['cells']} cells",
         "part MAE (percentage points) / R2: "
         + ", ".join(f"{p} {res['parts']['mae_pp'][p]:.1f} / {res['parts']['r2'][p]:.2f}" for p in PARTS)]
    for rule in RULES:
        s = res[rule]
        L += [f"--- {rule}: state accuracy {s['accuracy']:.3f}   macro F1 {s['macro_f1']:.3f}",
              f"{'state':<38}{'F1':>6}{'prec':>6}{'rec':>6}{'true':>7}{'pred':>7}"]
        for k, v in s["per_state"].items():
            L.append(f"{k:<38}{v['f1']:>6.2f}{v['precision']:>6.2f}{v['recall']:>6.2f}{v['true_cells']:>7}{v['pred_cells']:>7}")
        L.append("confusion (rows true 1-4, columns predicted 1-4): " + str(s["confusion"]))
    L.append(f"{'date':<12}{'acc50':>6}{'acc90':>6}" + "".join(f"{p[:10] + ' t/p %':>19}" for p in PARTS))
    for d, v in res["per_date"].items():
        L.append(f"{d:<12}{v['rule50']['accuracy']:>6.2f}{v['rule90']['accuracy']:>6.2f}" + "".join(
            f"{v['true_mean_pct'][p]:>11.1f}/{v['pred_mean_pct'][p]:<7.1f}" for p in PARTS))
    return "\n".join(L)


# ----------------------------------------------------------------------------- main

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shifts = {r["date"]: r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text())}
    cache, days = {}, {}
    run16 = [d for d in U.viirs_dates()
             if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    for date in run16:
        d = days[date] = load_day(date, shifts.get(date, [0, 0]), cache)
        d["state"] = {rule: states(d["pct"], d["good"], rule) for rule in RULES}
        write_pct(OUT / "truth", d, d["pct"])
        for rule in RULES:
            write_class(OUT / "truth" / rule, d, d["state"][rule])
        g = d["good"]
        print(date, f"{d['source']:<34} cells {int(g.sum())}  % " +
              " ".join(f"{p} {100 * d['pct'][g, k].mean():.1f}" for k, p in enumerate(PARTS)), flush=True)
        for rule in RULES:
            print("   ", rule, {STATES[s - 1]: int((d["state"][rule] == s).sum()) for s in range(1, 5)
                                if (d["state"][rule] == s).any()}, flush=True)

    dates = list(days)
    results, blocks = {}, []
    for model in ("rf", "unetpp"):
        tp_all, pp_all, st_all, per_date = [], [], {r: ([], []) for r in RULES}, {}
        oof = {k: [] for k in ("date", "cell", "true_pct", "pred_pct")}
        for test in dates:
            train = [d for d in dates if d != test]
            day = days[test]; g = day["good"]
            p = predict_rf(fit_rf(days, train), day) if model == "rf" else predict_unet(fit_unet(days, train), day)
            write_pct(OUT / model, day, p)
            per_date[test] = {"true_mean_pct": {n: float(100 * day["pct"][g, k].mean()) for k, n in enumerate(PARTS)},
                              "pred_mean_pct": {n: float(100 * p[g, k].mean()) for k, n in enumerate(PARTS)},
                              "parts": part_scores(day["pct"][g], p[g])}
            for rule in RULES:
                st = states(p, g, rule)
                write_class(OUT / model / rule, day, st)
                per_date[test][rule] = state_scores(day["state"][rule][g], st[g])
                st_all[rule][0].append(day["state"][rule][g]); st_all[rule][1].append(st[g])
                oof.setdefault(f"true_state_{rule}", []).append(day["state"][rule][g])
                oof.setdefault(f"pred_state_{rule}", []).append(st[g])
            tp_all.append(day["pct"][g]); pp_all.append(p[g])
            for k, a in (("date", np.full(g.sum(), test)), ("cell", np.flatnonzero(g)),
                         ("true_pct", day["pct"][g]), ("pred_pct", p[g])):
                oof[k].append(a)
            print(f"{model:<7}{test}  acc50 {per_date[test]['rule50']['accuracy']:.3f}  "
                  f"acc90 {per_date[test]['rule90']['accuracy']:.3f}  "
                  + "  ".join(f"{n} {per_date[test]['true_mean_pct'][n]:.1f}/{per_date[test]['pred_mean_pct'][n]:.1f}%"
                              for n in PARTS), flush=True)
        res = {"parts": {"cells": int(sum(len(a) for a in tp_all)),
                         **part_scores(np.concatenate(tp_all), np.concatenate(pp_all))},
               "per_date": per_date}
        for rule in RULES:
            res[rule] = state_scores(np.concatenate(st_all[rule][0]), np.concatenate(st_all[rule][1]))
        results[model] = res
        np.savez_compressed(OUT / model / "oof.npz", **{k: np.concatenate(v) for k, v in oof.items()})
        blocks.append(text_block({"rf": "Random Forest", "unetpp": "UNet++"}[model], res))
        (OUT / model / "metrics.txt").write_text(blocks[-1] + "\n")
        print(blocks[-1], flush=True)
    (OUT / "metrics_all.json").write_text(json.dumps(results, indent=1))
    head = ["Run 18: four river/land states per VIIRS 375 m cell from VIIRS alone (targets: run 16 Landsat map,",
            "cells and shift from run 17). Leave-one-date-out over " + ", ".join(dates) + ".",
            "rule50 = majority; rule90 = ice-free only if > 90 % open water, snow-free only if < 10 % snow.",
            "t/p % = true / predicted mean share of the box that day.", ""]
    (OUT / "summary.txt").write_text("\n".join(head) + "\n" + "\n\n".join(blocks) + "\n")
    print("wrote", OUT)
