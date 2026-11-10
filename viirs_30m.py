"""
Run 21: VIIRS upsampled to the Landsat 30 m grid, six run 16 classes predicted at 30 m by RF and UNet++,
with the run 16 zone rule.

    water zone  JRC occurrence >= 5 % (30 m river mask, run 16 water zone)   only river or ice (no snow)
    land        everything else                                            only snow, forest,
                                                                           shrub_grass_wetland, bare_developed (no ice)

Inputs per 30 m pixel (no Landsat band is an input):
  VIIRS, bilinear-upsampled from 375 m and moved by the run 17 VIIRS-Landsat shift:
    I1, I2, I3, I4, I5, NDVI_I, NDSI_I, I4-I5, solar zenith, view zenith,
    3 x 3 (375 m) mean and std of I1, I2, I3, I5
  static 30 m layers: JRC water occurrence, NLCD 2016 (10 groups, as in run 16)
Labels: the run 16 Landsat 30 m map of the day (with_daksh). Upsampling adds no detail by itself:
the 30 m detail can only come from the static layers (JRC, NLCD) and the zone rule.

Models (as run 16, with the zone rule built in):
  RF      one forest per zone, run 16 settings: top 25 inputs ranked on the training days, depth 16,
          min_samples_leaf 20, sqrt, 300 trees, balanced_subsample; trained on up to 20 000 pixels per
          class and day
  UNet++  smp resnet18, six classes, logits of the classes the zone does not allow are switched off,
          class-weighted cross-entropy (1 / sqrt frequency), 256 x 256 tiles, 3000 steps, flips,
          AdamW 1e-3; prediction in overlapping 512 x 512 windows with flip TTA

Evaluation, leave-one-date-out over the 6 run 16 dates with VIIRS:
  vs run 16 map    every valid 30 m pixel of the left-out day (accuracy, macro F1, 6 x 6 confusion)
  vs hand labels   labelled pixels of the left-out day, rule labels (as run 16) and original labels
  at 375 m         30 m predictions summed per VIIRS pixel -> open water / ice / snow / snow-free land %
                   and the rule50 / rule90 states, compared with the run 18 truth (same cells)
The 4 VIIRS-only dates are predicted by final models trained on all 6 days.

Outputs: output/21_viirs_30m/
  rf/, unetpp/   pred_<date>.tif (+ .qml), viirs_only/pred_<date>.tif, metrics.txt
  metrics_all.json, summary.txt, images/pred_<date>.jpg, images/conf_<model>.png

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/viirs_30m.py
"""

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
from scipy.ndimage import shift as nd_shift

import data as D
import rf_limited as L
import rf_six_classes as S6
import rf_viirs_unmix as U
import rf_water_zone as W
import viirs_states as V

gdal.UseExceptions()
OUT = U.HERE / "output" / "21_viirs_30m"
NAMES = ["river", "ice", "snow", "forest", "shrub_grass_wetland", "bare_developed"]
COLOURS = ["#0066ff", "#ff00ff", "#00ffff", "#1a7f37", "#b5d96b", "#c8a165"]
ZONE = W.ZONE                                          # (H, W) True = water zone
ALLOWED = {True: (0, 1), False: (2, 3, 4, 5)}
K, DEPTH, PER_CLASS, SEED = 25, 16, 20_000, 0
QML = U.RUN16 / "pred_2024-04-21.qml"


# ----------------------------------------------------------------------------- data

def run16_map(date):
    for p in (U.RUN16 / f"pred_{date}.tif", U.RUN16 / "new_dates" / date / "rf_pred.tif",
              U.OUT / "landsat_maps" / f"pred_{date}.tif"):
        if p.exists():
            return p
    return None


def bounds(ref):
    gt = ref.GetGeoTransform()
    return (gt[0], gt[3] + ref.RasterYSize * gt[5], gt[0] + ref.RasterXSize * gt[1], gt[3])


def upsample(day, ref, shift):
    """VIIRS inputs (C, hv, wv) -> (C, H, W) on the Landsat grid, bilinear, then moved by the run 17 shift."""
    x375, names = V.rf_table(day)                      # (n, 20) incl. the two 375 m JRC inputs
    keep = [i for i, n in enumerate(names) if not n.startswith("jrc")]
    hv, wv = day["shape"]
    arr = x375[:, keep].T.reshape(len(keep), hv, wv).astype(np.float32)
    src = gdal.GetDriverByName("MEM").Create("", wv, hv, len(keep), gdal.GDT_Float32)
    src.SetGeoTransform(day["gt_v"]); src.SetProjection(day["ds_v"].GetProjection())
    for k in range(len(keep)):
        b = src.GetRasterBand(k + 1); b.WriteArray(arr[k]); b.SetNoDataValue(np.nan)
    dst = gdal.Warp("", src, format="MEM", outputBounds=bounds(ref), width=ref.RasterXSize, height=ref.RasterYSize,
                    resampleAlg="bilinear", srcNodata=np.nan, dstNodata=np.nan)
    up = dst.ReadAsArray().astype(np.float32)
    # run 17: Landsat pixel r falls in VIIRS row (r + shift); so the VIIRS value for r is up[r + shift]
    dr, dc = shift
    if dr or dc:
        up = np.stack([nd_shift(a, (-dr, -dc), order=0, cval=np.nan) for a in up])
    return up, [names[i] + "_up" for i in keep]


STATIC = None


def static_inputs():
    global STATIC
    if STATIC is None:
        nl = S6.nlcd_inputs()
        STATIC = (np.stack([D._water_occurrence()] + [a for _, a in nl]).astype(np.float32),
                  ["jrc_occurrence_30m"] + [n for n, _ in nl])
    return STATIC


def load_day(date, shift, cache):
    day = V.load_day(date, shift, cache)
    ref = gdal.Open(str(run16_map(date)))
    cls = ref.ReadAsArray()
    up, names = upsample(day, ref, shift)
    st, snames = static_inputs()
    x = np.concatenate([up, st])
    valid = np.isfinite(x).all(0) & (cls > 0)
    return {"date": date, "x": x, "names": names + snames, "y": cls.astype(np.int16) - 1, "valid": valid,
            "ref": ref, "day375": day, "shift": shift}


def hand_labels():
    """{date: (H, W) six-class index labels, IGNORE elsewhere} from the with_daksh label files."""
    D.configure("four", "with_daksh")
    out = {}
    for group, date, folder, lab in D.scene_list():
        boxes = list(folder.glob("landsat_*_box.tif"))
        ref = gdal.Open(str(boxes[0] if boxes else folder / f"{date}_sr.tif"))
        y = D.load_labels(lab, ref)
        if (y != D.IGNORE).any() and y.shape == ZONE.shape:
            s = {"y": y}
            S6.prepare([s])
            out[date] = s["y"]
    return out


# ----------------------------------------------------------------------------- RF

def sample(day, rng):
    """Up to PER_CLASS pixels per class (run 16 map) of a day."""
    idx = []
    for k in range(6):
        cand = np.flatnonzero((day["y"].ravel() == k) & day["valid"].ravel())
        if len(cand):
            idx.append(rng.choice(cand, min(PER_CLASS, len(cand)), replace=False))
    idx = np.concatenate(idx)
    flat = day["x"].reshape(len(day["x"]), -1)
    return flat[:, idx].T, day["y"].ravel()[idx], ZONE.ravel()[idx]


class ZoneRF:
    def fit(self, days, train, seed=SEED):
        rng = np.random.default_rng(seed)
        parts = [sample(days[d], rng) for d in train]
        x = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        z = np.concatenate([p[2] for p in parts])
        self.parts = {}
        for zone in (True, False):
            t = (z == zone) & np.isin(y, ALLOWED[zone])
            keep = L.rank(x[t], y[t], seed)[:K]
            self.parts[zone] = (L.forest(DEPTH, seed=seed).fit(x[t][:, keep], y[t]), keep)
        return self

    def predict(self, day):
        flat = day["x"].reshape(len(day["x"]), -1)
        out = np.full(flat.shape[1], 255, np.uint8)
        for zone in (True, False):
            idx = np.flatnonzero(day["valid"].ravel() & (ZONE.ravel() == zone))
            f, keep = self.parts[zone]
            for i in range(0, len(idx), 400_000):
                j = idx[i:i + 400_000]
                out[j] = f.predict(flat[keep][:, j].T)
        return out.reshape(day["y"].shape)

    def inputs(self, names):
        return {("water" if z else "land"): [names[i] for i in keep] for z, (_, keep) in self.parts.items()}


# ----------------------------------------------------------------------------- UNet++

def zone_bias(dev):
    import torch
    b = np.full((6,) + ZONE.shape, -1e4, np.float32)
    for zone, ks in ALLOWED.items():
        for k in ks:
            b[k][ZONE == zone] = 0
    return torch.from_numpy(b).to(dev)


class ZoneUNet:
    TILE, STEPS, BATCH = 256, 3000, 8

    def fit(self, days, train, seed=SEED):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        self.dev = dev = "cuda" if torch.cuda.is_available() else "cpu"
        stack = np.concatenate([days[d]["x"][:, days[d]["valid"]][:, ::50] for d in train], 1)
        self.mean, self.std = stack.mean(1), stack.std(1) + 1e-6
        X = torch.from_numpy(np.stack([self.norm(days[d]) for d in train])).to(dev)
        Y = torch.from_numpy(np.stack([np.where(days[d]["valid"], days[d]["y"], -100)
                                       for d in train]).astype(np.int64)).to(dev)
        counts = np.bincount(np.concatenate([days[d]["y"][days[d]["valid"]] for d in train]), minlength=6).astype(float)
        w = 1 / np.sqrt(np.maximum(counts, 1)); w = w / w[counts > 0].mean()
        weight = torch.tensor(w, dtype=torch.float32, device=dev)
        bias = zone_bias(dev)
        self.net = net = V.make_net("unetpp", X.shape[1], classes=6).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=self.STEPS)
        H, Wd = ZONE.shape
        t = self.TILE
        net.train()
        for _ in range(self.STEPS):
            xs, ys, bs = [], [], []
            for _ in range(self.BATCH):
                i = rng.integers(len(train)); r = rng.integers(0, H - t); c = rng.integers(0, Wd - t)
                xs.append(X[i, :, r:r + t, c:c + t]); ys.append(Y[i, r:r + t, c:c + t]); bs.append(bias[:, r:r + t, c:c + t])
            x, y, b = torch.stack(xs), torch.stack(ys), torch.stack(bs)
            if rng.random() < 0.5:
                x, y, b = x.flip(-1), y.flip(-1), b.flip(-1)
            if rng.random() < 0.5:
                x, y, b = x.flip(-2), y.flip(-2), b.flip(-2)
            loss = F.cross_entropy(net(x) + b, y, weight=weight, ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        return self

    def norm(self, day):
        x = (day["x"] - self.mean[:, None, None]) / self.std[:, None, None]
        return np.nan_to_num(x).astype(np.float32)

    def predict(self, day, win=512, stride=448):
        import torch
        x = torch.from_numpy(self.norm(day)).to(self.dev)
        H, Wd = ZONE.shape
        prob = torch.zeros((6, H, Wd), device=self.dev); cnt = torch.zeros((H, Wd), device=self.dev)
        rs = sorted(set(list(range(0, H - win + 1, stride)) + [H - win]))
        cs = sorted(set(list(range(0, Wd - win + 1, stride)) + [Wd - win]))
        with torch.no_grad():
            for r in rs:
                for c in cs:
                    xb = x[None, :, r:r + win, c:c + win]
                    p = torch.softmax(self.net(xb), 1)
                    p = p + torch.softmax(self.net(xb.flip(-1)), 1).flip(-1)
                    p = p + torch.softmax(self.net(xb.flip(-2)), 1).flip(-2)
                    prob[:, r:r + win, c:c + win] += p[0] / 3; cnt[r:r + win, c:c + win] += 1
        logp = torch.log(prob / cnt.clamp(min=1) + 1e-9) + zone_bias(self.dev)
        out = logp.argmax(0).cpu().numpy().astype(np.uint8)
        out[~day["valid"]] = 255
        return out


# ----------------------------------------------------------------------------- scoring

def confusion(y, p, n=6):
    m = (y >= 0) & (y < n) & (p < n)
    return np.bincount(y[m].astype(int) * n + p[m].astype(int), minlength=n * n).reshape(n, n)


def summarise(c):
    c = np.asarray(c, float)
    tp, t, q = np.diag(c), c.sum(1), c.sum(0)
    pr = np.divide(tp, q, out=np.zeros(len(c)), where=q > 0)
    rc = np.divide(tp, t, out=np.zeros(len(c)), where=t > 0)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros(len(c)), where=(pr + rc) > 0)
    n = max(c.sum(), 1); po = tp.sum() / n; pe = (t * q).sum() / n ** 2
    present = t > 0
    return {"pixels": int(c.sum()), "accuracy": float(po), "kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan"),
            "macro_f1": float(f1[present].mean()), "balanced_accuracy": float(rc[present].mean()),
            "per_class": {NAMES[k]: {"f1": float(f1[k]), "precision": float(pr[k]), "recall": float(rc[k]),
                                     "true": int(t[k]), "pred": int(q[k])} for k in range(len(c))},
            "confusion": c.astype(int).tolist()}


def to_375(day, pred):
    """30 m prediction -> (n_cells, 4) fractions open water, ice, snow, snow-free land per VIIRS cell."""
    d = day["day375"]
    hv, wv = d["shape"]
    idx = V.cell_index(pred.shape, day["ref"].GetGeoTransform(), d["gt_v"], (hv, wv), tuple(day["shift"])).ravel()
    p = pred.ravel()
    ok = (idx >= 0) & (p < 6)
    n = np.bincount(idx[ok], minlength=hv * wv).astype(float)
    parts = [p == 0, p == 1, p == 2, np.isin(p, [3, 4, 5])]
    return np.stack([np.bincount(idx[ok], weights=q[ok].astype(float), minlength=hv * wv) / np.maximum(n, 1)
                     for q in parts], 1)


def score_375(day, pred):
    d = day["day375"]
    g = d["good"]
    pp = to_375(day, pred)
    out = {"parts": V.part_scores(d["pct"][g], pp[g])}
    for rule in V.RULES:
        out[rule] = float((V.states(d["pct"], g, rule)[g] == V.states(pp, g, rule)[g]).mean())
    return out


# ----------------------------------------------------------------------------- images

def stretch(rgb):
    ok = np.isfinite(rgb).all(-1)
    lo, hi = np.nanpercentile(rgb[ok], [2, 98], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb[~ok] = 0.6
    return rgb


def landsat_rgb(date):
    f = U.PATCHES / f"nenana_{date.replace('-', '')}" / f"{date}_sr.tif"
    if not f.exists():
        return None
    a = gdal.Open(str(f)).ReadAsArray().astype(np.float32)
    return stretch(np.stack([a[4], a[3], a[2]], -1))   # B6, B5, B4 = the bands of VIIRS I3, I2, I1


def viirs_up_rgb(day):
    i = {n: k for k, n in enumerate(day["names"])}
    return stretch(np.stack([day["x"][i["I3_up"]], day["x"][i["I2_up"]], day["x"][i["I1_up"]]], -1))


def hand_overlay(rgb, yh):
    """Landsat image dimmed, hand-labelled pixels painted in their class colour (ground truth)."""
    out = 0.45 * rgb + 0.55 * 0.92
    m = yh != D.IGNORE
    if m.any():
        from matplotlib.colors import to_rgb
        pal = np.array([to_rgb(c) for c in COLOURS])
        # labels are small polygons: grow them a little so they show at page size
        from scipy.ndimage import grey_dilation
        lab = np.where(m, yh.astype(int) + 1, 0)
        big = grey_dilation(lab, size=(3, 3))
        show = big > 0
        out[show] = pal[big[show] - 1]
    return out


def date_image(day, preds, scores, labelled, hand=None):
    cmap = ListedColormap(["#ffffff"] + COLOURS)
    ls = landsat_rgb(day["date"])
    yh = (hand or {}).get(day["date"])
    if ls is not None and yh is not None:
        gt_panel = (f"Ground truth: hand labels on Landsat\n{int((yh != D.IGNORE).sum()):,} labelled px", "rgb",
                    hand_overlay(ls, yh))
    else:
        gt_panel = ("Ground truth: no hand labels this day", "rgb", None if ls is None else 0.45 * ls + 0.5)
    hs = ""
    panels = [("VIIRS upsampled to 30 m (I3/I2/I1)", "rgb", viirs_up_rgb(day)),
              ("Landsat 30 m (B6/B5/B4, same bands)", "rgb", ls),
              gt_panel,
              ("Run 16 Landsat map" + (" (label map)" if labelled else " (display only)"), "cls", day["y"] + 1)]
    for m, lab in (("rf", "RF"), ("unetpp", "UNet++")):
        if m not in preds:
            continue
        t = f"{lab} from VIIRS at 30 m"
        if labelled and scores:
            t += f"\nvs run 16 map: acc {scores[m]['vs_run16']['accuracy']:.3f}"
            if scores[m].get("vs_hand_rule"):
                t += f"\nvs hand labels: acc {scores[m]['vs_hand_rule']['accuracy']:.3f}"
        panels.append((t, "cls", np.where(preds[m] == 255, 0, preds[m].astype(int) + 1)))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 8.4), constrained_layout=True)
    for a, (t, kind, img) in zip(ax, panels):
        if img is None:
            a.set_title(t + "\n(missing)", fontsize=10); a.axis("off"); continue
        if kind == "rgb":
            a.imshow(img, interpolation="nearest")
        else:
            a.imshow(img, cmap=cmap, vmin=-0.5, vmax=6.5, interpolation="nearest")
        a.contour(ZONE.astype(float), levels=[0.5], colors="yellow", linewidths=0.5)
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLOURS, NAMES)]
               + [Line2D([], [], color="yellow", lw=1.5, label="30 m water zone (JRC >= 5 %): river / ice only")],
               loc="outside lower center", ncol=7, fontsize=9, frameon=False)
    kind = "left-out test day" if labelled else "no run 16 label day: final models"
    fig.suptitle(f"Run 21: VIIRS upsampled to 30 m, six classes, {day['date']} ({kind})", fontsize=13)
    fig.savefig(OUT / "images" / f"pred_{day['date']}.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)


def conf_image(model, c, title):
    c = np.asarray(c, float)
    rowp = np.divide(c, c.sum(1, keepdims=True), out=np.zeros_like(c), where=c.sum(1, keepdims=True) > 0)
    fig, a = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    a.imshow(rowp, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            a.text(j, i, f"{int(c[i, j]):,}\n{100 * rowp[i, j]:.0f}%", ha="center", va="center", fontsize=7.5,
                   color="white" if rowp[i, j] > 0.55 else "black")
    short = ["river", "ice", "snow", "forest", "shrub", "bare"]
    a.set_xticks(range(6), short); a.set_yticks(range(6), short)
    a.set_xlabel("predicted"); a.set_ylabel("true (run 16 map)"); a.set_title(title, fontsize=10)
    f = OUT / "images" / f"conf_{model}.png"
    fig.savefig(f, dpi=140); plt.close(fig)
    return f


# ----------------------------------------------------------------------------- main

def text_block(name, r):
    s = r["vs_run16"]
    L_ = [f"=== {name}: leave-one-date-out, {s['pixels']:,} pixels (vs run 16 map)",
          f"accuracy {s['accuracy']:.3f}  kappa {s['kappa']:.3f}  macro F1 {s['macro_f1']:.3f}  "
          f"balanced acc {s['balanced_accuracy']:.3f}",
          f"{'class':<22}{'F1':>6}{'prec':>6}{'rec':>6}{'true':>11}{'pred':>11}"]
    for k, v in s["per_class"].items():
        L_.append(f"{k:<22}{v['f1']:>6.2f}{v['precision']:>6.2f}{v['recall']:>6.2f}{v['true']:>11,}{v['pred']:>11,}")
    L_.append("confusion (rows true, cols predicted): " + str(s["confusion"]))
    for lab in ("vs_hand_rule", "vs_hand_original"):
        h = r.get(lab)
        if h:
            L_.append(f"{lab}: {h['pixels']:,} px  accuracy {h['accuracy']:.3f}  macro F1 {h['macro_f1']:.3f}  "
                      f"kappa {h['kappa']:.3f}")
    L_.append(f"{'date':<12}{'acc':>7}{'F1':>7}{'hand acc':>10}{'375m acc50':>12}{'acc90':>7}  part MAE pp water/ice/snow/land")
    for d, v in r["per_date"].items():
        hh = v.get("vs_hand_rule")
        L_.append(f"{d:<12}{v['vs_run16']['accuracy']:>7.3f}{v['vs_run16']['macro_f1']:>7.3f}"
                  f"{(hh['accuracy'] if hh else float('nan')):>10.3f}{v['at_375m']['rule50']:>12.3f}"
                  f"{v['at_375m']['rule90']:>7.3f}  "
                  + " / ".join(f"{v['at_375m']['parts']['mae_pp'][k]:.1f}" for k in V.PARTS))
    if r.get("inputs"):
        L_.append("RF inputs (final model): " + json.dumps(r["inputs"]))
    return "\n".join(L_)


MODEL_CLASSES = {"rf": ZoneRF, "unetpp": ZoneUNet}
LONG = {"rf": "Random Forest", "unetpp": "UNet++"}


def read_pred(path):
    a = gdal.Open(str(path)).ReadAsArray().astype(np.int16) - 1
    return np.where(a < 0, 255, a).astype(np.uint8)


def train_model(model, days, labelled, viirs_only, hand):
    cls = MODEL_CLASSES[model]
    (OUT / model / "viirs_only").mkdir(parents=True, exist_ok=True)
    conf, ch_rule, ch_orig = np.zeros((6, 6), int), np.zeros((6, 6), int), np.zeros((6, 6), int)
    per_date = {}
    for test in labelled:
        day = days[test]
        pred = cls().fit(days, [d for d in labelled if d != test]).predict(day)
        D.write_prediction(OUT / model / f"pred_{test}.tif", np.where(pred == 255, 0, pred + 1).astype(np.uint8), day["ref"])
        shutil.copy(QML, OUT / model / f"pred_{test}.qml")
        v = day["valid"]
        c = confusion(day["y"][v], pred[v]); conf += c
        per_date[test] = {"vs_run16": summarise(c), "at_375m": score_375(day, pred)}
        if test in hand:
            yh = hand[test]; m = (yh != D.IGNORE) & v
            yr = W.rule_labels(yh, ZONE); mr = (yr != D.IGNORE) & v
            cr, co = confusion(yr[mr], pred[mr]), confusion(yh[m], pred[m])
            ch_rule += cr; ch_orig += co
            per_date[test]["vs_hand_rule"] = summarise(cr); per_date[test]["vs_hand_original"] = summarise(co)
        print(f"{model:<7}{test}  acc {per_date[test]['vs_run16']['accuracy']:.3f}  "
              f"F1 {per_date[test]['vs_run16']['macro_f1']:.3f}  375m acc50 {per_date[test]['at_375m']['rule50']:.3f}",
              flush=True)
    res = {"vs_run16": summarise(conf), "per_date": per_date}
    if ch_rule.sum():
        res["vs_hand_rule"] = summarise(ch_rule); res["vs_hand_original"] = summarise(ch_orig)
    final = cls().fit(days, labelled)
    if model == "rf":
        res["inputs"] = final.inputs(days[labelled[0]]["names"])
    for d in viirs_only:
        pred = final.predict(days[d])
        D.write_prediction(OUT / model / "viirs_only" / f"pred_{d}.tif",
                           np.where(pred == 255, 0, pred + 1).astype(np.uint8), days[d]["ref"])
        shutil.copy(QML, OUT / model / "viirs_only" / f"pred_{d}.qml")
    res["conf_image"] = str(conf_image(model, conf, f"Run 21 {LONG[model]}: 30 m, all left-out days vs run 16 map "
                                                    f"(acc {res['vs_run16']['accuracy']:.3f})"))
    (OUT / model / "metrics.txt").write_text(text_block(LONG[model], res) + "\n")
    print(text_block(LONG[model], res), flush=True)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "unetpp"], choices=list(MODEL_CLASSES))
    ap.add_argument("--images-only", action="store_true", help="redraw images from saved prediction tifs")
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    shifts = {r["date"]: r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text())}
    labelled = [d for d in U.viirs_dates()
                if (U.RUN16 / f"pred_{d}.tif").exists() or (U.RUN16 / "new_dates" / d / "rf_pred.tif").exists()]
    viirs_only = [d for d in U.viirs_dates() if d not in labelled]
    cache, days = {}, {}
    for date in labelled + viirs_only:
        days[date] = load_day(date, shifts.get(date, [0, 0]), cache)
        print(date, "valid px", int(days[date]["valid"].sum()), "inputs", len(days[date]["names"]), flush=True)
    hand = hand_labels()
    print("hand-labelled px:", {d: int((y != D.IGNORE).sum()) for d, y in hand.items() if d in labelled}, flush=True)

    mpath = OUT / "metrics_all.json"
    results = json.loads(mpath.read_text()) if mpath.exists() else {}
    if not args.images_only:
        for model in args.models:
            results[model] = train_model(model, days, labelled, viirs_only, hand)
            mpath.write_text(json.dumps(results, indent=1))          # saved after each model
    order = [m for m in ("rf", "unetpp") if m in results]
    (OUT / "summary.txt").write_text(
        "Run 21: VIIRS upsampled to 30 m, six run 16 classes with the zone rule, leave-one-date-out over "
        + ", ".join(labelled) + "\n\n" + "\n\n".join(text_block(LONG[m], results[m]) for m in order) + "\n")
    for d in days:
        preds = {}
        for m in order:
            f = OUT / m / (f"pred_{d}.tif" if d in labelled else f"viirs_only/pred_{d}.tif")
            if f.exists():
                preds[m] = read_pred(f)
        sc = {m: results[m]["per_date"][d] for m in order} if d in labelled else None
        date_image(days[d], preds, sc, d in labelled, hand)
    print("wrote", OUT, flush=True)
