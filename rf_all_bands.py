"""
Random Forest, four classes, with every band set against every other band.

Inputs per pixel (no water mask, no JRC water occurrence):
  - raw bands: blue, green, red, NIR, SWIR1 (+ SWIR2 and surface temperature in exclude_daksh)
  - every pair of reflectance bands: normalized difference (a-b)/(a+b) and difference a-b
    (a ratio a/b is left out: it orders pixels exactly like the normalized difference, so a
    tree gains nothing from it)
  - named indices: EVI, SAVI, MSAVI, BSI, visible brightness, NIR+SWIR; with SWIR2 also
    AWEI_nsh, AWEI_sh and Landsat 8 tasseled-cap brightness / greenness / wetness
  - texture: mean and standard deviation in 3x3 and 7x7 windows of NDVI, green/SWIR1
    (NDSI), green/NIR (NDWI), NIR and blue

Forest settings are chosen by 5-fold spatial cross-validation (same folds as train_rf.py);
the scores reported are those cross-validated ones. The external test trains on the patch
dates with the with_daksh settings and scores on 2024-03-12.

Outputs in output/5_rf_all_bands/<dataset folder>/four_class_river_ice_snow_land/random_forest/:
  metrics.txt (tuning table, scores, confusion, per date, input importance), oof.npz,
  pred_<date>.tif, new_dates/<date>/rf_pred.tif, images/*.jpg

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_all_bands.py
"""

import itertools, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
import data as D
import images as I

OUT = D.HERE / "output" / "5_rf_all_bands"
NEW = Path.home() / "Desktop/Patches"
NEW_DATES = ["2026-03-26", "2023-05-05"]
CONFIGS = [dict(max_features="sqrt", min_samples_leaf=1),
           dict(max_features="sqrt", min_samples_leaf=2),
           dict(max_features="sqrt", min_samples_leaf=5),
           dict(max_features=0.3, min_samples_leaf=2),
           dict(max_features="log2", min_samples_leaf=2)]
N_TREES = 500
EPS = 1e-6
H = None                                       # image rows, set in run()
EXTRA = None                                   # optional (x, date) -> [(name, (H, W) array)], extra inputs


def raw_bands(x):
    """Named reflectance bands (and thermal) from a D.load_features array."""
    ch = D.CHANNELS
    if "SR_B2" in ch:
        names = ["blue", "green", "red", "nir", "swir1", "swir2"]
        b = {n: x[ch.index(f"SR_B{i + 2}")] for i, n in enumerate(names)}
        b["thermal"] = x[ch.index("ST_B10")]
    else:
        b = {n: x[ch.index(n)] for n in ["blue", "green", "red", "nir", "swir1"]}
    return b


def texture(a, size):
    """Mean and standard deviation in a size x size window, ignoring missing pixels."""
    ok = np.isfinite(a)
    v = np.where(ok, a, 0).astype(np.float64)
    w = ndimage.uniform_filter(ok.astype(np.float64), size)
    m = ndimage.uniform_filter(v, size) / np.maximum(w, EPS)
    m2 = ndimage.uniform_filter(v * v, size) / np.maximum(w, EPS)
    sd = np.sqrt(np.maximum(m2 - m * m, 0))
    m[~ok] = np.nan; sd[~ok] = np.nan
    return m.astype(np.float32), sd.astype(np.float32)


def features(x, date=None):
    """(F, H, W) float32 features and their names, from one date's D.load_features array."""
    b = raw_bands(x)
    refl = [n for n in ["blue", "green", "red", "nir", "swir1", "swir2"] if n in b]
    f, names = [], []

    def add(name, arr):
        f.append(arr.astype(np.float32)); names.append(name)

    for n in refl + (["thermal"] if "thermal" in b else []):
        add(n, b[n])
    for a, c in itertools.combinations(refl, 2):
        add(f"nd_{a}_{c}", (b[a] - b[c]) / (b[a] + b[c] + EPS))
        add(f"diff_{a}_{c}", b[a] - b[c])
    B, G, R, N, S1 = (b[n] for n in ["blue", "green", "red", "nir", "swir1"])
    add("EVI", 2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1))
    add("SAVI", 1.5 * (N - R) / (N + R + 0.5))
    add("MSAVI", (2 * N + 1 - np.sqrt(np.maximum((2 * N + 1) ** 2 - 8 * (N - R), 0))) / 2)
    add("BSI", ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + EPS))
    add("vis_brightness", (B + G + R) / 3)
    add("nir_plus_swir1", N + S1)
    if "swir2" in b:
        S2 = b["swir2"]
        add("AWEI_nsh", 4 * (G - S1) - (0.25 * N + 2.75 * S2))
        add("AWEI_sh", B + 2.5 * G - 1.5 * (N + S1) - 0.25 * S2)
        bands6 = np.stack([B, G, R, N, S1, S2])                      # Baig et al. 2014, Landsat 8
        for name, c in (("tc_brightness", [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872]),
                        ("tc_greenness", [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608]),
                        ("tc_wetness", [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559])):
            add(name, np.tensordot(np.array(c, np.float32), bands6, 1))
    base = {"NDVI": (N - R) / (N + R + EPS), "NDSI": (G - S1) / (G + S1 + EPS),
            "NDWI": (G - N) / (G + N + EPS), "nir": N, "blue": B}
    for n, a in base.items():
        for size in (3, 7):
            m, sd = texture(a, size)
            add(f"{n}_mean{size}", m); add(f"{n}_std{size}", sd)
    if EXTRA:
        for n, a in EXTRA(x, date):
            add(n, a)
    return np.stack(f), names


def labelled_pixels(scenes):
    """Per labelled date: features (n, F), labels, rows, cols; and the feature names."""
    out, names = [], None
    for s in scenes:
        m = s["y"] != D.IGNORE
        if not m.any():
            continue
        fx, names = features(s["x"], s["date"])
        m &= np.isfinite(fx).all(0)
        rr, cc = np.nonzero(m)
        out.append(dict(date=s["date"], group=s["group"], x=fx[:, m].T, y=s["y"][m], r=rr, c=cc))
    return out, names


def make_rf(cfg, seed=0):
    return RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed, **cfg)


def spatial_cv(px, cfg):
    yt, yp, dt, rr, cc = [], [], [], [], []
    for k in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(H, k)
        tr_x = [p["x"][train_rows[p["r"]]] for p in px]
        tr_y = [p["y"][train_rows[p["r"]]] for p in px]
        rf = make_rf(cfg, k).fit(np.concatenate(tr_x), np.concatenate(tr_y))
        for p in px:
            t = test_rows[p["r"]]
            if t.any():
                yt.append(p["y"][t]); yp.append(rf.predict(p["x"][t]))
                dt += [p["date"]] * int(t.sum()); rr.append(p["r"][t]); cc.append(p["c"][t])
    return (np.concatenate(yt), np.concatenate(yp), np.array(dt),
            np.concatenate(rr), np.concatenate(cc))


def macro_f1(y, p):
    return float(D.report(y, p, "")[0].split("macro F1 ")[1].split()[0])


def predict_map(rf, x, date=None):
    fx, _ = features(x, date)
    valid = np.isfinite(fx).all(0)
    pred = np.zeros(valid.shape, np.uint8)
    idx = np.flatnonzero(valid)
    flat = fx.reshape(len(fx), -1)
    for i in range(0, len(idx), 400_000):
        j = idx[i:i + 400_000]
        pred.flat[j] = rf.predict(flat[:, j].T) + 1
    return pred


def jpeg(img, pred, title, path):
    I.panel(img, pred, I.rgb(D.COLOURS), D.CLASSES, title, path)


def run(dataset, cfg=None, out_root=OUT, task_dir="four_class_river_ice_snow_land",
        prepare=None, configs=CONFIGS):
    """Tune, cross-validate, fit and map one dataset. prepare(scenes) may relabel the
    scenes and set D.CLASSES / D.COLOURS (rf_six_classes.py does)."""
    global H
    D.configure("four", dataset)
    out = out_root / D.DATASET_DIRS[dataset] / task_dir / "random_forest"
    (out / "images").mkdir(parents=True, exist_ok=True)
    scenes = D.load_all()
    if prepare:
        prepare(scenes)
    H = scenes[0]["y"].shape[0]
    px, names = labelled_pixels(scenes)
    print(f"\n=== {dataset}: {len(names)} inputs, dates {', '.join(p['date'] for p in px)}")
    blocks = []

    if dataset == "external_daksh":
        train = [p for p in px if p["group"] == "mathus_naira"]
        test = [p for p in px if p["group"] == "daksh"]
        rf = make_rf(cfg).fit(np.concatenate([p["x"] for p in train]), np.concatenate([p["y"] for p in train]))
        yt = np.concatenate([p["y"] for p in test]); yp = rf.predict(np.concatenate([p["x"] for p in test]))
        D.save_oof(out / "oof.npz", [p["date"] for p in test for _ in p["y"]],
                   np.concatenate([p["r"] for p in test]), np.concatenate([p["c"] for p in test]), yt, yp)
        blocks.append(D.report(yt, yp, f"Random Forest, all bands — external test: trained on "
                               f"{', '.join(p['date'] for p in train)}, scored on 2024-03-12")[0])
        blocks.append(f"settings (from with_daksh tuning): {cfg}, {N_TREES} trees")
    else:
        tune = ["tuning (5-fold spatial CV, macro F1 / accuracy)"]
        best = None
        for c in configs:
            t0 = time.time()
            res = spatial_cv(px, c)
            f1 = macro_f1(res[0], res[1]); acc = float((res[0] == res[1]).mean())
            tune.append(f"  {str(c):50} macro F1 {f1:.3f}  accuracy {acc:.3f}  ({time.time() - t0:.0f} s)")
            print(tune[-1])
            if best is None or f1 > best[0]:
                best = (f1, c, res)
        _, cfg, (yt, yp, dt, rr, cc) = best
        D.save_oof(out / "oof.npz", dt, rr, cc, yt, yp)
        blocks.append(D.report(yt, yp, f"Random Forest, all bands — 5-fold spatial CV, all labelled dates "
                               f"(best settings {cfg}, {N_TREES} trees)")[0])
        for date in sorted(set(dt)):
            m = dt == date
            blocks.append(D.report(yt[m], yp[m], f"--- {date}")[0])
        blocks.append("\n".join(tune))
        rf = make_rf(cfg).fit(np.concatenate([p["x"] for p in px]), np.concatenate([p["y"] for p in px]))

    imp = sorted(zip(rf.feature_importances_, names), reverse=True)
    blocks.append(f"input importance (final model, {len(names)} inputs)\n" +
                  "\n".join(f"  {n:22} {v:.4f}" for v, n in imp))
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print(blocks[0])

    for s in scenes:
        pred = predict_map(rf, s["x"], s["date"]); pred[~s["valid"]] = 0
        D.write_prediction(out / f"pred_{s['date']}.tif", pred, s["ref"])
        img = I.image_for(s["date"])
        if img is not None:
            jpeg(img, pred, f"{s['date']}   Random Forest, all bands   {D.DATASET_DIRS[dataset]}",
                 out / "images" / f"rf_{s['date']}.jpg")
    if dataset != "external_daksh":
        for date in NEW_DATES:
            x, ref = D.load_features(NEW / f"nenana_{date.replace('-', '')}", date)
            pred = predict_map(rf, x, date)
            d = out / "new_dates" / date; d.mkdir(parents=True, exist_ok=True)
            D.write_prediction(d / "rf_pred.tif", pred, ref)
            jpeg(I.image_for(date), pred, f"{date} (no labels)   Random Forest, all bands   {D.DATASET_DIRS[dataset]}",
                 out / "images" / f"new_rf_{date}.jpg")
    print("wrote", out)
    return cfg


if __name__ == "__main__":
    run("exclude_daksh")
    cfg = run("with_daksh")
    run("external_daksh", cfg)
