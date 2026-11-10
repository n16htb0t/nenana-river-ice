"""
Which input groups give the best six-class Random Forest? Greedy forward search over groups
of inputs, every combination scored by the same 5-fold spatial cross-validation on the hand
labels (rf_six_classes.py labels). Each tried combination is logged with per-class F1, so it
also shows which inputs help river, ice, snow and each land type.

Groups (from the rf_terrain.py inputs: band pairs, indices, textures, NLCD, terrain, context):
  bands, thermal, nd_pairs, diff_pairs, veg_soil, water_idx, brightness, tasseled,
  tex3, tex7, nlcd, terrain, dist_water, context

Search: start empty; each round add the group that raises macro F1 most; stop when no group
adds at least MIN_GAIN. Search forest: 200 trees, max_features 0.1, min_samples_leaf 5.
The best combination is then refitted with 500 trees on every dataset and maps every date.

Outputs: output/11_feature_search/<dataset folder>/six_class/ (search.txt, metrics.txt,
oof.npz, pred_<date>.tif, new_dates/, images/)

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_feature_search.py
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import data as D
import rf_all_bands as R
import rf_six_classes as S
import rf_terrain as T

OUT = D.HERE / "output" / "11_feature_search"
MIN_GAIN = 0.002
SEARCH_RF = dict(n_estimators=200, max_features=0.1, min_samples_leaf=5)
FINAL_RF = dict(n_estimators=500, max_features=0.1, min_samples_leaf=5)
BANDS = {"blue", "green", "red", "nir", "swir1", "swir2"}


def group_of(n):
    if n in BANDS: return "bands"
    if n == "thermal": return "thermal"
    if n.startswith("nd_"): return "nd_pairs"
    if n.startswith("diff_"): return "diff_pairs"
    if n in ("EVI", "SAVI", "MSAVI", "BSI"): return "veg_soil"
    if n.startswith("AWEI"): return "water_idx"
    if n in ("vis_brightness", "nir_plus_swir1"): return "brightness"
    if n.startswith("tc_"): return "tasseled"
    if n.endswith(("_mean3", "_std3")): return "tex3"
    if n.endswith(("_mean7", "_std7")): return "tex7"
    if n.startswith("nlcd_"): return "nlcd"
    if n == "dist_nlcd_water_m": return "dist_water"
    if n in ("elev_m", "slope_deg", "northness", "eastness") or n.startswith("rel_elev"): return "terrain"
    if "_ctx" in n: return "context"
    raise ValueError(n)


def rf(params, seed=0):
    return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **params)


def cv(px, cols, params):
    yt, yp, dt, rr, cc = [], [], [], [], []
    for k in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(R.H, k)
        xtr = np.concatenate([p["x"][train_rows[p["r"]]][:, cols] for p in px])
        ytr = np.concatenate([p["y"][train_rows[p["r"]]] for p in px])
        m = rf(params, k).fit(xtr, ytr)
        for p in px:
            t = test_rows[p["r"]]
            if t.any():
                yt.append(p["y"][t]); yp.append(m.predict(p["x"][t][:, cols]))
                dt += [p["date"]] * int(t.sum()); rr.append(p["r"][t]); cc.append(p["c"][t])
    return np.concatenate(yt), np.concatenate(yp), np.array(dt), np.concatenate(rr), np.concatenate(cc)


def per_class_f1(y, p):
    out = []
    for i in range(len(D.CLASSES)):
        tp = np.sum((y == i) & (p == i)); pr = tp / max(np.sum(p == i), 1); rc = tp / max(np.sum(y == i), 1)
        out.append(2 * pr * rc / max(pr + rc, 1e-9))
    return out


def load(dataset):
    D.configure("four", dataset)
    scenes = D.load_all()
    S.prepare(scenes)
    R.H = scenes[0]["y"].shape[0]
    px, names = R.labelled_pixels(scenes)
    return scenes, px, names


def search(px, names, log, out):
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(group_of(n), []).append(i)
    cls = list(D.CLASSES.values())
    log.append("groups: " + ", ".join(f"{g} ({len(v)})" for g, v in groups.items()))
    log.append(f"{'combination':70} {'macroF1':>7} {'acc':>6}  " + " ".join(f"{c[:8]:>8}" for c in cls))
    chosen, best_f1, best_class = [], 0.0, {c: (0.0, "") for c in cls}
    while True:
        round_best = None
        for g in groups:
            if g in chosen:
                continue
            combo = chosen + [g]
            cols = sorted(i for gg in combo for i in groups[gg])
            t0 = time.time()
            y, p, *_ = cv(px, cols, SEARCH_RF)
            f1s = per_class_f1(y, p); f1 = float(np.mean(f1s)); acc = float((y == p).mean())
            log.append(f"{'+'.join(combo):70} {f1:7.3f} {acc:6.3f}  " + " ".join(f"{v:8.3f}" for v in f1s)
                       + f"  ({time.time() - t0:.0f} s)")
            print(log[-1])
            (out / "search.txt").write_text("\n".join(log) + "\n")
            for c, v in zip(cls, f1s):
                if v > best_class[c][0]:
                    best_class[c] = (v, "+".join(combo))
            if round_best is None or f1 > round_best[0]:
                round_best = (f1, g)
        if round_best is None or round_best[0] < best_f1 + MIN_GAIN:
            break
        best_f1 = round_best[0]; chosen.append(round_best[1])
        log.append(f"--> keep {round_best[1]}: {'+'.join(chosen)}  macro F1 {best_f1:.3f}")
        print(log[-1])
    log += ["", f"best combination: {'+'.join(chosen)}  macro F1 {best_f1:.3f}", "",
            "best F1 per class seen in the search (and the combination that gave it):"]
    log += [f"  {c:22} {v:.3f}  {combo}" for c, (v, combo) in best_class.items()]
    (out / "search.txt").write_text("\n".join(log) + "\n")
    return chosen, groups


def final(dataset, scenes, px, names, chosen, out):
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(group_of(n), []).append(i)
    use = [g for g in chosen if g in groups]
    cols = sorted(i for g in use for i in groups[g])
    blocks = [f"inputs: {'+'.join(use)} ({len(cols)} inputs); forest {FINAL_RF}"]
    if dataset == "external_daksh":
        train = [p for p in px if p["group"] == "mathus_naira"]; test = [p for p in px if p["group"] == "daksh"]
        model = rf(FINAL_RF).fit(np.concatenate([p["x"][:, cols] for p in train]), np.concatenate([p["y"] for p in train]))
        yt = np.concatenate([p["y"] for p in test]); yp = model.predict(np.concatenate([p["x"][:, cols] for p in test]))
        D.save_oof(out / "oof.npz", [p["date"] for p in test for _ in p["y"]],
                   np.concatenate([p["r"] for p in test]), np.concatenate([p["c"] for p in test]), yt, yp)
        blocks.insert(0, D.report(yt, yp, "best input combination — external test on 2024-03-12")[0])
    else:
        yt, yp, dt, rr, cc = cv(px, cols, FINAL_RF)
        D.save_oof(out / "oof.npz", dt, rr, cc, yt, yp)
        blocks.insert(0, D.report(yt, yp, "best input combination — 5-fold spatial CV, all labelled dates")[0])
        for date in sorted(set(dt)):
            m = dt == date
            blocks.append(D.report(yt[m], yp[m], f"--- {date}")[0])
        model = rf(FINAL_RF).fit(np.concatenate([p["x"][:, cols] for p in px]), np.concatenate([p["y"] for p in px]))
    imp = sorted(zip(model.feature_importances_, [names[i] for i in cols]), reverse=True)
    blocks.append("input importance\n" + "\n".join(f"  {n:24} {v:.4f}" for v, n in imp))
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print(blocks[0])

    def predict(x):
        fx, _ = R.features(x)
        fx = fx[cols]
        valid = np.isfinite(fx).all(0)
        pred = np.zeros(valid.shape, np.uint8)
        idx = np.flatnonzero(valid); flat = fx.reshape(len(fx), -1)
        for i in range(0, len(idx), 400_000):
            j = idx[i:i + 400_000]
            pred.flat[j] = model.predict(flat[:, j].T) + 1
        return pred

    for s in scenes:
        pred = predict(s["x"]); pred[~s["valid"]] = 0
        D.write_prediction(out / f"pred_{s['date']}.tif", pred, s["ref"])
        img = R.I.image_for(s["date"])
        if img is not None:
            R.jpeg(img, pred, f"{s['date']}   best inputs: {'+'.join(use)}   {D.DATASET_DIRS[dataset]}",
                   out / "images" / f"best_{s['date']}.jpg")
    if dataset != "external_daksh":
        for date in R.NEW_DATES:
            x, ref = D.load_features(R.NEW / f"nenana_{date.replace('-', '')}", date)
            pred = predict(x)
            d = out / "new_dates" / date; d.mkdir(parents=True, exist_ok=True)
            D.write_prediction(d / "best_pred.tif", pred, ref)
            R.jpeg(R.I.image_for(date), pred, f"{date} (no labels)   best inputs   {D.DATASET_DIRS[dataset]}",
                   out / "images" / f"new_best_{date}.jpg")
    print("wrote", out)


if __name__ == "__main__":
    R.EXTRA = T.extra
    chosen = None
    for ds in ("exclude_daksh", "with_daksh", "external_daksh"):
        out = OUT / D.DATASET_DIRS[ds] / "six_class"
        (out / "images").mkdir(parents=True, exist_ok=True)
        scenes, px, names = load(ds)
        if ds == "exclude_daksh":
            log = [f"forward search over input groups, {ds}, 5-fold spatial CV, forest {SEARCH_RF}"]
            chosen, _ = search(px, names, log, out)
        final(ds, scenes, px, names, chosen, out)
