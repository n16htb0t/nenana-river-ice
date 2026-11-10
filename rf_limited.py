"""
Six-class Random Forest with limited training parameters, so the model cannot memorise the
training pixels: few inputs and small trees.

Input pool: rf_angles.py (all bands, band pairs, indices, textures + per-pixel illumination
angles + NLCD land cover), without the per-date sun elevation / azimuth: those are one
value per date, so with 4-5 dates they mostly tell the forest which date a pixel is from. Limits searched by 5-fold spatial cross-validation:
  top_k      10 / 15 / 20 / 25 inputs, ranked by a quick forest on each fold's training
             pixels only (the test fold never picks the inputs)
  max_depth  8 / 12 / 16, min_samples_leaf 20, max_features sqrt, 300 trees
Each setting is reported with training accuracy next to the cross-validated accuracy:
a small gap means little overfitting.

Outputs: output/13_rf_six_classes_limited/<dataset folder>/six_class/random_forest/
(search.txt, metrics.txt with the chosen inputs, oof.npz, maps, images)

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_limited.py
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import data as D
import rf_all_bands as R
import rf_six_classes as S
import rf_angles as A

OUT = D.HERE / "output" / "13_rf_six_classes_limited"
TOP_K = (10, 15, 20, 25)
DEPTHS = (8, 12, 16)
LEAF = 20


def forest(depth, trees=300, seed=0):
    return RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=LEAF, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def rank(x, y, seed=0):
    """Input ranking from a quick forest on the given (training) pixels."""
    return np.argsort(forest(None, trees=100, seed=seed).fit(x, y).feature_importances_)[::-1]


def cv(px, k, depth):
    yt, yp, dt, rr, cc, gaps = [], [], [], [], [], []
    for f in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(R.H, f)
        xtr = np.concatenate([p["x"][train_rows[p["r"]]] for p in px])
        ytr = np.concatenate([p["y"][train_rows[p["r"]]] for p in px])
        keep = rank(xtr, ytr, f)[:k]
        m = forest(depth, seed=f).fit(xtr[:, keep], ytr)
        gaps.append(float((m.predict(xtr[:, keep]) == ytr).mean()))
        for p in px:
            t = test_rows[p["r"]]
            if t.any():
                yt.append(p["y"][t]); yp.append(m.predict(p["x"][t][:, keep]))
                dt += [p["date"]] * int(t.sum()); rr.append(p["r"][t]); cc.append(p["c"][t])
    return (np.concatenate(yt), np.concatenate(yp), np.array(dt), np.concatenate(rr), np.concatenate(cc),
            float(np.mean(gaps)))


def load(dataset):
    D.configure("four", dataset)
    scenes = D.load_all()
    S.prepare(scenes)
    R.H = scenes[0]["y"].shape[0]
    px, names = R.labelled_pixels(scenes)
    return scenes, px, names


def write_maps(model, keep, scenes, out, dataset, title):
    def predict(x, date):
        fx, _ = R.features(x, date)
        fx = fx[keep]
        valid = np.isfinite(fx).all(0)
        pred = np.zeros(valid.shape, np.uint8)
        idx = np.flatnonzero(valid); flat = fx.reshape(len(fx), -1)
        for i in range(0, len(idx), 400_000):
            j = idx[i:i + 400_000]
            pred.flat[j] = model.predict(flat[:, j].T) + 1
        return pred
    for s in scenes:
        pred = predict(s["x"], s["date"]); pred[~s["valid"]] = 0
        D.write_prediction(out / f"pred_{s['date']}.tif", pred, s["ref"])
        img = R.I.image_for(s["date"])
        if img is not None:
            R.jpeg(img, pred, f"{s['date']}   {title}   {D.DATASET_DIRS[dataset]}", out / "images" / f"rf_{s['date']}.jpg")
    if dataset != "external_daksh":
        for date in R.NEW_DATES:
            x, ref = D.load_features(R.NEW / f"nenana_{date.replace('-', '')}", date)
            pred = predict(x, date)
            d = out / "new_dates" / date; d.mkdir(parents=True, exist_ok=True)
            D.write_prediction(d / "rf_pred.tif", pred, ref)
            R.jpeg(R.I.image_for(date), pred, f"{date} (no labels)   {title}   {D.DATASET_DIRS[dataset]}",
                   out / "images" / f"new_rf_{date}.jpg")


def run(dataset, setting=None):
    out = OUT / D.DATASET_DIRS[dataset] / "six_class" / "random_forest"
    (out / "images").mkdir(parents=True, exist_ok=True)
    scenes, px, names = load(dataset)
    print(f"\n=== {dataset}: pool of {len(names)} inputs")
    blocks = []
    if dataset == "external_daksh":
        k, depth = setting
        train = [p for p in px if p["group"] == "mathus_naira"]; test = [p for p in px if p["group"] == "daksh"]
        x = np.concatenate([p["x"] for p in train]); y = np.concatenate([p["y"] for p in train])
        keep = rank(x, y)[:k]
        model = forest(depth).fit(x[:, keep], y)
        yt = np.concatenate([p["y"] for p in test]); yp = model.predict(np.concatenate([p["x"] for p in test])[:, keep])
        D.save_oof(out / "oof.npz", [p["date"] for p in test for _ in p["y"]],
                   np.concatenate([p["r"] for p in test]), np.concatenate([p["c"] for p in test]), yt, yp)
        blocks.append(D.report(yt, yp, f"Random Forest, limited (top {k} inputs, depth {depth}) — external test on 2024-03-12")[0])
        blocks.append(f"training accuracy {float((model.predict(x[:, keep]) == y).mean()):.3f}")
    else:
        log = [f"limited search on {dataset}: 5-fold spatial CV, six classes, min_samples_leaf {LEAF}, sqrt, 300 trees",
               f"{'top_k':>5} {'depth':>5}  {'CV acc':>6} {'macroF1':>7} {'train acc':>9} {'gap':>6}"]
        best = None
        for k in TOP_K:
            for depth in DEPTHS:
                t0 = time.time()
                yt, yp, dt, rr, cc, tr_acc = cv(px, k, depth)
                acc = float((yt == yp).mean()); f1 = R.macro_f1(yt, yp)
                log.append(f"{k:5d} {depth:5d}  {acc:6.3f} {f1:7.3f} {tr_acc:9.3f} {tr_acc - acc:6.3f}  ({time.time() - t0:.0f} s)")
                print(log[-1])
                (out / "search.txt").write_text("\n".join(log) + "\n")
                if best is None or f1 > best[0]:
                    best = (f1, (k, depth), (yt, yp, dt, rr, cc))
        f1, setting, (yt, yp, dt, rr, cc) = best
        k, depth = setting
        log += ["", f"best: top {k} inputs, depth {depth}, macro F1 {f1:.3f}"]
        (out / "search.txt").write_text("\n".join(log) + "\n")
        D.save_oof(out / "oof.npz", dt, rr, cc, yt, yp)
        blocks.append(D.report(yt, yp, f"Random Forest, limited (top {k} inputs, depth {depth}, leaf {LEAF}) — 5-fold spatial CV")[0])
        for date in sorted(set(dt)):
            m = dt == date
            blocks.append(D.report(yt[m], yp[m], f"--- {date}")[0])
        blocks.append("\n".join(log))
        x = np.concatenate([p["x"] for p in px]); y = np.concatenate([p["y"] for p in px])
        keep = rank(x, y)[:k]
        model = forest(depth).fit(x[:, keep], y)
        blocks.append(f"final model training accuracy {float((model.predict(x[:, keep]) == y).mean()):.3f}")
    imp = sorted(zip(model.feature_importances_, [names[i] for i in keep]), reverse=True)
    blocks.append(f"inputs used ({len(keep)}) and importance\n" + "\n".join(f"  {n:24} {v:.4f}" for v, n in imp))
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print(blocks[0])
    write_maps(model, keep, scenes, out, dataset, f"RF limited (top {k}, depth {depth})")
    print("wrote", out)
    return setting


DATE_CONSTANTS = {"sun_elevation_deg", "sun_azimuth_deg"}


def extra(x, date=None):
    return [(n, a) for n, a in A.extra(x, date) if n not in DATE_CONSTANTS]


if __name__ == "__main__":
    R.EXTRA = extra
    run("exclude_daksh")
    setting = run("with_daksh")
    run("external_daksh", setting)
