"""
Search for the highest accuracy and macro F1 on the six-class task (rf_six_classes.py
labels, rf_all_bands.py inputs, no NLCD input, no water mask). Every method is scored by
the same 5-fold spatial cross-validation; nothing is tuned on a test fold.

  1. Random Forest grid: min_samples_leaf 5/10/20/40 x max_features sqrt/0.1/0.2
  2. best of 1 with class_weight "balanced" and None
  3. best forest on the top 20 / 35 inputs, ranked inside each fold's training data
  4. best forest + per-class score weights tuned on that fold's own out-of-bag pixels
  (Random Forest only; et() and hgb() are kept for reference but not searched)

The best method by macro F1 is refitted on all labels and maps every date.
Outputs: output/8_best_six_class/<dataset folder>/six_class/ (search.txt, metrics.txt,
oof.npz, pred_<date>.tif, new_dates/, images/); with --nlcd-input (NLCD land cover also
given as inputs) in output/9_best_six_class_nlcd_input/.

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_tune.py
"""

import argparse, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import data as D
import rf_all_bands as R
import rf_six_classes as S

OUT = D.HERE / "output" / "8_best_six_class"
TREES = 300


def rf(leaf, mf, cw="balanced_subsample", trees=TREES, seed=0, oob=False):
    return RandomForestClassifier(n_estimators=trees, min_samples_leaf=leaf, max_features=mf,
                                  class_weight=cw, n_jobs=-1, random_state=seed, oob_score=oob)


def et(leaf, mf, seed=0):
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=mf,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def hgb(seed=0):
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                                          l2_regularization=1.0, class_weight="balanced",
                                          early_stopping=False, random_state=seed)


def tune_weights(proba, y, k):
    """Per-class multipliers on class scores that maximise macro F1 (coordinate search)."""
    w = np.ones(k)
    best = R.macro_f1(y, (proba * w).argmax(1))
    for _ in range(2):
        for c in range(k):
            for m in (0.5, 0.7, 0.85, 1.2, 1.5, 2.0, 3.0):
                w2 = w.copy(); w2[c] *= m
                f = R.macro_f1(y, (proba * w2).argmax(1))
                if f > best:
                    best, w = f, w2
    return w


def fit_predict(method, xtr, ytr, xte, seed):
    kind = method["kind"]
    if kind == "rf":
        return rf(method["leaf"], method["mf"], method.get("cw", "balanced_subsample"), seed=seed).fit(xtr, ytr).predict(xte)
    if kind == "et":
        return et(method["leaf"], method["mf"], seed).fit(xtr, ytr).predict(xte)
    if kind == "hgb":
        return hgb(seed).fit(xtr, ytr).predict(xte)
    if kind == "topk":
        imp = rf(method["leaf"], method["mf"], trees=150, seed=seed).fit(xtr, ytr).feature_importances_
        keep = np.argsort(imp)[::-1][:method["k"]]
        return rf(method["leaf"], method["mf"], seed=seed).fit(xtr[:, keep], ytr).predict(xte[:, keep])
    if kind == "oobw":
        m = rf(method["leaf"], method["mf"], seed=seed, oob=True).fit(xtr, ytr)
        oob = np.nan_to_num(m.oob_decision_function_)
        w = tune_weights(oob, ytr, oob.shape[1])
        return (m.predict_proba(xte) * w).argmax(1)
    raise ValueError(kind)


def cv(px, method):
    yt, yp, dt, rr, cc = [], [], [], [], []
    for k in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(R.H, k)
        xtr = np.concatenate([p["x"][train_rows[p["r"]]] for p in px])
        ytr = np.concatenate([p["y"][train_rows[p["r"]]] for p in px])
        te = [(p, test_rows[p["r"]]) for p in px if test_rows[p["r"]].any()]
        xte = np.concatenate([p["x"][t] for p, t in te])
        pred = fit_predict(method, xtr, ytr, xte, k)
        i = 0
        for p, t in te:
            n = int(t.sum())
            yt.append(p["y"][t]); yp.append(pred[i:i + n]); i += n
            dt += [p["date"]] * n; rr.append(p["r"][t]); cc.append(p["c"][t])
    return (np.concatenate(yt), np.concatenate(yp), np.array(dt), np.concatenate(rr), np.concatenate(cc))


def name(m):
    return ", ".join(f"{k}={v}" for k, v in m.items())


def search(dataset):
    D.configure("four", dataset)
    scenes = D.load_all()
    S.prepare(scenes)
    R.H = scenes[0]["y"].shape[0]
    px, names = R.labelled_pixels(scenes)
    out = OUT / D.DATASET_DIRS[dataset] / "six_class"
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    log, results = [f"search on {dataset}: {len(names)} inputs, 5-fold spatial CV, six classes"], []

    def score(m):
        t0 = time.time()
        res = cv(px, m)
        f1 = R.macro_f1(res[0], res[1]); acc = float((res[0] == res[1]).mean())
        results.append((f1, acc, m, res))
        log.append(f"  {name(m):60} macro F1 {f1:.3f}  accuracy {acc:.3f}  ({time.time() - t0:.0f} s)")
        print(log[-1])
        (out / "search.txt").write_text("\n".join(log) + "\n")

    for leaf in (5, 10, 20, 40):
        for mf in ("sqrt", 0.1, 0.2):
            score(dict(kind="rf", leaf=leaf, mf=mf))
    b = max((r for r in results if r[2]["kind"] == "rf"), key=lambda r: r[0])[2]
    for cw in ("balanced", None):
        score(dict(b, cw=cw))
    b = max((r for r in results if r[2]["kind"] == "rf"), key=lambda r: r[0])[2]
    for k in (20, 35):
        score(dict(kind="topk", leaf=b["leaf"], mf=b["mf"], k=k))
    score(dict(kind="oobw", leaf=b["leaf"], mf=b["mf"]))

    f1, acc, best, (yt, yp, dt, rr, cc) = max(results, key=lambda r: r[0])
    log += ["", f"best: {name(best)}  macro F1 {f1:.3f}  accuracy {acc:.3f}"]
    (out / "search.txt").write_text("\n".join(log) + "\n")
    D.save_oof(out / "oof.npz", dt, rr, cc, yt, yp)
    blocks = [D.report(yt, yp, f"best six-class model ({name(best)}) — 5-fold spatial CV, all labelled dates")[0]]
    for date in sorted(set(dt)):
        m = dt == date
        blocks.append(D.report(yt[m], yp[m], f"--- {date}")[0])
    blocks.append("\n".join(log))
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print(blocks[0])
    return best, scenes, px, names, out


def final_maps(best, scenes, px, names, out, dataset):
    """Refit the best method on all labels; map every date (and the unlabelled dates)."""
    x = np.concatenate([p["x"] for p in px]); y = np.concatenate([p["y"] for p in px])
    keep, w = np.arange(len(names)), None
    kind = best["kind"]
    if kind == "topk":
        imp = rf(best["leaf"], best["mf"], trees=150).fit(x, y).feature_importances_
        keep = np.argsort(imp)[::-1][:best["k"]]
    if kind in ("rf", "topk"):
        model = rf(best["leaf"], best["mf"], best.get("cw", "balanced_subsample"), trees=500).fit(x[:, keep], y)
    elif kind == "et":
        model = et(best["leaf"], best["mf"]).fit(x, y)
    elif kind == "hgb":
        model = hgb().fit(x, y)
    else:
        model = rf(best["leaf"], best["mf"], trees=500, oob=True).fit(x, y)
        w = tune_weights(np.nan_to_num(model.oob_decision_function_), y, len(D.CLASSES))

    def predict(xx):
        fx, _ = R.features(xx)
        valid = np.isfinite(fx).all(0)
        pred = np.zeros(valid.shape, np.uint8)
        idx = np.flatnonzero(valid); flat = fx.reshape(len(fx), -1)[keep]
        for i in range(0, len(idx), 400_000):
            j = idx[i:i + 400_000]
            pred.flat[j] = ((model.predict_proba(flat[:, j].T) * w).argmax(1) if w is not None
                            else model.predict(flat[:, j].T)) + 1
        return pred

    for s in scenes:
        pred = predict(s["x"]); pred[~s["valid"]] = 0
        D.write_prediction(out / f"pred_{s['date']}.tif", pred, s["ref"])
        img = R.I.image_for(s["date"])
        if img is not None:
            R.jpeg(img, pred, f"{s['date']}   best six-class ({name(best)})   {D.DATASET_DIRS[dataset]}",
                   out / "images" / f"best_{s['date']}.jpg")
    for date in R.NEW_DATES:
        xx, ref = D.load_features(R.NEW / f"nenana_{date.replace('-', '')}", date)
        pred = predict(xx)
        d = out / "new_dates" / date; d.mkdir(parents=True, exist_ok=True)
        D.write_prediction(d / "best_pred.tif", pred, ref)
        R.jpeg(R.I.image_for(date), pred, f"{date} (no labels)   best six-class   {D.DATASET_DIRS[dataset]}",
               out / "images" / f"new_best_{date}.jpg")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlcd-input", action="store_true", help="also give NLCD land cover as inputs")
    if ap.parse_args().nlcd_input:
        R.EXTRA = S.nlcd_inputs
        OUT = D.HERE / "output" / "9_best_six_class_nlcd_input"
    for ds in ("exclude_daksh", "with_daksh"):
        best, scenes, px, names, out = search(ds)
        final_maps(best, scenes, px, names, out, ds)
