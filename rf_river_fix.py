"""
Six-class limited Random Forest (rf_limited.py inputs: all bands + per-pixel angles + NLCD,
top 25 inputs, depth 16, leaf 20) with a land-cover penalty on river, so river stops
appearing all over land. Where NLCD says land, P(river) is multiplied by

    w = 1                                      within D0 = 60 m of NLCD water
    w = max(W_MIN, exp(-(dist - D0) / SCALE))  further out (W_MIN = 0.1)

and the pixel takes the class with the highest remaining probability (snow or ice in winter,
forest or shrub in summer: whatever the bands support next). Not a mask: river can still
win where the bands are sure enough.

One knob only, to keep it from overfitting: SCALE (150 / 300 / 600 / 1200 m), chosen on each
training fold's out-of-bag probabilities, never on the test fold, then scored on the held-out
fold next to the same forest without the penalty.

Outputs: output/14_rf_six_classes_river_fix/<dataset folder>/six_class/random_forest/
  metrics.txt         CV scores and confusion with and without the penalty
  river_on_land.txt   per date: % of NLCD-land pixels mapped as river, before / after
  pred_<date>.tif, pred_nofix_<date>.tif, new_dates/, images/

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_river_fix.py
"""

import numpy as np
from osgeo import gdal
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
import data as D
import rf_all_bands as R
import rf_six_classes as S
import rf_limited as L

OUT = D.HERE / "output" / "14_rf_six_classes_river_fix"
K, DEPTH = 25, 16
D0, W_MIN = 60.0, 0.1
SCALES = (150.0, 300.0, 600.0, 1200.0)
RIVER = 0                                       # class index of river
nlcd = gdal.Open(str(S.NLCD)).ReadAsArray()
LAND = ~np.isin(nlcd, [11, 12])
DIST = ndimage.distance_transform_edt(nlcd != 11, sampling=(30.0, 13.0)).astype(np.float32)


def forest(seed=0):
    return RandomForestClassifier(n_estimators=300, max_depth=DEPTH, min_samples_leaf=L.LEAF, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed, oob_score=True)


def apply(proba, land, dist, scale):
    if scale is None:
        return proba.argmax(1)
    w = np.where(dist <= D0, 1.0, np.maximum(W_MIN, np.exp(-(dist - D0) / scale)))
    p = proba.copy()
    p[:, RIVER] *= np.where(land, w, 1.0)
    return p.argmax(1)


def tune(proba, y, land, dist):
    """Best SCALE on training out-of-bag probabilities; None if no penalty helps."""
    best = (R.macro_f1(y, proba.argmax(1)), None)
    for s in SCALES:
        f = R.macro_f1(y, apply(proba, land, dist, s))
        if f > best[0]:
            best = (f, s)
    return best[1]


def fit(x, y, r, c, seed=0):
    keep = L.rank(x, y, seed)[:K]
    m = forest(seed).fit(x[:, keep], y)
    return m, keep, tune(np.nan_to_num(m.oob_decision_function_), y, LAND[r, c], DIST[r, c])


def stack(px, mask=None):
    sel = [(p, mask[p["r"]] if mask is not None else np.ones(len(p["y"]), bool)) for p in px]
    return (np.concatenate([p["x"][t] for p, t in sel]), np.concatenate([p["y"][t] for p, t in sel]),
            np.concatenate([p["r"][t] for p, t in sel]), np.concatenate([p["c"][t] for p, t in sel]))


def cv(px):
    out = {k: [] for k in ("y", "plain", "fixed", "date", "r", "c")}
    scales = []
    for f in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(R.H, f)
        m, keep, scale = fit(*stack(px, train_rows), seed=f)
        scales.append(scale)
        for p in px:
            t = test_rows[p["r"]]
            if not t.any():
                continue
            pr = m.predict_proba(p["x"][t][:, keep]); rr, cc = p["r"][t], p["c"][t]
            out["y"].append(p["y"][t]); out["plain"].append(pr.argmax(1))
            out["fixed"].append(apply(pr, LAND[rr, cc], DIST[rr, cc], scale))
            out["date"] += [p["date"]] * int(t.sum()); out["r"].append(rr); out["c"].append(cc)
    res = {k: np.concatenate(v) if k != "date" else np.array(v) for k, v in out.items()}
    return res, scales


def run(dataset):
    out = OUT / D.DATASET_DIRS[dataset] / "six_class" / "random_forest"
    (out / "images").mkdir(parents=True, exist_ok=True)
    scenes, px, names = L.load(dataset)
    print(f"\n=== {dataset}: pool {len(names)} inputs, top {K}, depth {DEPTH}")
    blocks = []
    if dataset == "external_daksh":
        train = [p for p in px if p["group"] == "mathus_naira"]; test = [p for p in px if p["group"] == "daksh"]
        model, keep, scale = fit(*stack(train))
        xt, yt, rt, ct = stack(test)
        pr = model.predict_proba(xt[:, keep])
        fixed = apply(pr, LAND[rt, ct], DIST[rt, ct], scale)
        D.save_oof(out / "oof.npz", ["2024-03-12"] * len(yt), rt, ct, yt, fixed)
        blocks.append(D.report(yt, fixed, f"external test 2024-03-12, WITH river penalty (scale {scale} m)")[0])
        blocks.append(D.report(yt, pr.argmax(1), "external test 2024-03-12, without penalty")[0])
    else:
        res, scales = cv(px)
        D.save_oof(out / "oof.npz", res["date"], res["r"], res["c"], res["y"], res["fixed"])
        blocks.append(D.report(res["y"], res["fixed"], "WITH river penalty — 5-fold spatial CV, all labelled dates")[0])
        blocks.append(D.report(res["y"], res["plain"], "without penalty (same forests) — 5-fold spatial CV")[0])
        blocks.append("penalty scale chosen per fold (m; None = no penalty helped): " + ", ".join(map(str, scales)))
        for date in sorted(set(res["date"])):
            m = res["date"] == date
            blocks.append(D.report(res["y"][m], res["fixed"][m], f"--- {date} (with penalty)")[0])
        model, keep, scale = fit(*stack(px))
    blocks.append(f"final penalty scale: {scale} m (D0 {D0:.0f} m, floor {W_MIN}); inputs ({K}): " +
                  ", ".join(names[i] for i in keep))
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print(blocks[0]); print(blocks[1])

    lines = ["date        river on NLCD land (% of land px): without penalty -> with penalty"]

    def predict(x, date):
        fx, _ = R.features(x, date)
        fx = fx[keep]
        valid = np.isfinite(fx).all(0)
        plain = np.zeros(valid.shape, np.uint8); fixed = plain.copy()
        idx = np.flatnonzero(valid); flat = fx.reshape(len(fx), -1)
        for i in range(0, len(idx), 300_000):
            j = idx[i:i + 300_000]
            pr = model.predict_proba(flat[:, j].T)
            plain.flat[j] = pr.argmax(1) + 1
            fixed.flat[j] = apply(pr, LAND.flat[j], DIST.flat[j], scale) + 1
        land = LAND & valid
        lines.append(f"{date}  {100 * (plain[land] == 1).mean():5.1f} %  ->  {100 * (fixed[land] == 1).mean():5.1f} %")
        return plain, fixed

    title = f"RF limited + river penalty (scale {scale} m)"
    for s in scenes:
        plain, fixed = predict(s["x"], s["date"])
        plain[~s["valid"]] = 0; fixed[~s["valid"]] = 0
        D.write_prediction(out / f"pred_{s['date']}.tif", fixed, s["ref"])
        D.write_prediction(out / f"pred_nofix_{s['date']}.tif", plain, s["ref"])
        img = R.I.image_for(s["date"])
        if img is not None:
            R.jpeg(img, fixed, f"{s['date']}   {title}", out / "images" / f"rf_{s['date']}.jpg")
            R.jpeg(img, plain, f"{s['date']}   without penalty", out / "images" / f"rf_nofix_{s['date']}.jpg")
    if dataset != "external_daksh":
        for date in R.NEW_DATES:
            x, ref = D.load_features(R.NEW / f"nenana_{date.replace('-', '')}", date)
            plain, fixed = predict(x, date)
            d = out / "new_dates" / date; d.mkdir(parents=True, exist_ok=True)
            D.write_prediction(d / "rf_pred.tif", fixed, ref); D.write_prediction(d / "rf_pred_nofix.tif", plain, ref)
            img = R.I.image_for(date)
            R.jpeg(img, fixed, f"{date} (no labels)   {title}", out / "images" / f"new_rf_{date}.jpg")
            R.jpeg(img, plain, f"{date} (no labels)   without penalty", out / "images" / f"new_rf_nofix_{date}.jpg")
    (out / "river_on_land.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    R.EXTRA = L.extra
    for ds in ("exclude_daksh", "with_daksh", "external_daksh"):
        run(ds)
