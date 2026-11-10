"""
Extra numbers for the run 16 report (rf_water_zone.py), written to
output/16_rf_water_zone/extra.json:

  scan        JRC cutoff 1 / 5 / 10 / 25 / 50 %: water-zone share, rule-label pixels, 5-fold spatial CV
              accuracy / macro F1 on rule labels and on original labels (exclude_daksh, with_daksh)
  gap         cutoff 5 %: per fold and zone, training accuracy vs held-out accuracy (overfitting check)
  importance  cutoff 5 %: importance of the 25 inputs of each final forest (water, land), per dataset
  counts      labelled pixels per date and class, original and after the rule

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_water_zone_extra.py
"""

import json
import numpy as np
import data as D
import rf_all_bands as R
import rf_limited as L
import rf_water_zone as W

CUTOFFS = (1.0, 5.0, 10.0, 25.0, 50.0)
WO = D._water_occurrence()


def macro(y, p):
    f1s = []
    for k in range(6):
        t, q = y == k, p == k
        if t.sum():
            tp = (t & q).sum(); pr = tp / max(1, q.sum()); rc = tp / t.sum()
            f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    return float((y == p).mean()), float(np.mean(f1s))


def cv(px, gap=None):
    ys, ps, zs = [], [], []
    for f in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(R.H, f)
        x, y, r, c, _ = W.stack(px, train_rows)
        m = W.ZoneModel(f).fit(x, y, r, c)
        xt, yt, rt, ct, _ = W.stack(px, test_rows)
        zt = W.ZONE[rt, ct]; pt = m.predict(xt, zt)
        ys.append(yt); ps.append(pt); zs.append(zt)
        if gap is not None:
            z = W.ZONE[r, c]; yr = W.rule_labels(y, z); k = yr != D.IGNORE
            ptr = m.predict(x[k], z[k])
            yrt = W.rule_labels(yt, zt); kt = yrt != D.IGNORE
            row = {"fold": f}
            for name, zm, zmt in (("water", z[k], zt[kt]), ("land", ~z[k], ~zt[kt])):
                row[f"{name}_train"] = float((ptr[zm] == yr[k][zm]).mean())
                row[f"{name}_test"] = float((pt[kt][zmt] == yrt[kt][zmt]).mean())
            row["all_train"] = float((ptr == yr[k]).mean()); row["all_test"] = float((pt[kt] == yrt[kt]).mean())
            gap.append(row)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(zs)


R.EXTRA = L.extra
out = {"scan": {}, "gap": {}, "importance": {}, "counts": {}}
for ds in ("exclude_daksh", "with_daksh", "external_daksh"):
    scenes, px, names = L.load(ds)
    W.ZONE = WO >= 5.0
    if ds == "external_daksh":
        x, y, r, c, _ = W.stack([p for p in px if p["group"] == "mathus_naira"])
        m = W.ZoneModel().fit(x, y, r, c)
    else:
        cnt = {}
        for p in px:
            z = W.ZONE[p["r"], p["c"]]; yr = W.rule_labels(p["y"], z)
            cnt[p["date"]] = {"original": [int((p["y"] == k).sum()) for k in range(6)],
                              "rule": [int((yr == k).sum()) for k in range(6)]}
        out["counts"][ds] = cnt
        gap = []
        print(ds, "gap pass", flush=True); cv(px, gap)
        out["gap"][ds] = gap
        out["scan"][ds] = []
        for t in CUTOFFS:
            W.ZONE = WO >= t
            y, p, z = cv(px)
            yr = W.rule_labels(y, z); k = yr != D.IGNORE
            ra, rf = macro(yr[k], p[k]); oa, of = macro(y, p)
            tp = ((p == 0) & (y == 0)).sum(); pr = tp / max(1, (p == 0).sum()); rc = tp / max(1, (y == 0).sum())
            row = {"cutoff": t, "zone_pct": float(100 * W.ZONE.mean()), "rule_px": int(k.sum()), "all_px": int(len(y)),
                   "rule_acc": ra, "rule_mf1": rf, "orig_acc": oa, "orig_mf1": of,
                   "river_f1_orig": float(2 * pr * rc / max(1e-9, pr + rc))}
            out["scan"][ds].append(row); print(ds, row, flush=True)
        W.ZONE = WO >= 5.0
        x, y, r, c, _ = W.stack(px)
        m = W.ZoneModel().fit(x, y, r, c)
    out["importance"][ds] = {name: sorted([[names[i], float(v)] for i, v in zip(keep, f.feature_importances_)],
                                          key=lambda a: -a[1]) for name, (f, keep) in m.parts.items()}
    (W.OUT / "extra.json").write_text(json.dumps(out, indent=1))
print("wrote", W.OUT / "extra.json", flush=True)
