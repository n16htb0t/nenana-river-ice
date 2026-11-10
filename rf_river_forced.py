"""
Run 14 with the river penalty forced on. In run 14 every fold's tuner chose "no penalty",
because ~43% of the hand-labelled river lies on NLCD-2016 land (the channel moved), so the
maps still showed river all over land. Here the penalty is always applied:

    w = 1                                      within D0 = 60 m of NLCD water
    w = max(W_MIN, exp(-(dist - D0) / SCALE))  further out, on NLCD land only

Same forest as run 14 / 13 (top 25 inputs, depth 16, leaf 20). First a scan: one 5-fold
spatial CV pass, the held-out probabilities scored at every SCALE (penalty_scan.txt), so the
cost in accuracy / F1 and the gain in river-on-land are visible side by side. Then the maps,
metrics and images with the chosen SCALE (FORCED).

Outputs: output/15_rf_six_classes_river_forced/<dataset folder>/six_class/random_forest/
  penalty_scan.txt    CV acc / macro F1 / river F1 per scale (exclude and with daksh)
  metrics.txt, river_on_land.txt, pred_<date>.tif, pred_nofix_<date>.tif, new_dates/, images/

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_river_forced.py
"""

import numpy as np
import data as D
import rf_all_bands as R
import rf_limited as L
import rf_river_fix as F

F.OUT = D.HERE / "output" / "15_rf_six_classes_river_forced"
FORCED = 600.0
SCAN = (None, 1200.0, 600.0, 300.0, 150.0)


def scan(dataset):
    out = F.OUT / D.DATASET_DIRS[dataset] / "six_class" / "random_forest"
    out.mkdir(parents=True, exist_ok=True)
    _, px, _ = L.load(dataset)
    ys, probs, rs, cs = [], [], [], []
    for f in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(R.H, f)
        x, y, r, c = F.stack(px, train_rows)
        keep = L.rank(x, y, f)[:F.K]
        m = F.forest(f).fit(x[:, keep], y)
        for p in px:
            t = test_rows[p["r"]]
            if t.any():
                ys.append(p["y"][t]); probs.append(m.predict_proba(p["x"][t][:, keep]))
                rs.append(p["r"][t]); cs.append(p["c"][t])
    y, pr, r, c = map(np.concatenate, (ys, probs, rs, cs))
    land, dist = F.LAND[r, c], F.DIST[r, c]
    lines = [f"forced river penalty scan, {dataset}: 5-fold spatial CV (top {F.K}, depth {F.DEPTH})",
             f"{'scale m':>8} {'CV acc':>7} {'macroF1':>8} {'river F1':>9} {'river recall':>13}"]
    for s in SCAN:
        p = F.apply(pr, land, dist, s)
        tp = ((p == F.RIVER) & (y == F.RIVER)).sum()
        rec = tp / max(1, (y == F.RIVER).sum()); prec = tp / max(1, (p == F.RIVER).sum())
        lines.append(f"{str(s):>8} {(p == y).mean():7.3f} {R.macro_f1(y, p):8.3f} "
                     f"{2 * prec * rec / max(1e-9, prec + rec):9.3f} {rec:13.3f}")
    (out / "penalty_scan.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    R.EXTRA = L.extra
    for ds in ("exclude_daksh", "with_daksh"):
        scan(ds)
    F.tune = lambda *a: FORCED
    for ds in ("exclude_daksh", "with_daksh", "external_daksh"):
        F.run(ds)
