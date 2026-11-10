"""
Random Forest per Landsat 30 m pixel, for one task and dataset (see data.py).

1. 5-fold spatial cross-validation on the hand labels -> metrics.
2. Final model on every labelled pixel -> prediction map for every date.

Outputs in <task folder>/random_forest/ (data.task_dir):
  metrics.txt            cross-validated scores (all dates, then per date)
  oof.npz                cross-validated prediction of every labelled pixel
  pred_<date>.tif/.qml   30 m class map
  scene_classes.csv      (four-class only) combined river/land class per date

Run:
    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/train_rf.py --task four --dataset exclude_daksh
"""

import csv, argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import data as D


def make_rf(seed=0):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed)


def pixels(scene, row_mask=None):
    """(n, C) features, (n,) labels and (rows, cols) of labelled pixels, optionally limited to rows."""
    m = scene["y"] != D.IGNORE
    if row_mask is not None:
        m &= row_mask[:, None]
    rr, cc = np.nonzero(m)
    return scene["x"][:, m].T, scene["y"][m], rr, cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(D.TASKS), default="four")
    ap.add_argument("--dataset", choices=D.DATASETS, default="exclude_daksh")
    args = ap.parse_args()
    D.configure(args.task, args.dataset)
    out = D.OUT / D.MODEL_DIRS["rf"]
    print(f"Random Forest — task {D.TASK}, dataset {D.DATASET} -> {out}")

    scenes = D.load_all()
    labelled = [s for s in scenes if (s["y"] != D.IGNORE).any()]
    print("labelled dates:", ", ".join(s["date"] for s in labelled))

    # 1. spatial cross-validation
    yt, yp, dt, rows, cols = [], [], [], [], []
    for k in range(D.N_FOLDS):
        tr_x, tr_y = [], []
        for s in labelled:
            _, train_rows = D.fold_masks(s["y"].shape[0], k)
            x, y, _, _ = pixels(s, train_rows)
            tr_x.append(x); tr_y.append(y)
        rf = make_rf(k).fit(np.concatenate(tr_x), np.concatenate(tr_y))
        for s in labelled:
            test_rows, _ = D.fold_masks(s["y"].shape[0], k)
            x, y, rr, cc = pixels(s, test_rows)
            if len(y):
                yt.append(y); yp.append(rf.predict(x)); dt += [s["date"]] * len(y)
                rows.append(rr); cols.append(cc)
        print(f"  fold {k + 1}/{D.N_FOLDS} done")
    yt, yp, dt = np.concatenate(yt), np.concatenate(yp), np.array(dt)
    out.mkdir(parents=True, exist_ok=True)
    D.save_oof(out / "oof.npz", dt, np.concatenate(rows), np.concatenate(cols), yt, yp)

    text, _ = D.report(yt, yp, "Random Forest — 5-fold spatial CV, all labelled dates")
    blocks = [text]
    for date in sorted(set(dt)):
        m = dt == date
        blocks.append(D.report(yt[m], yp[m], f"--- {date}")[0])

    # 2. final model on all labels, predict every date
    x = np.concatenate([pixels(s)[0] for s in labelled])
    y = np.concatenate([pixels(s)[1] for s in labelled])
    rf = make_rf().fit(x, y)
    imp = sorted(zip(rf.feature_importances_, D.CHANNELS), reverse=True)
    blocks.append("feature importance (final model)\n" +
                  "\n".join(f"  {n:10} {v:.3f}" for v, n in imp))
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print("\n" + blocks[0])

    table = []
    for s in scenes:
        pred = np.zeros(s["valid"].shape, np.uint8)
        pred[s["valid"]] = rf.predict(s["x"][:, s["valid"]].T) + 1
        D.write_prediction(out / f"pred_{s['date']}.tif", pred, s["ref"])
        if D.TASK == "four":
            cls, n = D.scene_class(pred)
            table.append([s["date"], cls] + [n[c] for c in D.CLASSES])
            print(f"  {s['date']}: {cls}")
    if table:
        with open(out / "scene_classes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "scene_class"] + [f"{n}_px" for n in D.CLASSES.values()])
            w.writerows(table)
    print("wrote", out)


if __name__ == "__main__":
    main()
