"""
Binary snow + binary river -> the four classes, per model.

  river & no snow -> river     river & snow -> ice
  not river & snow -> snow     not river & no snow -> land

Scores use the cross-validated predictions (oof.npz) of both binary models, so
each pixel is judged by models that never trained on it.

Outputs in <dataset folder>/binary/combined_to_four_class/<model folder>/:
  metrics.txt, pred_<date>.tif/.qml, scene_classes.csv

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/combine.py --dataset exclude_daksh
"""

import csv, argparse
import numpy as np
from osgeo import gdal
import data as D

NAMES = D.MODEL_NAMES


def four(river, snow):
    """0/1 river and snow -> four-class index (0 river, 1 ice, 2 snow, 3 land)."""
    return np.where(river == 1, np.where(snow == 1, 1, 0), np.where(snow == 1, 2, 3)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=D.DATASETS, default="exclude_daksh")
    args = ap.parse_args()
    D.configure("four", args.dataset)
    for model, name in NAMES.items():
        mdir = D.MODEL_DIRS[model]
        sd, rd = D.task_dir("snow", args.dataset) / mdir, D.task_dir("river", args.dataset) / mdir
        if not (sd / "oof.npz").exists() or not (rd / "oof.npz").exists():
            print(f"{name}: binary runs missing, skipped")
            continue
        out = D.task_dir("combined", args.dataset) / mdir
        out.mkdir(parents=True, exist_ok=True)
        s, r = np.load(sd / "oof.npz"), np.load(rd / "oof.npz")
        key = lambda z: {(d, a, b): i for i, (d, a, b) in enumerate(zip(z["date"], z["row"], z["col"]))}
        ks, kr = key(s), key(r)
        common = sorted(set(ks) & set(kr))
        i_s = np.array([ks[k] for k in common]); i_r = np.array([kr[k] for k in common])
        yt = four(r["y"][i_r], s["y"][i_s]); yp = four(r["p"][i_r], s["p"][i_s])
        dt = np.array([k[0] for k in common])
        blocks = [D.report(yt, yp, f"{name} binary snow + river combined — 5-fold spatial CV, all labelled dates")[0]]
        for date in sorted(set(dt)):
            m = dt == date
            blocks.append(D.report(yt[m], yp[m], f"--- {date}")[0])
        (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
        print("\n" + blocks[0])

        table = []
        for f in sorted(sd.glob("pred_*.tif")):
            date = f.stem[5:]
            fr = rd / f.name
            if not fr.exists():
                continue
            ds = gdal.Open(str(f))
            ps, pr = ds.ReadAsArray(), gdal.Open(str(fr)).ReadAsArray()
            pred = four(pr - 1, ps - 1) + 1
            pred[(ps == 0) | (pr == 0)] = 0
            D.write_prediction(out / f"pred_{date}.tif", pred, ds)
            cls, n = D.scene_class(pred)
            table.append([date, cls] + [n[c] for c in D.CLASSES])
        with open(out / "scene_classes.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "scene_class"] + [f"{n}_px" for n in D.CLASSES.values()])
            w.writerows(table)
        print("wrote", out)


if __name__ == "__main__":
    main()
