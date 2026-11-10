"""
Predict dates that have no labels with a task's final models (trained on every
labelled date of the dataset: exclude_daksh 11 inputs, with_daksh 9 inputs).

Per date, in <task folder>/new_dates/<date>/:
  rf_pred.tif / unetpp_pred.tif              the task's classes, 30 m
  rf_scene_mask.tif / unetpp_scene_mask.tif  (four-class and combined) river/land
                                             state per 375 m block

Imagery is read from ~/Desktop/Patches/nenana_<YYYYMMDD>/ (same grid as the patches).
The Random Forest is refitted on all labels (seconds); UNet++ loads unetpp_final.pt
and is skipped while that file does not exist. --task combined merges the binary
snow and river predictions (run those two first). Images: images.py.

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/predict_new.py --task four --dataset with_daksh
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from osgeo import gdal
import data as D
import train_rf as R
import train_unetpp as U
import scene_mask as S
import combine as C

SOURCE = Path.home() / "Desktop/Patches"
DATES  = ["2026-03-26", "2023-05-05"]


def state_mask(out, model, pred, ref):
    blocks, nb = S.block_ids(*pred.shape)
    mk = S.block_states(pred, blocks, nb)[blocks]
    mk[pred == 0] = 0
    D.write_prediction(out / f"{model}_scene_mask.tif", mk, ref, S.STATES, S.COLOURS)


def combined(dataset, dates):
    D.configure("four", dataset)
    for date in dates:
        out = D.task_dir("combined", dataset) / "new_dates" / date
        for model in ("rf", "unetpp"):
            fs = D.task_dir("snow", dataset) / "new_dates" / date / f"{model}_pred.tif"
            fr = D.task_dir("river", dataset) / "new_dates" / date / f"{model}_pred.tif"
            if not (fs.exists() and fr.exists()):
                continue
            out.mkdir(parents=True, exist_ok=True)
            ds = gdal.Open(str(fs)); ps, pr = ds.ReadAsArray(), gdal.Open(str(fr)).ReadAsArray()
            pred = C.four(pr - 1, ps - 1) + 1
            pred[(ps == 0) | (pr == 0)] = 0
            D.write_prediction(out / f"{model}_pred.tif", pred, ds)
            state_mask(out, model, pred, ds)
            print(f"{date} {model}: combined -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=DATES)
    ap.add_argument("--task", choices=list(D.TASKS) + ["combined"], default="four")
    ap.add_argument("--dataset", choices=D.DATASETS, default="exclude_daksh")
    args = ap.parse_args()
    if args.task == "combined":
        return combined(args.dataset, args.dates)
    D.configure(args.task, args.dataset)
    base = D.OUT

    # final models: RF refitted on every labelled pixel, UNet++ from its checkpoint
    labelled = [s for s in D.load_all() if (s["y"] != D.IGNORE).any()]
    rf = R.make_rf().fit(np.concatenate([R.pixels(s)[0] for s in labelled]),
                         np.concatenate([R.pixels(s)[1] for s in labelled]))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = None
    ckpt = base / D.MODEL_DIRS["unetpp"] / "unetpp_final.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        net = U.smp.UnetPlusPlus(encoder_name=ck["encoder"], encoder_weights=None,
                                 in_channels=len(D.CHANNELS), classes=len(D.CLASSES))
        net.load_state_dict(ck["state_dict"]); net.to(dev)
    else:
        print(f"UNet++ not trained yet for {args.dataset}/{args.task} - Random Forest only")

    for date in args.dates:
        out = base / "new_dates" / date
        out.mkdir(parents=True, exist_ok=True)
        x, ref = D.load_features(SOURCE / f"nenana_{date.replace('-', '')}", date)
        valid = np.isfinite(x).all(0) & D.corridor()
        preds = {}
        p = np.zeros(valid.shape, np.uint8); p[valid] = rf.predict(x[:, valid].T) + 1
        preds["rf"] = p
        if net is not None:
            p = U.predict(net, U.to_input(dict(x=x), ck["mean"], ck["std"]), dev) + 1; p[~valid] = 0
            preds["unetpp"] = p
        for m, p in preds.items():
            D.write_prediction(out / f"{m}_pred.tif", p, ref)
            if D.TASK == "four":
                state_mask(out, m, p, ref)
            print(f"{date} {m:6}: " + ", ".join(f"{n} {int((p == i).sum()):,}" for i, n in D.CLASSES.items()))
        print("wrote", out)


if __name__ == "__main__":
    main()
