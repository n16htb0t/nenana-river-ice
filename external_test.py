"""
External test cohort: train on the labelled patch dates only, score on the
labelled daksh date (2024-03-12), which the model never sees in training.

Inputs are the 9 shared ones (the daksh tiff has no SWIR2 and no temperature).
No cross-validation here: one model per task and method is trained on every
labelled patch pixel, then scored on every labelled daksh pixel.

Outputs in output/3_external_test_daksh/<task folder>/<model folder>/:
  metrics.txt            scores on the daksh labels + confusion matrix
  oof.npz                the daksh-label predictions (same format as the CV ones,
                         so combine.py and scene_mask.py read them unchanged)
  pred_<date>.tif/.qml   maps for every date (patch dates, labelled and unlabelled daksh dates)

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/external_test.py --task four
"""

import argparse
import numpy as np
import torch
import data as D
import train_rf as R
import train_unetpp as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(D.TASKS), default="four")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    args = ap.parse_args()
    D.configure(args.task, "external_daksh")
    scenes = D.load_all()
    lab = [s for s in scenes if (s["y"] != D.IGNORE).any()]
    train = [s for s in lab if s["group"] == "mathus_naira"]
    test = [s for s in lab if s["group"] == "daksh"]
    print(f"external test — task {D.TASK}: train {', '.join(s['date'] for s in train)}; "
          f"test {', '.join(s['date'] for s in test)} -> {D.OUT}")

    def score(model, name, preds):
        out = D.OUT / D.MODEL_DIRS[model]
        out.mkdir(parents=True, exist_ok=True)
        yt, yp, dt, rr, cc = [], [], [], [], []
        for s in test:
            m = s["y"] != D.IGNORE
            r, c = np.nonzero(m)
            yt.append(s["y"][m]); yp.append(preds[s["date"]][m] - 1)
            dt += [s["date"]] * len(r); rr.append(r); cc.append(c)
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        D.save_oof(out / "oof.npz", dt, np.concatenate(rr), np.concatenate(cc), yt, yp)
        text = D.report(yt, yp, f"{name} — external test: trained on "
                        f"{', '.join(s['date'] for s in train)}, scored on daksh "
                        f"{', '.join(s['date'] for s in test)}")[0]
        (out / "metrics.txt").write_text(text + "\n")
        print("\n" + text)
        for s in scenes:
            D.write_prediction(out / f"pred_{s['date']}.tif", preds[s["date"]], s["ref"])

    # Random Forest
    rf = R.make_rf().fit(np.concatenate([R.pixels(s)[0] for s in train]),
                         np.concatenate([R.pixels(s)[1] for s in train]))
    preds = {}
    for s in scenes:
        p = np.zeros(s["valid"].shape, np.uint8)
        p[s["valid"]] = rf.predict(s["x"][:, s["valid"]].T) + 1
        preds[s["date"]] = p
    score("rf", "Random Forest", preds)

    # UNet++
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = D.channel_stats(train)
    xs = [U.to_input(s, mean, std) for s in train]
    ys = [s["y"] for s in train]
    net = U.train(xs, ys, U.class_weights(train, [y != D.IGNORE for y in ys]), args, dev, 100)
    ud = D.OUT / D.MODEL_DIRS["unetpp"]
    ud.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std,
                "encoder": args.encoder, "channels": D.CHANNELS}, ud / "unetpp_final.pt")
    preds = {}
    for s in scenes:
        p = U.predict(net, U.to_input(s, mean, std), dev) + 1
        p[~s["valid"]] = 0
        preds[s["date"]] = p
    score("unetpp", "UNet++", preds)


if __name__ == "__main__":
    main()
