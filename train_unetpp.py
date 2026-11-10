"""
UNet++ per Landsat 30 m pixel, for one task and dataset (see data.py).

Same inputs, labels and spatial folds as train_rf.py. The labels
are sparse (only inside polygons), so the loss is computed on labelled pixels
only; unlabelled pixels still feed the network as context.

1. 5-fold spatial cross-validation: per fold, train on random 256 px crops whose
   labels come from training rows only, predict the whole image, score the test
   rows -> metrics.
2. Final model on every labelled pixel -> prediction map for every patch date.

Outputs in <task folder>/unet_plus_plus/ (data.task_dir):
  metrics.txt            cross-validated scores (all dates, then per date)
  oof.npz                cross-validated prediction of every labelled pixel
  pred_<date>.tif/.qml   30 m class map
  scene_classes.csv      (four-class only) combined river/land class per date
  unetpp_final.pt        final weights + channel scaling

Run:
    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/train_unetpp.py --task four --dataset exclude_daksh
    (--iters 3000 for longer training, --encoder resnet18 for a smaller net)
"""

import csv, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import data as D

CROP  = 256
TILE  = 512          # inference tile
HALO  = 64           # inference tile overlap


def to_input(scene, mean, std):
    """Scaled (C, H, W) float32 with missing imagery set to 0 (= channel mean)."""
    x = (scene["x"] - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def class_weights(scenes, masks):
    """Inverse square-root frequency of each class among training pixels."""
    n = np.zeros(len(D.CLASSES))
    for s, m in zip(scenes, masks):
        y = s["y"][m]
        n += np.bincount(y[y != D.IGNORE], minlength=len(D.CLASSES))
    w = 1.0 / np.sqrt(np.maximum(n, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class Crops:
    """Random crops centred near a training-labelled pixel, with flips / 90° turns."""

    def __init__(self, xs, ys, rng):
        self.xs, self.ys, self.rng = xs, ys, rng
        self.pts = [np.argwhere(y != D.IGNORE) for y in ys]
        self.use = [i for i, p in enumerate(self.pts) if len(p)]

    def batch(self, n):
        bx, by = [], []
        for _ in range(n):
            i = self.use[self.rng.integers(len(self.use))]
            x, y = self.xs[i], self.ys[i]
            r, c = self.pts[i][self.rng.integers(len(self.pts[i]))]
            H, W = y.shape
            r0 = int(np.clip(r - self.rng.integers(CROP), 0, H - CROP))
            c0 = int(np.clip(c - self.rng.integers(CROP), 0, W - CROP))
            xc, yc = x[:, r0:r0 + CROP, c0:c0 + CROP], y[r0:r0 + CROP, c0:c0 + CROP]
            k = self.rng.integers(4)
            xc, yc = np.rot90(xc, k, (1, 2)), np.rot90(yc, k)
            if self.rng.random() < 0.5:
                xc, yc = xc[:, :, ::-1], yc[:, ::-1]
            bx.append(xc.copy()); by.append(yc.copy())
        return torch.from_numpy(np.stack(bx)), torch.from_numpy(np.stack(by)).long()


def make_net(encoder):
    return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet",
                            in_channels=len(D.CHANNELS), classes=len(D.CLASSES))


def train(xs, ys, weights, args, dev, seed):
    torch.manual_seed(seed)
    net = make_net(args.encoder).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    data = Crops(xs, ys, np.random.default_rng(seed))
    w = weights.to(dev)
    net.train()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x, y = data.batch(args.batch)
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            loss = F.cross_entropy(net(x), y, weight=w, ignore_index=D.IGNORE)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        if it % 250 == 0 or it == args.iters:
            print(f"    iter {it:5}/{args.iters}  loss {loss.item():.4f}  ({time.time() - t0:.0f} s)")
    return net


@torch.no_grad()
def predict(net, x, dev):
    """Whole-image class index (H, W) from overlapping tiles."""
    net.eval()
    C, H, W = x.shape
    logits = torch.zeros(len(D.CLASSES), H, W)
    count = torch.zeros(1, H, W)
    step = TILE - 2 * HALO
    for r in range(0, H, step):
        for c in range(0, W, step):
            r0, c0 = max(0, min(r, H - TILE)), max(0, min(c, W - TILE))
            r1, c1 = min(H, r0 + TILE), min(W, c0 + TILE)
            t = torch.from_numpy(x[:, r0:r1, c0:c1])[None]
            ph, pw = (-t.shape[2]) % 32, (-t.shape[3]) % 32           # UNet++ needs /32
            t = F.pad(t, (0, pw, 0, ph), mode="reflect")
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = net(t.to(dev)).float().cpu()[0, :, :r1 - r0, :c1 - c0]
            logits[:, r0:r1, c0:c1] += out
            count[:, r0:r1, c0:c1] += 1
    return (logits / count).argmax(0).numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--skip-cv", action="store_true", help="only train the final model")
    ap.add_argument("--task", choices=list(D.TASKS), default="four")
    ap.add_argument("--dataset", choices=D.DATASETS, default="exclude_daksh")
    args = ap.parse_args()
    D.configure(args.task, args.dataset)
    OUT = D.OUT / D.MODEL_DIRS["unetpp"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"UNet++ — task {D.TASK}, dataset {D.DATASET}, device {dev} -> {OUT}")
    scenes = D.load_all()
    labelled = [s for s in scenes if (s["y"] != D.IGNORE).any()]
    print("labelled dates:", ", ".join(s["date"] for s in labelled))
    mean, std = D.channel_stats(labelled)          # input scaling from training dates only
    xs = [to_input(s, mean, std) for s in labelled]
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. spatial cross-validation
    blocks = []
    if not args.skip_cv:
        yt, yp, dt, rows, cols = [], [], [], [], []
        for k in range(D.N_FOLDS):
            print(f"  fold {k + 1}/{D.N_FOLDS}")
            ys, tests = [], []
            for s in labelled:
                test_rows, train_rows = D.fold_masks(s["y"].shape[0], k)
                y = s["y"].copy(); y[~train_rows] = D.IGNORE
                ys.append(y); tests.append(test_rows)
            net = train(xs, ys, class_weights(labelled, [y != D.IGNORE for y in ys]), args, dev, k)
            for s, x, test_rows in zip(labelled, xs, tests):
                m = (s["y"] != D.IGNORE) & test_rows[:, None]
                if m.any():
                    p = predict(net, x, dev)
                    yt.append(s["y"][m]); yp.append(p[m]); dt += [s["date"]] * int(m.sum())
                    rr, cc = np.nonzero(m); rows.append(rr); cols.append(cc)
            del net; torch.cuda.empty_cache()
        yt, yp, dt = np.concatenate(yt), np.concatenate(yp), np.array(dt)
        D.save_oof(OUT / "oof.npz", dt, np.concatenate(rows), np.concatenate(cols), yt, yp)
        blocks.append(D.report(yt, yp, "UNet++ — 5-fold spatial CV, all labelled dates")[0])
        for date in sorted(set(dt)):
            m = dt == date
            blocks.append(D.report(yt[m], yp[m], f"--- {date}")[0])
        print("\n" + blocks[0])

    # 2. final model on all labels, predict every date
    print("  final model")
    ys = [s["y"] for s in labelled]
    net = train(xs, ys, class_weights(labelled, [y != D.IGNORE for y in ys]), args, dev, 100)
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std,
                "encoder": args.encoder, "channels": D.CHANNELS}, OUT / "unetpp_final.pt")
    blocks.append(f"settings: {vars(args)}")
    (OUT / "metrics.txt").write_text("\n\n".join(blocks) + "\n")

    rows = []
    for s in scenes:
        pred = predict(net, to_input(s, mean, std), dev) + 1
        pred[~s["valid"]] = 0
        D.write_prediction(OUT / f"pred_{s['date']}.tif", pred, s["ref"])
        if D.TASK == "four":
            cls, n = D.scene_class(pred)
            rows.append([s["date"], cls] + [n[c] for c in D.CLASSES])
            print(f"  {s['date']}: {cls}")
    if rows:
        with open(OUT / "scene_classes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "scene_class"] + [f"{n}_px" for n in D.CLASSES.values()])
            w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
