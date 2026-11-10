"""
Run 21, UNet++: the same set-up as the run 21 RF (viirs_run16.py), with UNet++ instead of the zone forests.

  inputs       the same 46 VIIRS-based inputs at 30 m (viirs_run16.viirs_features), standardised
  labels       the same hand labels; the loss uses the rule labels (as run 16 trains on rule labels)
  water mask   logits of the classes the zone does not allow are switched off (water: river / ice;
               land: snow / forest / shrub / bare), in training and prediction
  evaluation   the same 5-fold spatial CV (data.fold_masks row blocks): a fold's model is trained on the labelled
               pixels of the training rows of all labelled dates and scored on the test rows
  network      smp UnetPlusPlus, resnet18 encoder (ImageNet start when cached), 6 classes, class-weighted
               cross-entropy (1 / sqrt of the labelled pixels per class), 256 x 256 tiles placed over labelled
               pixels, batch 8, 2000 steps, flips, AdamW 1e-3 one-cycle; prediction in 512 x 512 windows, flip TTA

Outputs: output/21_viirs_run16/2_with_daksh/unetpp/
  metrics.json, metrics.txt, oof.npz, pred_<date>.tif (+ .qml), images/pred_<date>.jpg, images/conf.png

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/viirs_run16_unet.py
"""

import json
import shutil

import numpy as np
from osgeo import gdal

import data as D
import rf_viirs_unmix as U
import rf_water_zone as W
import viirs_30m as M30
import viirs_run16 as V16
import viirs_states as V

gdal.UseExceptions()
DATASET = "with_daksh"
OUT = V16.OUT / V16.DATASETS[DATASET] / "unetpp"
ZONE = W.ZONE
TILE, STEPS, BATCH, SEED = 256, 2000, 8, 0


def zone_bias(dev):
    import torch
    b = np.full((6,) + ZONE.shape, -1e4, np.float32)
    for zone, ks in M30.ALLOWED.items():
        for k in ks:
            b[k][ZONE == zone] = 0
    return torch.from_numpy(b).to(dev)


class Net:
    def fit(self, X, Y, seed=SEED):
        """X: (n_dates, C, H, W) standardised inputs; Y: (n_dates, H, W) class index or -100 (no label)."""
        import torch
        import torch.nn.functional as F
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        self.dev = dev = "cuda" if torch.cuda.is_available() else "cpu"
        lab = [np.argwhere(y >= 0) for y in Y]
        counts = np.bincount(Y[Y >= 0], minlength=6).astype(float)
        w = 1 / np.sqrt(np.maximum(counts, 1)); w = w / w[counts > 0].mean()
        Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
        bias = zone_bias(dev)
        self.net = net = V.make_net("unetpp", X.shape[1], classes=6).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS)
        weight = torch.tensor(w, dtype=torch.float32, device=dev)
        H, Wd = ZONE.shape
        dates = [i for i in range(len(Y)) if len(lab[i])]
        net.train()
        for _ in range(STEPS):
            xs, ys, bs = [], [], []
            for _ in range(BATCH):
                i = dates[rng.integers(len(dates))]
                r0, c0 = lab[i][rng.integers(len(lab[i]))]          # tile placed over a labelled pixel
                r = int(np.clip(r0 - rng.integers(0, TILE), 0, H - TILE))
                c = int(np.clip(c0 - rng.integers(0, TILE), 0, Wd - TILE))
                xs.append(Xt[i, :, r:r + TILE, c:c + TILE]); ys.append(Yt[i, r:r + TILE, c:c + TILE])
                bs.append(bias[:, r:r + TILE, c:c + TILE])
            x, y, b = torch.stack(xs), torch.stack(ys), torch.stack(bs)
            if rng.random() < 0.5:
                x, y, b = x.flip(-1), y.flip(-1), b.flip(-1)
            if rng.random() < 0.5:
                x, y, b = x.flip(-2), y.flip(-2), b.flip(-2)
            loss = F.cross_entropy(net(x) + b, y, weight=weight, ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        return self

    def predict(self, x, win=512, stride=448):
        """x: (C, H, W) standardised -> (H, W) class index 0..5."""
        import torch
        xt = torch.from_numpy(x).to(self.dev)
        H, Wd = ZONE.shape
        prob = torch.zeros((6, H, Wd), device=self.dev); cnt = torch.zeros((H, Wd), device=self.dev)
        rs = sorted(set(list(range(0, H - win + 1, stride)) + [H - win]))
        cs = sorted(set(list(range(0, Wd - win + 1, stride)) + [Wd - win]))
        with torch.no_grad():
            for r in rs:
                for c in cs:
                    xb = xt[None, :, r:r + win, c:c + win]
                    p = torch.softmax(self.net(xb), 1)
                    p = p + torch.softmax(self.net(xb.flip(-1)), 1).flip(-1)
                    p = p + torch.softmax(self.net(xb.flip(-2)), 1).flip(-2)
                    prob[:, r:r + win, c:c + win] += p[0] / 3; cnt[r:r + win, c:c + win] += 1
        logp = torch.log(prob / cnt.clamp(min=1) + 1e-9) + zone_bias(self.dev)
        return logp.argmax(0).cpu().numpy().astype(np.uint8)


def standardise(feats, dates):
    stack = np.concatenate([feats[d][0].reshape(len(feats[d][0]), -1)[:, ::97] for d in dates], 1)
    mean, std = np.nanmean(stack, 1), np.nanstd(stack, 1) + 1e-6
    return {d: np.nan_to_num((feats[d][0] - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
            for d in feats}


if __name__ == "__main__":
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    shifts = {r["date"]: r["shift_fine_px"] for r in json.loads((U.OUT / "results.json").read_text())}
    cache, days30, feats = {}, {}, {}
    for date in U.viirs_dates():
        days30[date] = M30.load_day(date, shifts.get(date, [0, 0]), cache)
        feats[date] = V16.viirs_features(days30[date])
    hand = V16.labels(DATASET)
    dates = sorted(hand)
    Xn = standardise(feats, dates)
    valid = {d: np.isfinite(feats[d][0]).all(0) for d in feats}
    rule = {d: W.rule_labels(hand[d], ZONE) for d in dates}
    print("labelled dates", dates, "inputs", len(feats[dates[0]][1]), flush=True)

    ys, ps, ds, rs, cs = [], [], [], [], []
    H = ZONE.shape[0]
    for f in range(D.N_FOLDS):
        test_rows, train_rows = D.fold_masks(H, f)
        Y = np.stack([np.where((rule[d] != D.IGNORE) & valid[d] & train_rows[:, None], rule[d].astype(np.int64), -100).astype(np.int64)
                      for d in dates])
        net = Net().fit(np.stack([Xn[d] for d in dates]), Y, seed=f)
        for d in dates:
            pred = net.predict(Xn[d])
            m = (hand[d] != D.IGNORE) & valid[d] & test_rows[:, None]
            rr, cc = np.nonzero(m)
            ys.append(hand[d][m]); ps.append(pred[m]); ds.append(np.full(len(rr), d)); rs.append(rr); cs.append(cc)
        print(f"  fold {f + 1}/{D.N_FOLDS} done", flush=True)
    yt, pt, dt, rt, ct = map(np.concatenate, (ys, ps, ds, rs, cs))
    D.save_oof(OUT / "oof.npz", dt, rt, ct, yt, pt)
    zt = ZONE[rt, ct]
    yr = W.rule_labels(yt, zt); k = yr != D.IGNORE
    res = {"dates": dates, "labelled_px": int(len(yt)),
           "rule": M30.summarise(M30.confusion(yr[k], pt[k])), "original": M30.summarise(M30.confusion(yt, pt)),
           "per_date": {}}
    blocks = W.scores(yt, pt, zt, "VIIRS 30 m UNet++ (water mask) - 5-fold spatial CV, all labelled dates")
    for d in dates:
        m = dt == d
        yrd = W.rule_labels(yt[m], zt[m]); kd = yrd != D.IGNORE
        res["per_date"][d] = {"rule": M30.summarise(M30.confusion(yrd[kd], pt[m][kd])),
                              "original": M30.summarise(M30.confusion(yt[m], pt[m]))}
        blocks.append(D.report(yrd[kd], pt[m][kd], f"--- {d} (rule labels)")[0])
    (OUT / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print(f"rule labels: acc {res['rule']['accuracy']:.3f} F1 {res['rule']['macro_f1']:.3f} | original: acc "
          f"{res['original']['accuracy']:.3f} F1 {res['original']['macro_f1']:.3f}", flush=True)

    Y = np.stack([np.where((rule[d] != D.IGNORE) & valid[d], rule[d].astype(np.int64), -100).astype(np.int64) for d in dates])
    final = Net().fit(np.stack([Xn[d] for d in dates]), Y, seed=99)
    for d in feats:
        pred = final.predict(Xn[d]) + 1
        pred[~valid[d]] = 0
        D.write_prediction(OUT / f"pred_{d}.tif", pred.astype(np.uint8), days30[d]["ref"])
        shutil.copy(M30.QML, OUT / f"pred_{d}.qml")
    res["conf_image"] = str(V16.conf_image(OUT, res["rule"], f"Run 21 UNet++ ({DATASET}): 5-fold spatial CV, rule labels "
                                                             f"(acc {res['rule']['accuracy']:.3f})"))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    for d in days30:
        V16.date_image(OUT, days30[d], hand, res, DATASET, "UNet++")
    print("wrote", OUT, flush=True)
