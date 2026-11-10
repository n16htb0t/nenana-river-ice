"""
River / land state mask (the four classes below) from a trained model's 30 m map.

  1 ice_free_river_snow_free_land        open river, snow-free land
  2 ice_free_river_snow_covered_land     open river, snow-covered land
  3 ice_covered_river_snow_free_land     river frozen, snow-free land
  4 ice_covered_river_snow_covered_land  river frozen, snow-covered land
  0 no river in the block

The box is cut into 375 m blocks (12.5 x 12.5 Landsat pixels, the patch grid).
In each block: river state = ice if predicted ice pixels outnumber open-river
ones; land state = snow if predicted snow pixels outnumber snow-free land. A
block needs at least MIN_RIVER predicted river/ice pixels, otherwise it is 0.
Every 30 m pixel takes its block's class.

Evaluation: for blocks where the hand labels contain both river/ice and
snow/land, the hand labels give the true class and the cross-validated
predictions (oof.npz) of those same labelled pixels give the predicted class.

Inputs: four-class maps from the four-class and binary/combined_to_four_class model folders.
Outputs next to them: scene_mask_<date>.tif/.qml, scene_mask_metrics.txt

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/scene_mask.py --dataset exclude_daksh
"""

import argparse
import numpy as np
from osgeo import gdal
import data as D

STATES = {1: "ice_free_river_snow_free_land", 2: "ice_free_river_snow_covered_land",
          3: "ice_covered_river_snow_free_land", 4: "ice_covered_river_snow_covered_land"}
COLOURS = {1: "#0066ff", 2: "#00e5ff", 3: "#ff7f00", 4: "#ff00ff"}
SUB = 12.5          # Landsat pixels per 375 m block side
MIN_RIVER = 3


def block_ids(h, w):
    """(H, W) block index of every Landsat pixel and the number of blocks."""
    br, bc = (np.arange(h) // SUB).astype(int), (np.arange(w) // SUB).astype(int)
    nbc = bc.max() + 1
    return br[:, None] * nbc + bc[None, :], (br.max() + 1) * nbc


def block_states(cls, blocks, n_blocks, min_river=MIN_RIVER):
    """cls: four-class ids 1..4 (0 = none) at some pixels; blocks: their block index.
    Returns (n_blocks,) state 1..4, 0 where the block has no river."""
    n = {c: np.bincount(blocks[cls == c], minlength=n_blocks) for c in (1, 2, 3, 4)}
    ice = n[2] > n[1]
    snow = n[3] > n[4]
    state = 1 + snow.astype(int) + 2 * ice.astype(int)
    state[(n[1] + n[2]) < min_river] = 0
    return state.astype(np.uint8)


def evaluate(oof, scenes):
    """Block-level scores from the cross-validated predictions of labelled pixels."""
    dates = {s["date"]: s for s in scenes}
    yt, yp, dl = [], [], []
    for date in sorted(set(oof["date"])):
        m = oof["date"] == date
        s = dates[date]
        blocks, nb = block_ids(*s["y"].shape)
        b = blocks[oof["row"][m], oof["col"][m]]
        t, p = oof["y"][m] + 1, oof["p"][m] + 1
        # a block counts when its hand labels hold both water (river/ice) and land (snow/land)
        water = np.bincount(b[t <= 2], minlength=nb); land = np.bincount(b[t >= 3], minlength=nb)
        ok = (water > 0) & (land > 0)
        st, sp = block_states(t, b, nb, 1), block_states(p, b, nb, 1)
        keep = ok & (st > 0)
        yt.append(st[keep] - 1); yp.append(np.maximum(sp[keep], 1) - 1); dl += [date] * int(keep.sum())
    if not yt:
        return None
    yt, yp, dl = np.concatenate(yt), np.concatenate(yp), np.array(dl)
    blocks = [D.report(yt, yp, "River/land state per 375 m block — cross-validated, blocks with "
                               "labelled river and land", STATES)[0]]
    for date in sorted(set(dl)):
        m = dl == date
        blocks.append(D.report(yt[m], yp[m], f"--- {date}", STATES)[0])
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=D.DATASETS, default="exclude_daksh")
    args = ap.parse_args()
    D.configure("four", args.dataset)
    scenes = D.load_all()
    for folder in [D.task_dir(t, args.dataset) / m for t in ("four", "combined")
                   for m in D.MODEL_DIRS.values()]:
        maps = sorted(folder.glob("pred_*.tif"))
        if not maps:
            continue
        print(folder.relative_to(D.OUTPUT))
        for f in maps:
            date = f.stem[5:]
            ds = gdal.Open(str(f)); pred = ds.ReadAsArray()
            blocks, nb = block_ids(*pred.shape)
            state = block_states(pred, blocks, nb)
            mask = state[blocks]
            mask[pred == 0] = 0
            D.write_prediction(folder / f"scene_mask_{date}.tif", mask, ds, STATES, COLOURS)
            counts = np.bincount(state, minlength=5)
            print(f"  {date}: " + ", ".join(f"{STATES[c]} {counts[c]}" for c in STATES)
                  + f", no river {counts[0]} blocks")
        if (folder / "oof.npz").exists():
            text = evaluate(np.load(folder / "oof.npz"), scenes)
            if text:
                (folder / "scene_mask_metrics.txt").write_text(text + "\n")
                print(text.split("\n\n")[0])
        else:
            print("  no oof.npz here, so no block scores")


if __name__ == "__main__":
    main()
