"""
Run 16: six classes with the JRC water occurrence as a hard zone rule.

    water zone  JRC occurrence >= 5 %   only river or ice      (no snow, no land)
    land        JRC occurrence <  5 %   only snow, forest, shrub_grass_wetland, bare_developed
                                         (no ice, no river; land split by NLCD 2016 as before)

Two Random Forests, one per zone, each trained only on its zone's pixels:
  water RF  river vs ice
  land RF   snow / forest / shrub_grass_wetland / bare_developed
Same limited settings as runs 13-15 (rf_limited input pool, top 25 inputs ranked on the
training fold only, depth 16, min_samples_leaf 20, sqrt, 300 trees, balanced_subsample).

Labels are made to fit the rule ("rule labels"):
  in the water zone  snow -> ice (snow lying on frozen river); forest / shrub / bare dropped
  on land            river / ice dropped
Scores are given against the rule labels (main) and against the original hand labels
(every labelled pixel, so labels the rule cannot reproduce count as errors).

Outputs: output/16_rf_water_zone/<dataset folder>/six_class/random_forest/
  metrics.txt   label changes, CV scores (rule labels, original labels), per date, inputs
  oof.npz, pred_<date>.tif, new_dates/<date>/rf_pred.tif, images/
output/16_rf_water_zone/water_zone.tif   the zone (1 water, 0 land)

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_water_zone.py
"""

import numpy as np
from osgeo import gdal
import data as D
import rf_all_bands as R
import rf_limited as L

OUT = D.HERE / "output" / "16_rf_water_zone"
THRESHOLD = 5.0                                  # JRC occurrence % from which a pixel is water zone
K, DEPTH = 25, 16
RIVER, ICE, SNOW = 0, 1, 2                       # class indices (forest 3, shrub 4, bare 5)
WATER_CLASSES, LAND_CLASSES = (RIVER, ICE), (SNOW, 3, 4, 5)
ZONE = D._water_occurrence() >= THRESHOLD


def rule_labels(y, zone):
    """Hand labels moved onto the rule; IGNORE where the rule cannot hold them."""
    out = y.copy()
    out[zone & (y == SNOW)] = ICE
    out[zone & np.isin(y, (3, 4, 5))] = D.IGNORE
    out[~zone & np.isin(y, WATER_CLASSES)] = D.IGNORE
    return out


def stack(px, mask=None):
    sel = [(p, mask[p["r"]] if mask is not None else np.ones(len(p["y"]), bool)) for p in px]
    cat = lambda k: np.concatenate([p[k][t] for p, t in sel])
    x, y, r, c = cat("x"), cat("y"), cat("r"), cat("c")
    dates = np.concatenate([[p["date"]] * int(t.sum()) for p, t in sel])
    return x, y, r, c, dates


class ZoneModel:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, x, y, r, c):
        z = ZONE[r, c]; yr = rule_labels(y, z)
        self.parts = {}
        for name, m in (("water", z), ("land", ~z)):
            t = m & (yr != D.IGNORE)
            keep = L.rank(x[t], yr[t], self.seed)[:K]
            self.parts[name] = (L.forest(DEPTH, seed=self.seed).fit(x[t][:, keep], yr[t]), keep)
        return self

    def predict(self, x, zone):
        out = np.empty(len(x), np.uint8)
        for name, m in (("water", zone), ("land", ~zone)):
            if m.any():
                f, keep = self.parts[name]
                out[m] = f.predict(x[m][:, keep])
        return out


def change_table(y, z):
    names = D.CLASSES
    lines = [f"label changes for the rule (water zone = JRC >= {THRESHOLD:g} %)",
             f"{'class':<22}{'labelled':>9}{'in zone':>9}{'-> ice':>8}{'dropped':>9}"]
    yr = rule_labels(y, z)
    for k in range(len(names)):
        m = y == k
        lines.append(f"{names[k + 1]:<22}{m.sum():>9}{(m & z).sum():>9}"
                     f"{(m & (yr == ICE) & (y != ICE)).sum():>8}{(m & (yr == D.IGNORE)).sum():>9}")
    return "\n".join(lines)


def scores(y, p, z, title):
    yr = rule_labels(y, z); k = yr != D.IGNORE
    return [D.report(yr[k], p[k], f"{title}, rule labels")[0],
            D.report(y, p, f"{title}, original hand labels")[0]]


def run(dataset):
    out = OUT / D.DATASET_DIRS[dataset] / "six_class" / "random_forest"
    (out / "images").mkdir(parents=True, exist_ok=True)
    scenes, px, names = L.load(dataset)
    print(f"\n=== {dataset}: pool {len(names)} inputs, top {K}, depth {DEPTH}, zone JRC >= {THRESHOLD:g} %")
    blocks = []
    if dataset == "external_daksh":
        train = [p for p in px if p["group"] == "mathus_naira"]; test = [p for p in px if p["group"] == "daksh"]
        x, y, r, c, _ = stack(train)
        model = ZoneModel().fit(x, y, r, c)
        xt, yt, rt, ct, dt = stack(test)
        zt = ZONE[rt, ct]; pt = model.predict(xt, zt)
        blocks.append(change_table(yt, zt))
        blocks += scores(yt, pt, zt, "external test 2024-03-12")
        D.save_oof(out / "oof.npz", dt, rt, ct, yt, pt)
    else:
        x, y, r, c, _ = stack(px)
        blocks.append(change_table(y, ZONE[r, c]))
        ys, ps, ds, rs, cs = [], [], [], [], []
        for f in range(D.N_FOLDS):
            test_rows, train_rows = D.fold_masks(R.H, f)
            model = ZoneModel(f).fit(*stack(px, train_rows)[:4])
            xt, yt, rt, ct, dt = stack(px, test_rows)
            ps.append(model.predict(xt, ZONE[rt, ct])); ys.append(yt); ds.append(dt); rs.append(rt); cs.append(ct)
        yt, pt, dt, rt, ct = map(np.concatenate, (ys, ps, ds, rs, cs))
        zt = ZONE[rt, ct]
        D.save_oof(out / "oof.npz", dt, rt, ct, yt, pt)
        blocks += scores(yt, pt, zt, "water-zone RF — 5-fold spatial CV, all labelled dates")
        for date in sorted(set(dt)):
            m = dt == date
            yr = rule_labels(yt[m], zt[m]); k = yr != D.IGNORE
            blocks.append(D.report(yr[k], pt[m][k], f"--- {date} (rule labels)")[0])
        model = ZoneModel().fit(x, y, r, c)
    for name, (_, keep) in model.parts.items():
        blocks.append(f"{name} RF inputs ({K}): " + ", ".join(names[i] for i in keep))
    (out / "metrics.txt").write_text("\n\n".join(blocks) + "\n")
    print("\n\n".join(blocks[:3]))

    def predict(xs, date):
        fx, _ = R.features(xs, date)
        flat = fx.reshape(len(fx), -1)
        valid = np.isfinite(flat).all(0)
        pred = np.zeros(flat.shape[1], np.uint8)
        idx = np.flatnonzero(valid)
        for i in range(0, len(idx), 300_000):
            j = idx[i:i + 300_000]
            pred[j] = model.predict(flat[:, j].T, ZONE.flat[j]) + 1
        return pred.reshape(fx.shape[1:])

    title = f"RF water zone (JRC >= {THRESHOLD:g} %: river/ice; else snow/land)"
    for s in scenes:
        pred = predict(s["x"], s["date"]); pred[~s["valid"]] = 0
        D.write_prediction(out / f"pred_{s['date']}.tif", pred, s["ref"])
        img = R.I.image_for(s["date"])
        if img is not None:
            R.jpeg(img, pred, f"{s['date']}   {title}", out / "images" / f"rf_{s['date']}.jpg")
    if dataset != "external_daksh":
        for date in R.NEW_DATES:
            xs, ref = D.load_features(R.NEW / f"nenana_{date.replace('-', '')}", date)
            pred = predict(xs, date)
            d = out / "new_dates" / date; d.mkdir(parents=True, exist_ok=True)
            D.write_prediction(d / "rf_pred.tif", pred, ref)
            R.jpeg(R.I.image_for(date), pred, f"{date} (no labels)   {title}", out / "images" / f"new_rf_{date}.jpg")
    print("wrote", out)
    return scenes[0]["ref"]


def write_zone(ref):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(OUT / "water_zone.tif"), ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Byte,
                    options=["COMPRESS=DEFLATE"])
    ds.SetGeoTransform(ref.GetGeoTransform()); ds.SetProjection(ref.GetProjection())
    ds.GetRasterBand(1).WriteArray(ZONE.astype(np.uint8)); ds = None


if __name__ == "__main__":
    R.EXTRA = L.extra
    for ds in ("exclude_daksh", "with_daksh", "external_daksh"):
        ref = run(ds)
    write_zone(ref)
