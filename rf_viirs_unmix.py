"""
Run 17: unmix VIIRS 375 m pixels with the run 16 Landsat class maps
(Oleson et al. 1995, Remote Sens. Environ. 54:98-112, "Unmixing multiple land-cover type
reflectances from coarse spatial resolution satellite data").

Paper -> here
  classified TM image        run 16 Landsat 30 m map (with_daksh), six classes
  AVHRR 1.1 km pixels        VIIRS I-band 375 m grid file of the same day (I1..I5)
  Gaussian PSF filter        sigma = 375 m / 2.8 (Airy-disc rule, Eq. 1-3), cut at +-2 sigma
  cover type weights         blurred class maps read at each VIIRS cell centre (Step 3)
  multiple linear regression rho_viirs = sum_k w_k * rho_k, least squares, no intercept (Eq. 4-5)
  -> one mean reflectance per class, band and date; repeated over dates = temporal profiles

Added checks the paper did not have
  shift search      VIIRS vs Landsat offset (+-180 m) picked by the best fit, since the paper
                    names misregistration as a main error source
  box fractions     plain area fractions (each 30 m pixel to the VIIRS cell holding its centre),
                    to compare with the Gaussian weights
  spatial CV        5 row-band folds of VIIRS cells: fit on 4, predict the held-out cells
  standard errors   of each class reflectance, from the least-squares covariance
  Landsat check     unmixed I1 / I2 / I3 / I5 vs the Landsat red / NIR / SWIR1 / surface temperature
                    mean of the same class on the same day (VIIRS is top of atmosphere, Landsat
                    is surface: an offset is expected, the ranking of classes should agree)

Dates: every Patches folder with a VIIRS grid file. Six already have a run 16 map. The others
get one from the run 16 with_daksh model (the same final fit, seed 0), written to
output/17_viirs_unmix/landsat_maps/.

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_viirs_unmix.py
"""

import json
from pathlib import Path
import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter

gdal.UseExceptions()

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "17_viirs_unmix"
PATCHES = Path.home() / "Desktop" / "Patches"
RUN16 = HERE / "output" / "16_rf_water_zone" / "2_with_daksh" / "six_class" / "random_forest"
NAMES = ["river", "ice", "snow", "forest", "shrub_grass_wetland", "bare_developed"]
BANDS = ["I1", "I2", "I3", "I4", "I5"]                # 0.64, 0.865, 1.61, 3.74, 11.45 um
LANDSAT_PAIR = {"I1": "red", "I2": "nir", "I3": "swir1", "I5": "thermal"}
FOOTPRINT_M = 375.0
SIGMA_M = FOOTPRINT_M / 2.8                           # paper Eq. 1-3
MIN_VALID = 0.98                                      # share of a VIIRS cell that must be mapped
MIN_CLASS_CELLS = 5.0                                 # a class needs this many cells' worth of weight
N_FOLDS = 5


def read(path):
    ds = gdal.Open(str(path))
    return ds.ReadAsArray(), ds.GetGeoTransform(), ds


def m_per_deg(lat):
    phi = np.radians(lat)
    lat_m = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    lon_m = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi)
    return lat_m, lon_m


def viirs_dates():
    out = []
    for p in sorted(PATCHES.glob("nenana_*")):
        s = p.name.split("_")[-1]
        date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        if (p / f"{date}_viirs_375m.tif").exists():
            out.append(date)
    return out


def sun_angles(date):
    """Sun elevation / azimuth for a date missing from output/angles/sun_angles.json: NOAA solar
    position at the acquisition time (README) and the box centre. Path 069 scenes get the mean
    offset of the stored path 069 dates (scene-centre values): elevation +0.27, azimuth +1.54 deg."""
    import datetime as dt
    import re
    folder = PATCHES / f"nenana_{date.replace('-', '')}"
    text = (folder / "README.txt").read_text()
    t = dt.datetime.strptime(re.search(r"acquired (\S+ \S+) UTC", text).group(1), "%Y-%m-%d %H:%M:%S")
    lat, lon = 64.314, -149.20
    g = 2 * np.pi / 365 * (t.timetuple().tm_yday - 1 + (t.hour + t.minute / 60 - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                   - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    de = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
          + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians(((t.hour * 60 + t.minute + t.second / 60) + eq + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    zen = np.arccos(np.sin(la) * np.sin(de) + np.cos(la) * np.cos(de) * np.cos(ha))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(de) * np.cos(la))) + 180
    elev = 90 - np.degrees(zen)
    scene = re.search(r"(L[CO]0\d_L2SP_\d{6}_\S+)", text).group(1)
    if "_069015_" in scene:
        elev, az = elev + 0.27, az + 1.54
    return {"sun_elevation": float(elev), "sun_azimuth": float(az), "scene": scene,
            "source": "computed in run 17 (NOAA solar position)"}


def landsat_map(date, cache):
    """Run 16 with_daksh map for this date: existing file, else predicted with the final model."""
    for p in (RUN16 / f"pred_{date}.tif", RUN16 / "new_dates" / date / "rf_pred.tif",
              OUT / "landsat_maps" / f"pred_{date}.tif"):
        if p.exists():
            return p, "run 16" if "17_viirs" not in str(p) else "run 16 model, predicted in run 17"
    if "model" not in cache:
        import data as D
        import rf_all_bands as R
        import rf_limited as L
        import rf_water_zone as W
        R.EXTRA = L.extra
        _, px, _ = L.load("with_daksh")
        x, y, r, c, _ = W.stack(px)
        cache.update(model=W.ZoneModel().fit(x, y, r, c), D=D, R=R, W=W)
        print("fitted run 16 with_daksh final model", flush=True)
    D, R, W, model = cache["D"], cache["R"], cache["W"], cache["model"]
    import rf_angles as A
    if date not in A.SUN:
        A.SUN[date] = sun_angles(date)                    # in memory only, the json is left as it is
        cache.setdefault("sun", {})[date] = A.SUN[date]
        print(date, "sun angles", A.SUN[date], flush=True)
    xs, ref = D.load_features(PATCHES / f"nenana_{date.replace('-', '')}", date)
    fx, _ = R.features(xs, date)
    flat = fx.reshape(len(fx), -1)
    valid = np.isfinite(flat).all(0)
    pred = np.zeros(flat.shape[1], np.uint8)
    idx = np.flatnonzero(valid)
    for i in range(0, len(idx), 300_000):
        j = idx[i:i + 300_000]
        pred[j] = model.predict(flat[:, j].T, W.ZONE.flat[j]) + 1
    p = OUT / "landsat_maps" / f"pred_{date}.tif"
    p.parent.mkdir(parents=True, exist_ok=True)
    D.write_prediction(p, pred.reshape(fx.shape[1:]), ref)
    p.with_suffix(".qml").write_text((RUN16 / "pred_2024-04-21.qml").read_text())
    return p, "run 16 model, predicted in run 17"


def blur(cls, gt_f):
    """Gaussian-filtered class maps (paper Step 2) + filtered 'mapped' map, shape (7, H, W)."""
    lat_m, lon_m = m_per_deg(gt_f[3] + cls.shape[0] * gt_f[5] / 2)
    s = (SIGMA_M / (abs(gt_f[5]) * lat_m), SIGMA_M / (gt_f[1] * lon_m))
    maps = [(cls == k + 1).astype(np.float32) for k in range(6)] + [(cls > 0).astype(np.float32)]
    # zero outside the box, so the mapped share drops at the edges and those cells are left out
    return np.stack([gaussian_filter(m, s, mode="constant", cval=0.0, truncate=2.0) for m in maps]), s


def weights(cls, gt_f, gt_v, shape_v, shift=(0, 0), kind="psf", blurred=None):
    """(n_cells, 6) class weights and the mapped share per VIIRS cell (row-major).
    shift = (rows, cols) in fine pixels that the Landsat map is moved by (positive = south / east)."""
    H, Wd = cls.shape
    hv, wv = shape_v
    if kind == "box":
        lat = gt_f[3] + (np.arange(H) + 0.5 + shift[0]) * gt_f[5]
        lon = gt_f[0] + (np.arange(Wd) + 0.5 + shift[1]) * gt_f[1]
        rows = np.floor((lat - gt_v[3]) / gt_v[5]).astype(int)
        cols = np.floor((lon - gt_v[0]) / gt_v[1]).astype(int)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        ok = (rr >= 0) & (rr < hv) & (cc >= 0) & (cc < wv)
        cell = rr[ok] * wv + cc[ok]
        out = np.stack([np.bincount(cell, weights=(cls[ok] == k + 1), minlength=hv * wv) for k in range(6)], 1)
        mapped = np.bincount(cell, weights=(cls[ok] > 0), minlength=hv * wv)
        # cells hold 12 or 13 fine px per axis (375 m / 30 m = 12.5): divide by the real count, not 12.5^2
        # (floor at 12x12 so edge cells only partly inside the map still count as part-mapped)
        smallest = np.floor(gt_v[5] / gt_f[5]) * np.floor(gt_v[1] / gt_f[1])
        full = np.maximum(np.bincount(cell, minlength=hv * wv), smallest)
        return out / np.maximum(mapped, 1)[:, None], mapped / full
    lat_c = gt_v[3] + (np.arange(hv) + 0.5) * gt_v[5]
    lon_c = gt_v[0] + (np.arange(wv) + 0.5) * gt_v[1]
    fr = np.round((lat_c - gt_f[3]) / gt_f[5] - 0.5 - shift[0]).astype(int)
    fc = np.round((lon_c - gt_f[0]) / gt_f[1] - 0.5 - shift[1]).astype(int)
    rr, cc = np.meshgrid(fr, fc, indexing="ij")
    inside = ((rr >= 0) & (rr < H) & (cc >= 0) & (cc < Wd)).ravel()
    b = blurred[:, np.clip(rr, 0, H - 1), np.clip(cc, 0, Wd - 1)].reshape(7, -1)
    mapped = np.where(inside, b[6], 0)
    return (b[:6] / np.maximum(b[6], 1e-9)).T, mapped


def solve(w, y):
    """Least squares rho (Eq. 5), standard errors, R^2 (Eq. 8-11) and the fit."""
    rho, *_ = np.linalg.lstsq(w, y, rcond=None)
    fit = w @ rho
    res = y - fit
    s2 = (res @ res) / max(1, len(y) - w.shape[1])
    se = np.sqrt(np.maximum(np.diag(s2 * np.linalg.pinv(w.T @ w)), 0))
    sst = ((y - y.mean()) ** 2).sum()
    return rho, se, float(1 - (res @ res) / max(sst, 1e-12)), fit


def cv_r2(w, y, rows):
    """5 row-band folds with a 1-row gap: R^2 and RMSE on held-out cells."""
    edges = np.linspace(rows.min(), rows.max() + 1, N_FOLDS + 1)
    pred = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        test = (rows >= edges[f]) & (rows < edges[f + 1])
        train = (rows < edges[f] - 1) | (rows >= edges[f + 1] + 1)
        if test.any() and train.sum() > w.shape[1] * 5:
            rho, *_ = np.linalg.lstsq(w[train], y[train], rcond=None)
            pred[test] = w[test] @ rho
    ok = np.isfinite(pred)
    sse = ((y[ok] - pred[ok]) ** 2).sum()
    sst = ((y[ok] - y[ok].mean()) ** 2).sum()
    return float(1 - sse / max(sst, 1e-12)), float(np.sqrt(sse / max(ok.sum(), 1)))


def usable(w, mapped, v):
    """Cells fully mapped with finite VIIRS, the classes that can be solved, cells kept."""
    good = (mapped >= MIN_VALID) & np.isfinite(v.reshape(len(v), -1)).all(0)
    use = w[good].sum(0) >= MIN_CLASS_CELLS
    # cells holding a meaningful share of a left-out class cannot be explained, drop them
    good &= ~(w[:, ~use].sum(1) > 0.05)
    return good, use


def unmix(w, mapped, v, rows, bands=BANDS, cv=True):
    good, use = usable(w, mapped, v)
    out = {"n_cells": int(good.sum()), "classes_used": [NAMES[k] for k in np.flatnonzero(use)],
           "class_share_pct": {NAMES[k]: float(100 * w[good, k].sum() / max(good.sum(), 1)) for k in range(6)},
           "bands": {}}
    ww = w[good][:, use]
    for name in bands:
        y = v[BANDS.index(name)].ravel()[good]
        rho, se, r2, fit = solve(ww, y)
        row = {"r2": r2, "rmse": float(np.sqrt(((y - fit) ** 2).mean())),
               "rho": {NAMES[k]: float(x) for k, x in zip(np.flatnonzero(use), rho)},
               "se": {NAMES[k]: float(x) for k, x in zip(np.flatnonzero(use), se)}}
        if cv:
            row["cv_r2"], row["cv_rmse"] = cv_r2(ww, y, rows[good])
        out["bands"][name] = row
    return out, good, use


def landsat_means(date, cls):
    folder = PATCHES / f"nenana_{date.replace('-', '')}"
    sr = gdal.Open(str(folder / f"{date}_sr.tif")).ReadAsArray().astype(np.float32)
    st = gdal.Open(str(folder / f"{date}_thermal.tif")).ReadAsArray().astype(np.float32)
    out = {}
    for name, a in {"red": sr[2], "nir": sr[3], "swir1": sr[4], "thermal": st}.items():
        out[name] = {}
        for k in range(6):
            m = (cls == k + 1) & np.isfinite(a)
            out[name][NAMES[k]] = float(np.mean(a[m])) if m.sum() >= 50 else None
    return out


def write(path, arr, gt_v, ds_v, names):
    o = gdal.GetDriverByName("GTiff").Create(str(path), arr.shape[2], arr.shape[1], arr.shape[0],
                                             gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
    o.SetGeoTransform(gt_v); o.SetProjection(ds_v.GetProjection())
    for i, n in enumerate(names):
        band = o.GetRasterBand(i + 1); band.WriteArray(arr[i]); band.SetDescription(n); band.SetNoDataValue(np.nan)
    o = None


def load_date(date, cache):
    """Run 16 class map, VIIRS grid (9 bands), their geotransforms and the VIIRS dataset."""
    folder = PATCHES / f"nenana_{date.replace('-', '')}"
    map_path, source = landsat_map(date, cache)
    cls, gt_f, _ = read(map_path)
    v, gt_v, ds_v = read(folder / f"{date}_viirs_375m.tif")
    return cls, gt_f, v.astype(np.float64), gt_v, ds_v, map_path, source


def best_shift(cls, gt_f, v5, gt_v, bl):
    """Fine-pixel shift (+-6 rows ~ 180 m, +-14 cols ~ 180 m) with the best mean R^2 of I1..I3."""
    hv, wv = v5.shape[1:]
    rows = np.repeat(np.arange(hv), wv)
    best = None
    for dr in range(-6, 7):
        for dc in range(-14, 15, 2):
            w, mapped = weights(cls, gt_f, gt_v, (hv, wv), (dr, dc), "psf", bl)
            o, _, _ = unmix(w, mapped, v5, rows, ("I1", "I2", "I3"), cv=False)
            score = np.mean([o["bands"][b]["r2"] for b in ("I1", "I2", "I3")])
            if best is None or score > best[0]:
                best = (score, (dr, dc))
    return best[1]


def run_date(date, cache):
    cls, gt_f, v, gt_v, ds_v, map_path, source = load_date(date, cache)
    v5 = v[:5]
    hv, wv = v5.shape[1:]
    rows = np.repeat(np.arange(hv), wv)
    bl, sig = blur(cls, gt_f)
    shift = best_shift(cls, gt_f, v5, gt_v, bl)
    lat_m, lon_m = m_per_deg(gt_f[3])
    shift_m = [round(shift[0] * abs(gt_f[5]) * lat_m), round(shift[1] * gt_f[1] * lon_m)]
    res = {"date": date, "map": str(map_path), "map_source": source,
           "sigma_fine_px": [round(s, 2) for s in sig], "shift_fine_px": list(shift),
           "shift_m_south_east": shift_m, "vza_deg": float(np.nanmean(v[7]))}
    d = OUT / date; d.mkdir(parents=True, exist_ok=True)
    for kind in ("psf", "box"):
        for sh, tag in ((shift, "shifted"), ((0, 0), "no_shift")):
            w, mapped = weights(cls, gt_f, gt_v, (hv, wv), sh, kind, bl)
            o, good, use = unmix(w, mapped, v5, rows)
            res[f"{kind}_{tag}"] = o
            if kind == "psf" and tag == "shifted":
                fit = np.full((5, hv * wv), np.nan)
                for b, name in enumerate(BANDS):
                    rho = np.array([o["bands"][name]["rho"][NAMES[k]] for k in np.flatnonzero(use)])
                    fit[b, good] = w[good][:, use] @ rho
                write(d / "weights_psf.tif", np.where(good, w.T, np.nan).reshape(6, hv, wv), gt_v, ds_v, NAMES)
                write(d / "viirs_fit.tif", fit.reshape(5, hv, wv), gt_v, ds_v, BANDS)
                write(d / "viirs_residual.tif", (v5.reshape(5, -1) - fit).reshape(5, hv, wv), gt_v, ds_v, BANDS)
    res["landsat_class_mean"] = landsat_means(date, cls)
    m = res["psf_shifted"]
    print(f"{date}  {source:<34} cells {m['n_cells']:>5}  shift {shift_m} m  "
          + "  ".join(f"{b} R2 {m['bands'][b]['r2']:.2f} cv {m['bands'][b]['cv_r2']:.2f}" for b in BANDS[:3]), flush=True)
    return res


def summary(all_res):
    L = ["Run 17: VIIRS unmixing with the run 16 Landsat maps (Oleson et al. 1995)",
         f"Gaussian PSF sigma {SIGMA_M:.0f} m (375 m / 2.8), cut at 2 sigma; cells used need >= {MIN_VALID:.0%} mapped.",
         "Main result = psf_shifted (Gaussian weights, best VIIRS-Landsat shift).",
         "VIIRS I1-I3 = TOA reflectance, I4/I5 = brightness temperature (K).",
         "In brackets: Landsat mean of the same class that day (surface reflectance red/NIR/SWIR1, surface temperature K).", ""]
    for r in all_res:
        m = r["psf_shifted"]
        L.append(f"=== {r['date']}   map: {r['map_source']}   cells {m['n_cells']}   "
                 f"shift {r['shift_m_south_east']} m (south, east)   VZA {r['vza_deg']:.0f} deg")
        L.append("class share %: " + ", ".join(f"{k} {v:.1f}" for k, v in m["class_share_pct"].items()))
        L.append(f"{'band':<5}{'R2':>6}{'cvR2':>6}{'cvRMSE':>8}{'R2box':>7}{'R2noshift':>10}   class value +- se (Landsat)")
        for b in BANDS:
            x = m["bands"][b]
            ls = r["landsat_class_mean"].get(LANDSAT_PAIR.get(b, ""), {})
            parts = [f"{k[:6]} {x['rho'][k]:.3f}+-{x['se'][k]:.3f}"
                     + (f" ({ls[k]:.3f})" if ls.get(k) is not None else "") for k in NAMES if k in x["rho"]]
            L.append(f"{b:<5}{x['r2']:>6.2f}{x['cv_r2']:>6.2f}{x['cv_rmse']:>8.3f}"
                     f"{r['box_shifted']['bands'][b]['r2']:>7.2f}{r['psf_no_shift']['bands'][b]['r2']:>10.2f}   "
                     + ", ".join(parts))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cache, all_res = {}, []
    for date in viirs_dates():
        all_res.append(run_date(date, cache))
    (OUT / "results.json").write_text(json.dumps(all_res, indent=1))
    (OUT / "sun_angles_computed.json").write_text(json.dumps(cache.get("sun", {}), indent=1))
    (OUT / "summary.txt").write_text(summary(all_res) + "\n")
    print("wrote", OUT)
