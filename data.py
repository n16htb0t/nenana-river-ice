"""
Shared data loading for the Landsat 30 m models (Random Forest and UNet++).

Inputs per Landsat pixel:
  exclude_daksh (11): SR_B2..SR_B7 surface reflectance, ST_B10 (K),
                      JRC water occurrence (%), NDWI, NDSI, NDVI
  with_daksh    (9) : blue, green, red, NIR, SWIR1, water occurrence, NDWI, NDSI, NDVI
The daksh tiffs (landsat_*_box.tif) are Level-1 DN for B1..B6 with no SWIR2 and no
ST_B10, so with_daksh keeps only what every date has. Their DN become reflectance
with the Collection 2 constants (x 2.0e-5, - 0.1) divided by cos(SZA), SZA read
from the tiff itself (stored x 100); the patches use their surface reflectance.

Labels: the hand-drawn polygons, class_id 1 river, 2 ice, 3 snow, 4 land.
Anything else (0 not drawn, 9 no data) and pixels with missing imagery are ignored.
A TASK maps those four onto what a model learns:
  four   river / ice / snow / land
  snow   no snow (river, land)   vs snow (ice, snow)
  river  not river (snow, land)  vs river (river, ice)
The two binary answers together give the four classes back (see combine.py).
Independent one-vs-rest tasks, one model per class:
  ind_river  river (open water) vs rest     ind_ice   ice  vs rest
  ind_snow   snow vs rest                   ind_land  land vs rest

DATASET picks the dates:
  exclude_daksh   the labelled patch dates (patches/nenana_*)
  with_daksh      those + the daksh dates (nenana_crops/<YYYYMMDD>/landsat_*_box.tif)
  external_daksh  external test: train on the patch dates only (9 inputs), score on
                  the labelled daksh date (external_test.py)

Folds: the study box is cut into east-west bands of BLOCK rows and band i goes
to fold i % N_FOLDS, so test pixels are never next to training pixels of the
same polygon. Training pixels within BUFFER rows of a test band are dropped.

Outputs go to output/<dataset folder>/<task folder>/<model folder>/, names in
DATASET_DIRS, task_dir() and MODEL_DIRS.
"""

import os
from pathlib import Path
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")

HERE      = Path(__file__).resolve().parent
PATCHES   = Path.home() / "Desktop/RiverProject/Matus/narrow_rivers/patches"
CROPS     = HERE.parent                                   # ~/Desktop/nenana_crops
# RIVER_CORRIDOR_M=150 in the environment: train, score and predict only inside the river
# corridor (JRC water ever seen, widened by that many metres), outputs in their own folder
CORRIDOR_M = int(os.environ.get("RIVER_CORRIDOR_M", "0"))
OUTPUT    = HERE / "output" / (f"4_river_corridor_{CORRIDOR_M}m" if CORRIDOR_M else "")

IGNORE    = 255                      # label value for "no label" in index space
CHANNELS_ALL    = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                   "ST_B10", "water_occ", "NDWI", "NDSI", "NDVI"]
CHANNELS_COMMON = ["blue", "green", "red", "nir", "swir1", "water_occ", "NDWI", "NDSI", "NDVI"]
N_FOLDS   = 5
BLOCK     = 64                       # rows per fold band (~1.9 km)
BUFFER    = 12                       # rows dropped from training next to test bands

HAND      = {1: "river", 2: "ice", 3: "snow", 4: "land"}
TASKS = {
    # task: (classes {output id: name}, hand class_id -> output id, colours per output id)
    "four":  ({1: "river", 2: "ice", 3: "snow", 4: "land"}, {1: 1, 2: 2, 3: 3, 4: 4},
              {1: "#0066ff", 2: "#ff00ff", 3: "#00ffff", 4: "#8b4513"}),
    "snow":  ({1: "no_snow", 2: "snow"}, {1: 1, 2: 2, 3: 2, 4: 1},
              {1: "#8b4513", 2: "#00ffff"}),
    "river": ({1: "not_river", 2: "river"}, {1: 2, 2: 2, 3: 1, 4: 1},
              {1: "#999999", 2: "#0066ff"}),
}
for _cid, _name, _col in ((1, "river", "#0066ff"), (2, "ice", "#ff00ff"),
                          (3, "snow", "#00ffff"), (4, "land", "#8b4513")):
    TASKS[f"ind_{_name}"] = ({1: f"not_{_name}", 2: _name},
                             {c: (2 if c == _cid else 1) for c in HAND},
                             {1: "#999999", 2: _col})
DATASETS = ("exclude_daksh", "with_daksh", "external_daksh")
DATASET_DIRS = {"exclude_daksh": "1_exclude_daksh", "with_daksh": "2_with_daksh",
                "external_daksh": "3_external_test_daksh"}
MODEL_DIRS = {"rf": "random_forest", "unetpp": "unet_plus_plus"}
MODEL_NAMES = {"rf": "Random Forest", "unetpp": "UNet++"}

# set by configure()
TASK, DATASET, CLASSES, MAP, COLOURS, OUT, CHANNELS = ("four", "exclude_daksh",
                                                       None, None, None, None, None)


def configure(task="four", dataset="exclude_daksh"):
    global TASK, DATASET, CLASSES, MAP, COLOURS, OUT, CHANNELS
    TASK, DATASET = task, dataset
    CLASSES, MAP, COLOURS = TASKS[task]
    OUT = task_dir(task, dataset)
    CHANNELS = CHANNELS_ALL if dataset == "exclude_daksh" else CHANNELS_COMMON


def task_dir(task, dataset):
    """Output folder of a task ("combined" = binary snow + river merged)."""
    base = OUTPUT / DATASET_DIRS[dataset]
    if task == "four":
        return base / "four_class_river_ice_snow_land"
    if task.startswith("ind_"):
        return base / "one_vs_rest" / f"{task[4:]}_vs_rest"
    return base / "binary" / {"snow": "snow_vs_no_snow", "river": "river_vs_not_river",
                              "combined": "combined_to_four_class"}[task]


configure()


def _date(s):
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def scene_list():
    """[(group, date, folder, label file or None)] for the configured dataset."""
    out = []
    for d in sorted(PATCHES.glob("nenana_*")):
        date = _date(d.name.split("_")[1])
        if (d / f"{date}_sr.tif").exists():
            out.append(("mathus_naira", date, d, d / "labels" / f"labels_{date}.shp"))
    if DATASET in ("with_daksh", "external_daksh"):
        for d in sorted(p for p in CROPS.iterdir() if p.is_dir() and p.name.isdigit()):
            date = _date(d.name)
            if not any(d.glob("landsat_*_box.tif")):
                continue
            lab = d / "labels" / f"labels_{date}.shp"
            alt = Path.home() / f"labels_{d.name}.gpkg"           # where 2024-03-12 was saved
            if not _n_features(lab) and _n_features(alt):
                lab = alt
            out.append(("daksh", date, d, lab))
    return out


def _n_features(path):
    if not path or not path.exists():
        return 0
    from osgeo import ogr
    src = ogr.Open(str(path))
    return src.GetLayer(0).GetFeatureCount() if src else 0


def _water_occurrence():
    """JRC occurrence on the box grid (identical in every patch folder)."""
    f = next(PATCHES.glob("nenana_*/*_water_occurrence.tif"))
    wo = gdal.Open(str(f)).ReadAsArray().astype(np.float32)
    wo[wo == 255] = 0                                          # outside JRC = never water
    return wo


def _daksh_reflectance(folder):
    """landsat_*_box.tif DN -> TOA reflectance for B2..B6, + GDAL ref."""
    ds = gdal.Open(str(next(folder.glob("landsat_*_box.tif"))))
    band = {ds.GetRasterBand(i).GetDescription(): i for i in range(1, ds.RasterCount + 1)}
    sza = ds.GetRasterBand(band["SZA"]).ReadAsArray().astype(np.float32)
    sza[sza == -9999] = np.nan
    cos = np.cos(np.radians(sza / 100.0))
    out = []
    for b in ("B2", "B3", "B4", "B5", "B6"):
        dn = ds.GetRasterBand(band[b]).ReadAsArray().astype(np.float32)
        dn[(dn == -9999) | (dn == 0)] = np.nan
        out.append((dn * 2.0e-5 - 0.1) / cos)
    return np.stack(out), ds


def load_features(folder, date):
    """(C, H, W) float32 features (C = len(CHANNELS)), NaN where imagery is missing, + GDAL ref."""
    if any(folder.glob("landsat_*_box.tif")):
        refl, ref = _daksh_reflectance(folder)                 # blue, green, red, nir, swir1
        st = None
    else:
        ref = gdal.Open(str(folder / f"{date}_sr.tif"))
        sr = ref.ReadAsArray().astype(np.float32)              # B2..B7
        refl = sr if DATASET == "exclude_daksh" else sr[:5]
        st = gdal.Open(str(folder / f"{date}_thermal.tif")).ReadAsArray().astype(np.float32)
    wo = _water_occurrence()
    blue, green, red, nir, swir1 = refl[:5]
    eps = 1e-6
    ndwi = (green - nir) / (green + nir + eps)
    ndsi = (green - swir1) / (green + swir1 + eps)
    ndvi = (nir - red) / (nir + red + eps)
    parts = [refl] + ([st[None]] if DATASET == "exclude_daksh" else []) + \
            [wo[None], ndwi[None], ndsi[None], ndvi[None]]
    x = np.concatenate(parts)
    assert len(x) == len(CHANNELS)
    return x, ref


def load_labels(label_file, ref):
    """(H, W) uint8 task class index 0..K-1, IGNORE elsewhere."""
    ds = gdal.GetDriverByName("MEM").Create("", ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(ref.GetGeoTransform())
    ds.SetProjection(ref.GetProjection())
    if _n_features(label_file):
        gdal.Rasterize(ds, str(label_file), attribute="class_id", where="class_id IS NOT NULL")
    lab = ds.ReadAsArray()
    y = np.full(lab.shape, IGNORE, np.uint8)
    for hand, out in MAP.items():
        y[lab == hand] = out - 1
    return y


_CORRIDOR = None


def corridor():
    """River corridor mask on the box grid: JRC occurrence >= 1 %, widened by CORRIDOR_M
    (an ellipse, since rows are 30 m and columns about 13 m). All True when off."""
    global _CORRIDOR
    if _CORRIDOR is None:
        wo = _water_occurrence()
        if not CORRIDOR_M:
            _CORRIDOR = np.ones(wo.shape, bool)
        else:
            from scipy import ndimage
            ry, rx = CORRIDOR_M / 30.0, CORRIDOR_M / 13.0
            yy, xx = np.ogrid[-int(ry):int(ry) + 1, -int(rx):int(rx) + 1]
            _CORRIDOR = ndimage.binary_dilation(wo >= 1, (yy / ry) ** 2 + (xx / rx) ** 2 <= 1)
    return _CORRIDOR


def load_all():
    """Every date of the dataset with imagery: list of dicts with x, y, valid, ref."""
    scenes = []
    for group, date, folder, lab in scene_list():
        x, ref = load_features(folder, date)
        valid = np.isfinite(x).all(0) & corridor()
        y = load_labels(lab, ref)
        y[~valid] = IGNORE
        scenes.append(dict(group=group, date=date, folder=folder, x=x, y=y, valid=valid, ref=ref))
    return scenes


def fold_masks(h, k):
    """(test_rows, train_rows) boolean row masks for fold k."""
    rows = np.arange(h)
    test = (rows // BLOCK) % N_FOLDS == k
    near = np.zeros(h, bool)
    for r in np.flatnonzero(test):
        near[max(0, r - BUFFER): r + BUFFER + 1] = True
    return test, ~near


def channel_stats(scenes):
    """Per-channel mean / std over all valid pixels (for UNet++ input scaling)."""
    v = np.concatenate([s["x"][:, s["valid"]] for s in scenes], axis=1)
    return v.mean(1).astype(np.float32), (v.std(1) + 1e-6).astype(np.float32)


def pred_qml(classes=None, colours=None):
    classes, colours = classes or CLASSES, colours or COLOURS
    entries = "\n".join(f'    <paletteEntry value="{i}" color="{colours[i]}" alpha="255" label="{i} {n}"/>'
                        for i, n in classes.items())
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34" styleCategories="Symbology"><pipe>\n'
            '  <rasterrenderer type="paletted" band="1" opacity="1"><colorPalette>\n'
            f"{entries}\n  </colorPalette></rasterrenderer></pipe></qgis>\n")


def write_prediction(path, pred, ref, classes=None, colours=None):
    """pred: (H, W) uint8 output id 1..K, 0 = no prediction. Writes tif + qml."""
    classes = classes or CLASSES
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), ref.RasterXSize, ref.RasterYSize, 1,
                                              gdal.GDT_Byte, ["COMPRESS=DEFLATE"])
    ds.SetGeoTransform(ref.GetGeoTransform())
    ds.SetProjection(ref.GetProjection())
    b = ds.GetRasterBand(1)
    b.WriteArray(pred)
    b.SetNoDataValue(0)
    b.SetDescription(" ".join(f"{i}={n}" for i, n in classes.items()))
    ds = None
    path.with_suffix(".qml").write_text(pred_qml(classes, colours))


def scene_class(pred):
    """Whole-date combined class from a four-class 30 m prediction map.

    River state: ice-covered when predicted ice pixels outnumber open-river ones.
    Land state: snow-covered when predicted snow pixels outnumber snow-free land.
    """
    n = {c: int((pred == c).sum()) for c in HAND}
    river = "ice_covered_river" if n[2] > n[1] else "ice_free_river"
    land = "snow_covered_land" if n[3] > n[4] else "snow_free_land"
    return f"{river}_{land}", n


def report(y_true, y_pred, title, classes=None):
    """Text block: per-class precision / recall / F1, confusion matrix, totals.
    Classes with no true and no predicted pixels are left out of the macro F1."""
    classes = classes or CLASSES
    k = len(classes)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y_true, y_pred), 1)
    names = list(classes.values())
    w = max(9, max(len(n) for n in names) + 1)
    lines = [title, "", f"{'class':{w}}{'n':>9}{'precision':>11}{'recall':>9}{'F1':>7}"]
    f1s = []
    for i, n in enumerate(names):
        tp = cm[i, i]; p = tp / max(cm[:, i].sum(), 1); r = tp / max(cm[i].sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if cm[i].sum() or cm[:, i].sum():
            f1s.append(f1)
        lines.append(f"{n:{w}}{cm[i].sum():>9,}{p:>11.3f}{r:>9.3f}{f1:>7.3f}")
    lines += ["", f"overall accuracy {np.trace(cm) / max(cm.sum(), 1):.3f}   macro F1 {np.mean(f1s):.3f}",
              "", "confusion (rows = true, cols = predicted)",
              f"{'':{w}}" + "".join(f"{n:>{w}}" for n in names)]
    lines += [f"{n:{w}}" + "".join(f"{v:>{w},}" for v in cm[i]) for i, n in enumerate(names)]
    return "\n".join(lines), cm


def save_oof(path, dates, rows, cols, y, p):
    """Out-of-fold (cross-validated) predictions, used by combine.py."""
    np.savez_compressed(path, date=np.array(dates), row=np.asarray(rows), col=np.asarray(cols),
                        y=np.asarray(y), p=np.asarray(p))
