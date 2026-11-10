"""
Six-class Random Forest (rf_six_classes.py labels) with terrain and context inputs added to
the band pairs, indices, textures and NLCD land cover, so the forest itself learns where a
river can and cannot be. No mask or rule is applied after prediction.

Extra inputs:
  dist_nlcd_water_m       distance to the nearest NLCD water pixel (11), metres
  elev_m, slope_deg       Copernicus GLO-30 DEM on the box grid (output/dem/)
  northness, eastness     cos / sin of aspect
  rel_elev_1km_m          elevation above the lowest point within about 1 km (valley floor = 0)
  rel_elev_300m_m         the same within about 300 m
  <index>_ctx15/_ctx31    mean of MNDWI, NDWI, NDVI and visible brightness in 15 x 15 and
                          31 x 31 pixel windows (a river is a long connected strip; a dark
                          shadow pixel in forest is isolated)

Outputs: output/10_rf_six_classes_terrain/<dataset folder>/six_class/random_forest/

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_terrain.py
"""

import numpy as np
from osgeo import gdal
from scipy import ndimage
import data as D
import rf_all_bands as R
import rf_six_classes as S

OUT = D.HERE / "output" / "10_rf_six_classes_terrain"
DEM = D.HERE / "output" / "dem" / "copernicus_glo30_box.tif"
DY, DX = 30.0, 13.0                     # metres per row / column (0.00027027 deg at 64.3 N)
CONFIGS = [dict(max_features=0.1, min_samples_leaf=5),
           dict(max_features="sqrt", min_samples_leaf=5),
           dict(max_features="sqrt", min_samples_leaf=10)]
_STATIC = None


def static_inputs():
    """Terrain and distance-to-water inputs, the same on every date (cached)."""
    global _STATIC
    if _STATIC is None:
        dem = gdal.Open(str(DEM)).ReadAsArray().astype(np.float32)
        nlcd = gdal.Open(str(S.NLCD)).ReadAsArray()
        gy, gx = np.gradient(dem, DY, DX)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect = np.arctan2(-gx, gy)
        dist = ndimage.distance_transform_edt(nlcd != 11, sampling=(DY, DX))
        rel1k = dem - ndimage.minimum_filter(dem, size=(33, 77))
        rel300 = dem - ndimage.minimum_filter(dem, size=(11, 23))
        _STATIC = [("dist_nlcd_water_m", dist), ("elev_m", dem), ("slope_deg", slope),
                   ("northness", np.cos(aspect)), ("eastness", np.sin(aspect)),
                   ("rel_elev_1km_m", rel1k), ("rel_elev_300m_m", rel300)]
        _STATIC = [(n, a.astype(np.float32)) for n, a in _STATIC]
    return _STATIC


def context_inputs(x):
    """Neighbourhood means of water, vegetation and brightness indices for one date."""
    b = R.raw_bands(x)
    G, R_, N, S1, B = b["green"], b["red"], b["nir"], b["swir1"], b["blue"]
    idx = {"MNDWI": (G - S1) / (G + S1 + R.EPS), "NDWI": (G - N) / (G + N + R.EPS),
           "NDVI": (N - R_) / (N + R_ + R.EPS), "vis_brightness": (B + G + R_) / 3}
    out = []
    for n, a in idx.items():
        for size in (15, 31):
            out.append((f"{n}_ctx{size}", R.texture(a, size)[0]))
    return out


def extra(x, date=None):
    return S.nlcd_inputs(x) + static_inputs() + context_inputs(x)


if __name__ == "__main__":
    R.EXTRA = extra
    kw = dict(out_root=OUT, task_dir="six_class", prepare=S.prepare, configs=CONFIGS)
    R.run("exclude_daksh", **kw)
    cfg = R.run("with_daksh", **kw)
    R.run("external_daksh", cfg, **kw)
