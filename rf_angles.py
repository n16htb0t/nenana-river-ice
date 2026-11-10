"""
Six-class Random Forest on all bands + sun / illumination angles + NLCD land cover only
(no terrain elevation, no distance to water, no context windows, no water mask).

Inputs:
  all bands         rf_all_bands.py: bands, every band pair, indices, textures
  land cover        NLCD Alaska 2016, 10 groups (rf_six_classes.nlcd_inputs)
  angles            per date: sun_elevation_deg, sun_azimuth_deg (Landsat scene metadata,
                    output/angles/sun_angles.json); per pixel: slope_deg, northness,
                    eastness, cos_incidence (cos of the angle between sun and slope normal:
                    1 = slope faces the sun, <= 0 = self-shadowed) and shadowed (cos_i <= 0)
                    from the Copernicus GLO-30 DEM

Outputs: output/12_rf_six_classes_bands_angles_nlcd/<dataset folder>/six_class/random_forest/

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_angles.py
"""

import json
import numpy as np
from osgeo import gdal
import data as D
import rf_all_bands as R
import rf_six_classes as S

OUT = D.HERE / "output" / "12_rf_six_classes_bands_angles_nlcd"
DEM = D.HERE / "output" / "dem" / "copernicus_glo30_box.tif"
SUN = json.loads((D.HERE / "output" / "angles" / "sun_angles.json").read_text())
DY, DX = 30.0, 13.0                     # metres per row / column (0.00027027 deg at 64.3 N)
CONFIGS = [dict(max_features="sqrt", min_samples_leaf=10),
           dict(max_features=0.1, min_samples_leaf=5),
           dict(max_features="sqrt", min_samples_leaf=5)]
_TERRAIN = None


def terrain():
    """Slope and aspect (radians, aspect clockwise from north) from the DEM (cached)."""
    global _TERRAIN
    if _TERRAIN is None:
        dem = gdal.Open(str(DEM)).ReadAsArray().astype(np.float64)
        gy, gx = np.gradient(dem, DY, DX)          # rows run south, columns run east
        slope = np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)                # downhill direction, compass bearing
        _TERRAIN = slope, aspect
    return _TERRAIN


def angle_inputs(x, date):
    slope, aspect = terrain()
    sun = SUN[date]
    zen = np.radians(90.0 - sun["sun_elevation"]); az = np.radians(sun["sun_azimuth"])
    cos_i = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(az - aspect)
    full = lambda v: np.full(slope.shape, v, np.float32)
    return [("sun_elevation_deg", full(sun["sun_elevation"])), ("sun_azimuth_deg", full(sun["sun_azimuth"])),
            ("slope_deg", np.degrees(slope).astype(np.float32)), ("northness", np.cos(aspect).astype(np.float32)),
            ("eastness", np.sin(aspect).astype(np.float32)), ("cos_incidence", cos_i.astype(np.float32)),
            ("shadowed", (cos_i <= 0).astype(np.float32))]


def extra(x, date=None):
    return S.nlcd_inputs(x) + angle_inputs(x, date)


if __name__ == "__main__":
    R.EXTRA = extra
    kw = dict(out_root=OUT, task_dir="six_class", prepare=S.prepare, configs=CONFIGS)
    R.run("exclude_daksh", **kw)
    cfg = R.run("with_daksh", **kw)
    R.run("external_daksh", cfg, **kw)
