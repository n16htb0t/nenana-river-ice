"""
Random Forest, six classes: river, ice, snow and three kinds of land.

The hand labels give river, ice, snow and land. Inside the hand-drawn land polygons each
pixel is split by its NLCD Alaska 2016 land cover (30 m, made from Landsat; Earth Engine
USGS/NLCD_RELEASES/2016_REL/2016_AK, saved on the box grid in output/nlcd/):

  forest               NLCD 41 deciduous, 42 evergreen, 43 mixed forest
  shrub_grass_wetland  NLCD 51 dwarf shrub, 52 shrub, 71 grassland, 72 sedge, 81 pasture,
                       90 woody wetland, 95 emergent wetland
  bare_developed       NLCD 21-24 developed, 31 barren
  (land-label pixels that NLCD calls water, 11, are left out)

River, ice and snow stay exactly as drawn. NLCD is used only to split the land labels, not
as an input: the inputs are the same band pairs, indices and textures as rf_all_bands.py
(no water mask, no JRC water occurrence).

With --nlcd-input the NLCD land cover is also given to the forest as inputs, one 0/1 input
per NLCD group below (NLCD_INPUTS). Then the land types can be read straight off NLCD, so
their scores mainly show agreement with NLCD; river, ice and snow stay a fair test.
Outputs then go to output/7_rf_six_classes_nlcd_input/.

Outputs: output/6_rf_six_classes/<dataset folder>/six_class/random_forest/ (as rf_all_bands.py)

    ~/venvs/river-ml/bin/python ~/Desktop/nenana_crops/daksh/rf_six_classes.py [--nlcd-input]
"""

import argparse
import numpy as np
from osgeo import gdal
import data as D
import rf_all_bands as R

OUT = D.HERE / "output" / "6_rf_six_classes"
NLCD = D.HERE / "output" / "nlcd" / "nlcd_2016_ak_box.tif"
CLASSES = {1: "river", 2: "ice", 3: "snow", 4: "forest", 5: "shrub_grass_wetland", 6: "bare_developed"}
COLOURS = {1: "#0066ff", 2: "#ff00ff", 3: "#00ffff", 4: "#1a7f37", 5: "#b5d96b", 6: "#c8a165"}
LAND_GROUPS = {3: [41, 42, 43], 4: [51, 52, 71, 72, 81, 90, 95], 5: [21, 22, 23, 24, 31]}   # index space
NLCD_INPUTS = {"nlcd_water": [11, 12], "nlcd_developed": [21, 22, 23, 24], "nlcd_barren": [31],
               "nlcd_deciduous_forest": [41], "nlcd_evergreen_forest": [42], "nlcd_mixed_forest": [43],
               "nlcd_shrub": [51, 52], "nlcd_grass_sedge": [71, 72, 81], "nlcd_woody_wetland": [90],
               "nlcd_emergent_wetland": [95]}
CONFIGS = [dict(max_features="sqrt", min_samples_leaf=5),
           dict(max_features="sqrt", min_samples_leaf=2),
           dict(max_features="log2", min_samples_leaf=2)]


def prepare(scenes):
    """Split each scene's land labels (index 3) into the three NLCD land groups."""
    D.CLASSES, D.COLOURS = CLASSES, COLOURS
    nlcd = gdal.Open(str(NLCD)).ReadAsArray()
    for s in scenes:
        y = s["y"]
        land = y == 3
        y[land] = D.IGNORE
        for idx, codes in LAND_GROUPS.items():
            y[land & np.isin(nlcd, codes)] = idx


def nlcd_inputs(x=None, date=None):
    nlcd = gdal.Open(str(NLCD)).ReadAsArray()
    return [(n, np.isin(nlcd, codes).astype(np.float32)) for n, codes in NLCD_INPUTS.items()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlcd-input", action="store_true", help="also give NLCD land cover as inputs")
    args = ap.parse_args()
    if args.nlcd_input:
        R.EXTRA = nlcd_inputs
        OUT = D.HERE / "output" / "7_rf_six_classes_nlcd_input"
    kw = dict(out_root=OUT, task_dir="six_class", prepare=prepare, configs=CONFIGS)
    R.run("exclude_daksh", **kw)
    cfg = R.run("with_daksh", **kw)
    R.run("external_daksh", cfg, **kw)
