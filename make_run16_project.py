"""
Separate, self-contained QGIS project with only run 16 (water-zone RF):

  ~/Desktop/nenana_river_models/qgis_run16/
    run16.qgz
    data/...            every file the project uses, paths stored relative to run16.qgz

Layer tree:
  reference_layers/   run 16 water zone (JRC >= 5 %), JRC water occurrence, NLCD 2016
  <date groups>/      labels, study box, Landsat, Sentinel-2 (as in the main project)
    run16_water_zone/ with_daksh, exclude_daksh, external_test (2024-03-12 only)

Built from the self-contained main project (qgis_project/nenana_all.qgz, left untouched):
every other run's "predictions" / "rf_runs" groups, VIIRS and the DEM are left out.

    QT_QPA_PLATFORM=offscreen python3 ~/Desktop/nenana_crops/daksh/make_run16_project.py
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import glob
import shutil
from pathlib import Path
from qgis.core import (QgsApplication, QgsProject, QgsDataProvider, QgsRasterLayer,
                       QgsPalettedRasterRenderer)
from qgis.PyQt.QtGui import QColor

HOME = Path.home()
SOURCE = HOME / "Desktop/nenana_river_models/qgis_project/nenana_all.qgz"
DEST = HOME / "Desktop/nenana_river_models/qgis_run16"
PROJECT = DEST / "run16.qgz"
RUN = HOME / "Desktop/nenana_crops/daksh/output/16_rf_water_zone"
SUB = "six_class/random_forest"
DATASETS = [("with_daksh", "2_with_daksh"), ("exclude_daksh", "1_exclude_daksh"),
            ("external_test", "3_external_test_daksh")]
NEW_DATES = ["2026-03-26", "2023-05-05"]
DROP_GROUPS = {"predictions", "rf_runs", "VIIRS"}
MAP = [(str(HOME / "Desktop/nenana_river_models/qgis_project/data") + "/", ""),
       (str(RUN) + "/", "run16/")]


def target(path):
    for old, new in MAP:
        if path.startswith(old):
            return DEST / "data" / new / path[len(old):]
    return DEST / "data" / "other" / os.path.basename(path)


def copy_with_sidecars(src, dst):
    stem = os.path.splitext(src)[0]
    for f in glob.glob(glob.escape(stem) + ".*"):
        out = dst.parent / (os.path.splitext(dst.name)[0] + f[len(stem):])
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or os.path.getmtime(f) > os.path.getmtime(out):
            shutil.copy2(f, out)


def zone_style(lyr):
    lyr.setRenderer(QgsPalettedRasterRenderer(lyr.dataProvider(), 1, [
        QgsPalettedRasterRenderer.Class(1, QColor(0, 90, 255, 140), "water zone: river / ice only")]))


app = QgsApplication([], False); app.initQgis()
proj = QgsProject.instance()
proj.read(str(SOURCE))
root = proj.layerTreeRoot()


def drop(node):
    for lyr in node.findLayers():
        proj.removeMapLayer(lyr.layerId())
    node.parent().removeChildNode(node)


for g in [g for g in root.findGroups(recursive=True) if g.name() in DROP_GROUPS]:
    drop(g)
old_ref = root.findGroup("reference_layers")
if old_ref:
    drop(old_ref)

style = next((RUN / "2_with_daksh" / SUB).glob("pred_*.qml"))


def add(group, tif, name, on=False, borrow=True):
    lyr = QgsRasterLayer(str(tif), name)
    if borrow and not tif.with_suffix(".qml").exists():
        lyr.loadNamedStyle(str(style))
    proj.addMapLayer(lyr, False)
    group.addLayer(lyr).setItemVisibilityChecked(on)
    return lyr


ref = root.insertGroup(0, "reference_layers")
zone_style(add(ref, RUN / "water_zone.tif", "run 16 water zone (JRC >= 5 %)", True, borrow=False))
wo = next(glob.iglob(str(HOME / "Desktop/nenana_river_models/qgis_project/data/patches/nenana_*/*_water_occurrence.tif")))
add(ref, Path(wo), "JRC water occurrence (%)", borrow=False)
add(ref, HOME / "Desktop/nenana_river_models/qgis_project/data/model_outputs/nlcd/nlcd_2016_ak_box.tif",
    "NLCD 2016 land cover", borrow=False)

first_on = True
for group in [g for g in root.findGroups(recursive=True)
              if g.name().startswith("nenana_") or (g.name().isdigit() and len(g.name()) == 8)]:
    s = group.name().split("_")[-1]
    date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    layers = []
    for dlabel, ds in DATASETS:
        if dlabel == "external_test" and date != "2024-03-12":
            continue
        d = RUN / ds / SUB
        tif = d / "new_dates" / date / "rf_pred.tif" if date in NEW_DATES else d / f"pred_{date}.tif"
        if tif.exists():
            layers.append((f"{dlabel}  {date}", tif))
    if not layers:
        continue
    names = [c.name() for c in group.children()]
    at = next((i + 1 for i, n in enumerate(names) if n.startswith("labels")), 0)
    rg = group.insertGroup(at, "run16_water_zone")
    for i, (name, tif) in enumerate(layers):
        add(rg, tif, name, on=(i == 0 and first_on))
    first_on = False
    print(f"{group.name()}: {len(layers)} run 16 maps")

moved = 0
for lyr in proj.mapLayers().values():
    src = lyr.source(); path = src.split("|")[0]
    if not os.path.isfile(path):
        continue
    dst = target(path)
    copy_with_sidecars(path, dst)
    lyr.setDataSource(str(dst) + src[len(path):], lyr.name(), lyr.providerType(), QgsDataProvider.ProviderOptions())
    if lyr.name().startswith("run 16 water zone"):
        zone_style(lyr)
    moved += 1
# run16.qgz: full paths (opens on this computer from anywhere, links and copies included)
proj.writeEntryBool("Paths", "/Absolute", True)
proj.setFileName(str(PROJECT))
print("layers", moved, "saved", proj.write(str(PROJECT)), PROJECT)
# run16_portable.qgz: paths relative to the folder, for sending the whole qgis_run16 folder / zip
PORTABLE = DEST / "run16_portable.qgz"
proj.writeEntryBool("Paths", "/Absolute", False)
proj.setFileName(str(PORTABLE))
print("saved", proj.write(str(PORTABLE)), PORTABLE)
app.exitQgis()
