"""
Add the four-class prediction maps and river/land state masks to the QGIS project,
as a "predictions" subgroup inside each date's group (mathus_naira/nenana_<date> and
daksh/<YYYYMMDD>), just below the hand labels:

  predictions/
    with_daksh/       Random Forest, UNet++, RF state, UNet++ state
    exclude_daksh/    the same (patch dates only)
    external_test/    the same (2024-03-12 only: models trained without that date)
    river_only_*/     the same three, from the models trained and predicting only
                      inside the river corridor (river + 150 m of bank each side)

The unlabelled dates 2026-03-26 and 2023-05-05 (~/Desktop/Patches) get their own group
"new_dates (no labels)" with images and these predictions (river-only RF switched on).
Only Random Forest of with_daksh is switched on. Styles come from the .qml next to each tif.
Re-running replaces the "predictions" subgroups instead of adding new ones.
QGIS must not have the project open while this runs (it would overwrite it on save).

    QT_QPA_PLATFORM=offscreen python3 ~/Desktop/nenana_crops/daksh/add_to_project.py
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from qgis.core import QgsApplication, QgsProject, QgsRasterLayer, QgsVectorLayer

PROJECT = Path.home() / "Desktop/RiverProject/Matus/narrow_rivers/patches/all_patches.qgz"
OUT     = Path(__file__).resolve().parent / "output"
FOUR    = "four_class_river_ice_snow_land"
RIVER   = "4_river_corridor_150m"                     # river + 150 m banks only
DATASETS = [("with_daksh", "2_with_daksh"), ("exclude_daksh", "1_exclude_daksh"),
            ("external_test", "3_external_test_daksh"),
            ("river_only_with_daksh", f"{RIVER}/2_with_daksh"),
            ("river_only_exclude_daksh", f"{RIVER}/1_exclude_daksh"),
            ("river_only_external_test", f"{RIVER}/3_external_test_daksh")]
MODELS  = [("random_forest", "Random Forest", "rf"), ("unet_plus_plus", "UNet++", "unetpp")]
NEW     = Path.home() / "Desktop/Patches"                 # unlabelled dates, own group
NEW_DATES = ["2026-03-26", "2023-05-05"]


def add_new_date_groups(root):
    """Group "new_dates (no labels)" with each unlabelled date's study box and images."""
    top = root.findGroup("new_dates (no labels)") or root.insertGroup(2, "new_dates (no labels)")
    for date in NEW_DATES:
        s = date.replace("-", "")
        src = NEW / f"nenana_{s}"
        if top.findGroup(f"nenana_{s}") or not src.is_dir():
            continue
        g = top.addGroup(f"nenana_{s}")
        box = QgsVectorLayer(str(src / f"{date}_study_box.geojson"), f"{date}_study_box", "ogr")
        proj.addMapLayer(box, False); g.addLayer(box)
        for sub, names in (("Landsat", ["truecolour", "sr", "swir", "thermal", "water_occurrence", "qa_pixel"]),
                           ("VIIRS", ["viirs_375m", "viirs_wide_375m", "viirs_wide_falsecolour", "viirs_wide_thermal"])):
            sg = g.addGroup(sub)
            for n in names:
                f = src / f"{date}_{n}.tif"
                if f.exists():
                    lyr = QgsRasterLayer(str(f), f"{date}_{n}")
                    proj.addMapLayer(lyr, False)
                    sg.addLayer(lyr).setItemVisibilityChecked(n == "truecolour")
            sg.setExpanded(False)
        print(f"new group nenana_{s}")


def map_files(folder, mdir, short, date):
    """(classes map, state mask) of a model for a date: labelled-run maps or new_dates/."""
    d = OUT / folder / FOUR
    if date in NEW_DATES:
        return d / "new_dates" / date / f"{short}_pred.tif", d / "new_dates" / date / f"{short}_scene_mask.tif"
    return d / mdir / f"pred_{date}.tif", d / mdir / f"scene_mask_{date}.tif"

app = QgsApplication([], False); app.initQgis()
proj = QgsProject.instance()
proj.read(str(PROJECT))
root = proj.layerTreeRoot()
add_new_date_groups(root)

def is_date(n):
    return n.startswith("nenana_") or (n.isdigit() and len(n) == 8)


for group in [g for g in root.findGroups(recursive=True) if is_date(g.name())]:
    name = group.name()
    s = name.split("_")[-1]
    date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    for old in [g for g in group.findGroups(recursive=True) if g.name() == "predictions"]:
        for lyr in old.findLayers():
            proj.removeMapLayer(lyr.layerId())
        old.parent().removeChildNode(old)
    pg, added = None, []
    for label, folder in DATASETS:
        if label.endswith("external_test") and date != "2024-03-12":   # only meaningful on the test date
            continue
        tifs = [(map_files(folder, m, sh, date)[0], f"{n} {date}") for m, n, sh in MODELS]
        tifs += [(map_files(folder, m, sh, date)[1], f"{n} river/land state {date}") for m, n, sh in MODELS]
        tifs = [t for t in tifs if t[0].exists()]
        if not tifs:
            continue
        if pg is None:
            names = [c.name() for c in group.children()]
            at = next((i + 1 for i, n in enumerate(names) if n.startswith("labels")), 1)
            pg = group.insertGroup(at, "predictions")            # just under the labels layer
        dg = pg.addGroup(label)
        for i, (tif, lname) in enumerate(tifs):
            lyr = QgsRasterLayer(str(tif), lname)                # style from the .qml beside it
            proj.addMapLayer(lyr, False)
            first = "river_only_with_daksh" if date in NEW_DATES else "with_daksh"
            dg.addLayer(lyr).setItemVisibilityChecked(label == first and i == 0)
        dg.setExpanded(label == first)
        added.append(f"{label} ({len(tifs)})")
    if pg is not None:
        print(f"{name}: {', '.join(added)}")

print("saved", proj.write())
app.exitQgis()
