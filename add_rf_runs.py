"""
Add every Random Forest run's maps (runs 05-15) to the QGIS project, next to the older
"predictions" subgroup that add_to_project.py makes. In each date group:

  rf_runs/
    15 six-class, river penalty forced (best maps)   with_daksh / exclude_daksh / external_test
                                                     (+ "no penalty" versions for runs 14, 15)
    14 six-class, river penalty tuned
    13 six-class, limited params
    ...
    05 four-class, all bands

The unlabelled dates (new_dates (no labels) group) get the same from each run's new_dates/.
A top-level group "reference_layers" holds NLCD 2016 and the Copernicus DEM.
Only run 15 with_daksh is switched on. Styles come from the .qml next to each map; maps
without one (new dates) borrow the run's labelled-date style.
Re-running replaces the "rf_runs" subgroups. QGIS must not have the project open.

    QT_QPA_PLATFORM=offscreen python3 ~/Desktop/nenana_crops/daksh/add_rf_runs.py
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from qgis.core import QgsApplication, QgsProject, QgsRasterLayer

PROJECT = Path.home() / "Desktop/RiverProject/Matus/narrow_rivers/patches/all_patches.qgz"
OUT = Path(__file__).resolve().parent / "output"
RUNS = [  # (folder, label, sub-path inside each dataset, map name in new_dates/)
    ("15_rf_six_classes_river_forced", "15 six-class, river penalty forced (best maps)", "six_class/random_forest", "rf"),
    ("14_rf_six_classes_river_fix", "14 six-class, river penalty tuned", "six_class/random_forest", "rf"),
    ("13_rf_six_classes_limited", "13 six-class, limited params", "six_class/random_forest", "rf"),
    ("12_rf_six_classes_bands_angles_nlcd", "12 six-class, bands + angles + NLCD", "six_class/random_forest", "rf"),
    ("11_feature_search", "11 six-class, feature search", "six_class", "best"),
    ("10_rf_six_classes_terrain", "10 six-class, terrain (best scores)", "six_class/random_forest", "rf"),
    ("9_best_six_class_nlcd_input", "09 six-class, tuned + NLCD input", "six_class", "best"),
    ("8_best_six_class", "08 six-class, tuned", "six_class", "best"),
    ("7_rf_six_classes_nlcd_input", "07 six-class, NLCD input", "six_class/random_forest", "rf"),
    ("6_rf_six_classes", "06 six-class, all bands", "six_class/random_forest", "rf"),
    ("5_rf_all_bands", "05 four-class, all bands", "four_class_river_ice_snow_land/random_forest", "rf"),
]
DATASETS = [("with_daksh", "2_with_daksh"), ("exclude_daksh", "1_exclude_daksh"),
            ("external_test", "3_external_test_daksh")]
NEW_DATES = ["2026-03-26", "2023-05-05"]
SHOW = ("15_rf_six_classes_river_forced", "with_daksh")


def files(run, sub, short, ds, date):
    """[(tif, layer name, style fallback)] for one run / dataset / date."""
    d = OUT / run / ds / sub
    style = next(d.glob("pred_*.qml"), None)
    if date in NEW_DATES:
        n = d / "new_dates" / date
        out = [(n / f"{short}_pred.tif", date), (n / f"{short}_pred_nofix.tif", f"{date} no penalty")]
    else:
        out = [(d / f"pred_{date}.tif", date), (d / f"pred_nofix_{date}.tif", f"{date} no penalty")]
    return [(t, name, style) for t, name in out if t.exists()]


def add(group, tif, name, style, on):
    lyr = QgsRasterLayer(str(tif), name)
    if not tif.with_suffix(".qml").exists() and style is not None:
        lyr.loadNamedStyle(str(style))
    proj.addMapLayer(lyr, False)
    group.addLayer(lyr).setItemVisibilityChecked(on)


app = QgsApplication([], False); app.initQgis()
proj = QgsProject.instance()
proj.read(str(PROJECT))
root = proj.layerTreeRoot()


def drop(group):
    for lyr in group.findLayers():
        proj.removeMapLayer(lyr.layerId())
    group.parent().removeChildNode(group)


old = root.findGroup("reference_layers")
if old:
    drop(old)
ref = root.insertGroup(0, "reference_layers")
for tif, name in ((OUT / "nlcd/nlcd_2016_ak_box.tif", "NLCD 2016 land cover"),
                  (OUT / "dem/copernicus_glo30_box.tif", "Copernicus DEM 30 m")):
    if tif.exists():
        add(ref, tif, name, None, False)
ref.setExpanded(False)

for group in [g for g in root.findGroups(recursive=True)
              if g.name().startswith("nenana_") or (g.name().isdigit() and len(g.name()) == 8)]:
    s = group.name().split("_")[-1]
    date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    for g in [g for g in group.findGroups(recursive=True) if g.name() == "rf_runs"]:
        drop(g)
    rg, count = None, 0
    for run, label, sub, short in RUNS:
        layers = []
        for dlabel, ds in DATASETS:
            if dlabel == "external_test" and date != "2024-03-12":
                continue
            layers += [(dlabel, *f) for f in files(run, sub, short, ds, date)]
        if not layers:
            continue
        if rg is None:
            names = [c.name() for c in group.children()]
            at = next((i + 1 for i, n in enumerate(names) if n == "predictions"),
                      next((i + 1 for i, n in enumerate(names) if n.startswith("labels")), 1))
            rg = group.insertGroup(at, "rf_runs")
        g = rg.addGroup(label)
        first = True
        for dlabel, tif, name, style in layers:
            on = (run, dlabel) == SHOW and first and "no penalty" not in name
            first = first and not on
            add(g, tif, f"{dlabel}  {name}", style, on)
            count += 1
        g.setExpanded(run == SHOW[0])
    if rg is not None:
        print(f"{group.name()}: {count} layers")

print("saved", proj.write())
app.exitQgis()
