"""
Self-contained copy of the QGIS project: every file a layer uses is copied into one folder
next to the project, and the project stores paths relative to itself, so it opens from
anywhere (moved, copied to another computer, opened through a link) without broken layers.

  ~/Desktop/nenana_river_models/qgis_project/
    nenana_all.qgz
    data/patches/          hand labels + Landsat/Sentinel images of the patch dates (Matus/Naira)
    data/daksh_dates/      daksh dates 20240312, 20240420, 20240429
    data/new_dates/        unlabelled dates 2026-03-26, 2023-05-05
    data/model_outputs/    every model run's maps, NDVI, NLCD, DEM (same folder names as output/)
    data/other/            anything else the project used

Source project (left untouched): ~/Desktop/RiverProject/Matus/narrow_rivers/patches/all_patches.qgz
Re-running refreshes the copy (files are re-copied only when newer).

    QT_QPA_PLATFORM=offscreen python3 ~/Desktop/nenana_crops/daksh/make_qgis_package.py
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import glob
import shutil
from pathlib import Path
from qgis.core import QgsApplication, QgsProject, QgsDataProvider

HOME = str(Path.home())
SOURCE = Path.home() / "Desktop/RiverProject/Matus/narrow_rivers/patches/all_patches.qgz"
DEST = Path.home() / "Desktop/nenana_river_models/qgis_project"
PROJECT = DEST / "nenana_all.qgz"
MAP = [  # source folder prefix -> folder inside data/ (longest prefixes first)
    (f"{HOME}/Desktop/nenana_crops/daksh/output/", "model_outputs/"),
    (f"{HOME}/Desktop/RiverProject/Matus/narrow_rivers/patches/", "patches/"),
    (f"{HOME}/Desktop/RiverProject/Matus/", "matus/"),
    (f"{HOME}/Desktop/Patches/", "new_dates/"),
    (f"{HOME}/Desktop/nenana_crops/", "daksh_dates/"),
]


def target(path):
    for old, new in MAP:
        if path.startswith(old):
            return DEST / "data" / new / path[len(old):]
    return DEST / "data" / "other" / os.path.basename(path)


def copy_with_sidecars(src, dst):
    """Copy a data file plus every file sharing its stem (.qml, .aux.xml, shapefile parts)."""
    stem = os.path.splitext(src)[0]
    for f in glob.glob(glob.escape(stem) + ".*"):
        out = dst.parent / (os.path.splitext(dst.name)[0] + f[len(stem):])
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or os.path.getmtime(f) > os.path.getmtime(out):
            shutil.copy2(f, out)


app = QgsApplication([], False); app.initQgis()
proj = QgsProject.instance()
proj.read(str(SOURCE))
moved, skipped = 0, []
for lyr in proj.mapLayers().values():
    src = lyr.source()
    path = src.split("|")[0]
    if not os.path.isfile(path):
        skipped.append(f"{lyr.name()}: {src[:80]}")       # web tiles etc. need no file
        continue
    dst = target(path)
    copy_with_sidecars(path, dst)
    lyr.setDataSource(str(dst) + src[len(path):], lyr.name(), lyr.providerType(),
                      QgsDataProvider.ProviderOptions())
    moved += 1
proj.writeEntryBool("Paths", "/Absolute", False)        # paths relative to the .qgz
proj.setFileName(str(PROJECT))
ok = proj.write(str(PROJECT))
print(f"layers copied {moved}, kept as is {len(skipped)}: {skipped}")
print("saved", ok, PROJECT)
app.exitQgis()
