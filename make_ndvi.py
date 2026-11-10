"""
NDVI layer for every date, into output/ndvi/, and into the QGIS project:

  <date>_landsat_ndvi.tif    Landsat (B5 - B4) / (B5 + B4), 30 m, every date
                             (patch surface reflectance; daksh top-of-atmosphere)
  <date>_sentinel2_ndvi.tif  Sentinel-2 (B8 - B4) / (B8 + B4), 10 m, the daksh dates
                             (Sentinel-2 B8 is the NIR band that matches Landsat B5)

Each gets a .qml (red -> yellow -> green) and is added, switched off, at the top of
that date's Landsat / Sentinel-2 subgroup. Re-running replaces the layers.
QGIS must not have the project open while this runs.

    QT_QPA_PLATFORM=offscreen python3 ~/Desktop/nenana_crops/daksh/make_ndvi.py
"""

import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import numpy as np
from osgeo import gdal

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "ndvi"
PROJECT = Path.home() / "Desktop/RiverProject/Matus/narrow_rivers/patches/all_patches.qgz"
PATCHES = Path.home() / "Desktop/RiverProject/Matus/narrow_rivers/patches"
CROPS = HERE.parent
NEW = Path.home() / "Desktop/Patches"
NODATA = -9999.0
QML = """<!DOCTYPE qgis><qgis version="3.34" styleCategories="AllStyleCategories">
 <pipe><rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" classificationMin="-0.2" classificationMax="0.9">
  <rastershader><colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
   <item value="-0.2" label="-0.2 water / snow" color="#2b83ba"/>
   <item value="0" label="0" color="#d7191c"/>
   <item value="0.2" label="0.2 bare" color="#fdae61"/>
   <item value="0.45" label="0.45" color="#ffffbf"/>
   <item value="0.7" label="0.7" color="#a6d96a"/>
   <item value="0.9" label="0.9 dense vegetation" color="#1a9641"/>
  </colorrampshader></rastershader></rasterrenderer></pipe></qgis>
"""


def write(path, arr, ref):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32, ["COMPRESS=DEFLATE"])
    ds.SetGeoTransform(ref.GetGeoTransform()); ds.SetProjection(ref.GetProjection())
    b = ds.GetRasterBand(1); b.WriteArray(np.where(np.isfinite(arr), arr, NODATA)); b.SetNoDataValue(NODATA)
    b.SetDescription("NDVI"); ds = None
    path.with_suffix(".qml").write_text(QML)


def ndvi(nir, red):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (nir - red) / (nir + red)
    return np.clip(v, -1, 1)


def dates():
    """(date, landsat folder, kind, sentinel tif or None) for every date."""
    out = []
    for d in sorted(PATCHES.glob("nenana_*")):
        s = d.name[7:]; date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        if (d / f"{date}_sr.tif").exists() and any((d / "labels").glob("*.shp")):
            out.append((date, d, "patch", None))
    for d in sorted(CROPS.glob("2*")):
        if d.is_dir() and any(d.glob("landsat_*_box.tif")):
            date = f"{d.name[:4]}-{d.name[4:6]}-{d.name[6:]}"
            out.append((date, d, "daksh", next(d.glob("sentinel2_*_box.tif"), None)))
    for date in ("2026-03-26", "2023-05-05"):
        d = NEW / f"nenana_{date.replace('-', '')}"
        if (d / f"{date}_sr.tif").exists():
            out.append((date, d, "new", None))
    return out


def make():
    sys.path.insert(0, str(HERE))
    import data as D
    OUT.mkdir(parents=True, exist_ok=True)
    made = {}
    for date, folder, kind, s2 in dates():
        D.configure("four", "with_daksh" if kind == "daksh" else "exclude_daksh")
        x, ref = D.load_features(folder, date)
        ch = D.CHANNELS
        nir = x[ch.index("SR_B5" if "SR_B5" in ch else "nir")]          # Landsat band 5
        red = x[ch.index("SR_B4" if "SR_B4" in ch else "red")]          # Landsat band 4
        f = OUT / f"{date}_landsat_ndvi.tif"
        write(f, ndvi(nir, red), ref)
        made[date] = [("Landsat", f)]
        if s2 is not None:
            ds = gdal.Open(str(s2))
            b4 = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)   # B4 red
            b8 = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)   # B8 NIR
            v = ndvi(b8, b4); v[(b4 == 0) | (b8 == 0)] = np.nan          # 0 = no data
            f2 = OUT / f"{date}_sentinel2_ndvi.tif"
            write(f2, v, ds)
            made[date].append(("Sentinel-2", f2))
        print(date, ", ".join(p.name for _, p in made[date]))
    return made


def add_to_project(made):
    from qgis.core import QgsApplication, QgsProject, QgsRasterLayer
    app = QgsApplication([], False); app.initQgis()
    proj = QgsProject.instance(); proj.read(str(PROJECT))
    root = proj.layerTreeRoot()
    for g in root.findGroups(recursive=True):
        n = g.name()
        s = n.split("_")[-1]
        if not (s.isdigit() and len(s) == 8 and (n.startswith("nenana_") or n == s)):
            continue
        date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        for sub, f in made.get(date, []):
            sg = g.findGroup(sub)
            if sg is None:
                continue
            for c in [c for c in sg.children() if c.name().endswith("_ndvi")]:
                proj.removeMapLayer(c.layerId())
            lyr = QgsRasterLayer(str(f), f.stem)
            proj.addMapLayer(lyr, False)
            sg.insertLayer(0, lyr).setItemVisibilityChecked(False)
            print(f"{n}/{sub}: {f.stem}")
    print("saved", proj.write())
    app.exitQgis()


if __name__ == "__main__":
    add_to_project(make())
