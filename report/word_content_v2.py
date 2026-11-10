"""
v2 report for a river run (19 or 22): labels and how they are built, state scores, fraction statistics without rules
(VIIRS-predicted vs Landsat-derived class fractions), the VIIRS outcome figures and the v2 prediction images.

    ~/venvs/river-ml/bin/python report/word_content_v2.py 19 <scratch>/content_run19_v2.json
    NODE_PATH=<docx node_modules> node report/build_word.js <scratch>/content_run19_v2.json \
        ~/Desktop/nenana_river_models/reports/Nenana_VIIRS_run19_report_v2.docx
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE.parent / "output"
RUN = sys.argv[1]
ROOT = O / {"19": "19_viirs_river_states", "22": "22_viirs_river_gauss"}[RUN]
M = json.loads((ROOT / "metrics_all.json").read_text())
FS = json.loads((ROOT / "fraction_stats" / "stats.json").read_text())
FIG = O / "report_viirs_figures"
AGG = ROOT / "images" if RUN == "22" else O / "19_viirs_river_states" / "images"
MODELS = [m for m in ("rf", "unetpp", "cnn2d") if m in M]
MN = {"rf": "RF", "unetpp": "UNet++", "cnn2d": "2D CNN"}
PARTS = ["open water", "ice", "snow", "snow-free land"]
STATES = ["ice_free_river_snow_free_land", "ice_free_river_snow_covered_land",
          "ice_covered_river_snow_free_land", "ice_covered_river_snow_covered_land"]
SHORT = ["1 open river, no-snow land", "2 open river, snow land", "3 ice river, no-snow land", "4 ice river, snow land"]
SEASON = {"2023-04-19": "spring, snow", "2023-05-05": "spring break-up", "2023-07-24": "summer",
          "2023-10-04": "autumn freeze-up", "2024-04-21": "spring break-up", "2024-06-22": "summer",
          "2024-09-02": "late summer", "2025-07-04": "summer", "2026-03-26": "late winter", "2026-08-01": "summer"}
labelled = FS["dates"]
gauss = RUN == "22"
lab_word = "Gaussian-weighted" if gauss else "box"

B = []
h = lambda lvl, text: B.append({"t": f"h{lvl}", "text": text})
p = lambda text: B.append({"t": "p", "text": text})
bullets = lambda items: B.append({"t": "bullets", "items": items})
table = lambda header, rows, widths=None, align=None: B.append(
    {"t": "table", "header": header, "rows": rows, "widths": widths, "align": align})
pagebreak = lambda: B.append({"t": "pagebreak"})
f3 = lambda v: f"{v:.3f}"


def image(path, caption, w=620):
    if Path(path).exists():
        B.append({"t": "image", "path": str(path), "caption": caption, "width": w})


title = {"19": "run 19: four river / land states on the river + 1 px buffer",
         "22": "run 22: run 19 with Gaussian (run 17) labels"}[RUN]
h(0, f"Nenana river ice from VIIRS 375 m, {title} (v2)")
p("Version 2: adds the fraction statistics without rules (VIIRS-predicted vs Landsat-derived class fractions), the "
  "2D CNN fraction model, the VIIRS outcome figures with Landsat, the run 16 map and the hand labels next to them, and "
  "new prediction images (river ice in pink, no boxes).")

h(1, "1. Summary")
rows = []
for m in MODELS:
    a = FS["all"][m]["river + buffer"]
    rows.append([MN[m], f3(M[m]["rule50"]["accuracy"]), f3(M[m]["rule90"]["accuracy"]),
                 f"{a['mean of 4 classes']['mae']:.1f}", f"{a['open water']['r2']:.2f}", f"{a['ice']['r2']:.2f}",
                 f"{a['snow']['r2']:.2f}", f"{a['snow-free land']['r2']:.2f}"])
table(["model", "state acc rule50", "state acc rule90", "fraction MAE pp (mean)", "R² water", "R² ice", "R² snow",
       "R² snow-free"], rows, [1100, 1100, 1100, 1300, 1000, 1000, 1000, 1100], ["l"] + ["r"] * 7)
best = min(MODELS, key=lambda m: FS["all"][m]["river + buffer"]["mean of 4 classes"]["mae"])
ab = FS["all"][best]["river + buffer"]
bullets([
    f"Fractions (no rules): {MN[best]} matches the Landsat-derived fractions best, mean error "
    f"{ab['mean of 4 classes']['mae']:.1f} percentage points on the river + buffer pixels of the 6 left-out days "
    f"(water {ab['open water']['mae']:.1f}, ice {ab['ice']['mae']:.1f}, snow {ab['snow']['mae']:.1f}, "
    f"snow-free land {ab['snow-free land']['mae']:.1f} pp).",
    "States (rules): " + ", ".join(f"{MN[m]} {M[m]['rule50']['accuracy']:.3f} (rule50) / "
                                    f"{M[m]['rule90']['accuracy']:.3f} (rule90)" for m in MODELS) + ".",
    "Snow and snow-free land are the hardest fractions; water and ice are the closest.",
    "All scores compare against the run 16 Landsat map (a model prediction), not against ground truth; the hand labels "
    "are the ground truth and are shown next to the maps.",
])

h(1, "2. Labels: Landsat fractions per VIIRS pixel")
bullets([
    "Source: the run 16 Landsat 30 m map of the day (run 16 RF prediction: river, ice, snow, forest, shrub, bare). It is "
    "a prediction, not ground truth; the hand labels are the ground truth.",
    "Spatial step: " + ("each VIIRS pixel weighs the Landsat pixels around its centre with a Gaussian (sigma = 375 m / "
                        "2.8 = 134 m, cut at 2 sigma; Oleson et al. 1995)." if gauss else
                        "the 144-169 Landsat pixels (12-13 x 12-13) inside each VIIRS pixel are counted, after the run 17 "
                        "VIIRS-Landsat shift.") + " A VIIRS pixel is used only if >= 98 % of it is mapped.",
    "Class step: river -> open water, ice -> ice, snow -> snow, forest + shrub + bare -> snow-free land.",
    "States: river ice-free if open / (open + ice) > 0.5 (rule90: > 0.9); land snow-covered if snow / (snow + "
    "snow-free) > 0.5 (rule90: snow-free only if < 0.1); pixels with < 1 % river take the ratio of the whole day.",
    "Area: VIIRS pixels of the water mask alaska_occ_375m.tif (occurrence >= 0.05, 988 px) plus a 1-pixel buffer (678 px).",
])
for d in labelled:
    image(AGG / f"aggregation_{d}_overview.png",
          f"{d} ({SEASON.get(d, '')}): run 16 map -> {lab_word} fractions per VIIRS pixel (all panels from the run 16 map).", 640)
    image(AGG / f"aggregation_{d}_zoom.png", f"{d}: the Landsat pixels behind single VIIRS pixels and their fractions.", 640)

pagebreak()
h(1, "3. State scores (rules)")
p("RF, UNet++ and 2D CNN predict the four states at once from VIIRS (I1-I5, NDVI_I, NDSI_I, I4-I5, sun / view zenith, "
  "JRC water occurrence and zone share, water-mask occurrence; RF adds 3 x 3 mean / std). Leave-one-date-out over the "
  "6 labelled days.")
for rule in ("rule50", "rule90"):
    rows = []
    for m in MODELS:
        s = M[m][rule]
        rows.append([MN[m], f3(s["accuracy"]), f3(s["macro_f1"])]
                    + [f"{s['per_state'].get(st, {}).get('f1', 0):.2f}" for st in STATES])
    table([f"{rule}: model", "accuracy", "macro F1"] + [f"F1 {x}" for x in SHORT], rows,
          [1300, 900, 900, 1500, 1500, 1500, 1500], ["l"] + ["r"] * 6)
    image(FIG / f"conf_run{RUN}_{rule}.png", f"Run {RUN}: confusion matrices, {rule} (rows: run 16-derived state, "
                                             "columns: predicted from VIIRS).", 620)
rows = [[d, SEASON.get(d, "")] + [f"{M[m]['per_date'][d]['rule50']['accuracy']:.2f} / "
                                  f"{M[m]['per_date'][d]['rule90']['accuracy']:.2f}" for m in MODELS] for d in labelled]
table(["date", "season"] + [f"{MN[m]} acc50 / acc90" for m in MODELS], rows, [1400, 1900, 1900, 1900, 1900],
      ["l", "l", "r", "r", "r"])

pagebreak()
h(1, "4. Fraction statistics without rules")
p("For every VIIRS pixel the four class fractions predicted from VIIRS (RF, UNet++ and 2D CNN fraction models, "
  "trained on 5 days and tested on the 6th) are compared directly with the Landsat-derived fractions (the run 16 map "
  f"aggregated to 375 m, {lab_word} weights). No rule and no state: MAE and RMSE in percentage points, bias = VIIRS - "
  "Landsat, R², Pearson r.")
for i, area in enumerate(("river + buffer", "whole image")):
    h(2, f"4.{i + 1} {area[0].upper() + area[1:]}")
    rows = []
    for m in MODELS:
        for c in PARTS:
            s = FS["all"][m][area][c]
            rows.append([MN[m], c, f"{s['n']:,}", f"{s['landsat_mean']:.1f}", f"{s['viirs_mean']:.1f}", f"{s['mae']:.1f}",
                         f"{s['rmse']:.1f}", f"{s['bias']:+.1f}", f"{s['r2']:.2f}", f"{s['r']:.2f}"])
    table(["model", "class", "pixels", "Landsat mean %", "VIIRS mean %", "MAE pp", "RMSE pp", "bias pp", "R²", "r"], rows,
          [900, 1300, 900, 1000, 1000, 800, 800, 800, 700, 700], ["l", "l"] + ["r"] * 8)
image(ROOT / "fraction_stats" / "scatter.png",
      "Landsat-derived vs VIIRS-predicted fraction per pixel (river + buffer, 6 left-out days); dashed = 1:1.", 640)
h(2, "4.3 Per day (river + buffer), MAE pp water / ice / snow / snow-free land")
rows = [[d, SEASON.get(d, "")] + [" / ".join(f"{FS['per_day'][m][d][c]['mae']:.1f}" for c in PARTS) for m in MODELS]
        for d in labelled]
table(["date", "season"] + [MN[m] for m in MODELS], rows, [1300, 1800, 2000, 2000, 2000], ["l", "l", "r", "r", "r"])

pagebreak()
h(1, "5. VIIRS outcome: fractions and states predicted from VIIRS")
p("One row per model: VIIRS (I2/I2/I1, the model input), Landsat true colour of the same day, the run 16 map, the hand "
  "labels (where they exist), the four predicted fractions and the predicted state map. The zoom shows 5 x 5 VIIRS "
  "pixels with the same layers and the predicted % per pixel.")
for d in sorted(SEASON):
    image(ROOT / "images" / f"viirs_outcome_{d}_overview.png", f"{d} ({SEASON.get(d, '')}): VIIRS outcome.", 640)
    image(ROOT / "images" / f"viirs_outcome_{d}_zoom.png", f"{d}: VIIRS outcome, zoom.", 640)

pagebreak()
h(1, "6. Prediction images (v2)")
p("VIIRS, Landsat, the run 16 map and the state maps (run 16-derived on labelled days, then RF, UNet++, 2D CNN). State 3 "
  "is pink; ice-covered states on pixels with more than 10 % water are shown as river ice (pink), each panel using its "
  "own water fraction. rule50 shown; rule90 versions are in predict/images_v2/.")
for d in sorted(SEASON):
    image(ROOT / "predict" / "images_v2" / f"pred_{d}.jpg",
          f"{d} ({SEASON.get(d, '')}, {'left-out test day' if d in labelled else 'VIIRS only, final models'}).", 640)

h(1, "7. Things to know")
bullets([
    "All accuracies are agreement with the run 16 map (a Landsat model), not with the ground; the hand labels cover only "
    "4 days and small polygons.",
    "The state models and the fraction models are separate models, so a pixel's state and its fractions can disagree.",
    "Only 6 labelled days; ice and break-up come from 2-3 of them.",
    "No cloud mask on VIIRS.",
])
h(1, "8. Files")
table(["what", "where (~/Desktop/nenana_crops/daksh/)"], [
    ["state models", "viirs_river_states.py (RIVER_BUFFER=1" + (", RIVER_LABELS=psf" if gauss else "") + ")"],
    ["fraction models", "viirs_states.py (RF, UNet++), make_cnn_pct.py (2D CNN)"],
    ["fraction statistics", f"fraction_stats.py -> output/{ROOT.name}/fraction_stats/ (stats.json, stats.csv, scatter.png)"],
    ["figures", "make_aggregation_figs*.py, make_viirs_outcome_figs.py, make_pred_images_v2.py"],
    ["this report", "report/word_content_v2.py + report/build_word.js"],
], [2300, 6700])

json.dump(B, open(sys.argv[2], "w"))
print(len(B), "blocks")
