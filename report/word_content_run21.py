"""
Run 21 report: run 16 repeated on VIIRS upsampled to 30 m (viirs_run16.py). content.json for build_word.js
and the numbers for presentation/build_deck_run21.js. Uses every model folder present (random_forest now,
unetpp later).

    ~/venvs/river-ml/bin/python report/word_content_run21.py <scratch>/content21.json
    NODE_PATH=<docx node_modules> node report/build_word.js <scratch>/content21.json \
        ~/Desktop/nenana_river_models/reports/Nenana_VIIRS_run21_report.docx
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE.parent / "output"
R21 = O / "21_viirs_run16"
ALL = json.loads((R21 / "metrics_all.json").read_text())
DS = "with_daksh"
MODELS = {"random_forest": ("RF", ALL[DS])}
UNET = R21 / "2_with_daksh" / "unetpp" / "metrics.json"
if UNET.exists():
    MODELS["unetpp"] = ("UNet++", json.loads(UNET.read_text()))
NAMES = ["river", "ice", "snow", "forest", "shrub_grass_wetland", "bare_developed"]
SEASON = {"2023-04-19": "spring, snow", "2023-05-05": "spring break-up", "2023-07-24": "summer",
          "2023-10-04": "autumn freeze-up", "2024-04-21": "spring break-up", "2024-06-22": "summer",
          "2024-09-02": "late summer", "2025-07-04": "summer", "2026-03-26": "late winter", "2026-08-01": "summer"}
rf = ALL[DS]
r16 = rf["run16_same_dates"]
labelled = rf["dates"]

B = []
h = lambda lvl, text: B.append({"t": f"h{lvl}", "text": text})
p = lambda text: B.append({"t": "p", "text": text})
bullets = lambda items: B.append({"t": "bullets", "items": items})
table = lambda header, rows, widths=None, align=None: B.append(
    {"t": "table", "header": header, "rows": rows, "widths": widths, "align": align})
image = lambda path, caption, w=620: B.append({"t": "image", "path": str(path), "caption": caption, "width": w})
pagebreak = lambda: B.append({"t": "pagebreak"})
f3 = lambda v: f"{v:.3f}"
row = lambda name, s: [name, f"{s['pixels']:,}", f3(s["accuracy"]), f3(s["balanced_accuracy"]), f3(s["kappa"]),
                       f3(s["macro_f1"])]
HEAD = ["model / labels", "pixels", "accuracy", "balanced acc", "Cohen κ", "macro F1"]
WID = [3600, 1200, 1100, 1300, 1000, 1100]
deck = {"labelled": labelled, "models": [v[0] for v in MODELS.values()]}

h(0, "Nenana river ice from VIIRS: run 21, run 16 on VIIRS upsampled to 30 m")
p("Run 16 repeated with one change: the image is VIIRS 375 m upsampled to the Landsat 30 m grid instead of Landsat. "
  "Same six classes, same water-mask rule, same hand labels and rule labels, same Random Forest settings, same 5-fold "
  "spatial cross-validation." + ("" if "unetpp" in MODELS else " UNet++ on the same set-up follows in a later version."))

h(1, "1. Summary")
rows = []
for key, (name, m) in MODELS.items():
    rows += [row(f"{name} (VIIRS), rule labels", m["rule"]), row(f"{name} (VIIRS), original labels", m["original"])]
rows += [row("Run 16 RF (Landsat), rule labels, same dates", r16["rule"]),
         row("Run 16 RF (Landsat), original labels, same dates", r16["original"])]
table(HEAD, rows, WID, ["l", "r", "r", "r", "r", "r"])
deck["summary"] = rows
pc = rf["rule"]["per_class"]
bullets([
    f"RF on VIIRS at 30 m: {rf['rule']['accuracy']:.3f} accuracy, macro F1 {rf['rule']['macro_f1']:.3f} on the rule labels "
    f"(run 16 on Landsat, same dates: {r16['rule']['accuracy']:.3f} / {r16['rule']['macro_f1']:.3f}).",
    f"River ({pc['river']['f1']:.2f}), forest ({pc['forest']['f1']:.2f}), shrub ({pc['shrub_grass_wetland']['f1']:.2f}) "
    f"and bare ({pc['bare_developed']['f1']:.2f}) stay high: river is fixed by the water mask, the land classes come "
    "mostly from NLCD (the land forest's top inputs are NLCD layers).",
    f"Ice drops to F1 {pc['ice']['f1']:.2f} and snow fails (F1 {pc['snow']['f1']:.2f}): snow patches and the ice / open "
    "water mix are smaller than a 375 m VIIRS pixel, so the upsampled signal cannot place them at 30 m.",
    "Summer days are near perfect; freeze-up (2023-10-04) and break-up (2024-04-21) drop to about 0.72-0.74.",
])

h(1, "2. What is the same as run 16 and what changed")
table(["", "run 16", "run 21"], [
    ["image", "Landsat 30 m surface reflectance + thermal", "VIIRS I1-I5, 375 m, bilinear-upsampled to 30 m, moved by the run 17 shift"],
    ["classes", "river, ice, snow, forest, shrub_grass_wetland, bare_developed", "same"],
    ["water mask rule", "JRC occurrence >= 5 %: river / ice only; land: snow / forest / shrub / bare", "same (same 30 m mask)"],
    ["labels", "hand labels moved onto the rule (rule labels)", "same, on the dates that have VIIRS"],
    ["model", "ZoneModel: one RF per zone, top 25 inputs, depth 16, leaf 20, sqrt, 300 trees, balanced_subsample", "same code"],
    ["evaluation", "5-fold spatial CV (row blocks), rule and original labels", "same code"],
    ["dates", "7 labelled dates + 2 new", "4 labelled (2023-07-24, 2023-10-04, 2024-04-21, 2024-06-22); maps for all 10 VIIRS dates"],
    ["inputs", "66 (bands, pairs, indices, textures, angles, NLCD)", "46 from the VIIRS bands (below)"],
], [1600, 3700, 3700])
h(2, "2.1 Inputs")
p("The run 16 pool rebuilt on the VIIRS bands: red = I1, nir = I2, swir1 = I3, plus I4 and thermal = I5; all nd_ / diff_ "
  "pairs of red, nir, swir1; SAVI, MSAVI, nir + swir1, I4 - I5; NDVI, NDSI (I1 - I3), nir and red texture means / standard "
  "deviations in 3 x 3 and 7 x 7 windows (30 m pixels); terrain illumination (slope, northness, eastness, cos incidence, "
  "shadow) with the VIIRS sun angle; NLCD 2016 as 10 groups. VIIRS has no blue or green I-band, so EVI, BSI, NDWI, AWEI and "
  "the tasselled cap of run 16 do not exist here. As in run 16 the per-date sun constants are left out.")
for z, names in rf["inputs"].items():
    p(f"{z} RF, top 25 inputs (final model): " + ", ".join(names) + ".")
h(2, "2.2 Dates and labels")
rows = [[d, SEASON.get(d, ""), f"{rf['per_date'][d]['original']['pixels']:,}", f"{rf['per_date'][d]['rule']['pixels']:,}"]
        for d in labelled]
table(["date", "season", "labelled px", "rule-label px"], rows, [1800, 2600, 2200, 2200], ["l", "l", "r", "r"])
p("Run 16 dates without VIIRS (2024-03-12, 2024-04-20, 2024-04-29) are left out, so run 16's external test "
  "(2024-03-12) has no run 21 counterpart. Maps are made for all 10 VIIRS dates with the final model.")

pagebreak()
h(1, "3. Results (5-fold spatial cross-validation)")
for key, (name, m) in MODELS.items():
    h(2, name)
    for lab in ("rule", "original"):
        s = m[lab]
        table([f"{name}, {lab} labels: class", "true px", "pred px", "precision", "recall", "F1"],
              [[n, f"{v['true']:,}", f"{v['pred']:,}", f3(v["precision"]), f3(v["recall"]), f3(v["f1"])]
               for n, v in s["per_class"].items() if v["true"] or v["pred"]],
              [2800, 1200, 1200, 1200, 1200, 1200], ["l", "r", "r", "r", "r", "r"])
    if m.get("conf_image"):
        image(m["conf_image"], f"Run 21 {name}: confusion matrix, rule labels (counts and % of each true class).", 440)
h(2, "Run 16 (Landsat) on the same dates, for comparison")
table(["run 16 class (rule labels)", "true px", "precision", "recall", "F1"],
      [[n, f"{v['true']:,}", f3(v["precision"]), f3(v["recall"]), f3(v["f1"])] for n, v in r16["rule"]["per_class"].items() if v["true"]],
      [3000, 1500, 1500, 1500, 1500], ["l", "r", "r", "r", "r"])
h(2, "Per date (rule labels)")
rows = [[d, SEASON.get(d, ""), f"{rf['per_date'][d]['rule']['pixels']:,}"]
        + [f"{m['per_date'][d]['rule']['accuracy']:.3f} / {m['per_date'][d]['rule']['macro_f1']:.2f}" for _, m in MODELS.values()]
        for d in labelled]
table(["date", "season", "px"] + [f"{n} acc / F1" for n, _ in MODELS.values()], rows,
      [1500, 2200, 1200] + [2300] * len(MODELS), ["l", "l", "r"] + ["r"] * len(MODELS))
deck["per_date"] = rows
deck["per_class"] = [[n] + [f"{m['rule']['per_class'][n]['f1']:.2f}" for _, m in MODELS.values()]
                     + [f"{r16['rule']['per_class'][n]['f1']:.2f}"] for n in NAMES]
deck["conf"] = {n: m.get("conf_image") for n, m in MODELS.values()}

pagebreak()
h(1, "4. Maps")
p("Each image, left to right: VIIRS upsampled to 30 m; Landsat of the same day (B6/B5/B4, the bands of VIIRS I3/I2/I1); "
  "the ground truth = hand labels painted on the dimmed Landsat image; the run 16 (Landsat) map; the run 21 map from VIIRS. "
  "Yellow line = 30 m water mask (river / ice only inside). The per-date accuracy is the cross-validated accuracy on that "
  "day's hand labels; the maps themselves come from the final model trained on all labelled pixels.")
imgs = {}
for key in MODELS:
    for f in sorted((R21 / "2_with_daksh" / key / "images").glob("pred_*.jpg")):
        d = f.stem[5:]
        imgs.setdefault(d, []).append(str(f))
        image(f, f"Run 21 {MODELS[key][0]}, {d} ({SEASON.get(d, '')}, "
                 f"{'hand-labelled' if d in labelled else 'no hand labels'}).", 640)
deck["images"] = imgs

h(1, "5. Things to know")
bullets([
    "Upsampling copies each 375 m VIIRS value to its 12.5 x 12.5 Landsat pixels: VIIRS adds no 30 m detail. 30 m edges come "
    "from NLCD, the terrain, the texture windows and the water mask.",
    "Land classes largely reproduce NLCD; the high forest / shrub / bare scores measure agreement with NLCD as much as with VIIRS.",
    "Snow on land is usually patchy at 30 m and averaged away at 375 m, hence the low snow score.",
    "Only 4 labelled dates (vs 7 in run 16) and few ice / snow pixels: ice and snow scores move a lot with each polygon.",
    "Spatial CV holds out row blocks, not dates: the same date is in training and test (as in run 16).",
    "No cloud mask on VIIRS.",
])
h(1, "6. Files")
table(["what", "where (~/Desktop/nenana_crops/daksh/)"], [
    ["script", "viirs_run16.py (uses viirs_30m.py for the upsampling and image helpers)"],
    ["results", "output/21_viirs_run16/2_with_daksh/random_forest/ (metrics.txt, oof.npz, pred_<date>.tif + .qml, images/)"],
    ["all metrics", "output/21_viirs_run16/metrics_all.json"],
    ["report / deck", "report/word_content_run21.py + report/build_word.js; "
                      "~/Desktop/nenana_river_models/presentation/build_deck_run21.js"],
], [2300, 6700])

json.dump(B, open(sys.argv[1], "w"))
(O / "deck_numbers_run21.json").write_text(json.dumps(deck, indent=1))
print(len(B), "blocks")
