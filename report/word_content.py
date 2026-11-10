"""Collect everything for the Word report into content.json (blocks for build_word.js)."""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data as D
import summarise as S

B = []
h = lambda lvl, text: B.append({"t": f"h{lvl}", "text": text})
p = lambda text: B.append({"t": "p", "text": text})
bullets = lambda items: B.append({"t": "bullets", "items": items})
table = lambda header, rows, widths=None, align=None: B.append(
    {"t": "table", "header": header, "rows": rows, "widths": widths, "align": align})
def image(path, caption, w=620):
    """Add a figure; big JPEGs are shrunk to 1600 px wide for the report."""
    from PIL import Image
    path = Path(path)
    if path.suffix == ".jpg":
        im = Image.open(path)
        if im.width > 1600:
            small = Path(sys.argv[1]).parent / "fig" / path.name
            small.parent.mkdir(exist_ok=True)
            im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS).save(small, quality=85)
            path = small
    B.append({"t": "image", "path": str(path), "caption": caption, "width": w})
brk = lambda: B.append({"t": "pagebreak"})


def parse_block(text):
    """First (all-dates) block of a metrics file -> class rows, accuracy, macro F1, confusion."""
    lines = text.splitlines()
    rows, cm, names = [], [], []
    acc = mf1 = None
    i = 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r"overall accuracy ([\d.]+)\s+macro F1 ([\d.]+)", l)
        if m:
            acc, mf1 = m.groups()
        elif l.startswith("confusion"):
            names = lines[i + 1].split()
            for l2 in lines[i + 2:]:
                if not l2.strip():
                    break
                cm.append(l2.split())
            break
        else:
            q = l.split()
            if len(q) == 5 and re.match(r"[\d,]+$", q[1]) and "." in q[2]:
                rows.append(q)
        i += 1
    return rows, acc, mf1, names, cm


h(0, "Nenana river ice: Random Forest vs UNet++ on Landsat 30 m")
p("RiverProject · daksh models · 2026-10-27")
p("Random Forest and UNet++ models classify every Landsat 30 m pixel of the Nenana study box "
  "as river (open water), ice, snow or land, trained only on the hand-drawn label polygons, which "
  "are taken as ground truth. This report has every model's scores and confusion matrix, the "
  "labelled pixels per class, the model settings and the main figures.")

h(1, "1. Summary")
bullets([
    "Random Forest wins almost everywhere. Four classes, spatial cross-validation on the 4 patch dates: "
    "RF 0.952 accuracy / 0.750 macro F1, UNet++ 0.943 / 0.701. UNet++ only matches or edges RF on river vs not river (0.962 vs 0.957).",
    "River and land are easy (F1 about 0.96); ice (about 0.72) and snow (about 0.3 to 0.4) are hard: only 1,732 ice and 564 snow labelled pixels.",
    "Splitting the task does not help: two binary models merged to four classes score about the same as one four-class model; "
    "one-vs-rest models find ice no better than the four-class RF, and snow worse.",
    "Adding the labelled daksh date (with_daksh) costs about 1 point, mostly because the daksh image has no temperature band, so those models use 9 inputs instead of 11.",
    "The models do not carry over to a new winter date: trained on the patch dates and tested on the daksh 2024-03-12 labels, "
    "four-class accuracy is 0.34 (RF) and 0.47 (UNet++). RF calls most ice snow; none of the patch dates is midwinter.",
    "On unlabelled dates UNet++ marks much land as river and clouds come out as snow or ice.",
])

h(1, "2. Data")
h(2, "2.1 Datasets")
table(["Dataset (folder)", "Training dates", "Scored on", "Inputs"], [
    ["1_exclude_daksh", "4 patch dates", "spatial cross-validation", "11"],
    ["2_with_daksh", "4 patch dates + daksh 2024-03-12", "spatial cross-validation", "9"],
    ["3_external_test_daksh", "4 patch dates", "daksh 2024-03-12 labels (external cohort)", "9"],
], [2400, 2600, 2900, 1100])
p("11 inputs: surface reflectance SR_B2 to SR_B7, surface temperature ST_B10, JRC water occurrence, NDWI, NDSI, NDVI. "
  "9 inputs: blue, green, red, NIR, SWIR1, water occurrence, NDWI, NDSI, NDVI (the daksh image has no SWIR2 and no "
  "temperature; its Level-1 numbers are converted to reflectance with the Landsat Collection 2 constants and the sun angle). "
  "The unlabelled daksh dates 2024-04-20 and 2024-04-29, and 2026-03-26 and 2023-05-05, are only predicted.")

h(2, "2.2 Where the labels and images come from")
table(["Date", "Season", "QGIS group", "Labels", "Imagery"], [
    ["2023-07-24", "summer", "mathus_naira", "patches/nenana_20230724/labels/labels_2023-07-24.shp", "patch _sr.tif + _thermal.tif"],
    ["2023-10-04", "freeze-up", "mathus_naira", "patches/nenana_20231004/labels/", "patch"],
    ["2024-04-21", "breakup", "mathus_naira", "patches/nenana_20240421/labels/", "patch"],
    ["2024-06-22", "summer", "mathus_naira", "patches/nenana_20240622/labels/", "patch"],
    ["2024-03-12", "winter", "daksh", "~/labels_20240312.gpkg (3 polygons)", "nenana_crops/20240312/landsat_20240312_box.tif"],
    ["2024-04-20, 2024-04-29", "spring", "daksh", "none (predicted only)", "nenana_crops/<date>/landsat_*_box.tif"],
    ["2026-03-26, 2023-05-05", "midwinter, breakup", "not in project", "none (predicted only)", "~/Desktop/Patches/nenana_<date>/"],
], [1300, 1100, 1200, 3000, 2400])
p("patches/ = ~/Desktop/RiverProject/Matus/narrow_rivers/patches. Label classes (class_id): 1 river (open water), 2 ice, "
  "3 snow, 4 land; codes 0 and 9 are ignored. Study box lon -149.338 to -149.069, lat 64.094 to 64.535. Every date is on "
  "the same 995 x 1635 grid (0.00027027 degrees, EPSG:4326), so the same row and column is the same place on every date. "
  "The daksh image stores Level-1 numbers, converted to reflectance with (DN x 2.0e-5 - 0.1) / cos(sun zenith), the sun "
  "zenith read from the image (stored x 100); pixels with missing imagery are skipped.")

h(2, "2.3 How the dates are used")
p("Only dates with labels are trained on: in with_daksh the 4 patch dates + 2024-03-12; the unlabelled dates are only "
  "predicted. All labelled dates are pooled into one training set, so one model per task learns summer, freeze-up, "
  "breakup and (with daksh) winter at once. Each pixel is an example whose inputs come from its own date's image and whose "
  "label comes from its own date's polygons; a date's image is never paired with another date's labels. UNet++ draws each "
  "training crop from one date, and its input scaling (band means and spreads) is computed from the training dates only.")
p("Pixels labelled on more than one date: polygons of different dates often cover the same ground. Of the 85,890 labelled "
  "pixels in with_daksh, 49,688 locations are labelled on one date, 13,005 on two and 3,250 on three or more.")
bullets([
    "Training: each (date, location) is its own example with that date's inputs and label. The same spot can be river in "
    "July, ice in October and snow in March; all are correct and all are used. Nothing is averaged or merged across dates.",
    "Cross-validation: a 64-row band is in the same fold on every date, so a tested location is held out of training on "
    "all dates. A model cannot score well by remembering a place from another date.",
    "External test: the 2024-03-12 pixels share 1,108 location-date pairs with patch-date labels, mostly with another "
    "class. It tests a new date and season, not new ground.",
])
lab = {}
for ds_ in ("with_daksh",):
    D.configure("four", ds_)
    for s in D.load_all():
        if (s["y"] != D.IGNORE).any():
            lab[s["date"]] = s["y"]
rows = []
ds_list = sorted(lab)
for i, a in enumerate(ds_list):
    for b in ds_list[i + 1:]:
        m = (lab[a] != D.IGNORE) & (lab[b] != D.IGNORE)
        if m.any():
            rows.append([f"{a} & {b}", f"{int(m.sum()):,}", f"{(lab[a][m] == lab[b][m]).mean():.0%}"])
table(["Dates", "Shared pixels", "Same class on both"], rows, [4000, 2500, 2500], ["l", "r", "r"])

h(2, "2.4 Labelled pixels per class")
info = {}
for ds in D.DATASETS:
    counts, groups = S.label_counts(ds)
    info[ds] = (counts, groups)
for ds in D.DATASETS:
    counts, groups = info[ds]
    h(3, S.DATASET_TITLES[ds])
    rows = [[d, groups[d], *[f"{v:,}" for v in c], f"{c.sum():,}",
             "test" if ds == "external_daksh" and groups[d] == "daksh" else "train"]
            for d, c in sorted(counts.items())]
    tr = sum(c for d, c in counts.items() if not (ds == "external_daksh" and groups[d] == "daksh"))
    rows.append(["Training total", "", *[f"{v:,}" for v in tr], f"{tr.sum():,}", ""])
    if ds == "external_daksh":
        te = sum(c for d, c in counts.items() if groups[d] == "daksh")
        rows.append(["Test total", "", *[f"{v:,}" for v in te], f"{te.sum():,}", ""])
    table(["Date", "Group", "River", "Ice", "Snow", "Land", "Total", "Used for"], rows,
          [1300, 1400, 1100, 900, 900, 1100, 1100, 1200], ["l", "l", "r", "r", "r", "r", "r", "l"])

h(3, "Pixels per class of each task")
rows = []
c1, g1 = info["exclude_daksh"]; c2, g2 = info["with_daksh"]
t1 = sum(c1.values()); t2 = sum(c2.values()); te = sum(c for d, c in c1.items() if False) if False else None
ext_c, ext_g = info["external_daksh"]
te = sum(c for d, c in ext_c.items() if ext_g[d] == "daksh")
for t in S.TASKS:
    if t == "combined":
        continue
    f = lambda hand: ", ".join(f"{k} {v:,}" for k, v in S.task_counts(t, hand).items())
    rows.append([S.TASK_TITLES[t], f(t1), f(t2), f(te)])
table(["Task", "exclude_daksh (= external training)", "with_daksh", "external test"], rows,
      [2600, 2400, 2200, 1800])

h(1, "3. Models and evaluation")
table(["", "Random Forest", "UNet++"], [
    ["Library", "scikit-learn 1.9", "segmentation-models-pytorch 0.6, PyTorch 2.14, RTX 3080"],
    ["Architecture", "300 trees, minimum 2 samples per leaf", "UNet++ with ResNet-34 encoder (ImageNet weights)"],
    ["Sees", "one pixel's inputs", "256 x 256 px neighbourhood (shape, texture)"],
    ["Class imbalance", "balanced_subsample class weights", "cross-entropy weighted by 1/sqrt(class frequency)"],
    ["Training", "all labelled pixels of the training folds",
     "1,500 steps x 8 crops centred near labels, 90-degree turns and flips, AdamW lr 1e-3, one-cycle, mixed precision; loss only on labelled pixels"],
    ["Saved", "refitted when needed", "unetpp_final.pt (weights + input scaling)"],
], [1800, 3400, 3800])
h(2, "Tasks")
table(["Folder", "Classes", "From the hand labels"], [
    ["four_class_river_ice_snow_land", "river, ice, snow, land", "as drawn"],
    ["binary/snow_vs_no_snow", "no_snow, snow", "snow = ice + snow; no_snow = river + land"],
    ["binary/river_vs_not_river", "not_river, river", "river = open water + ice; not_river = snow + land"],
    ["binary/combined_to_four_class", "river, ice, snow, land", "the two binary answers merged (no training)"],
    ["one_vs_rest/<class>_vs_rest", "not_<class>, <class>", "one class vs everything else (river, ice, snow, land)"],
], [3200, 2300, 3500])
p("Why three set-ups: one four-class model must trade the classes off against each other; two binary models answer the two "
  "questions that define the river/land states (is it frozen, is it snowy); one-vs-rest models show how well each class can "
  "be found on its own, which matters for the rare classes (ice 1.7k px, snow 0.6k px against 40k river and 43k land).")
h(2, "How scores are measured")
p("Datasets 1 and 2: 5-fold spatial cross-validation. The box is cut into east-west bands 64 rows (about 1.9 km) tall; "
  "band i belongs to fold i mod 5 on every date. Each fold's model trains on the other 4 folds and is scored on its own; "
  "training pixels within 12 rows of the test band are dropped. Every labelled pixel is scored once by a model that never "
  "saw it. Afterwards one final model per task is trained on all labels and draws the maps.")
p("Dataset 3 (external cohort): one model trained on all labelled pixels of the 4 patch dates, scored on the 532 labelled "
  "pixels of daksh 2024-03-12, a date, season and labeller it never saw. Only 532 pixels and no river, so read it as a "
  "check of carrying over to a new date, not a precise number.")
p("Why spatial folds: a random pixel split would score pixels whose neighbours from the same polygon were in training, "
  "which inflates the numbers. Scores are always the cross-validated (or external) ones, never scores on training pixels, "
  "and they cover the labelled areas only; on the unlabelled dates there is nothing to score against.")
p("Confusion matrices: rows = true class (hand label), columns = predicted class.")

# 4-5. results per dataset
for n, ds in enumerate(D.DATASETS):
    brk()
    h(1, f"{4 + n}. Results: {S.DATASET_TITLES[ds].split(': ', 1)[0][3:]}")
    p(S.DATASET_TITLES[ds].split(": ", 1)[1].capitalize() + f". Folder output/{D.DATASET_DIRS[ds]}/.")
    overview, details = [], []
    for t in S.TASKS:
        for m, name in D.MODEL_NAMES.items():
            f = D.task_dir(t, ds) / D.MODEL_DIRS[m] / "metrics.txt"
            if not f.exists():
                continue
            rows, acc, mf1, names, cm = parse_block(S.first_block(f))
            overview.append([S.TASK_TITLES[t], name, acc, mf1])
            details.append((t, name, rows, acc, mf1, names, cm, f.parent.relative_to(D.OUTPUT)))
    h(2, "Overview")
    table(["Task", "Model", "Accuracy", "Macro F1"], overview, [4600, 1800, 1300, 1300], ["l", "l", "r", "r"])
    four = D.task_dir("four", ds) / "evaluation"
    image(four / "scores.png", "Four classes: F1, precision and recall per class, RF vs UNet++.")
    image(four / "confusion.png", "Four classes: confusion matrices, row-normalised, with pixel counts.")
    h(2, "Every model: scores and confusion matrix")
    for t, name, rows, acc, mf1, names, cm, rel in details:
        h(3, f"{S.TASK_TITLES[t]} · {name}")
        p(f"Accuracy {acc}, macro F1 {mf1}. Folder {rel}/.")
        k = len(names)
        table(["Class", "Pixels", "Precision", "Recall", "F1"], rows, [2200, 1500, 1500, 1500, 1500],
              ["l", "r", "r", "r", "r"])
        cw = 7800 // (k + 1)
        table(["true \\ predicted"] + names, cm, [cw] * (k + 1), ["l"] + ["r"] * k)

# state masks
brk()
h(1, "7. River/land state per 375 m block")
p("A single pixel cannot be 'frozen river with snowy land', so the box is cut into 375 m blocks (12.5 Landsat pixels). "
  "In each block: river frozen if predicted ice outnumbers open water; land snowy if predicted snow outnumbers bare land; "
  "blocks with fewer than 3 river or ice pixels are 'no river'. States: ice_free_river_snow_free_land, "
  "ice_free_river_snow_covered_land, ice_covered_river_snow_free_land, ice_covered_river_snow_covered_land. "
  "Scored on blocks where the hand labels hold both river and land.")
rows = []
for ds in D.DATASETS:
    for t in ("four", "combined"):
        for m, name in D.MODEL_NAMES.items():
            f = D.task_dir(t, ds) / D.MODEL_DIRS[m] / "scene_mask_metrics.txt"
            if f.exists():
                b = f.read_text().split("\n\n--- ")[0]
                acc, mf1 = re.search(r"overall accuracy ([\d.]+)\s+macro F1 ([\d.]+)", b).groups()
                nb = sum(int(l.split()[1].replace(",", "")) for l in b.splitlines()
                         if l.startswith("ice_") and len(l.split()) == 5 and "." in l.split()[2])
                rows.append([D.DATASET_DIRS[ds], "four classes" if t == "four" else "binary combined", name, str(nb), acc, mf1])
table(["Dataset", "From", "Model", "Blocks", "Accuracy", "Macro F1"], rows,
      [2300, 1700, 1600, 1000, 1200, 1200], ["l", "l", "l", "r", "r", "r"])
for ds in ("exclude_daksh", "external_daksh"):
    f = D.task_dir("four", ds) / "random_forest" / "scene_mask_metrics.txt"
    rows, acc, mf1, names, cm = parse_block(f.read_text().split("\n\n--- ")[0])
    short = {"ice_free_river_snow_free_land": "open / bare", "ice_free_river_snow_covered_land": "open / snowy",
             "ice_covered_river_snow_free_land": "frozen / bare", "ice_covered_river_snow_covered_land": "frozen / snowy"}
    h(3, f"State confusion: {D.DATASET_DIRS[ds]}, four classes, Random Forest (accuracy {acc})")
    table(["true \\ predicted"] + [short[n] for n in names], [[short[r[0]]] + r[1:] for r in cm],
          [2200, 1700, 1700, 1700, 1700], ["l", "r", "r", "r", "r"])
p("open = ice-free river, frozen = ice-covered river; bare = snow-free land, snowy = snow-covered land.")

# figures
brk()
h(1, "8. Test images: ground truth vs Random Forest vs UNet++")
p("Each figure is one labelled date, zoomed on a 3.6 x 3.6 km area holding many labels (up to two areas per date). "
  "Top row: the true-colour image, then what Random Forest and UNet++ predicted on the labelled pixels when they were "
  "tested (cross-validated: each pixel predicted by a model that never saw it; external: the model never saw the date). "
  "Bottom row: the ground truth (hand labels), then the full Random Forest and UNet++ maps. The accuracy in the titles is "
  "for all labelled pixels of that date. River blue, ice magenta, snow cyan, land brown. Four-class models.")
FIGS = HERE / "figures"
season = {"2023-07-24": "summer", "2023-10-04": "freeze-up", "2024-03-12": "winter, daksh",
          "2024-04-21": "breakup", "2024-06-22": "summer"}
h(2, "8.1 with_daksh: cross-validated, all 5 labelled dates")
for d in sorted(season):
    for f in sorted(FIGS.glob(f"gt_with_daksh_{d}_area*.png")):
        image(f, f"{d} ({season[d]}), {f.stem.split('_')[-1].replace('area', 'area ')}: image, RF and UNet++ test "
                 f"predictions (top); ground truth, RF and UNet++ maps (bottom).")
h(2, "8.2 external_test_daksh: trained on the patch dates, tested on 2024-03-12")
for f in sorted(FIGS.glob("gt_external_daksh_2024-03-12_area*.png")):
    image(f, "2024-03-12 (winter, daksh), external test: models trained without this date. RF calls most ice snow; "
             "UNet++ finds more ice but marks land as river.")
h(2, "8.3 Dates without labels")
p("No ground truth exists here, so these show only what the models find: true colour, RF classes, UNet++ classes, and the "
  "river/land state masks of both (with_daksh four-class models). State colours: open/snow-free blue, open/snowy light "
  "cyan, frozen/snow-free orange, frozen/snowy magenta.")
for d, cap in [("2024-04-20", "spring, daksh"), ("2024-04-29", "spring, daksh"),
               ("2026-03-26", "midwinter"), ("2023-05-05", "breakup")]:
    image(FIGS / f"new_{d}.png", f"{d} ({cap}), no labels: true colour, RF, UNet++, RF state, UNet++ state.")
h(2, "8.4 Prediction maps")
p("Image | image with the predicted classes (with_daksh four-class models). River blue, ice magenta, snow cyan, land brown.")
base = D.task_dir("four", "with_daksh")
for f, cap in [(base / "evaluation" / "zoom_2023-10-04.png", "2023-10-04 (freeze-up), zoom on the labels: true colour, hand labels, RF, UNet++ (with_daksh)."),
               (base / "evaluation" / "zoom_2024-03-12.png", "2024-03-12 (winter, daksh), zoom on the labels (with_daksh)."),
               (base / "images" / "rf_2024-04-20.jpg", "2024-04-20 (unlabelled daksh date), Random Forest, with_daksh."),
               (base / "images" / "new_rf_2026-03-26.jpg", "2026-03-26 (unlabelled, midwinter), Random Forest, with_daksh."),
               (base / "images" / "new_unetpp_2023-05-05.jpg", "2023-05-05 (unlabelled, breakup), UNet++, with_daksh."),
               (base / "images" / "rf_2024-04-20_state.jpg", "2024-04-20 river/land state mask, Random Forest, with_daksh.")]:
    image(f, cap)
h(2, "8.5 Whole box: true colour, hand labels, RF, UNet++")
for d in sorted(season):
    image(base / "evaluation" / f"compare_{d}.png", f"{d} ({season[d]}), whole study box (with_daksh four-class).")
image(D.task_dir("four", "external_daksh") / "evaluation" / "compare_2024-03-12.png",
      "2024-03-12, whole study box, external test models (trained without this date).")
p("Every map of every model and dataset is also a full-resolution JPEG in its task folder's images/ "
  "(new_ = unlabelled date, _state = river/land state mask).")

brk()
h(1, "9. Files and how to rerun")
p("All in ~/Desktop/nenana_crops/daksh/. README.md explains each file; output/RESULTS.md has every number in this report; "
  "each model folder has model_info.txt (method, settings, inputs, dates, pixels, scores, confusion matrix). "
  "Folders: output/<1_exclude_daksh | 2_with_daksh | 3_external_test_daksh>/<task folder>/.")
h(2, "Inside each task folder")
table(["Path", "What it is"], [
    ["random_forest/, unet_plus_plus/", "one folder per model"],
    ["<model>/model_info.txt", "the model in one page: method and settings, inputs, training/test dates and pixels per class, scores, confusion matrix, files"],
    ["<model>/metrics.txt", "precision, recall, F1 per class, accuracy, macro F1, confusion matrix; all dates, then per date; RF also lists input importance"],
    ["<model>/oof.npz", "the cross-validated prediction of every labelled pixel (date, row, col, true, predicted)"],
    ["<model>/pred_<date>.tif + .qml", "final model's 30 m map for every date; the .qml gives QGIS the class colours"],
    ["<model>/scene_mask_<date>.tif", "(four-class, combined) river/land state mask"],
    ["<model>/scene_mask_metrics.txt", "(four-class) state-mask scores on blocks where the labels hold both river and land"],
    ["<model>/scene_classes.csv", "(four-class, combined) one summary state for the whole box per date; coarse, use the mask"],
    ["<model>/unetpp_final.pt", "UNet++ final weights + input scaling"],
    ["evaluation/scores.png, confusion.png", "F1/precision/recall per class and both confusion matrices, RF vs UNet++"],
    ["evaluation/summary.csv", "every score of both models in one table"],
    ["evaluation/compare_<date>.png, zoom_<date>.png", "true colour, hand labels, RF, UNet++; whole box and zoomed on the labels"],
    ["new_dates/<date>/<model>_pred.tif", "maps of the unlabelled dates 2026-03-26 and 2023-05-05"],
    ["images/*.jpg", "full-resolution JPEGs, image next to image + mask"],
], [3400, 5600])
p("Colours: river blue, ice magenta, snow cyan, land brown, 'not ...' grey. State mask: open/snow-free blue, open/snowy "
  "light cyan, frozen/snow-free orange, frozen/snowy magenta.")
h(2, "Scripts")
p("Environment ~/venvs/river-ml (PyTorch with CUDA, segmentation-models-pytorch, scikit-learn, GDAL); "
  "PY=~/venvs/river-ml/bin/python.")
table(["Script", "Does"], [
    ["data.py", "dates, inputs, label mapping per task, spatial folds, folder names, score tables, map writing"],
    ["train_rf.py --task --dataset", "RF cross-validation + final maps"],
    ["train_unetpp.py --task --dataset", "UNet++ cross-validation + final maps (--iters, --encoder)"],
    ["external_test.py --task", "train on the patch dates, score on the daksh labels"],
    ["combine.py --dataset", "binary snow + river -> four classes, scored on the cross-validated predictions"],
    ["scene_mask.py --dataset", "river/land state masks + their scores"],
    ["evaluate.py --task --dataset", "figures in evaluation/ (--task combined too)"],
    ["predict_new.py --task --dataset [--dates]", "maps for unlabelled dates"],
    ["images.py --dataset", "all JPEGs"],
    ["summarise.py", "model_info.txt in every model folder + output/RESULTS.md"],
    ["add_to_project.py", "adds the maps to all_patches.qgz (close QGIS first)"],
    ["run_all.sh, run_rest.sh, run_external.sh", "the full runs, about 4 hours on the RTX 3080"],
    ["report/test_panels.py, word_content.py, build_word.js", "this report and its test figures"],
], [3800, 5200])
p("One model by hand: $PY train_rf.py --task ind_ice --dataset with_daksh, then train_unetpp.py and evaluate.py with the "
  "same arguments. Task names: four, snow, river, ind_river, ind_ice, ind_snow, ind_land; datasets: exclude_daksh, "
  "with_daksh, external_daksh.")

h(1, "10. Things to know when reading the results")
bullets([
    "Rare classes: ice (1,732 px) and snow (564 px, 714 with daksh) come from only 2023-10-04, 2024-04-21 and 2024-03-12, "
    "so their scores move a lot with each new polygon.",
    "2023-07-24 holds 31,021 of the 39,991 river pixels, so it shapes what the models call river. On the unlabelled dates "
    "the models mark some vegetated land as river.",
    "Clouds were never labelled, so models call them snow or ice.",
    "with_daksh has no temperature input, which the exclude_daksh Random Forest ranked among its 4 most important inputs; "
    "compare the two datasets with that in mind.",
    "Per-date macro F1 only averages classes present on that date (a summer date has no ice or snow).",
    "The external test has only 532 pixels and no river.",
    "VIIRS is not used yet.",
])

json.dump(B, open(sys.argv[1], "w"))
print(len(B), "blocks")
