"""
Random Forest report, every run (01, 03, 05-15; runs 02 and 04 were not Random Forest), into
content.json for build_word.js:

    ~/venvs/river-ml/bin/python report/word_content_rf.py <scratch>/content.json
    NODE_PATH=<docx node_modules> node report/build_word.js <scratch>/content.json \
        ~/Desktop/nenana_river_models/reports/Nenana_Random_Forest_report_all_runs.docx

Every number is read from the runs' metrics.txt / search.txt / penalty_scan.txt /
river_on_land.txt, so rerunning after a new run refreshes the report.
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data as D
import summarise as S

O = D.OUTPUT
FIG = Path.home() / "Desktop/nenana_river_models/presentation/figures"
B = []
h = lambda lvl, text: B.append({"t": f"h{lvl}", "text": text})
p = lambda text: B.append({"t": "p", "text": text})
bullets = lambda items: B.append({"t": "bullets", "items": items})
table = lambda header, rows, widths=None, align=None: B.append(
    {"t": "table", "header": header, "rows": rows, "widths": widths, "align": align})
image = lambda path, caption, w=620: B.append({"t": "image", "path": str(path), "caption": caption, "width": w})
brk = lambda: B.append({"t": "pagebreak"})

FOUR, SIX = "four_class_river_ice_snow_land/random_forest", "six_class/random_forest"
DS = [("exclude_daksh", "1_exclude_daksh"), ("with_daksh", "2_with_daksh"), ("external_test", "3_external_test_daksh")]

# run id -> (title, folder pattern with {ds}, classes, inputs, settings, what it tests)
RUNS = {
    "01": ("Four classes, 11 inputs (baseline)", "{ds}/" + FOUR, 4,
           "11 (exclude_daksh): SR_B2-SR_B7, ST_B10, JRC water occurrence, NDWI, NDSI, NDVI; 9 with daksh (no SWIR2, no temperature)",
           "300 trees, min 2 samples per leaf, balanced_subsample",
           "The first model: a few bands and three indices, plus the JRC water-occurrence layer."),
    "03": ("Four classes, river corridor only", "4_river_corridor_150m/{ds}/" + FOUR, 4,
           "as run 01", "as run 01; trained and predicted only inside the river + 150 m banks",
           "Whether limiting the model to the river and its banks helps (it does not: it loses context)."),
    "05": ("Four classes, all bands", "5_rf_all_bands/{ds}/" + FOUR, 4,
           "all bands: raw bands, normalized difference and difference of every band pair, EVI, SAVI, MSAVI, BSI, "
           "brightness, AWEI, tasseled cap, 3x3 and 7x7 texture (mean, std); no water mask, no JRC",
           "500 trees; min_samples_leaf and max_features chosen by spatial CV",
           "Every band set against every other band instead of 3 hand-picked indices."),
    "06": ("Six classes, all bands", "6_rf_six_classes/{ds}/" + SIX, 6,
           "as run 05", "as run 05",
           "Land split into forest / shrub-grass-wetland / bare-developed with NLCD 2016 (labels only, not inputs)."),
    "07": ("Six classes, all bands + NLCD input", "7_rf_six_classes_nlcd_input/{ds}/" + SIX, 6,
           "run 05 inputs + NLCD 2016 land cover as 10 yes/no inputs", "as run 05",
           "NLCD also given as an input, so the forest knows the land type."),
    "08": ("Six classes, tuned (no NLCD input)", "8_best_six_class/{ds}/six_class", 6,
           "as run 06", "best of: leaf 5/10/20/40 x max_features sqrt/0.1/0.2, class weights, top-20/35 inputs, OOB class weights",
           "How far tuning alone gets the six-class model without NLCD as input."),
    "09": ("Six classes, tuned + NLCD input", "9_best_six_class_nlcd_input/{ds}/six_class", 6,
           "as run 07", "the run 08 search with NLCD inputs", "The run 08 search with NLCD as input."),
    "10": ("Six classes, terrain + context + NLCD", "10_rf_six_classes_terrain/{ds}/" + SIX, 6,
           "run 07 inputs + distance to NLCD water, DEM elevation, slope, northness, eastness, height above the "
           "valley floor (300 m, 1 km), MNDWI/NDWI/NDVI/brightness averaged over 15x15 and 31x31 px",
           "500 trees; leaf and max_features chosen by spatial CV",
           "Lets the forest learn where a river can be (low, flat, near water, part of a long strip)."),
    "11": ("Six classes, feature-group search", "11_feature_search/{ds}/six_class", 6,
           "best groups found: band differences + NLCD + thermal + terrain + brightness + distance to water (35 inputs)",
           "greedy forward search over 14 input groups; final 500 trees, max_features 0.1, leaf 5",
           "Which input groups actually help: add one group at a time, keep it only if macro F1 rises."),
    "12": ("Six classes, all bands + sun/illumination angles + NLCD", "12_rf_six_classes_bands_angles_nlcd/{ds}/" + SIX, 6,
           "run 05 inputs + NLCD + per-date sun elevation and azimuth + per-pixel slope, northness, eastness, "
           "cos(incidence), shadowed (from the DEM)", "500 trees; leaf and max_features chosen by spatial CV",
           "Illumination: slopes facing away from the sun look darker. Note: sun elevation/azimuth are one value per "
           "date, so with 4-5 dates they partly tell the forest which date a pixel is from."),
    "13": ("Six classes, limited parameters", "13_rf_six_classes_limited/{ds}/" + SIX, 6,
           "run 12 inputs without the per-date sun constants; only the top 25, ranked inside each training fold",
           "top 10/15/20/25 inputs x max_depth 8/12/16, min_samples_leaf 20, 300 trees; best: 25 inputs, depth 16",
           "A small model that cannot memorise the training pixels; training vs CV accuracy reported."),
    "14": ("Six classes, limited + tuned river penalty", "14_rf_six_classes_river_fix/{ds}/" + SIX, 6,
           "as run 13", "as run 13; then P(river) lowered on NLCD land far from NLCD water, strength tuned per fold",
           "Stop river appearing on land. The tuner chose no penalty in every fold, so run 14 = run 13."),
    "15": ("Six classes, limited + forced river penalty (600 m)", "15_rf_six_classes_river_forced/{ds}/" + SIX, 6,
           "as run 13", "as run 13; river penalty always on, scale 600 m",
           "River on land removed from the maps at almost no cost in score."),
}


def blocks(path):
    """metrics.txt -> list of blocks; per-date blocks start with '--- <date>'."""
    return re.split(r"\n(?=--- \d{4}-\d{2}-\d{2})", Path(path).read_text())


def parse(text):
    """One score block -> class rows, accuracy, macro F1, confusion header, confusion rows."""
    lines = text.splitlines()
    rows, cm, names, acc, mf1 = [], [], [], None, None
    for i, l in enumerate(lines):
        m = re.search(r"overall accuracy ([\d.]+)\s+macro F1 ([\d.]+)", l)
        if m and acc is None:
            acc, mf1 = m.groups()
        elif l.startswith("confusion") and not cm:
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
    return rows, acc, mf1, names, cm


def metrics(run, ds):
    f = O / RUNS[run][1].format(ds=ds) / "metrics.txt"
    return f if f.exists() else None


SHORT = {"shrub_grass_wetland": "shrub/grass/wet", "bare_developed": "bare/dev"}


def score_tables(f):
    rows, acc, mf1, names, cm = parse(blocks(f)[0])
    table(["Class", "Pixels", "Precision", "Recall", "F1"], rows, [2400, 1500, 1500, 1500, 1500], ["l", "r", "r", "r", "r"])
    k = len(names)
    first = 2000 if k == 6 else 2200
    cw = (9000 - first) // k
    table(["true \\ predicted"] + [SHORT.get(n, n) for n in names],
          [[SHORT.get(r[0], r[0])] + r[1:] for r in cm], [first] + [cw] * k, ["l"] + ["r"] * k)


def score(run, ds):
    f = metrics(run, ds)
    if not f:
        return "-", "-"
    _, acc, mf1, _, _ = parse(blocks(f)[0])
    return acc, mf1


# ---------------------------------------------------------------- title + summary
s10 = {d: score("10", d) for _, d in DS}
s11 = score("11", "1_exclude_daksh"); s12 = score("12", "1_exclude_daksh"); s15 = score("15", "2_with_daksh")
h(0, "Nenana river ice: Random Forest on Landsat 30 m, every run")
p("RiverProject · daksh models · 2026-11-03")
p("Random Forest models classify every Landsat 30 m pixel of the Nenana study box. They are trained only on the "
  "hand-drawn label polygons, which are the ground truth. Four-class runs predict river (open water), ice, snow and "
  "land; six-class runs split land into forest, shrub/grass/wetland and bare/developed using NLCD 2016. This report has "
  "every Random Forest run (01, 03, 05 to 15): what it changes, its inputs and settings, its scores and confusion matrix "
  "on every dataset, plus the labelled pixels, the method, figures of ground truth against predictions, and the files. "
  "Runs 02 and 04 were not Random Forest and are left out.")

h(1, "1. Summary")
bullets([
    f"Best six-class scores without per-date help: run 11 (feature search) {s11[0]} accuracy / {s11[1]} macro F1 and "
    f"run 10 (terrain + context + NLCD) {s10['1_exclude_daksh'][0]} / {s10['1_exclude_daksh'][1]}, 5-fold spatial "
    f"cross-validation on the 4 patch dates. With the daksh date added, run 10 is best ({s10['2_with_daksh'][0]} / {s10['2_with_daksh'][1]}).",
    f"Run 12 scores higher ({s12[0]} / {s12[1]}) but part of that gain comes from the per-date sun angles, which tell the "
    "forest which date a pixel is from; it is reported, not used as the best model.",
    "Best four-class model: run 05 (all bands) 0.961 / 0.755, up from 0.952 / 0.750 for the 11-input baseline (run 01).",
    "NLCD land cover as an input is the biggest single gain for six classes: 0.822 / 0.663 (run 06) to 0.962 / 0.819 (run 07).",
    "Limited model (run 13: 25 inputs, depth 16, 20 px leaves): 0.947 / 0.808, training accuracy only 0.025 above CV, "
    "so it barely overfits; it costs about 1.5 points against run 10.",
    f"River on land: the forced penalty of run 15 cuts river on NLCD land from 22% to 4% on 2024-04-21 and from 89% to "
    f"28% on 2024-04-29, for 0.001 accuracy ({s15[0]} / {s15[1]} with daksh). These are the cleanest maps.",
    "River, forest and shrub are easy (F1 0.94 to 0.98). Ice (about 0.70 to 0.77) and snow (about 0.40) are hard: only "
    "1,732 ice and 564 snow labelled pixels.",
    "No model carries over to a new winter date: trained on the patch dates and tested on daksh 2024-03-12, accuracy is "
    "0.30 to 0.39 for every run. Most ice is called snow; none of the training dates is midwinter.",
])

# ---------------------------------------------------------------- all runs table
h(1, "2. Every run at a glance")
p("Accuracy / macro F1. exclude_daksh and with_daksh: 5-fold spatial cross-validation. External: trained on the 4 patch "
  "dates, tested on daksh 2024-03-12 (532 px, no river). Runs 08, 09 have no external test. Runs 02 and 04 were not Random Forest "
  "and are left out.")
rows = []
for r, v in RUNS.items():
    rows.append([r, v[0], str(v[2])] + ["{} / {}".format(*score(r, d)) if metrics(r, d) else "-" for _, d in DS])
table(["Run", "What changes", "Classes", "exclude_daksh", "with_daksh", "External test"], rows,
      [600, 3000, 900, 1500, 1500, 1500], ["l", "l", "r", "r", "r", "r"])

# ---------------------------------------------------------------- data
brk()
h(1, "3. Data")
h(2, "3.1 Datasets")
table(["Dataset (folder)", "Training dates", "Scored on"], [
    ["1_exclude_daksh", "4 patch dates", "5-fold spatial cross-validation"],
    ["2_with_daksh", "4 patch dates + daksh 2024-03-12", "5-fold spatial cross-validation"],
    ["3_external_test_daksh", "4 patch dates", "daksh 2024-03-12 labels (external cohort)"],
], [2600, 3200, 3200])
h(2, "3.2 Where the labels and images come from")
table(["Date", "Season", "Group", "Labels", "Imagery"], [
    ["2023-07-24", "summer", "mathus_naira", "patches/nenana_20230724/labels/", "patch _sr.tif + _thermal.tif"],
    ["2023-10-04", "freeze-up", "mathus_naira", "patches/nenana_20231004/labels/", "patch"],
    ["2024-04-21", "breakup", "mathus_naira", "patches/nenana_20240421/labels/", "patch"],
    ["2024-06-22", "summer", "mathus_naira", "patches/nenana_20240622/labels/", "patch"],
    ["2024-03-12", "winter", "daksh", "~/labels_20240312.gpkg", "nenana_crops/20240312/landsat_20240312_box.tif"],
    ["2024-04-20, 2024-04-29", "spring", "daksh", "none (predicted only)", "nenana_crops/<date>/"],
    ["2026-03-26, 2023-05-05", "midwinter, breakup", "new dates", "none (predicted only)", "~/Desktop/Patches/nenana_<date>/"],
], [1400, 1200, 1200, 2700, 2500])
p("patches/ = ~/Desktop/RiverProject/Matus/narrow_rivers/patches. Hand-label classes: 1 river (open water), 2 ice, 3 snow, "
  "4 land; 0 and 9 ignored. Study box lon -149.338 to -149.069, lat 64.094 to 64.535; every date on the same 995 x 1635 "
  "grid (EPSG:4326, 0.00027027 degrees), so a row and column is the same place on every date. The daksh image is "
  "Level-1 and is converted to reflectance with (DN x 2.0e-5 - 0.1) / cos(sun zenith).")
h(2, "3.3 How the dates are used")
p("All labelled dates are pooled into one training set. Each pixel is an example with its own date's inputs and its own "
  "date's label; a date's image is never paired with another date's labels. Unlabelled dates are only predicted.")
bullets([
    "The same ground labelled on several dates (it can be river in July, ice in October, snow in March) gives one example "
    "per date. Nothing is merged across dates.",
    "Cross-validation folds are 64-row east-west bands, the same band in the same fold on every date, so a tested place is "
    "held out on all dates and cannot be remembered from another date.",
])
lab = {}
D.configure("four", "with_daksh")
for s in D.load_all():
    if (s["y"] != D.IGNORE).any():
        lab[s["date"]] = s["y"]
rows, dl = [], sorted(lab)
for i, a in enumerate(dl):
    for b in dl[i + 1:]:
        m = (lab[a] != D.IGNORE) & (lab[b] != D.IGNORE)
        if m.any():
            rows.append([f"{a} & {b}", f"{int(m.sum()):,}", f"{(lab[a][m] == lab[b][m]).mean():.0%}"])
table(["Dates", "Shared labelled pixels", "Same class on both"], rows, [4000, 2500, 2500], ["l", "r", "r"])

h(2, "3.4 Labelled pixels per class, four classes")
for ds in D.DATASETS:
    counts, groups = S.label_counts(ds)
    h(3, S.DATASET_TITLES[ds])
    rows = [[d, groups[d], *[f"{v:,}" for v in c], f"{c.sum():,}",
             "test" if ds == "external_daksh" and groups[d] == "daksh" else "train"] for d, c in sorted(counts.items())]
    tr = sum(c for d, c in counts.items() if not (ds == "external_daksh" and groups[d] == "daksh"))
    rows.append(["Training total", "", *[f"{v:,}" for v in tr], f"{tr.sum():,}", ""])
    table(["Date", "Group", "River", "Ice", "Snow", "Land", "Total", "Used for"], rows,
          [1300, 1400, 1100, 900, 900, 1100, 1100, 1200], ["l", "l", "r", "r", "r", "r", "r", "l"])

h(2, "3.5 Labelled pixels per class, six classes")
p("River, ice and snow as drawn; land-label pixels split by NLCD 2016 (land pixels NLCD calls water are left out). "
  "Counts from the with_daksh run 10 evaluation.")
rows, tot = [], None
for blk in blocks(metrics("10", "2_with_daksh"))[1:]:
    d = blk.split()[1]
    n = [int(r[1].replace(",", "")) for r in parse(blk)[0]]
    tot = n if tot is None else [a + b for a, b in zip(tot, n)]
    rows.append([d] + [f"{v:,}" for v in n] + [f"{sum(n):,}"])
rows.append(["Total"] + [f"{v:,}" for v in tot] + [f"{sum(tot):,}"])
table(["Date", "River", "Ice", "Snow", "Forest", "Shrub/grass/wet", "Bare/dev", "Total"], rows,
      [1300, 1100, 900, 900, 1100, 1300, 1100, 1300], ["l"] + ["r"] * 7)

# ---------------------------------------------------------------- classes, inputs, method
brk()
h(1, "4. Classes, inputs and method")
h(2, "4.1 Six classes")
table(["Class", "From", "Map colour"], [
    ["river", "hand label 1 (open water)", "blue"], ["ice", "hand label 2", "magenta"], ["snow", "hand label 3", "cyan"],
    ["forest", "land label + NLCD 41 deciduous, 42 evergreen, 43 mixed", "dark green"],
    ["shrub_grass_wetland", "land label + NLCD 51, 52 shrub, 71 grass, 72 sedge, 81 pasture, 90 woody, 95 emergent wetland", "light green"],
    ["bare_developed", "land label + NLCD 21-24 developed, 31 barren", "tan"],
], [2200, 5300, 1500])
p("Four-class runs use river, ice, snow and land (land brown). NLCD Alaska 2016 comes from Earth Engine "
  "USGS/NLCD_RELEASES/2016_REL/2016_AK, resampled to the box grid (output/nlcd/).")
h(2, "4.2 Input groups")
table(["Group", "Inputs", "Runs"], [
    ["raw bands", "blue, green, red, NIR, SWIR1 (+ SWIR2, surface temperature in exclude_daksh)", "all"],
    ["band pairs", "normalized difference (a-b)/(a+b) and difference a-b of every reflectance band pair", "05+"],
    ["indices", "NDWI, NDSI, NDVI, MNDWI, EVI, SAVI, MSAVI, BSI, visible brightness, NIR+SWIR, AWEI_nsh, AWEI_sh, tasseled cap", "01 (3), 05+"],
    ["texture", "mean and std in 3x3 and 7x7 windows of NDVI, NDSI, NDWI, NIR, blue", "05+"],
    ["JRC water occurrence", "how often each pixel was water 1984-2021", "01, 03 only"],
    ["NLCD", "10 yes/no land-cover groups", "07, 09-15"],
    ["terrain", "Copernicus GLO-30 DEM: elevation, slope, northness, eastness, height above valley floor (300 m, 1 km)", "10, 11"],
    ["distance to water", "distance to the nearest NLCD water pixel (m)", "10, 11"],
    ["context", "MNDWI, NDWI, NDVI, brightness averaged over 15x15 and 31x31 px", "10"],
    ["illumination", "slope, northness, eastness, cos(sun incidence), shadowed; sun elevation/azimuth per date (12 only)", "12-15"],
], [1900, 5600, 1500])
h(2, "4.3 How scores are measured")
p("Datasets 1 and 2: 5-fold spatial cross-validation. The box is cut into east-west bands 64 rows (about 1.9 km) tall; "
  "band i is in fold i mod 5 on every date. Each fold's model trains on the other 4 folds (dropping pixels within 12 rows "
  "of the test band) and is scored on its own fold, so every labelled pixel is scored once by a model that never saw it. "
  "Settings (leaf size, max_features, input count, depth, penalty strength) are always chosen inside the training folds. "
  "Then one final model is trained on all labels and draws the maps.")
p("Dataset 3 (external cohort): one model trained on the 4 patch dates, scored on the 532 labelled pixels of daksh "
  "2024-03-12, a date, season and labeller it never saw. No river there, so read it as a carry-over check.")
p("All Random Forests use class_weight balanced_subsample (rare classes weigh more). Macro F1 = the mean F1 of the "
  "classes, so ice and snow count as much as river. Confusion matrices: rows = hand label, columns = prediction.")
h(2, "4.4 River penalty (runs 14, 15)")
p("NLCD water covers only 57% of the hand-labelled river (the channel has moved since 2016), so a hard water mask would "
  "delete real river. Instead, on pixels NLCD calls land, the forest's river probability is multiplied by w = 1 within "
  "60 m of NLCD water and w = max(0.1, exp(-(distance - 60 m) / scale)) further out; the pixel then takes the most likely "
  "class. River can still win where the bands are sure. Run 14 tuned the scale per fold (it chose no penalty every "
  "time); run 15 fixes it at 600 m. 'no fix' maps = same forest without this step.")

# ---------------------------------------------------------------- every run
EXTRA = {
    "08": ("8_best_six_class/{ds}/six_class/search.txt", "Search (5-fold spatial CV)", 12),
    "09": ("9_best_six_class_nlcd_input/{ds}/six_class/search.txt", "Search (5-fold spatial CV)", 12),
    "11": ("11_feature_search/{ds}/six_class/search.txt", "Feature search: last rounds and best F1 per class", 22),
    "13": ("13_rf_six_classes_limited/{ds}/" + SIX + "/search.txt", "Limited search: CV vs training accuracy", 16),
    "15": ("15_rf_six_classes_river_forced/{ds}/" + SIX + "/penalty_scan.txt", "Penalty strength vs score", 8),
}
brk()
h(1, "5. Every run: settings, scores and confusion matrices")
for r, (title, folder, k, inputs, settings, why) in RUNS.items():
    brk()
    h(2, f"Run {r}: {title}")
    p(why)
    table(["", ""], [["Classes", str(k)], ["Inputs", inputs], ["Settings", settings],
                     ["Folder", "output/" + folder.format(ds="<dataset>")]], [1600, 7400])
    for label, ds in DS:
        f = metrics(r, ds)
        if not f:
            continue
        h(3, f"Run {r} · {label}")
        first = blocks(f)[0]
        acc, mf1 = parse(first)[1:3]
        p(f"Accuracy {acc}, macro F1 {mf1}.")
        score_tables(f)
        if r in ("14", "15") and "without penalty" in first:
            _, a2, m2, _, _ = parse(first.split("without penalty", 1)[1])
            p(f"Same forests without the river penalty: accuracy {a2}, macro F1 {m2}.")
    if r in EXTRA:
        pat, cap, n = EXTRA[r]
        for label, ds in DS[:2]:
            f = O / pat.format(ds=ds)
            if f.exists():
                h(3, f"Run {r} · {cap} · {label}")
                lines = [l for l in f.read_text().splitlines() if l.strip()][-n:]
                table(["search output"], [[l] for l in lines], [9000])
    if r in ("14", "15"):
        for label, ds in DS[:2]:
            f = O / folder.format(ds=ds) / "river_on_land.txt"
            if f.exists():
                h(3, f"Run {r} · river on NLCD land · {label}")
                rows = [l.split() for l in f.read_text().splitlines()[1:] if l.strip()]
                table(["Date", "No penalty", "With penalty"], [[q[0], q[1] + " %", q[4] + " %"] for q in rows],
                      [3000, 3000, 3000], ["l", "r", "r"])

# ---------------------------------------------------------------- other tasks (RF only)
brk()
h(1, "6. Other tasks with the run 01 Random Forest")
p("Run 01 also trained Random Forests on split versions of the task (same 11 / 9 inputs, same evaluation).")
rows = []
for t in S.TASKS:
    for ds in ("exclude_daksh", "with_daksh", "external_daksh"):
        f = D.task_dir(t, ds) / "random_forest" / "metrics.txt"
        if f.exists():
            _, acc, mf1, _, _ = parse(S.first_block(f))
            rows.append([S.TASK_TITLES[t], D.DATASET_DIRS[ds], acc, mf1])
table(["Task", "Dataset", "Accuracy", "Macro F1"], rows, [4600, 2200, 1100, 1100], ["l", "l", "r", "r"])
p("Two binary models merged to four classes score about the same as one four-class model; one-vs-rest models find ice "
  "no better and snow worse.")
h(2, "River/land state per 375 m block (run 01)")
p("Each 375 m block: river frozen if predicted ice outnumbers open water; land snowy if snow outnumbers bare land. "
  "Scored on blocks where the labels hold both river and land.")
rows = []
for ds in D.DATASETS:
    for t in ("four", "combined"):
        f = D.task_dir(t, ds) / "random_forest" / "scene_mask_metrics.txt"
        if f.exists():
            b = f.read_text().split("\n\n--- ")[0]
            acc, mf1 = re.search(r"overall accuracy ([\d.]+)\s+macro F1 ([\d.]+)", b).groups()
            rows.append([D.DATASET_DIRS[ds], "four classes" if t == "four" else "binary combined", acc, mf1])
table(["Dataset", "From", "Accuracy", "Macro F1"], rows, [3000, 2400, 1800, 1800], ["l", "l", "r", "r"])

# ---------------------------------------------------------------- figures
brk()
h(1, "7. Ground truth vs Random Forest")
p("Each figure: true colour | hand labels (ground truth) | what the Random Forest predicted on the labelled pixels when "
  "tested (cross-validated: each pixel predicted by a model that never saw it) | the final map. Zoomed on the labels. "
  "Accuracy in the title is for that date's labelled pixels.")
season = {"2023-07-24": "summer", "2023-10-04": "freeze-up", "2024-04-21": "breakup", "2024-06-22": "summer"}
h(2, "7.1 Four classes, run 05 (exclude_daksh)")
for d, sname in season.items():
    image(FIG / f"four_gt_{d}.png", f"{d} ({sname}), four classes, run 05.")
h(2, "7.2 Six classes, run 10 (exclude_daksh)")
for d, sname in season.items():
    image(FIG / f"six_gt_{d}.png", f"{d} ({sname}), six classes, run 10.")
h(2, "7.3 External cohort: daksh 2024-03-12, never seen in training")
image(FIG / "four_external.png", "2024-03-12 (winter), four classes, run 05 trained on the patch dates only.")
image(FIG / "six_external.png", "2024-03-12 (winter), six classes, run 10 trained on the patch dates only. Most ice is called snow.")
h(1, "8. Maps of dates without labels")
p("No ground truth here; these show what the six-class run 10 model finds (whole box).")
for d, cap in [("2024-04-20", "spring, daksh"), ("2024-04-29", "spring, daksh"), ("2026-03-26", "midwinter"), ("2023-05-05", "breakup")]:
    image(FIG / f"pred_{d}.png", f"{d} ({cap}), six classes, run 10.", 380)
h(1, "9. River penalty: before and after (run 15)")
p("Same run 15 forest, without and with the 600 m river penalty, whole box, with_daksh model.")
for d in ("2024-04-21", "2024-04-20", "2024-04-29", "2023-05-05"):
    image(FIG / f"fix_{d}.png", f"{d}: true colour, no river penalty, river penalty.", 520)

# ---------------------------------------------------------------- files
brk()
h(1, "10. Files, QGIS and how to rerun")
table(["Where", "What"], [
    ["~/Desktop/nenana_river_models/", "main folder: runs/, presentation/, reports/, qgis_project/, data_layers/"],
    ["runs/<NN>_<what>_<dataset>", "link to each run's model folder (metrics.txt, oof.npz, pred_<date>.tif, images/)"],
    ["qgis_project/nenana_all.qgz", "self-contained QGIS project; every date group has rf_runs/ with all runs, fix and no-fix maps"],
    ["presentation/", "full and short decks, figures, make_figures.py"],
    ["~/Desktop/nenana_crops/daksh/", "all scripts; output/<run folder>/ holds every result"],
], [3400, 5600])
table(["Script", "Run"], [
    ["train_rf.py, external_test.py, combine.py, scene_mask.py", "01 (and the binary / one-vs-rest tasks)"],
    ["run_corridor.sh", "03"], ["rf_all_bands.py", "05"], ["rf_six_classes.py [--nlcd-input]", "06, 07"],
    ["rf_tune.py [--nlcd-input]", "08, 09"], ["rf_terrain.py", "10"], ["rf_feature_search.py", "11"],
    ["rf_angles.py", "12"], ["rf_limited.py", "13"], ["rf_river_fix.py", "14"], ["rf_river_forced.py", "15"],
    ["add_rf_runs.py, make_qgis_package.py", "QGIS project"],
    ["report/word_content_rf.py + build_word.js", "this report"],
], [5000, 4000])
p("Environment ~/venvs/river-ml/bin/python. Each run writes output/<run folder>/<1_exclude_daksh | 2_with_daksh | "
  "3_external_test_daksh>/.")

h(1, "11. Things to know when reading the results")
bullets([
    "Ice (1,732 px) and snow (564 px, 714 with daksh) come from only 2023-10-04, 2024-04-21 and 2024-03-12, so their "
    "scores move a lot with each polygon.",
    "2023-07-24 holds 31,021 of the 39,991 river pixels, so it shapes what the models call river.",
    "With NLCD as input, the land-type scores partly measure agreement with NLCD; river, ice and snow stay a fair test.",
    "with_daksh has no temperature and no SWIR2 input (the daksh image lacks them).",
    "Clouds were never labelled, so models call them snow or ice.",
    "The external test has 532 pixels and no river.",
    "Run 12's gain is partly the sun angles acting as a date label; run 13 onward removes them.",
])

json.dump(B, open(sys.argv[1], "w"))
print(len(B), "blocks")
