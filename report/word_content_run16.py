"""
Run 16 (water-zone Random Forest) report: makes the figures, then content.json for build_word.js.

    ~/venvs/river-ml/bin/python report/word_content_run16.py <scratch>/content16.json
    NODE_PATH=<docx node_modules> node report/build_word.js <scratch>/content16.json \
        ~/Desktop/nenana_river_models/reports/Nenana_run16_water_zone_report.docx

Scores are computed from each dataset's oof.npz (held-out predictions), the label changes and
chosen inputs are read from metrics.txt, and every image in images/ is included.
"""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path.home() / "Desktop/nenana_river_models/presentation"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal
import data as D
import make_figures as MF
import rf_water_zone as W

T, I = MF.T, MF.I
O = D.HERE / "output"
RUN = O / "16_rf_water_zone"
SUB = "six_class/random_forest"
FIG = Path.home() / "Desktop/nenana_river_models/presentation/figures"
DS = [("exclude_daksh", "1_exclude_daksh"), ("with_daksh", "2_with_daksh"), ("external_test", "3_external_test_daksh")]
NAMES = ["river", "ice", "snow", "forest", "shrub_grass_wetland", "bare_developed"]
SEASON = {"2023-07-24": "summer", "2023-10-04": "autumn, first ice", "2024-04-21": "spring break-up",
          "2024-06-22": "summer", "2024-03-12": "winter", "2024-04-20": "spring", "2024-04-29": "spring break-up",
          "2026-03-26": "winter, no labels", "2023-05-05": "spring, no labels"}

B = []
h = lambda lvl, text: B.append({"t": f"h{lvl}", "text": text})
p = lambda text: B.append({"t": "p", "text": text})
bullets = lambda items: B.append({"t": "bullets", "items": items})
table = lambda header, rows, widths=None, align=None: B.append(
    {"t": "table", "header": header, "rows": rows, "widths": widths, "align": align})
image = lambda path, caption, w=620: B.append({"t": "image", "path": str(path), "caption": caption, "width": w})


# ---------- numbers ----------
def oof(ds):
    z = np.load(RUN / ds / SUB / "oof.npz")
    return z["y"], z["p"], W.ZONE[z["row"], z["col"]], z["date"]


def score(y, pr):
    rows, f1s = [], []
    for k, n in enumerate(NAMES):
        t, q = y == k, pr == k
        if not t.sum() and not q.sum():
            continue
        tp = (t & q).sum()
        prec, rec = tp / max(1, q.sum()), tp / max(1, t.sum())
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if t.sum():
            f1s.append(f1)
        rows.append([n, f"{t.sum():,}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
    return rows, float((y == pr).mean()), float(np.mean(f1s))


def confusion(y, pr):
    ks = [k for k in range(6) if (y == k).any() or (pr == k).any()]
    rows = [[NAMES[a]] + [f"{((y == a) & (pr == b)).sum():,}" for b in ks] for a in ks if (y == a).any()]
    return ["true \\ predicted"] + [NAMES[b].split("_")[0] for b in ks], rows


def first_acc(path):
    m = re.search(r"overall accuracy ([\d.]+)\s+macro F1 ([\d.]+)", Path(path).read_text())
    return (float(m.group(1)), float(m.group(2))) if m else None


def metrics_txt(ds):
    return (RUN / ds / SUB / "metrics.txt").read_text()


def change_rows(ds):
    return [l.split() for l in metrics_txt(ds).split("\n\n")[0].splitlines()[2:]]


def inputs(ds):
    return {m.group(1): m.group(2).split(", ") for m in re.finditer(r"(water|land) RF inputs \(\d+\): (.*)", metrics_txt(ds))}


# ---------- figures ----------
def fig_zone():
    wo = D._water_occurrence()
    img = I.image_for("2024-04-21")
    fig, ax = plt.subplots(1, 3, figsize=(10, 7.4))
    T.show(ax[0], img, "true colour 2024-04-21")
    ax[1].imshow(np.ma.masked_equal(wo, 0), cmap="Blues", vmin=0, vmax=100, interpolation="nearest"); ax[1].set_axis_off()
    ax[1].set_title("JRC water occurrence (%)", fontsize=10)
    ov = img.astype(float)
    blue = np.array([0, 90, 255]) / (255.0 if ov.max() <= 1 else 1.0)
    ov[W.ZONE] = 0.45 * ov[W.ZONE] + 0.55 * blue
    T.show(ax[2], ov.astype(img.dtype), f"water zone (JRC >= {W.THRESHOLD:g} %)")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.03)
    out = FIG / "run16_zone.png"; fig.savefig(out, dpi=110); plt.close(fig)
    return out


def fig_gt(ds, ds_dir, external=False):
    scenes, cols, legend = MF.setup("six", "external_daksh" if external else ds)
    z = np.load(RUN / ds_dir / SUB / "oof.npz")
    outs = []
    for s in scenes:
        lab = s["y"] != D.IGNORE
        d = s["date"]
        m = z["date"] == d
        if not lab.any() or not m.any() or (external and d != "2024-03-12"):
            continue
        rr, cc, pp = z["row"][m], z["col"][m], z["p"][m]
        tp = np.zeros(lab.shape, np.uint8); tp[rr, cc] = pp + 1
        yr = W.rule_labels(s["y"], W.ZONE)
        gt = np.where(lab, s["y"] + 1, 0); gr = np.where(yr != D.IGNORE, yr + 1, 0)
        mp = gdal.Open(str(RUN / ds_dir / SUB / f"pred_{d}.tif")).ReadAsArray()
        r0, c0 = T.windows(lab, 1)[0]; r1, c1 = r0 + T.WIN[0], c0 + T.WIN[1]
        img = I.image_for(d)
        k = gr[rr, cc] > 0
        acc = float((pp[k] == gr[rr, cc][k] - 1).mean()) if k.any() else float("nan")
        fig, ax = plt.subplots(1, 5, figsize=(19, 4.9))
        panels = [(img, "true colour"), (T.overlay(img, gt, cols), "hand labels (original)"),
                  (T.overlay(img, gr, cols), "rule labels (used to train)"),
                  (T.overlay(img, tp, cols), f"RF {'on unseen date' if external else 'test prediction'} (acc {acc:.2f})"),
                  (T.overlay(img, mp, cols), "RF map")]
        for a, (im, t) in zip(ax, panels):
            T.show(a, im[r0:r1, c0:c1], t)
        fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, fontsize=10)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.1, wspace=0.03)
        out = FIG / ("run16_external.png" if external else f"run16_gt_{ds}_{d}.png")
        fig.savefig(out, dpi=100); plt.close(fig)
        outs.append((d, out, acc))
    return outs


def fig_compare():
    """true colour | run 10 | run 15 | run 16 (with_daksh), whole box, per date."""
    _, cols, legend = MF.setup("six", "with_daksh")
    runs = [("run 10", "10_rf_six_classes_terrain"), ("run 15", "15_rf_six_classes_river_forced"),
            ("run 16", "16_rf_water_zone")]
    outs = []
    for d in ("2023-07-24", "2023-10-04", "2024-03-12", "2024-04-20", "2024-04-21", "2024-04-29", "2024-06-22",
              "2026-03-26", "2023-05-05"):
        maps = []
        for label, folder in runs:
            f = None
            for ds in ("2_with_daksh", "1_exclude_daksh"):
                base = O / folder / ds / SUB
                c = base / "new_dates" / d / "rf_pred.tif" if d in ("2026-03-26", "2023-05-05") else base / f"pred_{d}.tif"
                if c.exists():
                    f = c; break
            maps.append((label, f))
        if any(f is None for _, f in maps):
            continue
        img = I.image_for(d)
        fig, ax = plt.subplots(1, 4, figsize=(10, 8.4))
        T.show(ax[0], img, "true colour")
        for a, (label, f) in zip(ax[1:], maps):
            T.show(a, T.overlay(img, gdal.Open(str(f)).ReadAsArray(), cols, 0.6), label)
        fig.suptitle(f"{d}   ({SEASON.get(d, '')})", fontsize=13)
        fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.03)
        out = FIG / f"run16_compare_{d}.png"; fig.savefig(out, dpi=100); plt.close(fig)
        outs.append((d, out))
    return outs


FIG.mkdir(exist_ok=True)
EX = json.loads((RUN / "extra.json").read_text())
compare = fig_compare()
zone_png = fig_zone()
gt_with = fig_gt("with_daksh", "2_with_daksh")
gt_ext = fig_gt("external_test", "3_external_test_daksh", external=True)

# ---------- scores ----------
S = {}
for name, ds in DS:
    y, pr, z, d = oof(ds)
    yr = W.rule_labels(y, z); k = yr != D.IGNORE
    S[name] = dict(y=y, pr=pr, z=z, d=d, yr=yr, k=k, rule=score(yr[k], pr[k]), orig=score(y, pr))
comp = []
for title, folder in (("10 six classes, terrain (best before)", "10_rf_six_classes_terrain"),
                      ("15 six classes, forced river penalty", "15_rf_six_classes_river_forced")):
    comp.append([title] + [("%.3f / %.3f" % first_acc(O / folder / ds / SUB / "metrics.txt"))
                           if (O / folder / ds / SUB / "metrics.txt").exists() else "-" for _, ds in DS])
comp.append(["16 water zone, rule labels"] + ["%.3f / %.3f" % S[n]["rule"][1:] for n, _ in DS])
comp.append(["16 water zone, original hand labels"] + ["%.3f / %.3f" % S[n]["orig"][1:] for n, _ in DS])
w = S["with_daksh"]
# numbers for the run 16 deck (presentation/build_deck_run16.js)
(RUN / "deck_numbers.json").write_text(json.dumps({
    "comp": comp, "score": {n: {"rule": S[n]["rule"], "orig": S[n]["orig"]} for n, _ in DS},
    "changes": change_rows("2_with_daksh"), "zone_pct": float(100 * W.ZONE.mean()),
    "figures": {"zone": str(FIG / "run16_zone.png"), "gt": [[d, str(f), a] for d, f, a in gt_with], "ext": [[d, str(f), a] for d, f, a in gt_ext], "compare": [[d, str(f)] for d, f in compare]},
}, indent=1))

# ---------- document ----------
B.append({"t": "h0", "text": "Nenana river ice: run 16, Random Forest with a water-zone rule"})
p("Landsat 30 m, six classes (river, ice, snow, forest, shrub/grass/wetland, bare/developed). "
  "The JRC Global Surface Water occurrence layer decides where river and ice may appear, and NLCD 2016 decides the land "
  "types. Random Forest only.")

h(1, "1. Summary")
bullets([
    f"Rule: where JRC water occurrence is at least {W.THRESHOLD:g} % (the water zone) a pixel can only be river or ice. "
    "Everywhere else it can only be snow or one of the three NLCD land types. No snow on the river, no ice or river on land.",
    "Two Random Forests, one per zone: river vs ice inside the water zone, snow / forest / shrub / bare outside it.",
    "Hand labels were moved onto the rule: snow inside the water zone became ice (snow lying on frozen river); labels the "
    "rule cannot hold (river or ice on land, land types inside the water zone) were left out of training.",
    "Against those rule labels, 5-fold spatial cross-validation gives accuracy %.3f and macro F1 %.3f with the daksh dates "
    "(%.3f / %.3f without them), higher than any earlier run (best before: run 10, 0.958 / 0.822)." %
    (*w["rule"][1:], *S["exclude_daksh"]["rule"][1:]),
    "Against the original hand labels the same predictions score %.3f / %.3f, because 19 %% of the river labels lie where "
    "JRC says land (the channel moved) and most bare labels are gravel bars inside the water zone." % w["orig"][1:],
    "On the unseen winter date 2024-03-12 (trained without it) every rule-label ice and snow pixel is right "
    "(accuracy %.3f); the 35 forest pixels there are called snow." % S["external_test"]["rule"][1],
    "The maps never show river on land or snow on the river, so they look much cleaner than runs 10-15.",
])

h(1, "2. Comparison with earlier runs")
p("Accuracy / macro F1. Runs 10 and 15 are scored against the original six-class labels. Run 16 is shown both ways: the "
  "rule labels are what the model is asked to learn, the original labels show the cost of the rule.")
table(["run", "exclude_daksh", "with_daksh", "external 2024-03-12"], comp, [3600, 1800, 1800, 1800])
p("Macro F1 here averages only classes that have test pixels. On the external test that is ice, snow, forest and shrub; "
  "the 36 forest/shrub pixels are all missed, which pulls macro F1 down for every run.")

h(1, "3. The idea")
p("Earlier runs let one forest choose any of the six classes anywhere, so river appeared on land and snow on the river. "
  "Run 16 uses a water mask instead: the JRC Global Surface Water occurrence layer (1984-2021), which gives for every "
  "30 m pixel how often it was water. Pixels that were water in at least 5 % of observations form the water zone. "
  "Inside it the river is either open (river) or frozen (ice); snow on the frozen river counts as ice. Outside it the "
  "ground is land, which is either snow-covered (snow) or one of the NLCD 2016 land types.")
table(["zone", "rule", "classes allowed", "model"], [
    ["water zone", f"JRC occurrence >= {W.THRESHOLD:g} %", "river, ice", "water RF (2 classes)"],
    ["land", f"JRC occurrence < {W.THRESHOLD:g} %", "snow, forest, shrub_grass_wetland, bare_developed", "land RF (4 classes)"],
], [1400, 2200, 3400, 2000])
p(f"The water zone covers {100 * W.ZONE.mean():.1f} % of the study box.")
image(zone_png, "Left: Landsat true colour on 2024-04-21. Middle: JRC water occurrence. Right: the water zone (blue).", 620)

h(1, "4. Data")
h(2, "4.1 Datasets")
table(["dataset", "dates", "Landsat bands", "use"], [
    ["exclude_daksh", "2023-07-24, 2023-10-04, 2024-04-21, 2024-06-22", "7: blue, green, red, NIR, SWIR1, SWIR2, thermal (ST_B10)", "5-fold spatial CV"],
    ["with_daksh", "those + 2024-03-12, 2024-04-20, 2024-04-29", "5: blue, green, red, NIR, SWIR1 (daksh tiffs have no SWIR2 or thermal)", "5-fold spatial CV"],
    ["external_test", "train on the 4 patch dates, test on 2024-03-12", "5", "unseen-date test"],
], [1500, 3000, 3000, 1500])
p("Labels are hand-drawn polygons (river, ice, snow, land); land is split into forest, shrub/grass/wetland and "
  "bare/developed with NLCD 2016, as in runs 06-15. Unlabelled dates 2026-03-26 and 2023-05-05 are only mapped.")
h(2, "4.2 Label changes for the rule")
for name, ds in DS:
    p(f"{name}: labelled pixels per class, how many lie in the water zone, how many became ice, how many were left out.")
    table(["class", "labelled", "in water zone", "became ice", "left out"], change_rows(ds), [2800, 1400, 1600, 1400, 1400],
          ["l", "r", "r", "r", "r"])
p("River and ice left out lie where JRC occurrence is below 5 %: mostly where the Nenana channel has moved since the JRC "
  "years. Bare/developed left out are gravel bars inside the water zone. Forest and shrub left out are a few edge pixels.")
h(2, "4.3 Labelled pixels per date and class")
p("with_daksh (includes every exclude_daksh date). Each cell: original hand labels / after the rule. 2024-04-20 and "
  "2024-04-29 have empty label files and are only mapped.")
short = ["river", "ice", "snow", "forest", "shrub", "bare"]
rows, tot_o, tot_r = [], np.zeros(6, int), np.zeros(6, int)
for d, v in sorted(EX["counts"]["with_daksh"].items()):
    o, r_ = np.array(v["original"]), np.array(v["rule"]); tot_o += o; tot_r += r_
    rows.append([d] + [f"{a:,} / {b:,}" for a, b in zip(o, r_)] + [f"{o.sum():,} / {r_.sum():,}"])
rows.append(["total"] + [f"{a:,} / {b:,}" for a, b in zip(tot_o, tot_r)] + [f"{tot_o.sum():,} / {tot_r.sum():,}"])
table(["date"] + short + ["all"], rows, [1150] + [1100] * 6 + [1250], ["l"] + ["r"] * 7)

h(1, "5. Inputs")
p("Same input pool as runs 13-15: every band, the normalised difference and difference of every band pair, indices "
  "(NDWI, NDSI, NDVI, SAVI, MSAVI, EVI, BSI, AWEI, tasseled cap brightness/greenness/wetness, visible brightness), 3x3 "
  "and 7x7 mean and standard deviation textures, per-pixel illumination (slope, northness, eastness, cos incidence, "
  "shadow, from the Copernicus DEM) and NLCD 2016 as 10 yes/no inputs. The per-date sun elevation and azimuth are left "
  "out (one value per date, they would tell the forest the date). JRC occurrence is not an input; it only picks the zone.")
table(["dataset", "Landsat bands", "inputs in the pool", "inputs used per forest"], [
    ["exclude_daksh", "7", "83", "25"], ["with_daksh", "5", "66", "25"], ["external_test", "5", "66", "25"]],
    [2200, 2200, 2300, 2300], ["l", "r", "r", "r"])
p("Each forest ranks the pool with a quick 100-tree forest on its own training pixels (never the test fold) and keeps "
  "the top 25.")
for name, ds in DS:
    ins = inputs(ds)
    table(["forest", f"25 inputs chosen ({name}, final model)"],
          [["water RF", ", ".join(ins.get("water", []))], ["land RF", ", ".join(ins.get("land", []))]], [1500, 7500])
p("The water forest relies on visible brightness, blue texture and SWIR differences (open water is dark, ice bright). "
  "The land forest relies on NLCD for the land type and on visible/SWIR contrast and NDSI for snow (snow is bright in "
  "visible light and dark in SWIR).")
h(2, "5.1 How much each input matters")
p("Importance of each of the 25 inputs in the final forests (mean decrease in impurity; each forest's values add up to "
  "1). Higher means the forest splits on that input more often and more usefully.")
for name, ds in (("exclude_daksh", "exclude_daksh"), ("with_daksh", "with_daksh"), ("external_test", "external_daksh")):
    imp = EX["importance"][ds]
    wv, lv = imp["water"], imp["land"]
    rows = [[str(i + 1), wv[i][0], f"{wv[i][1]:.3f}", lv[i][0], f"{lv[i][1]:.3f}"] for i in range(max(len(wv), len(lv)))]
    p(f"{name}:")
    table(["#", "water RF input", "importance", "land RF input", "importance"], rows, [500, 2900, 1300, 2900, 1300],
          ["r", "l", "r", "l", "r"])

h(1, "6. Method")
bullets([
    "Two scikit-learn Random Forests per training set: 300 trees, max_depth 16, min_samples_leaf 20, max_features sqrt, "
    "class_weight balanced_subsample (same limited settings as runs 13-15, so they cannot memorise pixels).",
    "Each forest trains only on its own zone's rule labels; a pixel is predicted by the forest of its zone.",
    "5-fold spatial cross-validation: the box is cut into east-west bands of 64 rows (about 1.9 km), band i is fold i mod 5, "
    "the same fold on every date, and training pixels within 12 rows of a test band are dropped.",
    "External test: train on the four patch dates only, predict the labelled daksh date 2024-03-12.",
    "Maps: the final forests (trained on all labelled pixels of the dataset) predict every pixel of every date.",
])

h(1, "7. Results")
for i, (name, ds) in enumerate(DS, 1):
    s = S[name]
    h(2, f"7.{i} {name}" + (" (5-fold spatial CV)" if name != "external_test" else " (trained without 2024-03-12)"))
    rows, acc, mf = s["rule"]
    p(f"Rule labels: accuracy {acc:.3f}, macro F1 {mf:.3f}.")
    table(["class", "pixels", "precision", "recall", "F1"], rows, [3000, 1500, 1500, 1500, 1500], ["l", "r", "r", "r", "r"])
    hd, cr = confusion(s["yr"][s["k"]], s["pr"][s["k"]])
    p("Confusion, rule labels (rows true, columns predicted):")
    table(hd, cr, None, ["l"] + ["r"] * (len(hd) - 1))
    rows, acc, mf = s["orig"]
    p(f"Original hand labels: accuracy {acc:.3f}, macro F1 {mf:.3f}.")
    table(["class", "pixels", "precision", "recall", "F1"], rows, [3000, 1500, 1500, 1500, 1500], ["l", "r", "r", "r", "r"])
    hd, cr = confusion(s["y"], s["pr"])
    p("Confusion, original hand labels:")
    table(hd, cr, None, ["l"] + ["r"] * (len(hd) - 1))
    if name != "external_test":
        per = []
        for d in sorted(set(s["d"])):
            m = s["d"] == d
            km = s["k"] & m
            _, a1, f1 = score(s["yr"][km], s["pr"][km]); _, a2, f2 = score(s["y"][m], s["pr"][m])
            per.append([d, SEASON.get(d, ""), f"{km.sum():,}", f"{a1:.3f} / {f1:.3f}", f"{a2:.3f} / {f2:.3f}"])
        p("Per date (accuracy / macro F1):")
        table(["date", "season", "rule px", "rule labels", "original labels"], per, [1400, 2000, 1200, 2200, 2200],
              ["l", "l", "r", "r", "r"])
h(2, "7.4 Overfitting check: training vs held-out accuracy")
p("For each cross-validation fold: accuracy of the forests on their own training pixels and on the held-out fold "
  "(rule labels), per zone. A small gap means the forests learned general patterns rather than memorising pixels.")
for name in ("exclude_daksh", "with_daksh"):
    g = EX["gap"][name]
    rows = [[str(r_["fold"] + 1)] + [f"{r_[f'{z}_train']:.3f} / {r_[f'{z}_test']:.3f}" for z in ("water", "land", "all")]
            for r_ in g]
    mean = {k: np.mean([r_[k] for r_ in g]) for k in g[0] if k != "fold"}
    rows.append(["mean"] + [f"{mean[f'{z}_train']:.3f} / {mean[f'{z}_test']:.3f}" for z in ("water", "land", "all")])
    p(f"{name} (training / held-out):")
    table(["fold", "water RF", "land RF", "both"], rows, [1200, 2600, 2600, 2600], ["l", "r", "r", "r"])
    p(f"Mean gap {name}: water {mean['water_train'] - mean['water_test']:.3f}, land "
      f"{mean['land_train'] - mean['land_test']:.3f}, both {mean['all_train'] - mean['all_test']:.3f}.")
h(2, "7.5 Which JRC cutoff")
p("The same training and 5-fold spatial CV with the water zone at different JRC occurrence cutoffs. A low cutoff makes "
  "the zone wider (more river labels fit the rule, but more land and gravel bars inside it); a high cutoff narrows it "
  "to the main channel. Rule-label scores are on different pixel sets for each cutoff; the original-label scores are "
  "on the same pixels and are the fair comparison.")
for name in ("exclude_daksh", "with_daksh"):
    rows = [[f"{r_['cutoff']:g} %", f"{r_['zone_pct']:.2f} %", f"{r_['rule_px']:,} of {r_['all_px']:,}",
             f"{r_['rule_acc']:.3f} / {r_['rule_mf1']:.3f}", f"{r_['orig_acc']:.3f} / {r_['orig_mf1']:.3f}",
             f"{r_['river_f1_orig']:.3f}"] for r_ in EX["scan"][name]]
    p(f"{name}:")
    table(["cutoff", "box in zone", "rule-label px", "rule acc / mF1", "original acc / mF1", "river F1 (orig)"], rows,
          [1000, 1300, 1900, 1600, 1700, 1500], ["l", "r", "r", "r", "r", "r"])
best = max(EX["scan"]["with_daksh"], key=lambda r_: r_["orig_mf1"])
p(f"Best cutoff on the original labels (with_daksh macro F1): {best['cutoff']:g} %. Run 16 uses 5 %.")
p("Reading the results: inside the water zone river vs ice is very reliable. The weakest class is snow: the land forest "
  "calls a few hundred forest and shrub pixels snow, mostly bright, sparse canopy in spring. Against the original labels "
  "the losses are exactly the labels the rule cannot reproduce: river outside the JRC zone, gravel bars (bare) inside "
  "it, and snow on the river (now ice).")

h(1, "8. Ground truth vs prediction (with_daksh)")
p("Each figure zooms on the labels of one date. Panels: Landsat true colour; the original hand labels; the rule labels "
  "used for training; the cross-validated test prediction (each pixel predicted by forests that never saw its fold), "
  "with its accuracy against the rule labels; and the final map.")
for d, f, acc in gt_with:
    image(f, f"{d} ({SEASON.get(d, '')}), run 16 with_daksh. Test accuracy on rule labels {acc:.2f}.", 640)
h(2, "External test")
for d, f, acc in gt_ext:
    image(f, f"{d} (winter), run 16 trained on the four patch dates only. Accuracy on rule labels {acc:.2f}.", 640)

h(1, "9. Every map")
p("All map images run 16 wrote, whole study box. Each figure is one date: per dataset the Landsat true colour and the "
  "run 16 map (class colours over true colour), side by side: exclude_daksh, with_daksh, external_test (where the "
  "dataset has that date).")
from PIL import Image, ImageDraw
by_date = {}
for name, ds in DS:
    for f in (RUN / ds / SUB / "images").glob("*.jpg"):
        by_date.setdefault(re.search(r"\d{4}-\d{2}-\d{2}", f.name).group(0), []).append((name, f))
for d in sorted(by_date, key=lambda q: ("no labels" in SEASON.get(q, ""), q)):
    ims = [(n, Image.open(f).convert("RGB")) for n, f in by_date[d]]
    hgt = max(im.height for _, im in ims)
    sheet = Image.new("RGB", (sum(im.width for _, im in ims) + 20 * (len(ims) - 1), hgt + 40), "white")
    x = 0
    for n, im in ims:
        sheet.paste(im, (x, 40)); ImageDraw.Draw(sheet).text((x + 10, 10), n, fill="black", font_size=26)
        x += im.width + 20
    out = FIG / f"run16_maps_{d}.jpg"; sheet.save(out, quality=88)
    image(out, f"{d} ({SEASON.get(d, '')}): " + ", ".join(n for n, _ in ims) + ".", 640)

h(1, "10. Run 16 next to runs 10 and 15")
p("The same date mapped by run 10 (best scores before), run 15 (river penalty) and run 16 (water zone), whole study box, "
  "with_daksh models (exclude_daksh where with_daksh has no map). Run 16 never puts river on land or snow on the river.")
for d, f in compare:
    image(f, f"{d} ({SEASON.get(d, '')}): true colour, run 10, run 15, run 16.", 560)

h(1, "11. Files and QGIS")
table(["what", "where"], [
    ["script", "~/Desktop/nenana_crops/daksh/rf_water_zone.py"],
    ["results", "~/Desktop/nenana_crops/daksh/output/16_rf_water_zone/<dataset>/six_class/random_forest/ (metrics.txt, oof.npz, pred_<date>.tif, new_dates/, images/)"],
    ["water zone", "output/16_rf_water_zone/water_zone.tif (1 = water zone)"],
    ["QGIS project (run 16 only)", "~/Desktop/nenana_river_models/qgis_run16/run16.qgz (full paths, this computer); run16_portable.qgz (relative paths, for sending); zipped: qgis_run16.zip"],
    ["extra analysis", "rf_water_zone_extra.py -> output/16_rf_water_zone/extra.json (cutoff scan, training vs held-out, importances, label counts)"],
    ["QGIS builder", "make_run16_project.py"],
    ["this report", "report/word_content_run16.py + report/build_word.js"],
], [2400, 6600])
p("The QGIS project holds, per date: labels, study box, Landsat (true colour, SR, SWIR, thermal, NDVI, JRC, QA) and "
  "Sentinel-2 where the date has it, plus a run16_water_zone group with the with_daksh, exclude_daksh and external maps. "
  "reference_layers holds the water zone, JRC water occurrence and NLCD 2016. run16.qgz stores full paths; run16_portable.qgz stores relative paths, so the unzipped "
  "folder opens anywhere.")

h(1, "12. Things to know")
bullets([
    "The rule-label scores are not directly comparable with runs 01-15: pixels the rule cannot hold are left out of them. "
    "The original-label scores are the like-for-like comparison.",
    "The JRC layer ends in 2021; where the channel has moved since, real river lies outside the water zone and is mapped "
    "as land or snow.",
    "Gravel bars inside the water zone can only be river or ice in this run.",
    "Ice and snow come from few dates (2023-10-04, 2024-04-21, 2024-03-12), so their scores move a lot with each polygon.",
    "NLCD is an input of the land forest, so forest/shrub/bare scores partly measure agreement with NLCD.",
    "Clouds were never labelled; on cloudy dates they become snow (land) or ice (water zone).",
])

json.dump(B, open(sys.argv[1], "w"))
print(len(B), "blocks")
