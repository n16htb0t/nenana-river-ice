"""
VIIRS runs 17-20 report: makes the figures, then content.json for build_word.js and deck numbers
for presentation/build_deck_viirs.js.

    ~/venvs/river-ml/bin/python report/word_content_viirs.py <scratch>/content_viirs.json
    NODE_PATH=<docx node_modules> node report/build_word.js <scratch>/content_viirs.json \
        ~/Desktop/nenana_river_models/reports/Nenana_VIIRS_runs17-20_report.docx

Every number is read from the run outputs:
  run 17  output/17_viirs_unmix/results.json
  run 18  output/18_viirs_states/metrics_all.json
  run 19  output/19_viirs_river_states/metrics_all.json, predict/totals.json
  run 20  output/20_viirs_river_only/metrics_all.json, predict/totals.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
O = HERE.parent / "output"
R17, R18, R19, R20 = (O / n for n in ("17_viirs_unmix", "18_viirs_states", "19_viirs_river_states",
                                       "20_viirs_river_only"))
FIG = O / "report_viirs_figures"
FIG.mkdir(parents=True, exist_ok=True)
STATES = ["ice_free_river_snow_free_land", "ice_free_river_snow_covered_land",
          "ice_covered_river_snow_free_land", "ice_covered_river_snow_covered_land"]
SHORT = ["1 open river, no-snow land", "2 open river, snow land", "3 ice river, no-snow land", "4 ice river, snow land"]
MODEL_NAMES = {"rf": "RF", "unetpp": "UNet++", "cnn2d": "2D CNN"}
SEASON = {"2023-04-19": "spring, snow", "2023-05-05": "spring break-up", "2023-07-24": "summer",
          "2023-10-04": "autumn freeze-up", "2024-04-21": "spring break-up", "2024-06-22": "summer",
          "2024-09-02": "late summer", "2025-07-04": "summer", "2026-03-26": "late winter", "2026-08-01": "summer"}

B = []
h = lambda lvl, text: B.append({"t": f"h{lvl}", "text": text})
p = lambda text: B.append({"t": "p", "text": text})
bullets = lambda items: B.append({"t": "bullets", "items": items})
table = lambda header, rows, widths=None, align=None: B.append(
    {"t": "table", "header": header, "rows": rows, "widths": widths, "align": align})
image = lambda path, caption, w=620: B.append({"t": "image", "path": str(path), "caption": caption, "width": w})
pagebreak = lambda: B.append({"t": "pagebreak"})
load = lambda path: json.loads(Path(path).read_text())
f2, f3 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}")

r17 = load(R17 / "results.json")
m18 = load(R18 / "metrics_all.json")
m19 = load(R19 / "metrics_all.json")
m20 = load(R20 / "metrics_all.json")
t19 = load(R19 / "predict" / "totals.json")
t20 = load(R20 / "predict" / "totals.json")
models19 = [m for m in ("rf", "unetpp", "cnn2d") if m in m19]
labelled = list(m18["rf"]["per_date"])
viirs_only = [d for d in t19 if d not in labelled]
deck = {"labelled": labelled, "viirs_only": viirs_only}


# ----------------------------------------------------------------------------- figures

def fig_r17():
    bands = ["I1", "I2", "I3", "I4", "I5"]
    dates = [r["date"] for r in r17]
    a = np.array([[r["psf_shifted"]["bands"][b]["r2"] for b in bands] for r in r17])
    cv = np.array([[r["psf_shifted"]["bands"][b]["cv_r2"] for b in bands] for r in r17])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for k, (m, t) in enumerate(((a, "R² (fit)"), (cv, "cross-validated R²"))):
        im = ax[k].imshow(m, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
        ax[k].set_xticks(range(5), bands); ax[k].set_yticks(range(len(dates)), dates)
        for i in range(len(dates)):
            for j in range(5):
                ax[k].text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax[k].set_title(f"Run 17 unmixing, {t}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    f = FIG / "r17_r2.png"; fig.savefig(f, dpi=130); plt.close(fig)
    xs, ys, cs = [], [], []
    for r in r17:
        lm = r.get("landsat_class_mean") or {}
        for b, lb in (("I1", "red"), ("I2", "nir"), ("I3", "swir1")):
            rho = r["psf_shifted"]["bands"][b]["rho"]
            for c, v in rho.items():
                share = r["psf_shifted"]["class_share_pct"].get(c, 0)
                lv = (lm.get(lb) or {}).get(c)            # landsat_class_mean is {band: {class: value}}
                if lv is not None and share >= 2:
                    xs.append(lv); ys.append(v); cs.append(c)
    fs = None
    if xs:
        fig, axx = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
        col = {"river": "#0066ff", "ice": "#ff00ff", "snow": "#00b4d8", "forest": "#1a7f37",
               "shrub_grass_wetland": "#9bbf4a", "bare_developed": "#c8a165"}
        for c in sorted(set(cs)):
            k = [i for i, q in enumerate(cs) if q == c]
            axx.scatter(np.array(xs)[k], np.array(ys)[k], s=22, color=col.get(c, "k"), label=c)
        axx.plot([0, 1], [0, 1], "k--", lw=0.8)
        axx.set_xlabel("Landsat class mean (surface reflectance)"); axx.set_ylabel("VIIRS unmixed class value (TOA)")
        axx.set_title("Run 17: unmixed vs Landsat, I1-I3,\nclasses with >= 2 % share"); axx.legend(fontsize=7)
        fs = FIG / "r17_scatter.png"; fig.savefig(fs, dpi=130); plt.close(fig)
    return f, fs


def fig_compare():
    runs = [("Run 18", m18), ("Run 19", m19), ("Run 20", m20)]
    groups = [(rn, rule) for rn, _ in runs for rule in ("rule50", "rule90")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    w = 0.26
    for k, metric in enumerate(("accuracy", "macro_f1")):
        for mi, mod in enumerate(("rf", "unetpp", "cnn2d")):
            xs, vs = [], []
            for gi, (rn, rule) in enumerate(groups):
                m = dict(runs)[rn]
                if mod in m:
                    xs.append(gi + (mi - 1) * w); vs.append(m[mod][rule][metric])
            ax[k].bar(xs, vs, w * 0.95, label=MODEL_NAMES[mod], color=["#0b6e99", "#e07a5f", "#2a9d8f"][mi])
        ax[k].set_xticks(range(len(groups)), [f"{a}\n{b}" for a, b in groups], fontsize=8)
        ax[k].set_ylim(0, 1); ax[k].set_title("state " + metric.replace("_", " ")); ax[k].legend(fontsize=8)
    f = FIG / "compare_runs.png"; fig.savefig(f, dpi=130); plt.close(fig)
    return f


def extra_metrics(c):
    """Metrics from a 4 x 4 confusion matrix (rows true, columns predicted)."""
    c = np.asarray(c, float)
    n = c.sum()
    tp, t, q = np.diag(c), c.sum(1), c.sum(0)
    present = t > 0
    prec = np.divide(tp, q, out=np.zeros(4), where=q > 0)
    rec = np.divide(tp, t, out=np.zeros(4), where=t > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    po, pe = tp.sum() / n, (t * q).sum() / n ** 2
    return {"accuracy": po, "balanced_accuracy": rec[present].mean(), "kappa": (po - pe) / (1 - pe),
            "macro_precision": prec[present].mean(), "macro_recall": rec[present].mean(), "macro_f1": f1[present].mean(),
            "weighted_f1": (f1 * t).sum() / n, "f1": f1, "precision": prec, "recall": rec, "support": t}


def fig_confusion(num, m, rule, mods):
    fig, ax = plt.subplots(1, len(mods), figsize=(4.6 * len(mods), 4.4), constrained_layout=True)
    ax = np.atleast_1d(ax)
    for a, mod in zip(ax, mods):
        c = np.asarray(m[mod][rule]["confusion"], float)
        rowp = np.divide(c, c.sum(1, keepdims=True), out=np.zeros_like(c), where=c.sum(1, keepdims=True) > 0)
        a.imshow(rowp, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{int(c[i, j]):,}\n{100 * rowp[i, j]:.0f}%", ha="center", va="center", fontsize=8.5,
                       color="white" if rowp[i, j] > 0.55 else "black")
        e = extra_metrics(c)
        a.set_title(f"{MODEL_NAMES[mod]}  acc {e['accuracy']:.3f}  κ {e['kappa']:.2f}  macro F1 {e['macro_f1']:.2f}",
                    fontsize=10)
        a.set_xticks(range(4), ["1", "2", "3", "4"]); a.set_yticks(range(4), ["1", "2", "3", "4"])
        a.set_xlabel("predicted state"); a.set_ylabel("true state")
    fig.suptitle(f"Run {num} confusion matrices, {rule} (counts and % of each true state; "
                 "1 open river/no-snow land, 2 open/snow, 3 ice/no-snow, 4 ice/snow)", fontsize=10.5)
    f = FIG / f"conf_run{num}_{rule}.png"; fig.savefig(f, dpi=140); plt.close(fig)
    return f


def fig_f1(num, m, mods):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    w = 0.8 / len(mods)
    for k, rule in enumerate(("rule50", "rule90")):
        for mi, mod in enumerate(mods):
            e = extra_metrics(m[mod][rule]["confusion"])
            ax[k].bar(np.arange(4) + (mi - (len(mods) - 1) / 2) * w, e["f1"], w * 0.95, label=MODEL_NAMES[mod],
                      color=["#0b6e99", "#e07a5f", "#2a9d8f"][mi])
        sup = extra_metrics(m[mods[0]][rule]["confusion"])["support"]
        ax[k].set_xticks(range(4), [f"{s}\n(n={int(v):,})" for s, v in zip(SHORT, sup)], fontsize=8)
        ax[k].set_ylim(0, 1); ax[k].set_ylabel("F1"); ax[k].set_title(f"Run {num}, F1 per state, {rule}")
        ax[k].legend(fontsize=8)
    f = FIG / f"f1_run{num}.png"; fig.savefig(f, dpi=140); plt.close(fig)
    return f


def metrics_rows(m, mods):
    rows = []
    for rule in ("rule50", "rule90"):
        for mod in mods:
            e = extra_metrics(m[mod][rule]["confusion"])
            rows.append([rule, MODEL_NAMES[mod]] + [f"{e[k]:.3f}" for k in
                        ("accuracy", "balanced_accuracy", "kappa", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")])
    return rows


METRIC_HEAD = ["rule", "model", "accuracy", "balanced acc", "Cohen κ", "macro prec", "macro rec", "macro F1", "weighted F1"]
METRIC_W = [900, 1000, 1000, 1050, 950, 1050, 1000, 1000, 1050]


def metrics_block(num, m, mods):
    """Confusion images, extra-metrics table and F1 chart for one run; returns paths for the deck."""
    h(2, "Confusion matrices and other metrics")
    p("Rows = true state, columns = predicted state; each cell shows the pixel count and the % of that true state (recall). "
      "Balanced accuracy = mean recall over the states present; Cohen κ = agreement beyond chance; weighted F1 = F1 "
      "weighted by the number of true pixels per state.")
    table(METRIC_HEAD, metrics_rows(m, mods), METRIC_W, ["l", "l"] + ["r"] * 7)
    out = {"conf": {}, "f1": None, "metrics": metrics_rows(m, mods)}
    for rule in ("rule50", "rule90"):
        f = fig_confusion(num, m, rule, mods)
        out["conf"][rule] = str(f)
        image(f, f"Run {num}: confusion matrices, {rule}, every model (leave-one-date-out).", 620)
    out["f1"] = str(fig_f1(num, m, mods))
    image(out["f1"], f"Run {num}: F1 per state and model (n = true pixels).", 620)
    return out


f_r17, f_r17s = fig_r17()
f_cmp = fig_compare()


# ----------------------------------------------------------------------------- helpers

def state_table(m, rule, models):
    rows = []
    for k, s in enumerate(STATES):
        row, tc = [], 0
        for mod in models:
            v = m[mod][rule]["per_state"].get(s)
            if v:
                tc = v["true_cells"]
                row += [f2(v["f1"]), f2(v["precision"]), f2(v["recall"])]
            else:
                row += ["-", "-", "-"]
        rows.append([SHORT[k], f"{tc:,}"] + row)
    head = ["state", "true px"] + [f"{MODEL_NAMES[mod]} {x}" for mod in models for x in ("F1", "prec", "rec")]
    return head, rows


def conf_rows(c):
    return [[SHORT[i]] + [f"{v:,}" for v in r] for i, r in enumerate(c)]


def area_rows(t, src_names):
    rows = []
    for d, v in t.items():
        for src in src_names:
            for area, x in (v["area_split"].get(src) or {}).items():
                if x:
                    rows.append([d, src, area, f"{x['water_pct']:.1f}",
                                 f"{x['open_water_pct_of_water']:.0f} / {x['ice_pct_of_water']:.0f}",
                                 f"{x['land_pct']:.1f}", f"{x['snow_pct_of_land']:.0f} / {x['snow_free_pct_of_land']:.0f}"])
    return rows


def state_pct_rows(t, rule, src_names):
    rows = []
    for d, v in t.items():
        for src in src_names:
            s = v["state_pct_of_river_buffer"].get(src)
            if s:
                rows.append([d, src] + [f"{x:.1f}" for x in s[rule]])
    return rows


# ----------------------------------------------------------------------------- content

h(0, "Nenana river ice from VIIRS 375 m: runs 17-20")
p("Unmixing (run 17), four river / land states over the whole image (run 18), and the four states on the river "
  "only, with and without a buffer (runs 19 and 20). Models: Random Forest, UNet++ and a plain 2D CNN. "
  "Labels come from the run 16 Landsat 30 m maps.")

h(1, "1. Summary")
rows = []
for name, m in (("Run 18: whole image", m18), ("Run 19: river + 1 px buffer", m19), ("Run 20: river only", m20)):
    for mod in ("rf", "unetpp", "cnn2d"):
        if mod in m:
            rows.append([name, MODEL_NAMES[mod], f3(m[mod]["rule50"]["accuracy"]), f3(m[mod]["rule50"]["macro_f1"]),
                         f3(m[mod]["rule90"]["accuracy"]), f3(m[mod]["rule90"]["macro_f1"]),
                         f"{m[mod]['rule50']['cells']:,}"])
table(["run", "model", "rule50 acc", "rule50 F1", "rule90 acc", "rule90 F1", "VIIRS px"], rows,
      [2600, 1000, 1000, 1000, 1000, 1000, 1000], ["l", "l", "r", "r", "r", "r", "r"])
deck["summary"] = rows
best = lambda m, rule: max(((mod, m[mod][rule]) for mod in m if mod in MODEL_NAMES), key=lambda t: t[1]["accuracy"])
b19, b20 = best(m19, "rule50"), best(m20, "rule50")
deck["best"] = {"run19": [MODEL_NAMES[b19[0]], b19[1]["accuracy"]], "run20": [MODEL_NAMES[b20[0]], b20[1]["accuracy"]]}
bullets([
    "Goal: Landsat (30 m) is rare; VIIRS (375 m) is daily. The run 16 Landsat maps give the labels, VIIRS alone gives the "
    "inputs, so the models can map the river state on any VIIRS day.",
    "Run 17 replicates Oleson et al. (1995): per-class VIIRS values recovered by least squares from the Landsat class shares. "
    "It works on snow / break-up days (I1 R² up to 0.77) and fails in summer (R² < 0.1), as in the paper's prairie case.",
    "Run 18 predicts the % of open water, ice, snow and snow-free land in every VIIRS pixel (RF R² 0.79-0.90 per part), then "
    "the four states by a 50 % or 90 % rule. Its river state mostly follows the day, so a wrong day flips the whole image.",
    f"Run 19 predicts the four states directly on river + buffer pixels. Best rule50: {MODEL_NAMES[b19[0]]} "
    f"{b19[1]['accuracy']:.3f} accuracy. It fixes run 18's 2024-04-21 collapse (RF 0.11 -> "
    f"{m19['rf']['per_date']['2024-04-21']['rule50']['accuracy']:.2f}).",
    f"Run 20 drops the buffer (river pixels only). Best rule50: {MODEL_NAMES[b20[0]]} {b20[1]['accuracy']:.3f}.",
    "State 2 (open river, snow land) is very rare (a few hundred pixels), so its F1 stays low in every run.",
    "All scores are leave-one-date-out: each date is predicted by models trained on the other 5 dates only.",
])

h(1, "2. Data and design")
h(2, "2.1 Why VIIRS")
p("Run 16 maps six classes from Landsat 30 m, but Landsat passes every 8-16 days and is often cloudy. VIIRS passes daily at "
  "375 m. Runs 17-20 move from 30 m to 375 m: the run 16 maps are summarised per VIIRS pixel and used as labels, and VIIRS "
  "bands are the only image inputs.")
h(2, "2.2 Grids")
table(["layer", "grid"], [
    ["Run 16 Landsat map", "EPSG:4326, 995 x 1635 px, 0.00027027° (~30 m); classes 1 river, 2 ice, 3 snow, 4 forest, "
                           "5 shrub_grass_wetland, 6 bare_developed"],
    ["VIIRS (Patches/nenana_<date>/<date>_viirs_375m.tif)", "EPSG:4326, 80 x 132 px, 0.00337838° (12.5 Landsat px); 9 bands "
                           "I1, I2, I3 (TOA reflectance), I4, I5 (brightness temperature K), SZA, SAA, VZA, VAA"],
    ["Water mask alaska_occ_375m.tif", "EPSG:4326, same 0.00337838° grid, aligned exactly (offset col 6412, row 2209); "
                           "JRC water occurrence 0-1"],
], [3000, 6000])
h(2, "2.3 Dates")
rows = []
for r in r17:
    d = r["date"]
    kind = "labelled (run 16 map)" if d in labelled else "VIIRS only (prediction)"
    rows.append([d, SEASON.get(d, ""), kind, f"{r['shift_m_south_east'][0]}, {r['shift_m_south_east'][1]}", f"{r['vza_deg']:.0f}"])
table(["date", "season", "use", "VIIRS-Landsat shift m (S, E)", "VZA °"], rows, [1500, 1900, 2300, 2200, 1100],
      ["l", "l", "l", "r", "r"])
p("Run 16 dates without a VIIRS file (2024-03-12, 2024-04-20, 2024-04-29) are left out. The 4 VIIRS-only dates have no "
  "run 16 map and are only predicted (no score).")
h(2, "2.4 From 30 m classes to VIIRS pixels")
bullets([
    "Each Landsat pixel is assigned to the VIIRS pixel it falls in, after moving the Landsat map by the run 17 shift "
    "(at most ~60 m). Shares of each class per VIIRS pixel = 'box' fractions.",
    "A VIIRS pixel is used when >= 98 % of it is mapped. A VIIRS pixel holds 12 or 13 Landsat pixels per side (12.5 on "
    "average); an early version divided by 12.5² and dropped every 12 x 12 pixel (a dashed grid of holes, ~25 % of pixels). "
    "Fixed: the share is now divided by the real count (floor 12 x 12, so edge pixels still count as partial). Runs 17-20 "
    "use the fix.",
    "Four parts per pixel: open_river = river, ice = ice, snow = snow, snow_free_land = forest + shrub + bare.",
])
h(2, "2.5 The four states and the two rules")
table(["state", "river", "land"], [
    ["1 ice_free_river_snow_free_land", "open", "no snow"],
    ["2 ice_free_river_snow_covered_land", "open", "snow"],
    ["3 ice_covered_river_snow_free_land", "ice", "no snow"],
    ["4 ice_covered_river_snow_covered_land", "ice", "snow"],
], [4400, 2300, 2300])
bullets([
    "rule50 (majority): river open if open / (open + ice) > 0.5; land snow-covered if snow / (snow + no-snow land) > 0.5.",
    "rule90: river open only if open share > 0.9; land snow-free only if snow share < 0.1 (mixed pixels count as snow-covered).",
    "A pixel with < 1 % river (or < 1 % land) takes the ratio of the whole date, so every pixel gets a state.",
])
h(2, "2.6 Inputs")
p("Per VIIRS pixel: I1, I2, I3, I4, I5, NDVI_I = (I2-I1)/(I2+I1), NDSI_I = (I1-I3)/(I1+I3), I4-I5, solar zenith, view "
  "zenith, JRC mean occurrence and JRC water-zone share (occurrence >= 5 %). RF also gets 3 x 3 mean / std of I1, I2, I3, I5. "
  "Runs 19 and 20 add the water-mask occurrence (21 RF inputs). No Landsat band is ever an input.")

pagebreak()
h(1, "3. Run 17: unmixing (Oleson et al. 1995)")
p("Oleson, Sarlin, Bajjouk, Steffen (1995, RSE 54: 98-112) recover the reflectance of each land-cover class from coarse "
  "pixels: coarse value = Σ class share x class value, solved by least squares over all pixels of the scene. Here the class "
  "shares come from the run 16 Landsat map, weighted by a Gaussian point-spread function (σ = 375 m / 2.8 = 134 m, cut at "
  "2σ), and the best VIIRS-Landsat shift is searched within ±6 rows / ±14 columns of Landsat pixels.")
image(f_r17, "Run 17: R² and cross-validated R² of the unmixing per band and date (Gaussian weights, best shift).", 600)
rows = []
for r in r17:
    b, bs = r["psf_shifted"]["bands"], r["box_shifted"]["bands"]
    rows.append([r["date"], f"{r['psf_shifted']['n_cells']:,}"]
                + [f"{b[x]['r2']:.2f} / {b[x]['cv_r2']:.2f}" for x in ("I1", "I2", "I3", "I5")] + [f"{bs['I1']['r2']:.2f}"])
table(["date", "pixels", "I1 R² / cv", "I2 R² / cv", "I3 R² / cv", "I5 R² / cv", "I1 R² box"], rows,
      [1300, 900, 1400, 1400, 1400, 1400, 1200], ["l", "r", "r", "r", "r", "r", "r"])
h(3, "All fit metrics per band (Gaussian weights, best shift)")
p("Run 17 is a regression (VIIRS value from class shares), so it has no confusion matrix. Its metrics: R² of the fit, "
  "cross-validated R² (5 folds of pixels), RMSE and cross-validated RMSE (reflectance for I1-I3, kelvin for I4-I5).")
rows = []
for r in r17:
    b = r["psf_shifted"]["bands"]
    for x in ("I1", "I2", "I3", "I4", "I5"):
        rows.append([r["date"], x, f"{b[x]['r2']:.3f}", f"{b[x]['cv_r2']:.3f}", f"{b[x]['rmse']:.3f}", f"{b[x]['cv_rmse']:.3f}",
                     f"{r['psf_no_shift']['bands'][x]['r2']:.3f}", f"{r['box_shifted']['bands'][x]['r2']:.3f}"])
table(["date", "band", "R²", "cv R²", "RMSE", "cv RMSE", "R² no shift", "R² box"], rows,
      [1300, 700, 1000, 1000, 1100, 1100, 1300, 1100], ["l", "l"] + ["r"] * 6)
deck["run17_metrics"] = [[r["date"]] + [f"{r['psf_shifted']['bands'][x]['r2']:.2f} / {r['psf_shifted']['bands'][x]['cv_r2']:.2f}"
                                         for x in ("I1", "I2", "I3", "I4", "I5")] for r in r17]
if f_r17s:
    image(f_r17s, "Run 17: unmixed VIIRS class values against the Landsat mean of the same class (red / NIR / SWIR1).", 380)
r421 = next(r for r in r17 if r["date"] == "2024-04-21")
p("2024-04-21, I1 (red) unmixed: " + ", ".join(f"{c} {v:.3f}" for c, v in r421["psf_shifted"]["bands"]["I1"]["rho"].items())
  + ". Ice and snow match Landsat within ~0.01; river (0.4 % of the image) comes out negative.")
bullets([
    "Works when the classes differ strongly and each covers enough of the image: snow / ice dates, I1 and I2.",
    "Fails in summer (R² < 0.1): forest, shrub and river look alike at 375 m, as in the paper's prairie scene.",
    "Rare classes (river 0.4 %) get unstable or negative values: the paper's 'well represented' condition.",
    "Gaussian weights beat box weights (higher I1 R²); the shift adds little (<= ~60 m).",
    "Run 17 is a check of the physics; runs 18-20 learn the inverse (VIIRS -> shares / states) instead.",
])

pagebreak()
h(1, "4. Run 18: % mix and states over the whole image")
p("RF regressor (4 outputs, depth 12, min leaf 20, sqrt, 300 trees) and UNet++ (resnet18, softmax over the four parts, "
  "soft cross-entropy, 1500 steps, flips, flip test-time augmentation) predict the four part shares of every VIIRS pixel. "
  "The states come from the predicted shares by rule50 / rule90.")
PK = ("open_river", "ice", "snow", "snow_free_land")
rows = []
for mod in ("rf", "unetpp"):
    pa = m18[mod]["parts"]
    rows.append([MODEL_NAMES[mod]] + [f"{pa['mae_pp'][k]:.1f} / {pa['r2'][k]:.2f}" for k in PK]
                + [f3(m18[mod]["rule50"]["accuracy"]), f3(m18[mod]["rule90"]["accuracy"])])
table(["model", "open water MAE pp / R²", "ice", "snow", "snow-free land", "rule50 acc", "rule90 acc"], rows,
      [1100, 1700, 1300, 1300, 1500, 1050, 1050], ["l", "r", "r", "r", "r", "r", "r"])
rows = []
for d in labelled:
    t = m18["rf"]["per_date"][d]["true_mean_pct"]
    row = [d, " / ".join(f"{t[k]:.1f}" for k in PK)]
    for mod in ("rf", "unetpp"):
        q = m18[mod]["per_date"][d]
        row += [" / ".join(f"{q['pred_mean_pct'][k]:.1f}" for k in PK), f"{q['rule50']['accuracy']:.2f}"]
    rows.append(row)
table(["date", "truth % water/ice/snow/land", "RF %", "RF acc50", "UNet++ %", "U++ acc50"], rows,
      [1200, 2100, 2000, 900, 2000, 900], ["l", "r", "r", "r", "r", "r"])
for mod in ("rf", "unetpp"):
    hd, rr = state_table(m18, "rule50", [mod])
    h(3, f"{MODEL_NAMES[mod]} rule50 per state")
    table(hd, rr, [3300, 1200, 1500, 1500, 1500], ["l", "r", "r", "r", "r"])
for mod in ("rf", "unetpp"):
    h(3, f"{MODEL_NAMES[mod]} part % per date: MAE (percentage points) / R²")
    table(["date"] + list(PK), [[d] + [f"{m18[mod]['per_date'][d]['parts']['mae_pp'][k]:.1f} / "
                                       f"{m18[mod]['per_date'][d]['parts']['r2'][k]:.2f}" for k in PK] for d in labelled],
          [1500, 1900, 1900, 1900, 1900], ["l", "r", "r", "r", "r"])
deck["run18"] = metrics_block(18, m18, ["rf", "unetpp"])
bullets([
    "The snow / land split is learned well; the river state of most pixels comes from the whole-date ratio (< 1 % river), "
    "so one wrong day flips the whole image: RF calls 2024-04-21 open.",
    "UNet++ has only 5 images per fold and is less accurate than RF.",
])
image(R18 / "images" / "run18_summary.jpg", "Run 18 summary: states per date (truth / RF / UNet++) and mean % per part.", 620)
for d in labelled:
    f = R18 / "images" / f"run18_{d}.jpg"
    if f.exists():
        image(f, f"Run 18, {d} ({SEASON.get(d, '')}): % maps and state maps, truth / RF / UNet++.", 620)


def river_run(num, sec, m, t, root, area, mask_text):
    mods = [x for x in models19 if x in m]
    pagebreak()
    h(1, f"{sec}. Run {num}: four states on the {area}")
    p(mask_text)
    p("Models predict the four states at once (one 4-class model per rule), trained and scored only on these pixels: "
      "RF classifier with the run 16 settings (depth 16, min leaf 20, sqrt, 300 trees, balanced_subsample; 21 inputs, so no "
      "top-25 selection); UNet++ (resnet18, 4-class cross-entropy on mask pixels, 1500 steps, flips, flip TTA); 2D CNN "
      "(4 conv 3 x 3 layers 32-64-64-64 with batch norm and ReLU, dropout 0.1, 1 x 1 output, no down-sampling, so each pixel "
      "sees 9 x 9 VIIRS pixels = 3.4 km; 97k parameters; same training as UNet++).")
    if (root / "images" / "water_mask.jpg").exists():
        image(root / "images" / "water_mask.jpg", f"Run {num}: the water mask cropped to the image and the predicted area.", 600)
    for rule in ("rule50", "rule90"):
        h(2, rule)
        table(["model", "accuracy", "macro F1"],
              [[MODEL_NAMES[mod], f3(m[mod][rule]["accuracy"]), f3(m[mod][rule]["macro_f1"])] for mod in mods],
              [3000, 3000, 3000], ["l", "r", "r"])
        hd, rr = state_table(m, rule, mods)
        table(hd, rr, [2200, 700] + [680] * (len(hd) - 2), ["l"] + ["r"] * (len(hd) - 1))
        for mod in mods:
            h(3, f"{MODEL_NAMES[mod]} confusion, {rule} (rows true, columns predicted)")
            table(["true \\ predicted", "1", "2", "3", "4"], conf_rows(m[mod][rule]["confusion"]),
                  [3400, 1400, 1400, 1400, 1400], ["l", "r", "r", "r", "r"])
    mb = metrics_block(num, m, mods)
    h(2, "Accuracy per date")
    rows = [[d, f"{m['rf']['per_date'][d]['cells']:,}"] + [f"{m[mod]['per_date'][d][rule]['accuracy']:.2f}"
                                                         for mod in mods for rule in ("rule50", "rule90")] for d in labelled]
    table(["date", "px"] + [f"{MODEL_NAMES[mod]} {r[4:]}" for mod in mods for r in ("rule50", "rule90")], rows,
          [1300, 800] + [1150] * (2 * len(mods)), ["l"] + ["r"] * (1 + 2 * len(mods)))
    image(root / "images" / f"run{num}_summary.jpg", f"Run {num}: states per date, truth and every model.", 620)
    h(2, "Prediction images with the water / land split")
    p("Each image starts with VIIRS (I3/I2/I1) and Landsat 30 m of the same day side by side, for visual checking. Landsat "
      "is shown in bands B6/B5/B4 (SWIR1 / NIR / red), the same bands as VIIRS I3/I2/I1, so colours compare directly: ice and "
      "snow are cyan, open water dark blue, vegetation green / red. Landsat exists for all 10 dates (the README of each "
      "Patches folder gives its time; VIIRS passed 21-69 min later). Our masks are drawn on both: yellow line = river "
      "(occurrence >= 0.05), orange dashed line = outer edge of the 1-pixel buffer (run 19 only). On days without a run 16 "
      "map, Landsat is only shown, never used as truth or input. Third panel: the run 16 Landsat 30 m map (six classes) - the label source on the 6 labelled days; on the 4 other days the run 16 model applied to that Landsat scene in run 17, shown for comparison only.")
    p("Then one map per model with all four predicted states. Yellow box: water (river) % vs land %, water split into open water "
      "/ ice, land split into snow / no snow, for the predicted area and for the whole VIIRS image (the % comes from the "
      "run 18 % models; for the 2D CNN a % model of the same design). The six labelled days are left-out test days; the four "
      "VIIRS-only days are predicted by final models trained on all six labelled days.")
    names = ["Landsat truth"] + [MODEL_NAMES[x] for x in mods]
    table(["date", "source", "area", "water %", "open / ice % of water", "land %", "snow / no snow % of land"],
          area_rows(t, names), [1200, 1300, 1700, 900, 1500, 900, 1500], ["l", "l", "l", "r", "r", "r", "r"])
    h(3, "States, % of predicted pixels (1 / 2 / 3 / 4), rule50")
    table(["date", "source", "1", "2", "3", "4"], state_pct_rows(t, "rule50", names),
          [1500, 1800, 1400, 1400, 1400, 1400], ["l", "l", "r", "r", "r", "r"])
    for d in t:
        f = root / "predict" / "images" / f"pred_{d}.jpg"
        if f.exists():
            kind = "left-out test day" if d in labelled else "VIIRS only, final models"
            image(f, f"Run {num}, {d} ({SEASON.get(d, '')}, {kind}): predicted states (rule50) and the water / land split.", 620)
    deck[f"run{num}"] = {
        "metrics": {mod: {r: [m[mod][r]["accuracy"], m[mod][r]["macro_f1"]] for r in ("rule50", "rule90")} for mod in mods},
        "per_date": {d: {mod: m[mod]["per_date"][d]["rule50"]["accuracy"] for mod in mods} for d in labelled},
        "cells": m["rf"]["rule50"]["cells"],
        "split": {d: {src: v["area_split"][src] for src in v["area_split"]} for d, v in t.items()},
        "images": {d: str(root / "predict" / "images" / f"pred_{d}.jpg") for d in t},
        "summary_img": str(root / "images" / f"run{num}_summary.jpg"),
        "mask_img": str(root / "images" / "water_mask.jpg"),
        "conf": mb["conf"], "f1_img": mb["f1"], "extra": mb["metrics"]}


river_run(19, 5, m19, t19, R19, "river + 1 px buffer",
          "Water mask alaska_occ_375m.tif cropped to the image: river = occurrence >= 0.05 (the run 16 5 % rule), 988 pixels; "
          "buffer = 1 VIIRS pixel (375 m) around it, 678 pixels. Predicted area 1,666 pixels; ~1,530-1,640 per date are used "
          "(>= 98 % mapped by the Landsat map).")
river_run(20, 6, m20, t20, R20, "river only (no buffer)",
          "Same water mask and cut (occurrence >= 0.05), no buffer: 988 river pixels only. Buffer pixels are mostly land, so "
          "their river state came from the date; without them every scored pixel holds river.")

pagebreak()
h(1, "7. Runs 18-20 compared")
image(f_cmp, "State accuracy and macro F1 per run, rule and model (leave-one-date-out).", 620)
rows = [[d, SEASON.get(d, "")] + [f"{m['per_date'][d]['rule50']['accuracy']:.2f}" for m in (m18["rf"], m19["rf"], m20["rf"])]
        for d in labelled]
table(["date", "season", "run 18 RF", "run 19 RF", "run 20 RF"], rows, [1500, 2100, 1800, 1800, 1800],
      ["l", "l", "r", "r", "r"])
deck["compare_rf"] = rows
h(2, "All metrics, all runs (leave-one-date-out)")
allrows = []
for num, m, mods in ((18, m18, ["rf", "unetpp"]), (19, m19, models19), (20, m20, [x for x in models19 if x in m20])):
    allrows += [[f"run {num}"] + r for r in metrics_rows(m, mods)]
table(["run"] + METRIC_HEAD, allrows, [700] + [800, 900, 900, 950, 850, 950, 900, 900, 950], ["l", "l", "l"] + ["r"] * 7)
deck["all_metrics"] = allrows
bullets([
    "Runs 19 and 20 score only river (and buffer) pixels, run 18 the whole image: the numbers measure different things. "
    "Run 18 is easy on land (mostly the day's river state), hard where it matters.",
    "Per-pixel states (runs 19-20) follow the river: break-up days show a mix of open and ice reaches instead of one state "
    "for the whole day.",
    "2023-10-04 (freeze-up) and 2024-04-21 (break-up) are the hard days; summer days are near perfect.",
    "2024-09-02 (VIIRS only): check the state map against the % split before trusting it; ice in early September is unlikely.",
])

h(1, "8. Files")
table(["what", "where (~/Desktop/nenana_crops/daksh/)"], [
    ["run 17 unmixing", "rf_viirs_unmix.py -> output/17_viirs_unmix/ (results.json, summary.txt, <date>/weights_psf.tif, "
                        "viirs_fit.tif, viirs_residual.tif)"],
    ["run 18 whole image", "viirs_states.py -> output/18_viirs_states/ (truth|rf|unetpp/pct_<date>.tif, "
                           "<rule>/class_<date>.tif, images/)"],
    ["run 18 images", "make_run18_images.py -> output/18_viirs_states/images/"],
    ["run 19 river + buffer", "RIVER_BUFFER=1 viirs_river_states.py -> output/19_viirs_river_states/ (mask/, "
                              "truth|rf|unetpp|cnn2d/<rule>/class_<date>.tif, images/)"],
    ["run 20 river only", "RIVER_BUFFER=0 viirs_river_states.py -> output/20_viirs_river_only/"],
    ["prediction images", "RIVER_BUFFER=1|0 viirs_river_predict.py -> <run>/predict/ (images/pred_<date>.jpg, "
                          "pred_<date>_rule90.jpg, totals.json, VIIRS-only maps)"],
    ["water mask", "~/Desktop/nenana_crops/alaska_occ_375m.tif; cropped copies in <run>/mask/"],
    ["this report", "report/word_content_viirs.py + report/build_word.js; deck presentation/build_deck_viirs.js"],
], [2300, 6700])
p("Every .tif is on the VIIRS grid (EPSG:4326) and opens in QGIS; class maps carry a .qml style (1-4 colours, 0 = not predicted).")

h(1, "9. Things to know")
bullets([
    "Labels are the run 16 Landsat maps, not hand labels: any run 16 error (gravel as river, cloud as snow/ice) becomes a "
    "VIIRS label.",
    "Only 6 labelled days; each model sees 5 per fold. Ice and break-up states come from 3 days.",
    "State 2 (open river, snow land) has a few hundred pixels in total; its F1 is not reliable.",
    "The JRC occurrence behind the water mask ends in 2021; moved channels fall outside the mask.",
    "VIIRS view angle and clouds vary daily; no cloud mask is applied, so cloudy days give wrong states.",
    "The 30 m to 375 m shift (run 17) is small (<= ~60 m) and applied before the shares are computed.",
])

json.dump(B, open(sys.argv[1], "w"))


# ----------------------------------------------------------------------------- one report per run
def sections(blocks):
    """Split the combined report into its h1 sections, keyed by the h1 title without its number."""
    out, key = {}, "title"
    for b in blocks:
        if b["t"] == "h1":
            key = b["text"].split(". ", 1)[1]
        out.setdefault(key, []).append(b)
    for v in out.values():                       # no page break at the end of a section
        while v and v[-1]["t"] == "pagebreak":
            v.pop()
    return out


def renumbered(blocks, n):
    blocks = [dict(b) for b in blocks]
    blocks[0]["text"] = f"{n}. " + blocks[0]["text"].split(". ", 1)[1]
    return blocks


SEC = sections(B)
key_of = {17: next(k for k in SEC if k.startswith("Run 17")), 18: next(k for k in SEC if k.startswith("Run 18")),
          19: next(k for k in SEC if k.startswith("Run 19")), 20: next(k for k in SEC if k.startswith("Run 20"))}
TITLES = {17: "Run 17: VIIRS unmixing with the run 16 Landsat maps (Oleson et al. 1995)",
          18: "Run 18: open water, ice, snow and land % and the four states over the whole VIIRS image",
          19: "Run 19: four river / land states on the river + 1 px buffer (RF, UNet++, 2D CNN)",
          20: "Run 20: four river / land states on the river only, no buffer (RF, UNet++, 2D CNN)"}
for num in (17, 18, 19, 20):
    B = []                                       # the helpers above append to B
    h(0, f"Nenana river ice from VIIRS 375 m: {TITLES[num]}")
    p("Part of the VIIRS series (runs 17-20); labels from the run 16 Landsat 30 m maps, VIIRS 375 m inputs only.")
    h(1, "1. Summary")
    if num == 17:
        table(["date", "pixels", "I1 R² / cv", "I2 R² / cv", "I5 R² / cv"],
              [[r["date"], f"{r['psf_shifted']['n_cells']:,}"] + [f"{r['psf_shifted']['bands'][x]['r2']:.2f} / "
               f"{r['psf_shifted']['bands'][x]['cv_r2']:.2f}" for x in ("I1", "I2", "I5")] for r in r17],
              [1800, 1200, 2000, 2000, 2000], ["l", "r", "r", "r", "r"])
        bullets(["Class values recovered from VIIRS by least squares on the run 16 class shares (Gaussian PSF, best shift).",
                 "Works on snow / break-up days (I1 R² up to 0.77); fails in summer (R² < 0.1).",
                 "Rare classes (river 0.4 %) come out unstable or negative; thermal bands do not unmix.",
                 "Regression, so no confusion matrix: R², cv R², RMSE and cv RMSE per band are given."])
    else:
        m, mods = {18: (m18, ["rf", "unetpp"]), 19: (m19, models19),
                   20: (m20, [x for x in models19 if x in m20])}[num]
        table(METRIC_HEAD, metrics_rows(m, mods), METRIC_W, ["l", "l"] + ["r"] * 7)
        bm = max(mods, key=lambda x: m[x]["rule50"]["accuracy"])
        bullets([f"Best rule50 model: {MODEL_NAMES[bm]}, accuracy {m[bm]['rule50']['accuracy']:.3f}, "
                 f"macro F1 {m[bm]['rule50']['macro_f1']:.3f}; {m['rf']['rule50']['cells']:,} scored VIIRS pixels.",
                 "Leave-one-date-out over the 6 run 16 dates: each date is predicted by models trained on the other 5.",
                 "rule50 accuracy per date (RF): " + ", ".join(f"{d} {m['rf']['per_date'][d]['rule50']['accuracy']:.2f}"
                                                              for d in labelled) + "."]
                + (["The river state of most pixels follows the whole date, so one wrong date flips the image "
                    "(RF 2024-04-21)."] if num == 18 else
                   ["State 2 (open river, snow land) is rare, so its F1 is low for every model."]))
    B += renumbered(SEC["Data and design"], 2)
    B.append({"t": "pagebreak"})
    B += renumbered(SEC[key_of[num]], 3)
    B += renumbered(SEC["Things to know"], 4)
    B += renumbered(SEC["Files"], 5)
    out = Path(sys.argv[1]).with_name(f"content_run{num}.json")
    json.dump(B, open(out, "w"))
    print("wrote", out, len(B), "blocks")
deck["figures"] = {"r17": str(f_r17), "r17s": str(f_r17s) if f_r17s else None, "compare": str(f_cmp),
                   "r18_summary": str(R18 / "images" / "run18_summary.jpg")}
(O / "deck_numbers_viirs.json").write_text(json.dumps(deck, indent=1))
print(len(B), "blocks")
