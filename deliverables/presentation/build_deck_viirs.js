// VIIRS runs 17-20 deck, pptxgenjs. Same theme and layouts as build_deck_run16.js.
// Numbers: ~/Desktop/nenana_crops/daksh/output/deck_numbers_viirs.json (report/word_content_viirs.py). Run:
//   NODE_PATH=<dir with pptxgenjs, image-size> PPTX_SKILL=<pptx skill dir> node build_deck_viirs.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("image-size");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HOME = process.env.HOME;
const O = `${HOME}/Desktop/nenana_crops/daksh/output`;
const OUT = `${HOME}/Desktop/nenana_river_models/presentation/Nenana_VIIRS_runs17-20.pptx`;
const N = JSON.parse(fs.readFileSync(`${O}/deck_numbers_viirs.json`, "utf8"));
const R17 = JSON.parse(fs.readFileSync(`${O}/17_viirs_unmix/results.json`, "utf8"));
const M18 = JSON.parse(fs.readFileSync(`${O}/18_viirs_states/metrics_all.json`, "utf8"));

const THEME = {
  name: "Arctic River",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14213D", lt1: "FFFFFF", dk2: "0B2E4F", lt2: "EAF4F8",
    accent1: "0B6E99", accent2: "00B4D8", accent3: "E07A5F", accent4: "2A9D8F",
    accent5: "8D99AE", accent6: "F4A261", hlink: "0B6E99", folHlink: "5E548E",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "RiverProject";
pres.title = "Nenana river ice from VIIRS: runs 17-20";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6, fontSize: 44, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.6, fontSize: 20,
      color: "BFE6F2", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION", background: { color: C.accent1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.3, fontSize: 40, bold: true,
      color: C.background1, fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.0, fontSize: 18,
      color: "DDF3FA" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 28, bold: true,
      color: C.text2, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8D99AE" },
});

const T = (s, txt, o) => s.addText(txt, Object.assign({ isTextBox: true, fontFace: THEME.bodyFontFace, color: C.text1, margin: 0 }, o));
function picture(s, file, box, name) {
  const { width, height } = imageSize(fs.readFileSync(file));
  let w = box.w, h = (box.w * height) / width;
  if (h > box.h) { h = box.h; w = (box.h * width) / height; }
  const x = box.x + (box.w - w) / 2, y = box.y + (box.h - h) / 2;
  s.addImage({ path: file, x, y, w, h, altText: name, objectName: name });
  return { x, y, w, h };
}
function card(s, x, y, w, h, head, body, color, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: name });
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  T(s, head, { x: x + 0.8, y: y + 0.2, w: w - 1.0, h: 0.5, fontSize: 17, bold: true, color: C.text2, valign: "middle" });
  T(s, body, { x: x + 0.3, y: y + 0.85, w: w - 0.6, h: h - 1.0, fontSize: 14, valign: "top" });
}
function stat(s, x, y, w, big, small, color, name) {
  T(s, big, { x, y, w, h: 1.1, fontSize: 50, bold: true, color, fontFace: THEME.headFontFace, objectName: name });
  T(s, small, { x, y: y + 1.1, w, h: 0.8, fontSize: 14, color: C.text1, valign: "top" });
}
function bullets(s, items, box, size = 16) {
  T(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    Object.assign({ fontSize: size, valign: "top", paraSpaceAfter: 8 }, box));
}
const head = { bold: true, color: "FFFFFF", fill: { color: HEX.accent1 }, fontSize: 12 };
function tableOf(header, rows, size = 12) {
  return [header.map((t) => ({ text: t, options: head }))].concat(
    rows.map((r, i) => r.map((t, j) => ({ text: String(t), options: { fontSize: size, align: j === 0 ? "left" : "right",
      fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 } } }))));
}
const chartFrame = (o) => Object.assign({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", catAxisLabelColor: "14213D", valAxisLabelColor: "8D99AE",
  valGridLine: { color: "E3E8EE", size: 0.75 }, catGridLine: { style: "none" }, showTitle: true, titleFontSize: 14,
  titleColor: "0B2E4F", showLegend: true, legendPos: "b", legendFontSize: 12, showValue: true, dataLabelFontSize: 9,
  dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1,
}, o);
const f3 = (v) => v.toFixed(3);
const NAMES = { rf: "RF", unetpp: "UNet++", cnn2d: "2D CNN" };
const COLORS = [HEX.accent1, HEX.accent3, HEX.accent4];
const content = (title, section) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const sectionSlide = (title, body) => {
  pres.addSection({ title });
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: title });
  s.addText(title, { placeholder: "title" });
  s.addText(body, { placeholder: "body" });
  return s;
};

// ---------- title + summary ----------
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Nenana river ice from daily VIIRS", { placeholder: "title" });
s.addText("Runs 17-20: unmixing, whole-image states, and four river / land states on the river with and without a buffer. "
  + "RF, UNet++ and a 2D CNN; labels from the run 16 Landsat maps.", { placeholder: "body" });

const r19 = N.run19, r20 = N.run20;
s = content("VIIRS can map the river state on days Landsat cannot", "Overview");
stat(s, 0.7, 1.45, 3.9, f3(r19.metrics.rf.rule50[0]), "run 19 RF accuracy, river + buffer, rule50 (unseen days)", C.accent1, "stat 19");
stat(s, 4.75, 1.45, 3.9, f3(r20.metrics.rf.rule50[0]), "run 20 RF accuracy, river only, rule50 (unseen days)", C.accent4, "stat 20");
stat(s, 8.8, 1.45, 3.9, f3(M18.rf.rule50.accuracy), "run 18 RF accuracy, whole image, rule50", C.accent3, "stat 18");
bullets(s, [
  "Landsat 30 m passes every 8-16 days; VIIRS 375 m passes daily. Run 16 maps become VIIRS labels; VIIRS alone is the input.",
  "Run 17 (Oleson 1995 unmixing): class values recovered on snow / break-up days (I1 R² up to 0.77), not in summer.",
  "Per-pixel river states (runs 19, 20) fix run 18's whole-day flips, e.g. RF on 2024-04-21: 0.11 -> "
    + r19.per_date["2024-04-21"].rf.toFixed(2) + ".",
  "Every score is leave-one-date-out: 6 labelled days, train on 5, test on the 6th.",
], { x: 0.7, y: 4.15, w: 11.9, h: 2.7 }, 16);

// ---------- design ----------
pres.addSection({ title: "Design" });
s = content("From 30 m classes to 375 m labels", "Design");
card(s, 0.6, 1.45, 3.95, 2.55, "Labels", "Run 16 Landsat map per date, summed per VIIRS pixel (12.5 x 12.5 Landsat px) after the "
  + "run 17 shift (<= ~60 m). Pixels >= 98 % mapped.", C.accent1, "card labels");
card(s, 4.7, 1.45, 3.95, 2.55, "Inputs (VIIRS only)", "I1-I5, NDVI_I, NDSI_I, I4-I5, sun and view zenith, JRC occurrence; "
  + "RF adds 3 x 3 mean / std. Runs 19-20 add the water mask.", C.accent4, "card inputs");
card(s, 8.8, 1.45, 3.95, 2.55, "Dates", `${N.labelled.length} labelled days (run 16 dates with VIIRS); `
  + `${N.viirs_only.length} VIIRS-only days predicted by final models.`, C.accent3, "card dates");
const states = [["1", "ice_free_river_snow_free_land", "open", "no snow"], ["2", "ice_free_river_snow_covered_land", "open", "snow"],
  ["3", "ice_covered_river_snow_free_land", "ice", "no snow"], ["4", "ice_covered_river_snow_covered_land", "ice", "snow"]];
s.addTable(tableOf(["state", "name", "river", "land"], states, 12), { x: 0.6, y: 4.25, w: 7.2, colW: [0.8, 3.9, 1.2, 1.3] });
bullets(s, ["rule50: open if open > 50 % of river; snow if snow > 50 % of land.",
  "rule90: open only if > 90 % open; snow-free only if < 10 % snow.",
  "< 1 % river or land in a pixel: the date's ratio decides."], { x: 8.1, y: 4.3, w: 4.7, h: 2.6 }, 14);

s = content("Dates and the VIIRS-Landsat shift", "Design");
s.addTable(tableOf(["date", "use", "shift m (S, E)", "VZA °"], R17.map((r) => [r.date,
  N.labelled.includes(r.date) ? "labelled" : "VIIRS only", `${r.shift_m_south_east[0]}, ${r.shift_m_south_east[1]}`,
  r.vza_deg.toFixed(0)]), 13), { x: 0.6, y: 1.45, w: 7.0, colW: [1.8, 2.0, 1.9, 1.3] });
bullets(s, ["Run 16 dates without VIIRS (2024-03-12, 2024-04-20, 2024-04-29) are left out.",
  "VIIRS-only days have no truth: maps and % only.",
  "Bug fixed on the way: a VIIRS pixel holds 12 or 13 Landsat pixels per side; dividing by 12.5² dropped every 12 x 12 pixel "
    + "(~25 %). Now divided by the real count."], { x: 8.0, y: 1.5, w: 4.8, h: 5.2 }, 15);

// ---------- run 17 ----------
sectionSlide("Run 17: unmixing", "Oleson et al. 1995 replicated with the run 16 maps");
s = content("Unmixing works on snow days, fails in summer", "Run 17: unmixing");
picture(s, N.figures.r17, { x: 0.5, y: 1.35, w: 8.3, h: 3.6 }, "run 17 R2");
bullets(s, ["VIIRS value = Σ class share x class value; least squares over the image.",
  "Gaussian PSF σ = 134 m (375 m / 2.8), best shift searched.",
  "2024-04-21 I1: ice 0.422 vs Landsat 0.434; snow 0.187 vs 0.177.",
  "Summer R² < 0.1: forest, shrub and river look alike at 375 m.",
  "River (0.4 %) comes out negative: too rare to unmix."], { x: 9.0, y: 1.45, w: 3.9, h: 5.4 }, 14);
if (N.figures.r17s) picture(s, N.figures.r17s, { x: 2.5, y: 5.0, w: 4.0, h: 2.2 }, "run 17 scatter");
s = content("Run 17: fit metrics per band (R² / cross-validated R²)", "Run 17: unmixing");
s.addTable(tableOf(["date", "I1", "I2", "I3", "I4", "I5"], N.run17_metrics, 12), { x: 0.6, y: 1.4, w: 8.6,
  colW: [1.6, 1.4, 1.4, 1.4, 1.4, 1.4] });
bullets(s, ["Regression, not classification: no confusion matrix.",
  "cv R² = 5 folds of pixels; full RMSE and cv RMSE per band are in the report.",
  "Thermal bands (I4, I5) do not unmix: surface temperature is not a fixed class value."], { x: 9.5, y: 1.5, w: 3.4, h: 5.3 }, 14);

// ---------- run 18 ----------
sectionSlide("Run 18: whole image", "% of open water, ice, snow and snow-free land per VIIRS pixel, then the states");
s = content("Good % mix, but the river state follows the day", "Run 18: whole image");
const PK = ["open_river", "ice", "snow", "snow_free_land"];
s.addTable(tableOf(["model", "water MAE / R²", "ice", "snow", "no-snow land", "acc50", "acc90"], ["rf", "unetpp"].map((m) =>
  [NAMES[m]].concat(PK.map((k) => `${M18[m].parts.mae_pp[k].toFixed(1)} / ${M18[m].parts.r2[k].toFixed(2)}`),
    [f3(M18[m].rule50.accuracy), f3(M18[m].rule90.accuracy)])), 12), { x: 0.6, y: 1.4, w: 12.1 });
picture(s, N.figures.r18_summary, { x: 0.6, y: 2.6, w: 8.0, h: 4.6 }, "run 18 summary");
bullets(s, ["Most pixels hold < 1 % river, so their river state is the date's ratio.",
  "One wrong date flips the whole image: RF calls 2024-04-21 open (0.11).",
  "UNet++ trails RF: 5 training images per fold."], { x: 8.9, y: 2.8, w: 4.0, h: 4.2 }, 14);


// ---------- confusion matrices + other metrics (one set per run) ----------
const MHEAD = ["rule", "model", "acc", "bal. acc", "κ", "macro P", "macro R", "macro F1", "wtd F1"];
function metricSlides(num, D, sec) {
  for (const rule of ["rule50", "rule90"]) {
    const s = content(`Run ${num}: confusion matrices, ${rule}`, sec);
    picture(s, D.conf[rule], { x: 0.3, y: 1.25, w: 12.7, h: 5.4 }, `run ${num} confusion ${rule}`);
    T(s, "Rows true state, columns predicted; cell = pixels and % of the true state. 1 open river / no-snow land, "
      + "2 open / snow, 3 ice / no-snow, 4 ice / snow.", { x: 0.4, y: 6.75, w: 12.4, h: 0.4, fontSize: 11, color: "555555" });
  }
  const s = content(`Run ${num}: all metrics and F1 per state`, sec);
  s.addTable(tableOf(MHEAD, D.metrics, 11), { x: 0.4, y: 1.3, w: 12.5, colW: [1.0, 1.2, 1.3, 1.4, 1.3, 1.6, 1.6, 1.5, 1.6] });
  picture(s, D.f1, { x: 0.4, y: 1.45 + 0.32 * (D.metrics.length + 1), w: 12.5, h: 5.75 - 0.32 * (D.metrics.length + 1) },
    `run ${num} f1`);
}

metricSlides(18, N.run18, "Run 18: whole image");

// ---------- runs 19 / 20 ----------
function riverSection(num, R, area, body) {
  const sec = `Run ${num}: ${area}`;
  sectionSlide(sec, body);
  const mods = Object.keys(R.metrics);
  let s = content(`Run ${num}: the predicted area`, sec);
  picture(s, R.mask_img, { x: 0.5, y: 1.35, w: 8.6, h: 5.8 }, `run ${num} mask`);
  bullets(s, [body, "Pixels outside the mask are not predicted.", `${R.cells.toLocaleString()} scored pixels over 6 days.`],
    { x: 9.3, y: 1.5, w: 3.6, h: 5.4 }, 14);

  s = content(`Run ${num}: four states at once, three models`, sec);
  s.addTable(tableOf(["model", "rule50 acc", "rule50 F1", "rule90 acc", "rule90 F1"], mods.map((m) =>
    [NAMES[m], f3(R.metrics[m].rule50[0]), f3(R.metrics[m].rule50[1]), f3(R.metrics[m].rule90[0]), f3(R.metrics[m].rule90[1])]), 13),
    { x: 0.6, y: 1.4, w: 6.0, colW: [1.4, 1.15, 1.15, 1.15, 1.15] });
  s.addChart(pres.charts.BAR, mods.map((m) => ({ name: NAMES[m], labels: N.labelled,
    values: N.labelled.map((d) => R.per_date[d][m]) })), chartFrame({ x: 6.8, y: 1.3, w: 6.0, h: 5.6, barDir: "col",
    barGrouping: "clustered", chartColors: COLORS.slice(0, mods.length), title: "rule50 accuracy per left-out day",
    catAxisLabelFontSize: 10 }));
  bullets(s, ["RF: depth 16, leaf 20, balanced classes (run 16 settings).",
    "UNet++: resnet18, masked cross-entropy.",
    "2D CNN: 4 conv 3 x 3 layers, 9 x 9 pixel view, 97k params.",
    "State 2 (open river, snow land) is rare: low F1 everywhere."], { x: 0.6, y: 3.6, w: 6.0, h: 3.3 }, 14);

  metricSlides(num, { conf: R.conf, f1: R.f1_img, metrics: R.extra }, sec);

  s = content(`Run ${num}: states per day, truth and every model`, sec);
  picture(s, R.summary_img, { x: 0.4, y: 1.3, w: 12.5, h: 5.9 }, `run ${num} summary`);

  for (const d of ["2024-04-21", "2023-10-04", "2023-04-19", "2025-07-04"]) {
    if (!R.images[d] || !fs.existsSync(R.images[d])) continue;
    const kind = N.labelled.includes(d) ? "left-out test day" : "VIIRS only";
    s = content(`Run ${num}, ${d}: predicted states (${kind})`, sec);
    picture(s, R.images[d], { x: 0.3, y: 1.2, w: 12.7, h: 6.1 }, `run ${num} ${d}`);
  }

  s = content(`Run ${num}: water and land split (${area})`, sec);
  const srcs = ["Landsat truth"].concat(mods.map((m) => NAMES[m]));
  const area0 = (sp) => sp && Object.values(sp)[0];
  const rows = Object.keys(R.split).map((d) => [d].concat(srcs.map((src) => {
    const x = area0(R.split[d][src]);
    return x ? `${x.water_pct.toFixed(0)} % (${x.open_water_pct_of_water.toFixed(0)}/${x.ice_pct_of_water.toFixed(0)}) | `
      + `${x.snow_pct_of_land.toFixed(0)}/${x.snow_free_pct_of_land.toFixed(0)}` : "-";
  })));
  s.addTable(tableOf(["date"].concat(srcs), rows, 11), { x: 0.4, y: 1.35, w: 12.5, colW: [1.5].concat(srcs.map(() => 11.0 / srcs.length)) });
  T(s, "Cell: water % of the area (open / ice % of the water) | snow / no-snow % of the land. Labelled days: left-out "
    + "predictions. VIIRS-only days: final models.", { x: 0.4, y: 6.7, w: 12.5, h: 0.5, fontSize: 12, color: "555555" });
}
riverSection(19, r19, "river + buffer", "Water mask occurrence >= 0.05 (988 px) plus a 1-pixel (375 m) buffer (678 px).");
riverSection(20, r20, "river only", "Water mask occurrence >= 0.05 only: 988 river pixels, no buffer.");

// ---------- comparison ----------
pres.addSection({ title: "Comparison" });
s = content("Runs 18-20 side by side", "Comparison");
picture(s, N.figures.compare, { x: 0.4, y: 1.3, w: 8.2, h: 3.4 }, "compare runs");
s.addTable(tableOf(["date", "season", "run 18 RF", "run 19 RF", "run 20 RF"], N.compare_rf, 12),
  { x: 0.6, y: 4.85, w: 8.0, colW: [1.6, 2.4, 1.33, 1.33, 1.34] });
bullets(s, ["Run 18 scores the whole image, runs 19-20 only the river area: different questions.",
  "Freeze-up (2023-10-04) and break-up (2024-04-21) are the hard days.",
  "Summer days are near perfect in every run."], { x: 8.9, y: 1.5, w: 4.0, h: 5.3 }, 14);

s = content("All metrics, all runs", "Comparison");
s.addTable(tableOf(["run"].concat(MHEAD), N.all_metrics, 10), { x: 0.4, y: 1.3, w: 12.5,
  colW: [0.9, 0.9, 1.1, 1.2, 1.3, 1.1, 1.3, 1.3, 1.3, 1.3] });
s = content("Things to know", "Comparison");
card(s, 0.6, 1.45, 3.9, 2.55, "Labels are model maps", "Run 16 errors (gravel as river, cloud as snow) become VIIRS labels.", C.accent3, "card labels2");
card(s, 4.72, 1.45, 3.9, 2.55, "Six days only", "Each model sees 5 days per fold; ice states come from 3 days.", C.accent6, "card days");
card(s, 8.84, 1.45, 3.9, 2.55, "State 2 is rare", "A few hundred pixels: its F1 is not reliable.", C.accent2, "card rare");
card(s, 0.6, 4.25, 6.0, 2.5, "No cloud mask", "Cloudy VIIRS days give wrong states; check the % split (e.g. 2024-09-02).", C.accent4, "card cloud");
card(s, 6.75, 4.25, 6.0, 2.5, "Mask from JRC (to 2021)", "Moved channels fall outside the water mask.", C.accent5, "card jrc");

s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Comparison" });
s.addText("Daily river state from VIIRS", { placeholder: "title" });
s.addText(`Best on unseen days, rule50: run 19 ${N.best.run19[0]} ${f3(N.best.run19[1])}, run 20 ${N.best.run20[0]} `
  + `${f3(N.best.run20[1])}. Next: more labelled days (ice and break-up), a cloud mask, and daily VIIRS downloads.`,
  { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
