// Nenana river ice classification: concise deck (11 slides) with evaluations and images.
// Run: NODE_PATH=<dir with pptxgenjs, image-size> PPTX_SKILL=<pptx skill dir> node build_deck_short.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("image-size");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HOME = process.env.HOME;
const R = `${HOME}/Desktop/nenana_crops/daksh`;
const FIG = `${R}/report/figures`;
const OWN = `${HOME}/Desktop/nenana_river_models/presentation/figures`;
const RUN10 = `${R}/output/10_rf_six_classes_terrain/1_exclude_daksh/six_class/random_forest/images`;
const OUT = `${HOME}/Desktop/nenana_river_models/presentation/Nenana_river_ice_RF_short.pptx`;

const THEME = {
  name: "Arctic River", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "14213D", lt1: "FFFFFF", dk2: "0B2E4F", lt2: "EAF4F8", accent1: "0B6E99", accent2: "00B4D8",
    accent3: "E07A5F", accent4: "2A9D8F", accent5: "8D99AE", accent6: "F4A261", hlink: "0B6E99", folHlink: "5E548E" },
};
const HEX = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "RiverProject";
pres.title = "Nenana river ice classification (short)";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6, fontSize: 44, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.6, fontSize: 20,
      color: "BFE6F2", valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 30, bold: true,
    color: C.text2, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } }],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8D99AE" },
});

const T = (s, txt, o) => s.addText(txt, Object.assign({ isTextBox: true, fontFace: THEME.bodyFontFace, color: C.text1, margin: 0 }, o));
function picture(s, file, box, name) {
  const { width, height } = imageSize(fs.readFileSync(file));
  let w = box.w, h = (box.w * height) / width;
  if (h > box.h) { h = box.h; w = (box.h * width) / height; }
  s.addImage({ path: file, x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h, altText: name, objectName: name });
}
function caption(s, txt, y) { T(s, txt, { x: 0.6, y, w: 12.1, h: 0.45, fontSize: 13, color: "4A5568", italic: true }); }
function stat(s, x, y, w, big, small, color) {
  T(s, big, { x, y, w, h: 0.85, fontSize: 44, bold: true, color, fontFace: THEME.headFontFace });
  T(s, small, { x, y: y + 0.85, w, h: 0.6, fontSize: 13, color: "4A5568" });
}
function mix(f) {
  const a = [255, 255, 255], b = [11, 110, 153];
  return a.map((v, i) => Math.round(v + (b[i] - v) * f).toString(16).padStart(2, "0")).join("").toUpperCase();
}
function confusionSlide(title, sub, names, cm, prf, note) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
  s.addText(title, { placeholder: "title" });
  T(s, sub, { x: 0.6, y: 1.1, w: 12.1, h: 0.4, fontSize: 13, color: "4A5568" });
  const k = names.length, cell = { fontSize: k > 4 ? 12 : 14, fontFace: THEME.bodyFontFace, align: "center", valign: "middle" };
  const hd = (t) => ({ text: t, options: Object.assign({}, cell, { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 } }) });
  const rows = [[hd("true \\ predicted")].concat(names.map(hd))].concat(cm.map((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0);
    return [{ text: names[i], options: Object.assign({}, cell, { bold: true, align: "left", fill: { color: HEX.lt2 } }) }]
      .concat(row.map((v) => { const f = v / tot; return { text: `${v.toLocaleString("en-US")}\n${(100 * f).toFixed(0)}%`,
        options: Object.assign({}, cell, { color: f > 0.5 ? "FFFFFF" : "14213D", bold: f > 0.5, fill: { color: mix(Math.sqrt(f)) } }) }; }));
  }));
  const first = k > 4 ? 1.9 : 1.6;
  s.addTable(rows, { x: 0.6, y: 1.65, w: 7.9, colW: [first].concat(Array(k).fill((7.9 - first) / k)),
    rowH: Math.min(0.75, 4.9 / (k + 1)), border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "confusion" });
  const ph = [["class", "prec.", "recall", "F1"]].concat(prf.map((r, i) => [names[i]].concat(r.map((v) => v.toFixed(3)))));
  s.addTable(ph.map((r, i) => r.map((c, j) => ({ text: c, options: { fontSize: 13, fontFace: THEME.bodyFontFace, bold: i === 0 || j === 0,
    color: i === 0 ? "FFFFFF" : "14213D", align: j === 0 ? "left" : "right", fill: { color: i === 0 ? HEX.accent1 : (i % 2 ? "FFFFFF" : HEX.lt2) } } }))),
    { x: 8.85, y: 1.65, w: 3.85, colW: [1.6, 0.75, 0.75, 0.75], rowH: 0.36, border: { type: "solid", color: "D5E3EA", pt: 0.5 }, objectName: "prf" });
  T(s, note, { x: 8.85, y: 1.75 + 0.36 * (k + 1) + 0.3, w: 3.85, h: 1.8, fontSize: 14, color: C.accent3, valign: "top" });
  s.addNotes("Rows: true class (hand label). Columns: predicted class. Cell: pixel count and share of that true class. Spatial cross-validation: every pixel predicted by a model that never saw it.");
}

// 1. title
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Mapping river ice on the Nenana River", { placeholder: "title" });
s.addText("Random Forest on Landsat 30 m · best: 0.962 accuracy, 0.826 macro F1 (6 classes)", { placeholder: "body" });

// 2. data + classes
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Overview" });
s.addText("Ground truth: 85,358 hand-labelled pixels on 4 dates", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "pixels", labels: ["river", "ice", "snow", "land"], values: [39991, 1732, 564, 43071] }],
  { x: 0.6, y: 1.3, w: 6.4, h: 5.4, barDir: "col", chartColors: ["0066FF", "C026D3", "00B4D8", "8B4513"], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", showLegend: false, showTitle: true, title: "Labelled pixels per class",
    titleFontSize: 14, titleColor: "14213D", titleFontFace: "+mn-lt", valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
    valAxisLabelFormatCode: "#,##0", catAxisLabelColor: "4A5568", valAxisLabelColor: "4A5568", catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 13, dataLabelFontSize: 12, objectName: "chart_pixels" });
T(s, [{ text: "Dates", options: { bold: true, color: HEX.accent1, breakLine: true } },
  { text: "2023-07-24 summer · 2023-10-04 freeze-up · 2024-04-21 breakup · 2024-06-22 summer · 2024-03-12 winter (external test)", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Classes", options: { bold: true, color: HEX.accent1, breakLine: true } },
  { text: "4 classes: river, ice, snow, land (as drawn).", options: { breakLine: true } },
  { text: "6 classes: land split by NLCD Alaska 2016 into forest, shrub/grass/wetland, bare/developed.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Ice and snow are 2.7 % of labels: they limit macro F1.", options: { bold: true, color: HEX.accent3 } }],
  { x: 7.4, y: 1.5, w: 5.3, h: 5.2, fontSize: 15, valign: "top" });

// 3. technique
pres.addSection({ title: "Technique" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Technique" });
s.addText("Technique: many inputs per pixel, honest scoring", { placeholder: "title" });
const inputs = [["Bands", "Blue to SWIR2 + surface temperature", HEX.accent1], ["Band pairs", "All 15 as (a-b)/(a+b) and a-b: NDVI, NDWI, NDSI…", HEX.accent2],
  ["Indices", "EVI, SAVI, MSAVI, BSI, AWEI, tasseled cap", HEX.accent4], ["Texture", "3x3 / 7x7 mean and spread", HEX.accent6],
  ["Land cover", "NLCD Alaska 2016, 10 groups", HEX.accent3], ["Terrain + context", "DEM, height above valley floor, 15/31 px windows", "5E548E"]];
inputs.forEach(([h, b, col], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.35 + Math.floor(i / 3) * 1.7;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.85, h: 1.5, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `in_${i}` });
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.27, w: 0.38, h: 0.38, fill: { color: col }, line: { color: col } });
  T(s, h, { x: x + 0.8, y: y + 0.2, w: 2.9, h: 0.5, fontSize: 17, bold: true, color: C.text2, valign: "middle" });
  T(s, b, { x: x + 0.3, y: y + 0.8, w: 3.3, h: 0.6, fontSize: 14 });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.95, w: 12.1, h: 1.75, rectRadius: 0.12, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: "scoring" });
T(s, [{ text: "Scoring  ", options: { bold: true, color: "BFE6F2" } },
  { text: "5-fold spatial cross-validation: 64-row east-west bands, 12-row buffer, same fold on every date, so no pixel is scored by a model that saw it or its neighbours.", options: { color: "FFFFFF", breakLine: true } },
  { text: "External test  ", options: { bold: true, color: "BFE6F2" } },
  { text: "Train on the 4 patch dates, test on the unseen winter date 2024-03-12.", options: { color: "FFFFFF" } }],
  { x: 0.9, y: 5.1, w: 11.5, h: 1.45, fontSize: 15, valign: "middle" });
s.addNotes("Model: Random Forest (500 trees, class-balanced, settings searched by spatial CV). No water mask in the best models.");

// 4. all runs
pres.addSection({ title: "Evaluation" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Every run: accuracy / macro F1", { placeholder: "title" });
const runs = [
  ["01", "4", "Random Forest, 11 inputs", "0.952 / 0.750", "0.944 / 0.731", "0.338 / 0.180"],
  ["03", "4", "Random Forest, river + 150 m only", "0.940 / 0.739", "0.927 / 0.725", "0.329 / 0.173"],
  ["05", "4", "Random Forest, all bands (68)", "0.961 / 0.755", "0.957 / 0.745", "0.299 / 0.191"],
  ["06", "6", "Random Forest, all bands (68)", "0.822 / 0.663", "0.817 / 0.653", "0.301 / 0.144"],
  ["07", "6", "Random Forest, + NLCD (78)", "0.962 / 0.819", "0.958 / 0.813", "0.305 / 0.147"],
  ["08", "6", "Random Forest, all bands, tuned", "0.823 / 0.667", "0.815 / 0.659", "-"],
  ["09", "6", "Random Forest, + NLCD, tuned", "0.961 / 0.825", "0.957 / 0.819", "-"],
  ["10", "6", "Random Forest, + NLCD + terrain + context (93)", "0.962 / 0.826", "0.958 / 0.822", "0.312 / 0.144"]];
const best = new Set([2, 7]);
s.addTable([["Run", "Classes", "Model and inputs", "Without daksh", "With daksh", "External test"]].concat(runs).map((r, i) =>
  r.map((c, j) => ({ text: c, options: { fontSize: 13, fontFace: THEME.bodyFontFace, bold: i === 0 || best.has(i - 1),
    color: i === 0 ? "FFFFFF" : "14213D", align: j >= 3 || j < 2 ? "center" : "left",
    fill: { color: i === 0 ? HEX.dk2 : best.has(i - 1) ? "D8F0F7" : (i % 2 ? "FFFFFF" : "F5F9FB") } } }))),
  { x: 0.6, y: 1.3, w: 12.1, colW: [0.7, 1.0, 4.4, 2.0, 2.0, 2.0], rowH: 0.5, border: { type: "solid", color: "D5E3EA", pt: 0.5 }, objectName: "runs" });
caption(s, "Highlighted: best four-class (05) and six-class (10). Spatial CV except the external test. NLCD also made the land-type labels, so 07-10 land scores show agreement with NLCD.", 6.4);

// 5-6. confusion matrices
confusionSlide("Best four-class Random Forest", "Run 05, 68 inputs, spatial CV, 4 patch dates · accuracy 0.961 · macro F1 0.755",
  ["river", "ice", "snow", "land"], [[38560, 166, 1, 1264], [250, 1316, 125, 41], [18, 347, 160, 39], [930, 104, 36, 42001]],
  [[0.970, 0.964, 0.967], [0.681, 0.760, 0.718], [0.497, 0.284, 0.361], [0.969, 0.975, 0.972]],
  "Snow is mostly called ice (347 of 564). River and land swap about 2,200 pixels at the banks.");
confusionSlide("Best six-class Random Forest", "Run 10, 93 inputs, spatial CV, 4 patch dates · accuracy 0.962 · macro F1 0.826",
  ["river", "ice", "snow", "forest", "shrub/grass/wet.", "bare/dev."],
  [[38749, 226, 3, 314, 440, 259], [163, 1374, 178, 8, 8, 1], [17, 325, 212, 2, 5, 3],
   [255, 38, 45, 21749, 80, 6], [188, 44, 32, 252, 15110, 0], [205, 11, 21, 2, 6, 2481]],
  [[0.979, 0.969, 0.974], [0.681, 0.793, 0.733], [0.432, 0.376, 0.402], [0.974, 0.981, 0.977], [0.966, 0.967, 0.966], [0.902, 0.910, 0.906]],
  "Ice and snow stay the confused pair (325 snow called ice). River F1 0.97.");

// 7-10. images: every labelled test date, external test, predictions
pres.addSection({ title: "Images" });
function pairSlide(title, files, stats, note) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Images" });
  s.addText(title, { placeholder: "title" });
  picture(s, `${OWN}/${files[0]}`, { x: 0.6, y: 1.15, w: 10.0, h: 2.95 }, files[0]);
  picture(s, `${OWN}/${files[1]}`, { x: 0.6, y: 4.15, w: 10.0, h: 2.95 }, files[1]);
  stat(s, 10.85, 1.6, 2.0, stats[0][0], stats[0][1], stats[0][2]);
  stat(s, 10.85, 4.6, 2.0, stats[1][0], stats[1][1], stats[1][2]);
  s.addNotes(note);
  return s;
}
const NOTE = "Panels: true colour, hand labels, RF test prediction (cross-validated: each pixel predicted by a model that never saw it), RF map. Zoomed on the densest labels (3.6 x 3.6 km).";
pairSlide("Four classes: ground truth vs Random Forest", ["four_gt_2023-10-04.png", "four_gt_2024-04-21.png"],
  [["0.87", "2023-10-04 freeze-up", HEX.accent1], ["0.86", "2024-04-21 breakup", HEX.accent1]], NOTE + " Run 05.");
pairSlide("Six classes: ground truth vs Random Forest", ["six_gt_2024-06-22.png", "six_gt_2023-10-04.png"],
  [["0.97", "2024-06-22 summer", HEX.accent1], ["0.85", "2023-10-04 freeze-up", HEX.accent1]], NOTE + " Run 10.");
pairSlide("External test: unseen winter date 2024-03-12", ["four_external.png", "six_external.png"],
  [["0.30", "four classes", HEX.accent3], ["0.31", "six classes", HEX.accent3]],
  "Models trained on the 4 patch dates only, scored on 532 daksh pixels (342 ice, 150 snow, 40 land). Ice is called snow: none of the training dates is midwinter.");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Images" });
s.addText("Predictions on dates with no labels", { placeholder: "title" });
["2024-04-20", "2024-04-29", "2026-03-26", "2023-05-05"].forEach((d, i) =>
  picture(s, `${OWN}/pred_${d}.png`, { x: 0.6 + i * 3.05, y: 1.15, w: 2.95, h: 5.35 }, `pred_${d}`));
caption(s, "Best six-class Random Forest, whole box. No ground truth, so no score. Clouds and dark forest are the main false ice and river.", 6.6);
s.addNotes("2024-04-20 and 2024-04-29: daksh spring dates, with_daksh model. 2026-03-26 midwinter and 2023-05-05 breakup: model trained without daksh.");

// 11. findings
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
s.addText("Findings and next steps", { placeholder: "title" });
s.addText([{ text: "Many band pairs and indices lift Random Forest most; tuning settings adds < 0.01.", options: { breakLine: true } },
  { text: "Snow (F1 0.40) and ice (0.73) limit the score; winter dates fail without winter training data.", options: { breakLine: true } },
  { text: "Next: label a midwinter date, more snow and clouds; finish the input-combination search.", options: { bold: true, color: "FFFFFF" } }],
  { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
