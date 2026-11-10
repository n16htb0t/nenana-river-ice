// Nenana river ice classification: presentation deck (pptxgenjs).
// Run: NODE_PATH=<dir with pptxgenjs, image-size> PPTX_SKILL=<pptx skill dir> node build_deck.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("image-size");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HOME = process.env.HOME;
const R = `${HOME}/Desktop/nenana_crops/daksh`;
const FIG = `${R}/report/figures`;
const OWN = `${HOME}/Desktop/nenana_river_models/presentation/figures`;
const OUT = `${HOME}/Desktop/nenana_river_models/presentation/Nenana_river_ice_RF.pptx`;

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
pres.layout = "LAYOUT_WIDE";                       // 13.33 x 7.5 in
pres.author = "RiverProject";
pres.title = "Nenana river ice classification";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6, fontSize: 44, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.4, fontSize: 20,
      color: "BFE6F2", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.accent1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.3, fontSize: 40, bold: true,
      color: C.background1, fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.0, fontSize: 18,
      color: "DDF3FA" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true,
      color: C.text2, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8D99AE" },
});

// ---------- helpers ----------
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
function stat(s, x, y, w, big, small, color) {
  T(s, big, { x, y, w, h: 0.9, fontSize: 48, bold: true, color, fontFace: THEME.headFontFace });
  T(s, small, { x, y: y + 0.9, w, h: 0.6, fontSize: 13, color: "4A5568" });
}
function chip(s, x, y, col, name, desc) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 0.5, h: 0.5, rectRadius: 0.08, fill: { color: col }, line: { color: "BBBBBB", width: 0.5 } });
  T(s, [{ text: name, options: { bold: true, breakLine: true } }, { text: desc, options: { fontSize: 12, color: "4A5568" } }],
    { x: x + 0.7, y: y - 0.05, w: 4.8, h: 0.6, fontSize: 15, valign: "middle" });
}
const AX = { catAxisLabelColor: "4A5568", valAxisLabelColor: "4A5568", catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontSize: 11, dataLabelColor: "14213D", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt" };

// ---------- 1. title ----------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Mapping river ice on the Nenana River", { placeholder: "title" });
s.addText("Random Forest on Landsat 30 m, trained only on hand-labelled ground truth", { placeholder: "body" });
s.addNotes("Goal: label every Landsat 30 m pixel of the Nenana study box as river, ice, snow or land (and later three land types), using only our hand-drawn polygons as ground truth.");

// ---------- 2. classes ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("What we classify", { placeholder: "title" });
T(s, "Four classes from the hand labels", { x: 0.6, y: 1.45, w: 6, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
T(s, "Six classes: land split by NLCD land cover", { x: 6.9, y: 1.45, w: 6, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
[["0066FF", "river", "open water"], ["FF00FF", "ice", "frozen river"], ["00FFFF", "snow", "snow on land or ice"],
 ["8B4513", "land", "everything else"]].forEach(([c, n, d], i) => chip(s, 0.6, 2.15 + i * 0.85, c, n, d));
[["0066FF", "river", "as drawn"], ["FF00FF", "ice", "as drawn"], ["00FFFF", "snow", "as drawn"],
 ["1A7F37", "forest", "NLCD 41-43"], ["B5D96B", "shrub / grass / wetland", "NLCD 51-95"], ["C8A165", "bare / developed", "NLCD 21-31"]]
  .forEach(([c, n, d], i) => chip(s, 6.9, 2.15 + i * 0.7, c, n, d));
T(s, "Hand labels are ground truth. The six-class land types come from NLCD Alaska 2016 inside the hand-drawn land polygons.",
  { x: 0.6, y: 6.55, w: 12.1, h: 0.5, fontSize: 13, italic: true, color: "4A5568" });
s.addNotes("Four classes come straight from the hand labels. For six classes, each land pixel takes its NLCD 2016 land-cover group (forest, shrub/grass/wetland, bare/developed); river, ice and snow stay as drawn.");

// ---------- 3. data ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Labelled data: 85,358 pixels on 4 dates", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "labelled pixels", labels: ["river", "ice", "snow", "land"], values: [39991, 1732, 564, 43071] }],
  Object.assign({ x: 0.6, y: 1.45, w: 6.6, h: 5.2, barDir: "col", chartColors: ["0066FF", "C026D3", "00B4D8", "8B4513"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", showLegend: false, showTitle: true,
    title: "Labelled pixels per class", titleFontSize: 14, titleColor: "14213D", valGridLine: { color: "E2E8F0", size: 0.5 },
    catGridLine: { style: "none" }, valAxisLabelFormatCode: "#,##0", objectName: "chart_pixels" }, AX));
const rows = [["Date", "Season", "Labelled px"], ["2023-07-24", "summer", "31,028"], ["2023-10-04", "freeze-up", "8,020"],
  ["2024-04-21", "breakup", "3,081"], ["2024-06-22", "summer", "43,229"], ["2024-03-12", "winter (daksh)", "532"]];
s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : "14213D",
  fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? "FFFFFF" : HEX.lt2) }, fontSize: 14, fontFace: THEME.bodyFontFace } }))),
  { x: 7.6, y: 1.6, w: 5.1, colW: [1.7, 1.9, 1.5], rowH: 0.45, border: { type: "solid", color: "D5E3EA", pt: 0.5 }, objectName: "table_dates" });
T(s, "Ice and snow are rare (2.7 % of labels), so they drive the macro F1.", { x: 7.6, y: 4.6, w: 5.1, h: 0.9, fontSize: 15, color: C.accent3, bold: true });
T(s, "Landsat grid 995 x 1635 px, EPSG:4326. Unlabelled dates 2024-04-20, 2024-04-29, 2026-03-26 and 2023-05-05 are only predicted.",
  { x: 7.6, y: 5.5, w: 5.1, h: 1.1, fontSize: 13, color: "4A5568" });
s.addNotes("2023-07-24 holds most of the river pixels. 2024-03-12 is the only midwinter date; we use it either in training (with_daksh) or as an external test.");

// ---------- 4. pipeline ----------
pres.addSection({ title: "Technique" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Technique" });
s.addText("Pipeline", { placeholder: "title" });
const steps = [["1", "Labels", "Hand polygons rasterised to the Landsat 30 m grid"], ["2", "Inputs", "Up to 93 per pixel: bands, band pairs, indices, texture, NLCD, terrain"],
  ["3", "Train", "Random Forest, 500 trees, class-balanced"], ["4", "Score", "5-fold spatial cross-validation + external winter test"],
  ["5", "Map", "Every date, unlabelled dates, JPEGs, QGIS layers"]];
steps.forEach(([n, h, b], i) => {
  const x = 0.6 + i * 2.5;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.1, w: 2.2, h: 3.4, rectRadius: 0.15, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `step_${n}` });
  s.addShape(pres.shapes.OVAL, { x: x + 0.75, y: 2.35, w: 0.7, h: 0.7, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  T(s, n, { x: x + 0.75, y: 2.35, w: 0.7, h: 0.7, fontSize: 22, bold: true, color: C.background1, align: "center", valign: "middle" });
  T(s, h, { x: x + 0.15, y: 3.2, w: 1.9, h: 0.5, fontSize: 18, bold: true, color: C.text2, align: "center" });
  T(s, b, { x: x + 0.2, y: 3.75, w: 1.8, h: 1.6, fontSize: 13, align: "center", valign: "top" });
  if (i < 4) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.23, y: 3.6, w: 0.24, h: 0.4, fill: { color: C.accent2 }, line: { color: C.accent2 } });
});
T(s, "No water mask is used in the best models: river comes from the pixels themselves.", { x: 0.6, y: 6.0, w: 12.1, h: 0.5, fontSize: 15, italic: true, color: C.accent3 });
s.addNotes("Everything is reproducible from scripts in nenana_crops/daksh. The Random Forest runs in about a minute per cross-validation.");

// ---------- 5. inputs ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Technique" });
s.addText("Inputs: every band set against every other", { placeholder: "title" });
const cards = [
  ["Bands", "Blue, green, red, NIR, SWIR1, SWIR2 and surface temperature (B10)", HEX.accent1],
  ["Band pairs", "All 15 pairs as (a - b)/(a + b) and a - b: NDVI, NDWI, MNDWI, NDSI and more", HEX.accent2],
  ["Indices", "EVI, SAVI, MSAVI, BSI, AWEI x2, tasseled-cap brightness / greenness / wetness", HEX.accent4],
  ["Texture", "Mean and spread in 3x3 and 7x7 windows of NDVI, NDSI, NDWI, NIR, blue", HEX.accent6],
  ["Land cover", "NLCD Alaska 2016 (30 m, from Landsat): 10 land-cover groups", HEX.accent3],
  ["Terrain + context", "Copernicus DEM: elevation, slope, height above valley floor; 15 / 31 px neighbourhood", "5E548E"],
];
cards.forEach(([h, b, col], i) => card(s, 0.6 + (i % 3) * 4.1, 1.5 + Math.floor(i / 3) * 2.65, 3.85, 2.4, h, b, col, `card_${i}`));
s.addNotes("Different surfaces separate on different band pairs: snow is bright in visible and dark in SWIR (NDSI), water is dark in NIR and SWIR (NDWI, MNDWI), vegetation is bright in NIR (NDVI). A ratio is left out: it orders pixels like the normalised difference, so a tree gains nothing.");

// ---------- 6. scoring ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Technique" });
s.addText("Honest scoring: spatial cross-validation", { placeholder: "title" });
const foldCols = [HEX.accent1, HEX.accent2, HEX.accent4, HEX.accent6, HEX.accent3];
for (let i = 0; i < 15; i++) {
  const f = i % 5;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.55 + i * 0.33, w: 3.4, h: 0.31, fill: { color: foldCols[f], transparency: f === 2 ? 0 : 55 },
    line: { color: "FFFFFF", width: 0.5 }, objectName: `band_${i}` });
  if (i < 5) T(s, `fold ${f + 1}`, { x: 4.35, y: 1.55 + i * 0.33, w: 1.0, h: 0.31, fontSize: 11, color: "4A5568", valign: "middle" });
}
T(s, "fold 3 = test", { x: 0.8, y: 6.6, w: 3.4, h: 0.35, fontSize: 12, color: "4A5568", align: "center" });
T(s, [{ text: "64-row east-west bands (1.9 km)", options: { bold: true, breakLine: true } },
  { text: "Band i belongs to fold i mod 5 on every date. Train on 4 folds, score the 5th; drop training pixels within 12 rows of a test band.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Every labelled pixel is scored once, by a model that never saw it or its polygon's neighbours.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "External test", options: { bold: true, breakLine: true } },
  { text: "Train on the 4 patch dates, test on the unseen winter date 2024-03-12 (532 px)." }],
  { x: 5.6, y: 1.6, w: 7.1, h: 4.9, fontSize: 16, valign: "top" });
s.addNotes("A random pixel split would put neighbours from the same polygon in train and test and inflate the scores. Spatial bands avoid that. Macro F1 averages the classes equally, so rare ice and snow count as much as river.");

// ---------- 7. section ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Results" });
s.addText("Results", { placeholder: "title" });
s.addText("Four classes, then six classes, then the hardest test", { placeholder: "body" });

// ---------- 8. four-class chart ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Four classes: more inputs, better scores", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "accuracy", labels: ["RF, 11 inputs", "RF, river only", "RF, all bands (68)"], values: [0.952, 0.940, 0.961] },
  { name: "macro F1", labels: ["RF, 11 inputs", "RF, river only", "RF, all bands (68)"], values: [0.750, 0.739, 0.755] }],
  Object.assign({ x: 0.6, y: 1.45, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.5, valAxisMaxVal: 1,
    showLegend: true, legendPos: "b", legendFontSize: 12, showTitle: true, title: "Spatial CV, 4 patch dates", titleFontSize: 14,
    titleColor: "14213D", valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" }, objectName: "chart_four" }, AX));
stat(s, 8.7, 1.7, 4.0, "0.961", "accuracy, Random Forest with 68 inputs", HEX.accent1);
stat(s, 8.7, 3.3, 4.0, "0.755", "macro F1 (river 0.97, ice 0.72, snow 0.36, land 0.97)", HEX.accent3);
T(s, "River only = trained and scored inside the river + 150 m corridor (fewer, harder pixels).", { x: 8.7, y: 5.0, w: 4.0, h: 1.4, fontSize: 14, color: "4A5568" });
s.addNotes("11 inputs: bands, temperature, JRC water, NDWI, NDSI, NDVI. All bands: 68 inputs, no water mask. With the daksh date added: 0.957 / 0.745.");

// ---------- 9. four-class ground truth, every labelled date ----------
function pairSlide(title, files, stats, note) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText(title, { placeholder: "title" });
  picture(s, `${OWN}/${files[0]}`, { x: 0.6, y: 1.15, w: 10.0, h: 2.95 }, files[0]);
  picture(s, `${OWN}/${files[1]}`, { x: 0.6, y: 4.15, w: 10.0, h: 2.95 }, files[1]);
  stat(s, 10.85, 1.6, 2.0, stats[0][0], stats[0][1], stats[0][2]);
  stat(s, 10.85, 4.6, 2.0, stats[1][0], stats[1][1], stats[1][2]);
  s.addNotes(note);
}
const GTNOTE = "Panels: true colour, hand labels, RF test prediction (cross-validated: each pixel predicted by a model that never saw it), RF map. Zoomed on the densest labels (3.6 x 3.6 km).";
pairSlide("Four classes, summer: truth vs Random Forest", ["four_gt_2023-07-24.png", "four_gt_2024-06-22.png"],
  [["0.99", "2023-07-24", HEX.accent1], ["0.96", "2024-06-22", HEX.accent1]], GTNOTE + " Run 05.");
pairSlide("Four classes, ice seasons: truth vs Random Forest", ["four_gt_2023-10-04.png", "four_gt_2024-04-21.png"],
  [["0.87", "2023-10-04 freeze-up", HEX.accent1], ["0.86", "2024-04-21 breakup", HEX.accent1]], GTNOTE + " Run 05.");

// ---------- 10. six-class progression ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Six classes: what moved the score", { placeholder: "title" });
const sixLabels = ["all bands", "+ tuned", "+ NLCD", "+ NLCD tuned", "+ terrain, context"];
s.addChart(pres.charts.BAR, [
  { name: "accuracy", labels: sixLabels, values: [0.822, 0.823, 0.962, 0.961, 0.962] },
  { name: "macro F1", labels: sixLabels, values: [0.663, 0.667, 0.819, 0.825, 0.826] }],
  Object.assign({ x: 0.6, y: 1.45, w: 8.0, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.5, valAxisMaxVal: 1,
    showLegend: true, legendPos: "b", legendFontSize: 12, showTitle: true, title: "Six classes, spatial CV, 4 patch dates (runs 06-10)",
    titleFontSize: 14, titleColor: "14213D", valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" }, objectName: "chart_six" }, AX));
stat(s, 9.0, 1.7, 3.8, "0.826", "best macro F1: all bands + NLCD + terrain + context", HEX.accent3);
T(s, [{ text: "Forest settings moved F1 by under 0.01.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "NLCD lifts the land types most, because the same map made their labels: read their scores as agreement with NLCD." }],
  { x: 9.0, y: 3.4, w: 3.8, h: 3.2, fontSize: 14, color: "4A5568", valign: "top" });
s.addNotes("Run 06: 68 inputs. 08: forest settings searched. 07/09: NLCD land cover as inputs. 10: + distance to NLCD water, Copernicus DEM terrain, 15/31 px context. Input-combination search (run 11) and bands + angles + land cover (run 12) still running.");

// ---------- 11. per-class ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Best six-class model: F1 per class", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "F1", labels: ["river", "ice", "snow", "forest", "shrub/grass/wetland", "bare/developed"],
  values: [0.974, 0.733, 0.402, 0.977, 0.966, 0.906] }],
  Object.assign({ x: 0.6, y: 1.45, w: 7.8, h: 5.3, barDir: "bar", chartColors: ["0066FF", "C026D3", "00B4D8", "1A7F37", "8DB848", "C8A165"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: false,
    showTitle: true, title: "Run 10, spatial CV, 4 patch dates", titleFontSize: 14, titleColor: "14213D",
    valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" }, objectName: "chart_perclass" }, AX));
card(s, 8.8, 1.6, 3.9, 2.4, "Strong", "River 0.97 and the land types 0.91-0.98.", HEX.accent4, "card_strong");
card(s, 8.8, 4.25, 3.9, 2.45, "Weak", "Snow 0.40: only 564 labels and confused with ice (325 snow pixels called ice).", HEX.accent3, "card_weak");
s.addNotes("Accuracy 0.962, macro F1 0.826. Snow and ice limit macro F1; more snow and midwinter labels would help more than any model change.");

// ---------- confusion matrices ----------
function mix(frac) {                                   // white -> accent1 by row share
  const a = [255, 255, 255], b = [11, 110, 153];
  return a.map((v, i) => Math.round(v + (b[i] - v) * frac).toString(16).padStart(2, "0")).join("").toUpperCase();
}
function confusion(s, names, cm, prf, box) {
  const k = names.length, cell = { fontSize: k > 4 ? 12 : 14, fontFace: THEME.bodyFontFace, align: "center", valign: "middle" };
  const head = [{ text: "true \\ predicted", options: Object.assign({}, cell, { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 } }) }]
    .concat(names.map((n) => ({ text: n, options: Object.assign({}, cell, { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 } }) })));
  const body = cm.map((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0);
    return [{ text: names[i], options: Object.assign({}, cell, { bold: true, align: "left", fill: { color: HEX.lt2 } }) }]
      .concat(row.map((v) => {
        const f = tot ? v / tot : 0;
        return { text: `${v.toLocaleString("en-US")}\n${(100 * f).toFixed(0)}%`,
          options: Object.assign({}, cell, { color: f > 0.5 ? "FFFFFF" : "14213D", bold: f > 0.5, fill: { color: mix(Math.sqrt(f)) } }) };
      }));
  });
  const first = k > 4 ? 1.9 : 1.6;
  const cw = (box.w - first) / k;
  s.addTable([head].concat(body), { x: box.x, y: box.y, w: box.w, colW: [first].concat(Array(k).fill(cw)),
    rowH: Math.min(0.75, box.h / (k + 1)), border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "confusion_table" });
  const ph = [["class", "prec.", "recall", "F1"]].concat(prf.map((r, i) => [names[i]].concat(r.map((v) => v.toFixed(3)))));
  return ph;
}
function prfTable(s, ph, box) {
  s.addTable(ph.map((r, i) => r.map((c, j) => ({ text: c, options: { fontSize: 13, fontFace: THEME.bodyFontFace,
    bold: i === 0 || j === 0, color: i === 0 ? "FFFFFF" : "14213D", align: j === 0 ? "left" : "right",
    fill: { color: i === 0 ? HEX.accent1 : (i % 2 ? "FFFFFF" : HEX.lt2) } } }))),
    { x: box.x, y: box.y, w: box.w, colW: [box.w * 0.43, box.w * 0.19, box.w * 0.19, box.w * 0.19], rowH: 0.36,
      border: { type: "solid", color: "D5E3EA", pt: 0.5 }, objectName: "prf_table" });
}
const CMS = [
  ["Confusion matrix: best four-class Random Forest", "Run 05, all bands (68 inputs), spatial CV, 4 patch dates. Accuracy 0.961, macro F1 0.755.",
   ["river", "ice", "snow", "land"],
   [[38560, 166, 1, 1264], [250, 1316, 125, 41], [18, 347, 160, 39], [930, 104, 36, 42001]],
   [[0.970, 0.964, 0.967], [0.681, 0.760, 0.718], [0.497, 0.284, 0.361], [0.969, 0.975, 0.972]],
   "Snow is mostly called ice (347 of 564). River and land swap about 2,200 pixels at the banks."],
  ["Confusion matrix: best six-class Random Forest", "Run 10, 93 inputs (all bands + NLCD + terrain + context), spatial CV, 4 patch dates. Accuracy 0.962, macro F1 0.826.",
   ["river", "ice", "snow", "forest", "shrub/grass/wet.", "bare/dev."],
   [[38749, 226, 3, 314, 440, 259], [163, 1374, 178, 8, 8, 1], [17, 325, 212, 2, 5, 3],
    [255, 38, 45, 21749, 80, 6], [188, 44, 32, 252, 15110, 0], [205, 11, 21, 2, 6, 2481]],
   [[0.979, 0.969, 0.974], [0.681, 0.793, 0.733], [0.432, 0.376, 0.402], [0.974, 0.981, 0.977], [0.966, 0.967, 0.966], [0.902, 0.910, 0.906]],
   "Ice and snow remain the confused pair (325 snow called ice). Land types are clean because NLCD is an input."],
];
CMS.forEach(([title, sub, names, cm, prf, note]) => {
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText(title, { placeholder: "title" });
  T(s, sub, { x: 0.6, y: 1.2, w: 12.1, h: 0.4, fontSize: 13, color: "4A5568" });
  const ph = confusion(s, names, cm, prf, { x: 0.6, y: 1.75, w: 7.9, h: 4.9 });
  prfTable(s, ph, { x: 8.85, y: 1.75, w: 3.85 });
  T(s, note, { x: 8.85, y: 1.85 + 0.36 * (names.length + 1) + 0.3, w: 3.85, h: 1.6, fontSize: 14, color: C.accent3, valign: "top" });
  s.addNotes("Rows are the true class (hand label), columns the predicted class; each cell shows the pixel count and its share of that true class. Every pixel was predicted by a model that never saw it (spatial cross-validation).");
});

// ---------- all runs ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Every run: accuracy / macro F1", { placeholder: "title" });
const runs = [
  ["01", "4", "Random Forest, 11 inputs", "0.952 / 0.750", "0.944 / 0.731", "0.338 / 0.180"],
  ["03", "4", "Random Forest, river + 150 m only", "0.940 / 0.739", "0.927 / 0.725", "0.329 / 0.173"],
  ["05", "4", "Random Forest, all bands (68)", "0.961 / 0.755", "0.957 / 0.745", "0.299 / 0.191"],
  ["06", "6", "Random Forest, all bands (68)", "0.822 / 0.663", "0.817 / 0.653", "0.301 / 0.144"],
  ["07", "6", "Random Forest, + NLCD (78)", "0.962 / 0.819", "0.958 / 0.813", "0.305 / 0.147"],
  ["08", "6", "Random Forest, all bands, tuned", "0.823 / 0.667", "0.815 / 0.659", "-"],
  ["09", "6", "Random Forest, + NLCD, tuned", "0.961 / 0.825", "0.957 / 0.819", "-"],
  ["10", "6", "Random Forest, + NLCD + terrain + context (93)", "0.962 / 0.826", "0.958 / 0.822", "0.312 / 0.144"],
];
const best = new Set([2, 7]);
s.addTable([["Run", "Classes", "Model and inputs", "Without daksh (CV)", "With daksh (CV)", "External test"]].concat(runs).map((r, i) =>
  r.map((c, j) => ({ text: c, options: { fontSize: 13, fontFace: THEME.bodyFontFace, bold: i === 0 || best.has(i - 1),
    color: i === 0 ? "FFFFFF" : "14213D", align: j >= 3 || j < 2 ? "center" : "left",
    fill: { color: i === 0 ? HEX.dk2 : best.has(i - 1) ? "D8F0F7" : (i % 2 ? "FFFFFF" : "F5F9FB") } } }))),
  { x: 0.6, y: 1.35, w: 12.1, colW: [0.7, 1.0, 4.4, 2.0, 2.0, 2.0], rowH: 0.5, border: { type: "solid", color: "D5E3EA", pt: 0.5 }, objectName: "table_runs" });
T(s, "Highlighted: best four-class (05) and best six-class (10). External test = trained without 2024-03-12, scored on it.",
  { x: 0.6, y: 6.55, w: 12.1, h: 0.45, fontSize: 13, italic: true, color: "4A5568" });
s.addNotes("CV = 5-fold spatial cross-validation. With daksh adds the 2024-03-12 labels to training (9 inputs instead of 11 for runs 01-05 because that image has no SWIR2 or thermal band). Run 11 (input-combination search) and run 12 (bands + angles + land cover) are still training.");

// ---------- six-class ground truth, every labelled date ----------
pairSlide("Six classes, summer: truth vs Random Forest", ["six_gt_2023-07-24.png", "six_gt_2024-06-22.png"],
  [["0.99", "2023-07-24", HEX.accent1], ["0.97", "2024-06-22", HEX.accent1]], GTNOTE + " Run 10.");
pairSlide("Six classes, ice seasons: truth vs Random Forest", ["six_gt_2023-10-04.png", "six_gt_2024-04-21.png"],
  [["0.85", "2023-10-04 freeze-up", HEX.accent1], ["0.87", "2024-04-21 breakup", HEX.accent1]], GTNOTE + " Run 10.");

// ---------- external test ----------
pairSlide("External test: unseen winter date 2024-03-12", ["four_external.png", "six_external.png"],
  [["0.30", "four classes (run 05)", HEX.accent3], ["0.31", "six classes (run 10)", HEX.accent3]],
  "Trained on the 4 patch dates only, scored on 532 daksh pixels (342 ice, 150 snow, 40 land). Ice is called snow: none of the training dates is midwinter. More inputs did not fix it; a labelled midwinter date would.");

// ---------- predictions on unlabelled dates ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Predictions on dates with no labels", { placeholder: "title" });
["2024-04-20", "2024-04-29", "2026-03-26", "2023-05-05"].forEach((d, i) =>
  picture(s, `${OWN}/pred_${d}.png`, { x: 0.6 + i * 3.05, y: 1.15, w: 2.95, h: 5.35 }, `pred_${d}`));
T(s, "Best six-class Random Forest, whole box. No ground truth, so no score. Clouds and dark forest are the main false ice and river.",
  { x: 0.6, y: 6.6, w: 12.1, h: 0.45, fontSize: 13, color: "4A5568", italic: true });
s.addNotes("2024-04-20 and 2024-04-29: daksh spring dates, with_daksh model. 2026-03-26 midwinter and 2023-05-05 breakup: model trained without daksh.");

// ---------- 16. findings ----------
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
s.addText("What we learned", { placeholder: "title" });
[["Many inputs win", "All band pairs, indices and land cover lift Random Forest the most.", HEX.accent1],
 ["Inputs matter, settings don't", "68-93 inputs raised the scores; tuning the forest moved F1 by under 0.01.", HEX.accent4],
 ["Snow and ice are the limit", "Few labels: snow F1 about 0.40, ice about 0.73.", HEX.accent3],
 ["Season coverage matters most", "No midwinter training date, so the winter test fails for every model.", HEX.accent6]]
  .forEach(([h, b, col], i) => card(s, 0.6 + (i % 2) * 6.15, 1.5 + Math.floor(i / 2) * 2.65, 5.9, 2.4, h, b, col, `find_${i}`));
s.addNotes("NLCD as an input makes the land-type scores mostly agreement with NLCD itself; river, ice and snow remain a fair test.");

// ---------- 17. next ----------
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
s.addText("Next steps", { placeholder: "title" });
s.addText("Label a midwinter date and more snow  ·  label clouds  ·  finish the input-combination search  ·  add VIIRS", { placeholder: "body" });
s.addNotes("All outputs: ~/Desktop/nenana_river_models (runs, reports, NDVI/NLCD layers, QGIS project, scripts).");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
