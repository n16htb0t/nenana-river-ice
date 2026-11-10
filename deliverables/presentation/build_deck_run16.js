// Run 16 (water-zone Random Forest) deck, pptxgenjs. Same theme and layouts as build_deck.js.
// Numbers: output/16_rf_water_zone/deck_numbers.json (report/word_content_run16.py) and extra.json
// (rf_water_zone_extra.py). Run:
//   NODE_PATH=<dir with pptxgenjs, image-size> PPTX_SKILL=<pptx skill dir> node build_deck_run16.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("image-size");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HOME = process.env.HOME;
const RUN = `${HOME}/Desktop/nenana_crops/daksh/output/16_rf_water_zone`;
const OUT = `${HOME}/Desktop/nenana_river_models/presentation/Nenana_run16_water_zone.pptx`;
const N = JSON.parse(fs.readFileSync(`${RUN}/deck_numbers.json`, "utf8"));
const X = JSON.parse(fs.readFileSync(`${RUN}/extra.json`, "utf8"));
const FIGS = N.figures;
const fig = (list, d) => (list.find((g) => g[0] === d) || [])[1];

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
pres.title = "Nenana river ice: run 16, water-zone Random Forest";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6, fontSize: 44, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.6, fontSize: 20,
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
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30, bold: true,
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
function stat(s, x, y, w, big, small, color, name) {
  T(s, big, { x, y, w, h: 1.1, fontSize: 54, bold: true, color, fontFace: THEME.headFontFace, objectName: name });
  T(s, small, { x, y: y + 1.1, w, h: 0.8, fontSize: 14, color: C.text1, valign: "top" });
}
function bullets(s, items, box, size = 16) {
  T(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    Object.assign({ fontSize: size, valign: "top", paraSpaceAfter: 8 }, box));
}
const head = { bold: true, color: "FFFFFF", fill: { color: HEX.accent1 }, fontSize: 13 };
function tableOf(header, rows, opts) {
  return [header.map((t) => ({ text: t, options: head }))].concat(
    rows.map((r, i) => r.map((t, j) => ({ text: String(t), options: { fontSize: 13, align: j === 0 ? "left" : "right",
      fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 }, bold: opts && opts.boldRow === i } }))));
}
const chartFrame = (o) => Object.assign({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", catAxisLabelColor: "14213D", valAxisLabelColor: "8D99AE",
  valGridLine: { color: "E3E8EE", size: 0.75 }, catGridLine: { style: "none" }, showTitle: true, titleFontSize: 14,
  titleColor: "0B2E4F", showLegend: true, legendPos: "b", legendFontSize: 12, showValue: true, dataLabelFontSize: 10,
  dataLabelFormatCode: "0.00",
}, o);

const ws = N.score.with_daksh, ex = N.score.external_test;
const f3 = (v) => v.toFixed(3);

// ---------- 1 title ----------
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Run 16: river ice with a water-zone rule", { placeholder: "title" });
s.addText("Landsat 30 m, six classes, Random Forest. JRC water occurrence decides where river and ice can be; "
  + "NLCD 2016 decides the land types.", { placeholder: "body" });

// ---------- 2 summary ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Overview" });
s.addText("Run 16 is the cleanest and best-scoring model so far", { placeholder: "title" });
stat(s, 0.7, 1.5, 3.8, f3(ws.rule[1]), "accuracy with daksh dates, 5-fold spatial CV (run 10: 0.958)", C.accent1, "stat acc");
stat(s, 4.75, 1.5, 3.8, f3(ex.rule[1]), "accuracy on unseen winter date 2024-03-12 (run 10: 0.312)", C.accent4, "stat external");
const g = X.gap.with_daksh, gap = g.reduce((a, r) => a + r.all_train - r.all_test, 0) / g.length;
stat(s, 8.8, 1.5, 3.8, `${(100 * gap).toFixed(1)} %`, "training vs held-out accuracy gap: no overfitting", C.accent3, "stat gap");
bullets(s, [
  "No river on land and no snow on the river: the water zone only allows river or ice, land only snow or a land type.",
  "Scores are against the rule labels (snow on the river counted as ice). Against the original hand labels: "
    + `${f3(ws.orig[1])} accuracy, macro F1 ${f3(ws.orig[2])}.`,
  "The loss comes from labels the rule cannot reproduce: river where the channel has moved, gravel bars in the water zone.",
], { x: 0.7, y: 4.2, w: 11.9, h: 2.6 }, 16);

// ---------- 3 rule ----------
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("The JRC water zone decides what each pixel can be", { placeholder: "title" });
picture(s, FIGS.zone, { x: 0.5, y: 1.35, w: 6.6, h: 5.7 }, "water zone figure");
card(s, 7.4, 1.45, 5.4, 2.1, "Water zone: JRC >= 5 %", "River or ice only. Snow on the frozen river counts as ice. "
  + `${N.zone_pct.toFixed(1)} % of the study box.`, C.accent1, "card water");
card(s, 7.4, 3.75, 5.4, 2.1, "Land: JRC < 5 %", "Snow, forest, shrub/grass/wetland or bare/developed. Land types from "
  + "NLCD 2016. No ice, no river.", C.accent4, "card land");
T(s, "One Random Forest per zone, each trained only on its own zone's pixels.",
  { x: 7.4, y: 6.05, w: 5.4, h: 0.7, fontSize: 14, italic: true, color: C.text2 });

// ---------- 4 labels ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Hand labels were moved onto the rule", { placeholder: "title" });
s.addTable(tableOf(["class", "labelled", "in water zone", "became ice", "left out"],
  N.changes.map((r) => [r[0].replace(/_/g, " ")].concat(r.slice(1).map((v) => Number(v).toLocaleString("en-US"))))),
  { x: 0.6, y: 1.45, w: 7.4, colW: [2.4, 1.25, 1.35, 1.2, 1.2], border: { type: "solid", color: "D5DDE5", pt: 0.75 },
    fontFace: THEME.bodyFontFace, color: C.text1 });
bullets(s, [
  "Snow in the water zone became ice: it is snow lying on the frozen river.",
  "River and ice left out lie where JRC says land, mostly where the channel moved after the JRC years (to 2021).",
  "Bare left out are gravel bars inside the water zone.",
  "with_daksh labels shown; 2024-04-20 and 2024-04-29 have no labels.",
], { x: 8.4, y: 1.5, w: 4.4, h: 5.3 }, 15);

// ---------- 5 inputs / model ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Inputs and model: small forests, spatial test folds", { placeholder: "title" });
card(s, 0.6, 1.45, 3.9, 2.55, "Bands", "7 without daksh: blue, green, red, NIR, SWIR1, SWIR2, thermal. 5 with daksh "
  + "(no SWIR2, no thermal).", C.accent1, "card bands");
card(s, 4.72, 1.45, 3.9, 2.55, "Inputs", "Pool of 83 / 66: band pairs, indices, textures, illumination, NLCD. Each "
  + "forest keeps its top 25. JRC only picks the zone.", C.accent2, "card inputs");
card(s, 8.84, 1.45, 3.9, 2.55, "Forests", "300 trees, depth 16, min 20 pixels per leaf, balanced class weights.",
  C.accent4, "card forest");
card(s, 0.6, 4.2, 6.0, 2.55, "Testing", "5-fold spatial CV: 1.9 km east-west bands, same fold on every date, 12-row "
  + "buffer. External: train on 4 patch dates, test on 2024-03-12.", C.accent3, "card cv");
const wi = X.importance.with_daksh;
card(s, 6.75, 4.2, 6.0, 2.55, "Top inputs (with daksh)", `Water RF: ${wi.water.slice(0, 4).map((a) => a[0]).join(", ")}. `
  + `Land RF: ${wi.land.slice(0, 4).map((a) => a[0]).join(", ")}.`, C.accent6, "card importance");

// ---------- 6 comparison ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Run 16 beats every earlier run on its rule labels", { placeholder: "title" });
s.addTable(tableOf(["run (accuracy / macro F1)", "exclude_daksh", "with_daksh", "external 2024-03-12"], N.comp, { boldRow: 2 }),
  { x: 0.6, y: 1.5, w: 12.1, colW: [5.0, 2.3, 2.3, 2.5], border: { type: "solid", color: "D5DDE5", pt: 0.75 },
    fontFace: THEME.bodyFontFace, color: C.text1 });
bullets(s, [
  "Rule labels leave out pixels the rule cannot hold, so they are not a like-for-like comparison with runs 10 and 15.",
  "The original-label row is like-for-like: lower, because 19 % of river labels sit outside the JRC zone and most bare "
    + "labels are gravel bars inside it.",
  "External test: every ice and snow pixel right; the 36 forest/shrub pixels are called snow.",
], { x: 0.6, y: 4.3, w: 12.1, h: 2.5 }, 16);

// ---------- 7 per class ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("River, ice and land near perfect; snow weakest", { placeholder: "title" });
const cls = ws.rule[0].map((r) => r[0].replace(/_/g, " "));
const origF1 = Object.fromEntries(ws.orig[0].map((r) => [r[0].replace(/_/g, " "), Number(r[4])]));
s.addChart(pres.charts.BAR, [
  { name: "rule labels", labels: cls, values: ws.rule[0].map((r) => Number(r[4])) },
  { name: "original hand labels", labels: cls, values: cls.map((c) => origF1[c] || 0) },
], chartFrame({ x: 0.5, y: 1.35, w: 8.2, h: 5.6, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent5],
  title: "F1 per class, with_daksh, 5-fold spatial CV", valAxisMaxVal: 1, valAxisMinVal: 0, dataLabelPosition: "outEnd" }));
const snow = ws.rule[0].find((r) => r[0] === "snow");
bullets(s, [
  `Snow: recall ${snow[3]} but precision ${snow[2]}. The land forest calls a few hundred bright forest and shrub pixels snow.`,
  "Ice vs river inside the water zone: F1 0.91 / 0.99.",
  "Original-label losses: river outside the zone, gravel bars, snow on the river.",
], { x: 9.0, y: 1.6, w: 3.9, h: 5.2 }, 15);

// ---------- 8 cutoff ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("A 5 % JRC cutoff fits the hand labels best", { placeholder: "title" });
const cut = X.scan.with_daksh.map((r) => `${r.cutoff} %`);
s.addChart(pres.charts.LINE, [
  { name: "with_daksh", labels: cut, values: X.scan.with_daksh.map((r) => r.orig_mf1) },
  { name: "exclude_daksh", labels: cut, values: X.scan.exclude_daksh.map((r) => r.orig_mf1) },
], chartFrame({ x: 0.5, y: 1.35, w: 8.2, h: 5.6, chartColors: [HEX.accent1, HEX.accent3], lineSize: 3, lineDataSymbolSize: 8,
  title: "macro F1 on the original hand labels vs JRC cutoff", valAxisMinVal: 0.55, valAxisMaxVal: 0.75,
  dataLabelPosition: "t", dataLabelFormatCode: "0.000" }));
const w50 = X.scan.with_daksh.find((r) => r.cutoff === 50), w5 = X.scan.with_daksh.find((r) => r.cutoff === 5);
bullets(s, [
  "Same training at JRC 1, 5, 10, 25 and 50 %.",
  "Higher cutoffs narrow the zone to the main channel and push real river onto land.",
  `River F1 on original labels: ${w5.river_f1_orig.toFixed(2)} at 5 %, ${w50.river_f1_orig.toFixed(2)} at 50 %.`,
], { x: 9.0, y: 1.6, w: 3.9, h: 5.2 }, 15);

// ---------- 9 overfitting ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Training and held-out accuracy match: no overfitting", { placeholder: "title" });
const mean = (rows, k) => rows.reduce((a, r) => a + r[k], 0) / rows.length;
const zones = ["water", "land", "all"], zl = ["water RF", "land RF", "both"];
s.addChart(pres.charts.BAR, [
  { name: "training pixels", labels: zl, values: zones.map((z) => mean(g, `${z}_train`)) },
  { name: "held-out fold", labels: zl, values: zones.map((z) => mean(g, `${z}_test`)) },
], chartFrame({ x: 0.5, y: 1.35, w: 8.2, h: 5.6, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent5, HEX.accent1],
  title: "accuracy, with_daksh, mean of 5 folds (rule labels)", valAxisMinVal: 0.9, valAxisMaxVal: 1,
  dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000" }));
const ge = X.gap.exclude_daksh, gapx = ge.reduce((a, r) => a + r.all_train - r.all_test, 0) / ge.length;
bullets(s, [
  `Mean gap ${(100 * gap).toFixed(1)} % with daksh, ${(100 * gapx).toFixed(1)} % without.`,
  "Limited forests (25 inputs, depth 16, leaf 20) cannot memorise pixels.",
  "Held-out folds are 1.9 km bands never next to training pixels.",
], { x: 9.0, y: 1.6, w: 3.9, h: 5.2 }, 15);

// ---------- maps ----------
pres.addSection({ title: "Maps" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Maps" });
s.addText("Maps", { placeholder: "title" });
s.addText("Ground truth vs prediction, then run 16 next to runs 10 and 15", { placeholder: "body" });

function gtSlide(title, items) {
  const sl = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Maps" });
  sl.addText(title, { placeholder: "title" });
  T(sl, "true colour | hand labels | rule labels | test prediction | map", { x: 0.6, y: 1.2, w: 12.1, h: 0.35, fontSize: 13, color: C.accent5 });
  items.forEach(([file, cap], i) => {
    if (!file) return;
    const y = 1.6 + i * 2.75;
    picture(sl, file, { x: 0.6, y, w: 12.1, h: 2.35 }, cap);
    T(sl, cap, { x: 0.6, y: y + 2.35, w: 12.1, h: 0.3, fontSize: 12, italic: true, align: "center" });
  });
}
const acc = (d) => { const r = FIGS.gt.find((x) => x[0] === d); return r ? r[2].toFixed(2) : "-"; };
gtSlide("Spring and autumn: river vs ice in the water zone", [
  [fig(FIGS.gt, "2024-04-21"), `2024-04-21 spring break-up, test accuracy ${acc("2024-04-21")}`],
  [fig(FIGS.gt, "2023-10-04"), `2023-10-04 first ice, test accuracy ${acc("2023-10-04")}`]]);
gtSlide("Summer: open river and NLCD land types", [
  [fig(FIGS.gt, "2023-07-24"), `2023-07-24 summer, test accuracy ${acc("2023-07-24")}`],
  [fig(FIGS.gt, "2024-06-22"), `2024-06-22 summer, test accuracy ${acc("2024-06-22")}`]]);
gtSlide("Winter: ice on the river, snow on land", [
  [fig(FIGS.gt, "2024-03-12"), `2024-03-12 with daksh in training, test accuracy ${acc("2024-03-12")}`],
  [FIGS.ext[0] && FIGS.ext[0][1], `2024-03-12 unseen date (trained on patch dates only), accuracy ${FIGS.ext[0] ? FIGS.ext[0][2].toFixed(2) : "-"}`]]);

function compareSlide(d, title, notes) {
  const sl = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Maps" });
  sl.addText(title, { placeholder: "title" });
  picture(sl, fig(FIGS.compare, d), { x: 0.4, y: 1.25, w: 8.4, h: 6.0 }, `${d} run 10, 15, 16`);
  bullets(sl, notes, { x: 9.0, y: 1.6, w: 3.9, h: 5.2 }, 15);
}
compareSlide("2024-03-12", "Winter: only run 16 keeps ice off the land", [
  "True colour, run 10, run 15, run 16.", "Runs 10/15 paint large areas of land as ice (magenta).",
  "Run 16: ice only inside the water zone, snow (cyan) on land."]);
compareSlide("2024-04-21", "Break-up: run 16 keeps river and ice in the channel", [
  "Runs 10/15 show river and ice patches far from the river.", "Run 16 keeps them in the JRC water zone.",
  "Snow stays on high, open ground."]);
compareSlide("2026-03-26", "Unlabelled winter date: same picture", [
  "No labels: a check of how the models generalise.", "Run 16 gives a single ice ribbon along the river and snow elsewhere."]);

// ---------- caveats + next ----------
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
s.addText("Things to know", { placeholder: "title" });
card(s, 0.6, 1.45, 3.9, 2.6, "JRC ends in 2021", "Where the channel moved since, real river lies outside the zone and "
  + "is mapped as land or snow.", C.accent3, "card jrc");
card(s, 4.72, 1.45, 3.9, 2.6, "Gravel bars", "Inside the water zone they can only be river or ice.", C.accent6, "card gravel");
card(s, 8.84, 1.45, 3.9, 2.6, "Few ice/snow dates", "Ice and snow come from 3 dates, so their scores move with each polygon.",
  C.accent2, "card few");
card(s, 0.6, 4.25, 6.0, 2.5, "NLCD as input", "Forest/shrub/bare scores partly measure agreement with NLCD.", C.accent4, "card nlcd");
card(s, 6.75, 4.25, 6.0, 2.5, "Clouds", "Never labelled: cloudy pixels become snow on land or ice in the water zone.",
  C.accent5, "card clouds");

s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
s.addText("Run 16: clean maps, best scores", { placeholder: "title" });
s.addText(`${f3(ws.rule[1])} accuracy with daksh, ${f3(ex.rule[1])} on the unseen date. Next: update the JRC zone where the `
  + "channel moved, add a gravel-bar class in the water zone, label more ice and snow dates.", { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
