// Run 21 deck (VIIRS upsampled to 30 m, six classes), pptxgenjs. Same theme as build_deck_viirs.js.
// Numbers: ~/Desktop/nenana_crops/daksh/output/deck_numbers_run21.json (report/word_content_run21.py). Run:
//   NODE_PATH=<dir with pptxgenjs, image-size> PPTX_SKILL=<pptx skill dir> node build_deck_viirs.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("image-size");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HOME = process.env.HOME;
const O = `${HOME}/Desktop/nenana_crops/daksh/output`;
const OUT = `${HOME}/Desktop/nenana_river_models/presentation/Nenana_VIIRS_run21.pptx`;
const N = JSON.parse(fs.readFileSync(`${O}/deck_numbers_run21.json`, "utf8"));
const M18 = JSON.parse(fs.readFileSync(`${O}/18_viirs_states/metrics_all.json`, "utf8"));
const ALL = JSON.parse(fs.readFileSync(`${O}/21_viirs_run16/metrics_all.json`, "utf8"));
const RF = ALL.with_daksh, R16 = RF.run16_same_dates;

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
pres.title = "Nenana river ice from VIIRS: run 21, 30 m";
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
const SEASON = { "2023-04-19": "spring", "2023-05-05": "break-up", "2023-07-24": "summer", "2023-10-04": "freeze-up",
  "2024-04-21": "break-up", "2024-06-22": "summer", "2024-09-02": "late summer", "2025-07-04": "summer",
  "2026-03-26": "late winter", "2026-08-01": "summer" };
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Run 21: run 16 on VIIRS at 30 m", { placeholder: "title" });
s.addText("Run 16 repeated with VIIRS upsampled to the Landsat 30 m grid instead of Landsat: same six classes, water-mask "
  + "rule, hand labels, Random Forest and spatial cross-validation.", { placeholder: "body" });

s = content("VIIRS keeps river and land, loses snow and some ice", "Overview");
stat(s, 0.7, 1.45, 3.9, f3(RF.rule.accuracy), "run 21 RF accuracy (VIIRS), rule labels, 5-fold spatial CV", C.accent1, "stat rf");
stat(s, 4.75, 1.45, 3.9, f3(R16.rule.accuracy), "run 16 RF accuracy (Landsat), same 4 dates", C.accent4, "stat r16");
stat(s, 8.8, 1.45, 3.9, RF.rule.per_class.snow.f1.toFixed(2), "snow F1 on VIIRS (Landsat: " + R16.rule.per_class.snow.f1.toFixed(2) + ")", C.accent3, "stat snow");
bullets(s, [
  `Macro F1 ${f3(RF.rule.macro_f1)} on VIIRS vs ${f3(R16.rule.macro_f1)} on Landsat (rule labels).`,
  "River, forest, shrub, bare stay high: the water mask fixes river, NLCD drives the land classes.",
  "Snow patches and the ice / open-water mix are smaller than a 375 m pixel, so they cannot be placed at 30 m.",
  "Summer days near perfect; freeze-up and break-up drop to about 0.72.",
], { x: 0.7, y: 4.15, w: 11.9, h: 2.7 }, 16);

// ---------- design ----------
pres.addSection({ title: "Design" });
s = content("Same as run 16, only the image changed", "Design");
s.addTable(tableOf(["", "run 16", "run 21"], [
  ["image", "Landsat 30 m", "VIIRS I1-I5 upsampled to 30 m (bilinear + run 17 shift)"],
  ["classes / rule", "6 classes; water mask: river / ice only", "same"],
  ["labels", "hand labels -> rule labels", "same (dates with VIIRS)"],
  ["model", "one RF per zone, top 25, depth 16, leaf 20", "same code"],
  ["evaluation", "5-fold spatial CV", "same code"],
  ["dates", "7 labelled", "4 labelled; maps for all 10 VIIRS dates"],
  ["inputs", "66 Landsat-based", "46 VIIRS-based (no blue / green I-band)"],
], 12), { x: 0.6, y: 1.45, w: 12.1, colW: [2.0, 4.6, 5.5] });
s.addTable(tableOf(["date", "season", "labelled px"], N.labelled.map((d) => [d, SEASON[d] || "",
  RF.per_date[d].original.pixels.toLocaleString()]), 12), { x: 0.6, y: 5.0, w: 6.0, colW: [2.0, 2.0, 2.0] });
bullets(s, ["No VIIRS for 2024-03-12, 2024-04-20, 2024-04-29: left out (incl. run 16's external test)."],
  { x: 7.0, y: 5.1, w: 5.7, h: 1.6 }, 13);

// ---------- results ----------
pres.addSection({ title: "Results" });
s = content("Scores: VIIRS vs Landsat on the same dates", "Results");
s.addTable(tableOf(["model / labels", "pixels", "accuracy", "balanced acc", "κ", "macro F1"], N.summary, 12),
  { x: 0.6, y: 1.4, w: 12.1, colW: [4.4, 1.4, 1.5, 1.7, 1.3, 1.8] });
s.addTable(tableOf(["class (rule labels)"].concat(N.models.map((m) => `${m} F1 (VIIRS)`)).concat(["run 16 F1 (Landsat)"]), N.per_class, 12),
  { x: 0.6, y: 1.8 + 0.36 * (N.summary.length + 1), w: 7.5 });
for (const [name, img] of Object.entries(N.conf)) {
  if (!img) continue;
  s = content(`${name}: confusion matrix (rule labels)`, "Results");
  picture(s, img, { x: 0.6, y: 1.3, w: 7.2, h: 5.9 }, `conf ${name}`);
  bullets(s, ["Rows: true class (hand labels, rule).", "Columns: predicted.", "Cell: pixels and % of the true class.",
    "Snow is mostly called forest / shrub; ice partly river."], { x: 8.2, y: 1.6, w: 4.6, h: 4.5 }, 14);
}
s = content("Per date (rule labels)", "Results");
s.addTable(tableOf(["date", "season", "px"].concat(N.models.map((m) => `${m} acc / F1`)), N.per_date, 13),
  { x: 0.6, y: 1.45, w: 9.0 });

// ---------- maps ----------
pres.addSection({ title: "Maps" });
for (const d of Object.keys(N.images).sort()) {
  for (const f of N.images[d]) {
    s = content(`${d} (${SEASON[d] || ""}): ${N.labelled.includes(d) ? "hand-labelled" : "no hand labels"}`, "Maps");
    picture(s, f, { x: 0.25, y: 1.2, w: 12.85, h: 5.95 }, `map ${d}`);
  }
}

// ---------- end ----------
pres.addSection({ title: "Wrap-up" });
s = content("Things to know", "Wrap-up");
card(s, 0.6, 1.45, 3.9, 2.55, "No 30 m detail from VIIRS", "Upsampling copies the 375 m value; 30 m edges come from NLCD, terrain, the mask.", C.accent3, "k1");
card(s, 4.72, 1.45, 3.9, 2.55, "Land = NLCD", "High forest / shrub / bare scores largely measure agreement with NLCD.", C.accent6, "k2");
card(s, 8.84, 1.45, 3.9, 2.55, "4 labelled dates", "Run 16 had 7; ice and snow scores move with each polygon.", C.accent2, "k3");
card(s, 0.6, 4.25, 6.0, 2.5, "Spatial CV", "Row blocks held out, not dates (as run 16).", C.accent4, "k4");
card(s, 6.75, 4.25, 6.0, 2.5, "No cloud mask", "Cloudy VIIRS days give wrong snow / ice.", C.accent5, "k5");

s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Wrap-up" });
s.addText("Run 16 on VIIRS", { placeholder: "title" });
s.addText(`RF ${f3(RF.rule.accuracy)} on VIIRS vs ${f3(R16.rule.accuracy)} on Landsat (rule labels, same dates); snow is lost at 375 m. `
  + "Next: UNet++ on the same set-up.", { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
