// Run 22 deck (run 19 with Gaussian run 17 labels), pptxgenjs. Same theme as build_deck_viirs.js.
// Numbers: ~/Desktop/nenana_crops/daksh/output/deck_numbers_viirs.json (report/word_content_viirs.py). Run:
//   NODE_PATH=<dir with pptxgenjs, image-size> PPTX_SKILL=<pptx skill dir> node build_deck_viirs.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("image-size");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HOME = process.env.HOME;
const O = `${HOME}/Desktop/nenana_crops/daksh/output`;
const OUT = `${HOME}/Desktop/nenana_river_models/presentation/Nenana_VIIRS_run22.pptx`;
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
pres.title = "Nenana river ice from VIIRS: run 22";
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
const R = N.run22;
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Run 22: Gaussian labels", { placeholder: "title" });
s.addText("Run 19 (river + 1 px buffer, RF / UNet++ / 2D CNN, four states at once) with the labels from run 17's Gaussian "
  + "footprint shares instead of box counts.", { placeholder: "body" });
s = content("Gaussian labels help the strict rule", "Overview");
s.addTable(tableOf(["model", "run 22 rule50 acc / F1", "run 22 rule90", "run 19 rule50", "run 19 rule90"], R.vs19, 13),
  { x: 0.6, y: 1.45, w: 12.1, colW: [1.8, 2.7, 2.5, 2.5, 2.6] });
bullets(s, ["Box labels (run 19): every Landsat pixel inside the VIIRS pixel counts equally.",
  "Gaussian labels (run 22): weights fall off from the pixel centre (sigma 134 m, cut at 2 sigma), as in run 17.",
  "rule90 improves for every model; rule50 about the same.",
  "Totals boxes in the images still use the run 18 box % models."], { x: 0.6, y: 3.6, w: 12.1, h: 3.2 }, 16);
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

riverSection(22, R, "river + buffer, Gaussian labels", "Water mask occurrence >= 0.05 (988 px) plus a 1-pixel buffer (678 px); labels from Gaussian shares.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
