// content.json (from word_content.py) -> Word report
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, ImageRun, PageBreak, LevelFormat, BorderStyle, Footer, PageNumber,
  TableOfContents,
} = require("docx");

const [, , input, output] = process.argv;
const blocks = JSON.parse(fs.readFileSync(input, "utf8"));
const FONT = "Calibri";
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function pngSize(buf, type) {
  if (type === "png") return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
  let i = 2;                                   // jpeg: find SOF marker
  while (i < buf.length) {
    const marker = buf[i + 1], len = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return [buf.readUInt16BE(i + 7), buf.readUInt16BE(i + 5)];
    i += 2 + len;
  }
  throw new Error("no jpeg size");
}

function table(b) {
  const k = b.header.length;
  const widths = b.widths || Array(k).fill(Math.floor(9000 / k));
  const total = widths.reduce((a, c) => a + c, 0);
  const align = (j) => ((b.align && b.align[j] === "r") ? AlignmentType.RIGHT : AlignmentType.LEFT);
  const cell = (text, j, head, bold) => new TableCell({
    borders, width: { size: widths[j], type: WidthType.DXA },
    shading: head ? { fill: "DCE6F1", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: align(j), children: [
      new TextRun({ text: String(text), bold: head || bold, font: FONT, size: 18 })] })],
  });
  const rows = [new TableRow({ tableHeader: true, children: b.header.map((t, j) => cell(t, j, true)) })];
  b.rows.forEach((r) => {
    const bold = /total/i.test(String(r[0]));
    rows.push(new TableRow({ cantSplit: true, children: r.map((t, j) => cell(t, j, false, bold || (j === 0 && b.header[0].includes("predicted")))) }));
  });
  return [new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows }),
          new Paragraph({ spacing: { after: 120 }, children: [] })];
}

const children = [];
for (const b of blocks) {
  if (b.t === "h0") {
    children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun({ text: b.text })] }));
  } else if (b.t.startsWith("h")) {
    const lvl = { h1: HeadingLevel.HEADING_1, h2: HeadingLevel.HEADING_2, h3: HeadingLevel.HEADING_3 }[b.t];
    children.push(new Paragraph({ heading: lvl, keepNext: true, children: [new TextRun({ text: b.text })] }));
    if (b.text === "1. Summary") {
      // table of contents just before the summary
      children.splice(children.length - 1, 0,
        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents")] }),
        new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({ children: [new PageBreak()] }));
    }
  } else if (b.t === "p") {
    children.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: b.text })] }));
  } else if (b.t === "bullets") {
    b.items.forEach((it) => children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 },
      spacing: { after: 60 }, children: [new TextRun({ text: it })] })));
  } else if (b.t === "table") {
    children.push(...table(b));
  } else if (b.t === "image") {
    const type = b.path.endsWith(".png") ? "png" : "jpg";
    const buf = fs.readFileSync(b.path);
    const [w, h] = pngSize(buf, type);
    let width = b.width, height = Math.round(width * h / w);
    if (height > 820) { height = 820; width = Math.round(height * w / h); }
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
      children: [new ImageRun({ type, data: buf, transformation: { width, height } })] }));
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: b.caption, italics: true, size: 18, color: "555555" })] }));
  } else if (b.t === "pagebreak") {
    children.push(new Paragraph({ children: [new PageBreak()] }));
  }
}

const doc = new Document({
  creator: "RiverProject",
  title: "Nenana river ice: RF vs UNet++",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 40, bold: true, color: "1F3864" },
        paragraph: { spacing: { after: 160 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "2E5597" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 21, bold: true, color: "404040" }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 1100, bottom: 1100, left: 1300, right: 1300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "777777" })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(output, buf); console.log("wrote", output); });
