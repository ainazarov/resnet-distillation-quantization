"""Typeset the Russian Markdown report, experiment figures and Android screenshots."""

import re
from shutil import copyfile
from html import escape
from io import BytesIO
from pathlib import Path
import os

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, Fit, NameObject,
    NumberObject, RectangleObject, TextStringObject,
)
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf"
SOURCE = ROOT / "report/HW1_Food101_v5.md"
OUT = OUTPUT / "HW1_Food101_v5.pdf"
FIG = ROOT / "report/figures"
W, H = A4
M = 44
BODY = W - 2 * M
INK, SUB, TEAL = "#162C3C", "#435B65", "#007365"
PALE, GRID, LIGHT = "#E9F3F0", "#CEDBD9", "#F4F7F7"
FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
for name, file in [("UI", "segoeui.ttf"), ("UIB", "seguisb.ttf"), ("Serif", "georgia.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FONTS / file))
pdfmetrics.registerFontFamily("UI", normal="UI", bold="UIB")

TABLE_WIDTHS = {
    "Модель": (.56, .22, .22),
    "Среднее время этапа на Galaxy S24+, мс": (.68, .16, .16),
    "Что меняли": (.45, .55),
}


def sections(source):
    """Read the headings and consecutive body blocks in source order."""
    parts = re.split(r"(?m)^## (.+)$", source)
    if parts[0].strip() or len(parts) < 3:
        raise ValueError("The report must start with a level-two section heading")
    return [(parts[i].strip(), re.split(r"\n\s*\n", parts[i+1].strip()))
            for i in range(1, len(parts), 2)]


def inline(text):
    text = escape(text, quote=False)
    # Bind digit groups and their measurement units in running Russian prose.
    text = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "\u00a0", text)
    text = re.sub(r"(?<=\d) (?=(?:мс|МиБ|МБ|ГиБ|п\. п\.)\b)", "\u00a0", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                  rf'<link href="\2" color="{TEAL}"><u>\1</u></link>', text)
    # Typeset the authors' plain-text probability subscripts as mathematical indices.
    text = re.sub(r"p_(учителя|ученика)", r"p<sub>\1</sub>", text)
    return text.replace(chr(96), "")


def paragraph(text, size, leading, color=INK, font="UI", align=TA_LEFT):
    style = ParagraphStyle("text", fontName=font, fontSize=size, leading=leading,
                           textColor=HexColor(color), alignment=align, splitLongWords=0)
    return Paragraph(text, style)


class Report:
    def __init__(self, page_count):
        self.stream = BytesIO()
        self.c = Canvas(self.stream, pagesize=A4, pageCompression=True)
        self.page_count = page_count
        self.figures = []
        self.titles = []
        self.attachment_links = {}

    def page(self):
        c = self.c
        c.setStrokeColor(HexColor(GRID))
        c.line(M, H - 40, W-M, H-40)
        c.setFillColor(HexColor(TEAL))
        c.setFont("UIB", 8.1)
        c.drawString(M, H-31, "НИУ ВШЭ   /   ДИСТИЛЛЯЦИЯ И УСКОРЕНИЕ МОДЕЛЕЙ")
        c.setFillColor(HexColor(SUB))
        c.drawRightString(W-M, H-31, "ДОМАШНЕЕ ЗАДАНИЕ № 1   /   V5")
        self.y = 56

    def heading(self, title):
        if self.y > 56:
            self.y += 12
        self.titles.append((self.c.getPageNumber()-1, title, H-self.y+12))
        self.txt(escape(title), 25.5, 30.5, font="Serif", gap=12)

    def advance(self, height, gap):
        self.ink_bottom = self.y + height
        self.y = self.ink_bottom + gap

    def txt(self, text, size=11.6, lead=17.0, color=INK, font="UI", gap=11):
        p = paragraph(text, size, lead, color, font)
        _, height = p.wrap(BODY, 800)
        p.drawOn(self.c, M, H-self.y-height)
        self.advance(height, gap)

    def box(self, text):
        p = paragraph(text, 11.6, 17)
        _, height = p.wrap(BODY-24, 800)
        self.c.setFillColor(HexColor(PALE))
        self.c.roundRect(M, H-self.y-height-18, BODY, height+18, 5, fill=1, stroke=0)
        p.drawOn(self.c, M+12, H-self.y-height-9)
        self.advance(height + 18, 11)

    def table(self, markdown):
        lines = [[cell.strip() for cell in line.strip("|").split("|")]
                 for line in markdown.splitlines()]
        alignments = [TA_RIGHT if col.endswith(":") else TA_LEFT for col in lines[1]]
        header = lines[0][0]
        size = 9.9 if header == "Что меняли" else 10.2
        data = []
        for i, row in enumerate([lines[0], *lines[2:]]):
            color, font = ("#FFFFFF", "UIB") if i == 0 else (INK, "UI")
            data.append([paragraph(inline(cell), size, size+3.5, color, font, align)
                         for cell, align in zip(row, alignments, strict=True)])
        table = Table(data, colWidths=[BODY*w for w in TABLE_WIDTHS[header]])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor(INK)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor(LIGHT)]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, -1), (-1, -1), .7, HexColor(GRID)),
        ]))
        _, height = table.wrap(BODY, 800)
        table.drawOn(self.c, M, H-self.y-height)
        self.advance(height, 10)

    def screenshots(self, markdown):
        items = [re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", line).groups()
                 for line in markdown.splitlines()]
        gap = 8
        width = (BODY - gap * (len(items) - 1)) / len(items)
        height = .80 * width
        bottom = H - self.y - height
        caption_heights = []
        for i, (label, filename) in enumerate(items):
            x = M + i * (width + gap)
            image = ImageReader(str(SOURCE.parent / filename))
            iw, ih = image.getSize()
            # The result panel starts below Android's status bar and occupies
            # the upper part of all three screenshots. Clip only in the layout.
            self.c.saveState()
            frame = self.c.beginPath()
            frame.rect(x, bottom, width, height)
            self.c.clipPath(frame, stroke=0, fill=0)
            self.c.drawImage(image, x, H-self.y+.10*width-ih*width/iw,
                             width=width, height=ih*width/iw)
            self.c.restoreState()
            self.c.setStrokeColor(HexColor(GRID))
            self.c.setLineWidth(.6)
            self.c.rect(x, bottom, width, height, stroke=1, fill=0)
            caption = paragraph(inline(label), 10.1, 13.2, font="UIB")
            _, label_height = caption.wrap(width, H)
            caption.drawOn(self.c, x, bottom-6-label_height)
            caption_heights.append(label_height)
        self.advance(height + 6 + max(caption_heights), 17)

    def fig(self, spec):
        stem, caption = re.fullmatch(r"\[Рисунок: ([\w-]+)(?:\. Подпись: «(.+)»\.)?\]", spec).groups()
        f = PdfReader(FIG / f"{stem}.pdf").pages[0]
        width = BODY * (.92 if stem == "architecture-stages-v5" else 1)
        scale = width / float(f.mediabox.width)
        height = float(f.mediabox.height) * scale
        # Place the drawing invocation inline, so text extraction follows the
        # same order as the visible diagrams, captions and body paragraphs.
        key = NameObject(f"/ReportFigure{len(self.figures)+1}")
        self.c.saveState()
        self.c.translate(M+(BODY-width)/2, H-self.y-height)
        self.c.scale(scale, scale)
        self.c.addLiteral(f"{key} Do")
        self.c.restoreState()
        self.figures.append((self.c.getPageNumber()-1, key, f))
        self.advance(height, 5)
        if caption:
            self.txt(inline(caption), 9.0, 12.8, SUB,
                     gap=8 if stem in {"architecture-stages-v5", "conv-groups-v5"} else 13)

    def attachment_link(self, filename, label):
        rect = (M, H-self.y-16, M+pdfmetrics.stringWidth(label, "UI", 9.6), H-self.y)
        self.attachment_links[filename] = (self.c.getPageNumber()-1, rect)
        self.txt(f"<u>{label}</u>", 9.6, 14, TEAL, gap=2)

    def finish_page(self):
        if self.ink_bottom > H-36:
            raise ValueError(f"Page {self.c.getPageNumber()} exceeds body by {self.ink_bottom-H+36:.1f} pt")
        c = self.c
        c.setStrokeColor(HexColor(GRID))
        c.line(M, 29, W-M, 29)
        c.setFillColor(HexColor(SUB))
        c.setFont("UI", 8)
        c.drawString(M, 16, "Food-101   /   Galaxy S24+")
        c.drawRightString(W-M, 16, f"{c.getPageNumber()} / {self.page_count}")
        c.showPage()

    def save(self):
        self.c.save()
        self.stream.seek(0)
        w = PdfWriter()
        w.append(PdfReader(self.stream))
        for page, key, figure in self.figures:
            form = DecodedStreamObject()
            form.set_data(figure.get_contents().get_data())
            form.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Form"),
                         NameObject("/BBox"): RectangleObject(figure.mediabox),
                         NameObject("/Resources"): figure["/Resources"].clone(w)})
            if "/Group" in figure:
                form[NameObject("/Group")] = figure["/Group"].clone(w)
            resources = w.pages[page]["/Resources"]
            if "/XObject" not in resources:
                resources[NameObject("/XObject")] = DictionaryObject()
            resources["/XObject"][key] = w._add_object(form.flate_encode())
        for page, title, top in self.titles:
            w.add_outline_item(title, page, fit=Fit.fit_horizontally(top))
        for name, path in [("Android-README.md", ROOT / "android/README.md"),
                           ("README.md", ROOT / "README.md"),
                           ("layers.csv", ROOT / "results/pilot/layers/layers.csv"),
                           ("components.csv", ROOT / "results/pilot/components/components.csv"),
                           ("stages.csv", ROOT / "results/pilot/stages/stages.csv")]:
            file = w.add_attachment(name, path.read_bytes())
            if name in self.attachment_links:
                page, rect = self.attachment_links[name]
                add_attachment_link(w, file.pdf_object, page, rect)
        w.add_metadata({"/Title": self.titles[0][1], "/Subject": "Дистилляция и ускорение моделей · Домашнее задание № 1",
                        "/Keywords": "Food-101, ResNet, Galaxy S24+, XNNPACK, INT8"})
        with OUT.open("wb") as dest:
            w.write(dest)
        main_report = ROOT / "HW1_Report_Nazarov.pdf"
        copyfile(OUT, main_report)
        print(main_report)


def add_attachment_link(writer, file_spec, page_index, rectangle):
    # Reuse the embedded file's FileSpec. The page already paints the underlined
    # label, so its activation rectangle gets an empty normal appearance.
    left, bottom, right, top = rectangle
    appearance = DecodedStreamObject()
    appearance.set_data(b"")
    appearance.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Form"),
                       NameObject("/BBox"): RectangleObject((0, 0, right-left, top-bottom)),
                       NameObject("/Resources"): DictionaryObject()})
    annotation = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/FileAttachment"),
        NameObject("/Rect"): RectangleObject(rectangle), NameObject("/FS"): file_spec.indirect_reference,
        NameObject("/Contents"): TextStringObject(str(file_spec["/F"])), NameObject("/F"): NumberObject(4),
        NameObject("/AP"): DictionaryObject({NameObject("/N"): writer._add_object(appearance)}),
    })
    page = writer.pages[page_index]
    if "/Annots" not in page:
        page[NameObject("/Annots")] = ArrayObject()
    page["/Annots"].append(writer._add_object(annotation))


def block_groups(blocks):
    """Keep the explanation of a table or convolution chart with the visual."""
    remaining = iter(blocks)
    for block in remaining:
        group = [block]
        if block.startswith(("|", "[Рисунок: conv-groups-")):
            following = next(remaining, None)
            if following is not None:
                group.append(following)
        yield group


def draw_group(r, group):
    for block in group:
        if block.startswith("[Рисунок:"):
            r.fig(block)
        elif block.startswith("!["):
            r.screenshots(block)
        elif block.startswith("|"):
            r.table(block)
        elif block.startswith("¹"):
            r.txt(inline(block), 9.1, 12.1, SUB, gap=6)
        elif block.startswith("**Функция потерь."):
            r.box(inline(block))
        elif block.startswith("**Источники и код."):
            r.txt(inline(block), 9.6, 14.2, gap=4)
            r.attachment_link("README.md", "README проекта (вложение)")
        else:
            r.txt(inline(block))
    if group[0].startswith("[Рисунок: conv-groups-"):
        r.attachment_link("layers.csv", "Все 73 слоя: имена, вызовы и время (вложенный CSV)")
        r.attachment_link("components.csv", "Время частей сети и полного прохода (вложенный CSV)")


def paginate(contents):
    # Use the same rendering methods to measure groups and to paint them. A
    # caption and its drawing move together when a section needs another page.
    measure = Report(0)
    pages = []
    continuation = False
    y = 56
    for title, blocks in contents:
        for group in block_groups(blocks):
            measure.y = 0
            if title:
                measure.heading(title)
            draw_group(measure, group)
            height = measure.y
            heading_gap = 12 if title and y > 56 else 0
            if not pages or (title and not continuation) or y + heading_gap + measure.ink_bottom > H-36:
                pages.append([])
                y = 56
                heading_gap = 0
                continuation = title is None
            if y + heading_gap + measure.ink_bottom > H-36:
                raise ValueError(f"Block in {title!r} is taller than an A4 page")
            pages[-1].append((title, group))
            y += heading_gap + height
            if title:
                continuation = False
                title = None
    return pages


def render():
    pages = paginate(sections(SOURCE.read_text(encoding="utf-8-sig")))
    r = Report(len(pages))
    for page in pages:
        r.page()
        for title, group in page:
            if title:
                r.heading(title)
            draw_group(r, group)
        r.finish_page()
    r.save()


if __name__ == "__main__":
    render()
