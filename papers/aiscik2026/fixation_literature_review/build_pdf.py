#!/usr/bin/env python3
"""Render the fixation literature review to a polished, linked PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "report-source.md"
OUTPUT = HERE.parents[2] / "output/pdf/llm_agent_fixation_literature_review.pdf"

NAVY = colors.HexColor("#17233D")
BLUE = colors.HexColor("#2C5F8A")
CYAN = colors.HexColor("#49A6B5")
INK = colors.HexColor("#17202A")
MUTED = colors.HexColor("#536273")
LIGHT = colors.HexColor("#EAF1F6")
PALE = colors.HexColor("#F5F8FA")
LINE = colors.HexColor("#CAD6DF")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        ),
        (
            Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
            Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"),
        ),
    ]
    for normal, bold, italic in candidates:
        if all(path.exists() for path in (normal, bold, italic)):
            pdfmetrics.registerFont(TTFont("ReviewSans", str(normal)))
            pdfmetrics.registerFont(TTFont("ReviewSans-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("ReviewSans-Italic", str(italic)))
            pdfmetrics.registerFontFamily(
                "ReviewSans",
                normal="ReviewSans",
                bold="ReviewSans-Bold",
                italic="ReviewSans-Italic",
                boldItalic="ReviewSans-Bold",
            )
            return "ReviewSans", "ReviewSans-Bold", "ReviewSans-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


class ReviewDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=0.72 * inch,
            rightMargin=0.72 * inch,
            topMargin=0.72 * inch,
            bottomMargin=0.66 * inch,
            title="Fixation in LLM Research Agents",
            author="Research synthesis",
            subject="Trajectory lock-in, defixation, memory, and search diversity",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="body",
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=draw_page))

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            if style in {"H1", "H2"}:
                level = 0 if style == "H1" else 1
                text = flowable.getPlainText()
                key = f"heading-{self.page}-{abs(hash(text))}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def draw_page(canvas, doc) -> None:
    page = canvas.getPageNumber()
    canvas.saveState()
    if page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        canvas.setFillColor(CYAN)
        canvas.rect(0, 0, 0.18 * inch, letter[1], fill=1, stroke=0)
    else:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, letter[1] - 0.48 * inch, letter[0] - doc.rightMargin, letter[1] - 0.48 * inch)
        canvas.setFont(FONT_BOLD, 7.2)
        canvas.setFillColor(BLUE)
        canvas.drawString(doc.leftMargin, letter[1] - 0.36 * inch, "FIXATION IN LLM RESEARCH AGENTS")
        canvas.setFont(FONT, 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(letter[0] - doc.rightMargin, 0.34 * inch, f"{page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName=FONT,
    fontSize=9.35,
    leading=12.3,
    textColor=INK,
    alignment=TA_JUSTIFY,
    spaceAfter=6.1,
    allowWidows=0,
    allowOrphans=0,
)
LEAD = ParagraphStyle(
    "Lead",
    parent=BODY,
    fontSize=11.1,
    leading=15,
    textColor=NAVY,
    spaceAfter=11,
)
H1 = ParagraphStyle(
    "H1",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=16,
    leading=19,
    textColor=NAVY,
    spaceBefore=14,
    spaceAfter=7,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=11.8,
    leading=14,
    textColor=BLUE,
    spaceBefore=10,
    spaceAfter=4.5,
    keepWithNext=True,
)
BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    leftIndent=14,
    firstLineIndent=-9,
    bulletIndent=2,
    spaceAfter=3.2,
)
NUMBER = ParagraphStyle(
    "Number",
    parent=BODY,
    leftIndent=18,
    firstLineIndent=-13,
    bulletIndent=0,
    spaceAfter=3.8,
)
SMALL = ParagraphStyle(
    "Small",
    parent=BODY,
    fontSize=7.8,
    leading=9.8,
    spaceAfter=3.4,
    alignment=TA_LEFT,
)
TABLE_HEAD = ParagraphStyle(
    "TableHead",
    parent=SMALL,
    fontName=FONT_BOLD,
    textColor=WHITE,
    leading=9,
)
TABLE_CELL = ParagraphStyle(
    "TableCell",
    parent=SMALL,
    fontSize=7.3,
    leading=8.8,
    textColor=INK,
)
COVER_KICKER = ParagraphStyle(
    "CoverKicker",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=10,
    leading=12,
    textColor=CYAN,
    alignment=TA_LEFT,
    tracking=1.3,
)
COVER_TITLE = ParagraphStyle(
    "CoverTitle",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=31,
    leading=34,
    textColor=WHITE,
    alignment=TA_LEFT,
    spaceAfter=16,
)
COVER_SUB = ParagraphStyle(
    "CoverSub",
    parent=BODY,
    fontSize=15,
    leading=20,
    textColor=colors.HexColor("#D7E6EF"),
    alignment=TA_LEFT,
)
COVER_META = ParagraphStyle(
    "CoverMeta",
    parent=BODY,
    fontSize=9,
    leading=13,
    textColor=colors.HexColor("#AFC8D7"),
    alignment=TA_LEFT,
)


def inline(text: str) -> str:
    value = html.escape(text.strip())
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(
        r"(https?://[^\s<]+)",
        lambda m: f'<link href="{m.group(1).rstrip(".,;)")}" color="#2C5F8A"><u>source</u></link>'
        + m.group(1)[len(m.group(1).rstrip(".,;)")) :],
        value,
    )
    return value


def make_table(rows: list[list[str]], width: float) -> Table:
    ncols = max(len(row) for row in rows)
    rows = [row + [""] * (ncols - len(row)) for row in rows]
    if ncols == 2:
        ratios = [0.31, 0.69]
    elif ncols == 3:
        ratios = [0.25, 0.37, 0.38]
    elif ncols == 4:
        ratios = [0.20, 0.23, 0.31, 0.26]
    else:
        ratios = [1 / ncols] * ncols
    data = []
    for r_index, row in enumerate(rows):
        style = TABLE_HEAD if r_index == 0 else TABLE_CELL
        data.append([Paragraph(inline(cell), style) for cell in row])
    table = Table(
        data,
        colWidths=[width * ratio for ratio in ratios],
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def parse_markdown(text: str, width: float) -> list:
    lines = text.splitlines()
    story = []
    paragraph: list[str] = []
    table_rows: list[list[str]] = []
    first_body_para = True

    def flush_paragraph() -> None:
        nonlocal paragraph, first_body_para
        if paragraph:
            joined = " ".join(item.strip() for item in paragraph)
            style = LEAD if first_body_para else BODY
            story.append(Paragraph(inline(joined), style))
            paragraph = []
            first_body_para = False

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            cleaned = [row for row in table_rows if not all(set(cell) <= {"-", ":", " "} for cell in row)]
            story.append(Spacer(1, 4))
            story.append(make_table(cleaned, width))
            story.append(Spacer(1, 8))
            table_rows = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("# Fixation in LLM Research Agents"):
            continue
        if line.startswith("## A cross-disciplinary evidence map"):
            continue
        if line.startswith("**Research synthesis prepared") or line.startswith("**Scope:"):
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        flush_table()
        if not line.strip():
            flush_paragraph()
            continue
        if line.startswith("## "):
            flush_paragraph()
            title = line[3:].strip()
            if title == "Selected annotated bibliography":
                story.append(PageBreak())
            story.append(Paragraph(inline(title), H1))
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline(line[4:]), H2))
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            flush_paragraph()
            story.append(Paragraph(inline(numbered.group(2)), NUMBER, bulletText=numbered.group(1) + "."))
            continue
        if line.startswith("- "):
            flush_paragraph()
            content = line[2:].strip()
            style = SMALL if content.startswith("**[S") or content.startswith("**[P") else BULLET
            story.append(Paragraph(inline(content), style, bulletText="•"))
            continue
        paragraph.append(line)
    flush_paragraph()
    flush_table()
    return story


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReviewDoc(str(OUTPUT))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName=FONT_BOLD,
            fontSize=9.2,
            leading=12,
            textColor=NAVY,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=3,
        ),
        ParagraphStyle(
            "TOC2",
            fontName=FONT,
            fontSize=8.2,
            leading=10.5,
            textColor=MUTED,
            leftIndent=16,
            firstLineIndent=0,
        ),
    ]

    story = [
        Spacer(1, 0.95 * inch),
        Paragraph("RESEARCH EVIDENCE MAP", COVER_KICKER),
        Spacer(1, 0.28 * inch),
        Paragraph("Fixation in LLM<br/>Research Agents", COVER_TITLE),
        HRFlowable(width="28%", thickness=3, color=CYAN, hAlign="LEFT", spaceBefore=2, spaceAfter=20),
        Paragraph(
            "Trajectory lock-in, defixation, memory, search diversity, and the evidence surrounding state-matched autonomous model-search forks",
            COVER_SUB,
        ),
        Spacer(1, 2.2 * inch),
        Paragraph("CROSS-DISCIPLINARY SYNTHESIS", COVER_KICKER),
        Spacer(1, 7),
        Paragraph(
            "Prepared 3 September 2026<br/>Primary literature through 2 September 2026<br/>Includes an annotated bibliography and project-specific experiment agenda",
            COVER_META,
        ),
        PageBreak(),
        Paragraph("How to read this report", H1),
        Paragraph(
            "The report separates evidence that directly studies longitudinal LLM agents from shorter LLM decision experiments, human design-fixation analogues, and evolutionary-search analogies. Source IDs [S01-S50] correspond to the annotated bibliography. [P2] denotes the repository's exact state-matched Tiny Addition analysis. Recent arXiv-only findings are labeled as preliminary.",
            LEAD,
        ),
        KeepTogether(
            [
                Paragraph("Evidence hierarchy", H2),
                make_table(
                    [
                        ["Level", "Interpretation"],
                        ["Direct", "Longitudinal LLM/agent behavior or an exact project fork"],
                        ["Near-direct", "Controlled LLM mechanism likely to operate in trajectories"],
                        ["Analogue", "Human design or algorithmic search mechanism; hypothesis source, not proof"],
                    ],
                    doc.width,
                ),
            ]
        ),
        Spacer(1, 8),
        Paragraph("Contents", H1),
        toc,
        PageBreak(),
    ]
    story.extend(parse_markdown(SOURCE.read_text(encoding="utf-8"), doc.width))
    doc.multiBuild(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
