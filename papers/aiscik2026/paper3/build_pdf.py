#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the AISciK-ready Paper 3 PDF in an eight-page NeurIPS-like layout."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
OUTPUT = HERE.parents[2] / "output/pdf/paper3_population_memory.pdf"
MAIN_PAGES = 8


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


AGG = json.loads((DERIVED / "aggregate.json").read_text(encoding="utf-8"))
SUMMARIES = {
    (row["task"], row["memory_system"], row["prompt_policy"]): row
    for row in rows(DERIVED / "system_prompt_summaries.csv")
}
CONTRASTS = {
    (row["task"], row["architecture"], row["prompt_policy"], row["metric"]): row
    for row in rows(DERIVED / "paired_contrasts.csv")
}
EXEMPLARS = rows(DERIVED / "alternative_branch_improvements.csv")


def f(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def signed(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def contrast(task: str, prompt: str, metric: str) -> dict[str, str]:
    return CONTRASTS[(task, "greedy", prompt, metric)]


def ci(row: dict[str, str], *, scale: float = 1.0, digits: int = 3) -> str:
    low = scale * float(row["bootstrap_low"])
    high = scale * float(row["bootstrap_high"])
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def summary(task: str, system: str, prompt: str) -> dict[str, str]:
    return SUMMARIES[(task, system, prompt)]


def register_fonts() -> tuple[str, str, str, str]:
    font_dir = Path("/System/Library/Fonts/Supplemental")
    paths = {
        "AISciKTimes": font_dir / "Times New Roman.ttf",
        "AISciKTimes-Bold": font_dir / "Times New Roman Bold.ttf",
        "AISciKTimes-Italic": font_dir / "Times New Roman Italic.ttf",
        "AISciKTimes-BoldItalic": font_dir / "Times New Roman Bold Italic.ttf",
    }
    if all(path.exists() for path in paths.values()):
        for name, path in paths.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "AISciKTimes",
            normal="AISciKTimes",
            bold="AISciKTimes-Bold",
            italic="AISciKTimes-Italic",
            boldItalic="AISciKTimes-BoldItalic",
        )
        return (
            "AISciKTimes",
            "AISciKTimes-Bold",
            "AISciKTimes-Italic",
            "AISciKTimes-BoldItalic",
        )
    return "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"


FONT, FONT_BOLD, FONT_ITALIC, FONT_BOLD_ITALIC = register_fonts()


class NeuripsDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=1.5 * inch,
            rightMargin=1.0 * inch,
            topMargin=0.72 * inch,
            bottomMargin=0.72 * inch,
            title="Pluralism Without a Free Lunch",
            author="Anonymous Authors",
            subject="AISciK Workshop (NeurIPS 2026) submission",
        )
        frame = Frame(
            1.5 * inch,
            0.6 * inch,
            5.5 * inch,
            9.0 * inch,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="main",
        )
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=draw_page))


def draw_page(canvas, doc) -> None:
    page = canvas.getPageNumber()
    canvas.saveState()
    canvas.setFont(FONT, 7)
    canvas.setFillColor(colors.HexColor("#4b5563"))
    canvas.drawCentredString(letter[0] / 2, 0.28 * inch, str(page))
    canvas.setFont(FONT, 4.8)
    canvas.setFillColor(colors.HexColor("#9ca3af"))
    line_start = (page - 1) * 55 + 1
    for index in range(55):
        y = 9.47 * inch - index * (9.0 * inch / 54)
        canvas.drawRightString(1.42 * inch, y, str(line_start + index))
    if page == 1:
        canvas.setFont(FONT, 7.3)
        canvas.setFillColor(colors.black)
        canvas.drawString(
            1.5 * inch,
            0.28 * inch,
            "Submitted to the AISciK Workshop (NeurIPS 2026).",
        )
    canvas.restoreState()


styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName=FONT,
    fontSize=9.65,
    leading=10.7,
    alignment=TA_JUSTIFY,
    spaceAfter=4.7,
    allowWidows=0,
    allowOrphans=0,
)
SMALL = ParagraphStyle(
    "Small",
    parent=BODY,
    fontSize=8.15,
    leading=9.1,
    spaceAfter=3.0,
)
TINY = ParagraphStyle(
    "Tiny",
    parent=BODY,
    fontSize=7.15,
    leading=8.0,
    spaceAfter=2.3,
)
H1 = ParagraphStyle(
    "H1",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=11.8,
    leading=12.8,
    alignment=TA_LEFT,
    spaceBefore=5,
    spaceAfter=3.7,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=9.9,
    leading=10.9,
    alignment=TA_LEFT,
    spaceBefore=3.7,
    spaceAfter=2.2,
    keepWithNext=True,
)
TITLE = ParagraphStyle(
    "Title",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=17,
    leading=19,
    alignment=TA_CENTER,
    spaceBefore=5,
    spaceAfter=7,
)
AUTHOR = ParagraphStyle(
    "Author",
    parent=BODY,
    fontSize=10,
    leading=11,
    alignment=TA_CENTER,
    spaceAfter=5,
)
ABSTRACT = ParagraphStyle(
    "Abstract",
    parent=BODY,
    fontSize=9.2,
    leading=10.2,
    leftIndent=0.42 * inch,
    rightIndent=0.42 * inch,
    spaceAfter=4.5,
)
ABSTRACT_HEAD = ParagraphStyle(
    "AbstractHead",
    parent=H1,
    alignment=TA_CENTER,
    fontSize=10.8,
    spaceBefore=2,
    spaceAfter=3,
)
CAPTION = ParagraphStyle(
    "Caption",
    parent=SMALL,
    fontSize=7.55,
    leading=8.45,
    alignment=TA_JUSTIFY,
    spaceBefore=2,
    spaceAfter=3.5,
)
BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    leftIndent=12,
    firstLineIndent=-7,
    bulletIndent=2,
    spaceAfter=2.4,
)
QUOTE = ParagraphStyle(
    "Quote",
    parent=SMALL,
    leftIndent=11,
    rightIndent=9,
    backColor=colors.HexColor("#f3f4f6"),
    borderPadding=4,
    spaceBefore=2,
    spaceAfter=4,
)
REF = ParagraphStyle(
    "Reference",
    parent=SMALL,
    fontSize=7.9,
    leading=8.8,
    leftIndent=10,
    firstLineIndent=-10,
    spaceAfter=3.0,
)


def P(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def h1(text: str) -> Paragraph:
    return Paragraph(text, H1)


def h2(text: str) -> Paragraph:
    return Paragraph(text, H2)


def bullet(text: str) -> Paragraph:
    return Paragraph("- " + text, BULLET)


def pagebreak(story: list[Flowable]) -> None:
    story.append(PageBreak())


def ruled_title(story: list[Flowable]) -> None:
    story.append(HRFlowable(width="100%", thickness=4, color=colors.black, spaceAfter=10))
    story.append(P(
        "Pluralism Without a Free Lunch<br/>"
        "<font size='13'>A Lineage Audit of Population Memory in Autonomous ML Research</font>",
        TITLE,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=2, spaceAfter=9))
    story.append(P("Anonymous Authors", AUTHOR))


def data_table(
    data: list[list[Any]],
    widths: list[float],
    *,
    header_rows: int = 1,
    font_size: float = 7.1,
) -> Table:
    table = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, header_rows - 1), FONT_BOLD),
        ("FONTNAME", (0, header_rows), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.05),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e5e7eb")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3),
    ]))
    return table


def figure(path: Path, width: float, caption: str) -> list[Flowable]:
    image_width, image_height = ImageReader(str(path)).getSize()
    height = width * image_height / image_width
    image = Image(str(path), width=width, height=height)
    image.hAlign = "CENTER"
    return [image, P(caption, CAPTION)]


def build_story() -> list[Flowable]:
    story: list[Flowable] = []

    # Main page 1: title, abstract, question, and contribution.
    ruled_title(story)
    story.append(P("Abstract", ABSTRACT_HEAD))
    story.append(P(
        "Autonomous research agents need memory, but memory also decides which failed ideas survive long enough to become possible stepping stones. "
        "We ask how recorded search behavior differs under a one-incumbent controller, a four-lineage portfolio policy, and a native island population. "
        "We reconstruct 8,480 LLM proposals from 84 trajectories across transformer compression and Fashion-MNIST. In the controlled greedy search system, "
        "the K=4 selector mechanically allocated work across 3.97-3.99 effective top-level branches versus 1.00 under K=1 and chose a non-incumbent parent on 73-75% of proposals. "
        "Agents used literal reference/alternative-design phrases in 7.5-26.2% of portfolio messages versus at most 0.3% of single-incumbent messages. Yet "
        "yield did not transport: K=4-minus-K=1 normalized endpoint contrasts were 0.044 [-0.033, 0.131] under ordinary prompting and 0.010 [-0.028, 0.055] "
        "under assumption challenges, while Fashion-MNIST differences were slightly negative. Portfolio context cost 25-38% more subject-agent tokens per proposal. "
        "A native OpenEvolve stress test maintained 27-29 candidates across nearly five occupied islands and repeatedly revived old parents, but it also exposed "
        "multiple designs in 80-82% of nominal single-incumbent proposals, collapsing the intended factor. These population-memory systems therefore implement procedural lineage diversity under our operationalization, "
        "not a free performance gain. We propose lineage, attribution, cost, and prompt-composition audits as epistemic guardrails for AI scientist evaluations.",
        ABSTRACT,
    ))
    story.append(h1("1  Introduction"))
    story.append(P(
        "Scientific communities preserve competing programs because a currently weak line may contain a later stepping stone. The division-of-cognitive-labor tradition "
        "therefore treats diversity as an epistemic institution, not merely a count of people [1,2]. Autonomous scientific agents instantiate the same institutional choice "
        "in software: a controller can retain one best design, a small portfolio, or an island-based population. Those choices determine what evidence is visible, what source "
        "becomes the next parent, and which failures remain available for reconsideration. Systems such as AlphaEvolve, OpenEvolve, LoongFlow, and EvoScientist make such "
        "memory central to long-horizon discovery [8-11], but published capability results rarely audit the final visible population, parent use, attribution record, cost, and outcome together.",
    ))
    story.append(P(
        "AISciK asks how agent evaluations can measure epistemic properties and prevent homogenization of research questions, methods, and hypotheses [17]. We study the AI "
        "research process itself. Our unit is a trajectory, not a proposal treated as independent. Our claim is behavioral: a lineage is an executable parent-descendant chain, "
        "not private reasoning; pluralism means multiple such chains remain selectable. We do not claim globally novel science or causal effects from nonrandomized framework comparisons.",
    ))
    story.append(h2("Contributions"))
    story.append(bullet("A trace-level audit of research pluralism: effective selected branches, non-incumbent parent use, dormant-parent revival, literal alternative phrases, recorded task yield, and token cost."))
    story.append(bullet("A descriptive K=1 versus K=4 whole-system controller contrast across two tasks, reported within prompt policy rather than pooled across it."))
    story.append(bullet("A compositional audit showing that a native population engine can override a nominal memory factor, plus source-audited cases in which old branches preceded large recorded improvements."))
    pagebreak(story)

    # Main page 2: related work and conceptual frame.
    story.append(h1("2  Related work and conceptual frame"))
    story.append(h2("Epistemic diversity and path dependence"))
    story.append(P(
        "Kitcher models scientific labor as allocation across competing approaches [1]. Zollman's network models show a benefit to transient diversity: rapid consensus can "
        "eliminate a promising alternative before its evidence matures [2]. Organizational learning likewise separates exploitation of known returns from exploration of uncertain "
        "alternatives [3], while network structure changes the balance [4]. These accounts do not imply that more alternatives always improve output. They predict a mechanism: "
        "alternatives must survive, remain reachable, and sometimes re-enter active inquiry. Our lineage measures translate that mechanism into auditable agent traces.",
    ))
    story.append(h2("Divergent and quality-diversity search"))
    story.append(P(
        "Novelty search can escape deceptive objectives by rewarding behavioral difference [5]. MAP-Elites stores high-quality solutions across chosen behavior niches [6], and the "
        "quality-diversity literature emphasizes that container, selection, and mutation policies jointly define the search process [7]. A population is thus not a neutral cache. "
        "Its feature map, parent sampler, migration, prompt renderer, and retention rule are institutional rules governing epistemic attention.",
    ))
    story.append(h2("Autonomous research agents"))
    story.append(P(
        "The AI Scientist and its successor automate hypothesis, experiment, and writing loops, with the latter using progressive tree search [12,13]. MLAgentBench and recent "
        "research-agent evaluations treat experimental iteration as a search policy over code and evidence [14,15]. AlphaEvolve couples LLM code edits to evaluator selection [8]. "
        "OpenEvolve operationalizes MAP-Elites, islands, archives, migration, and inspiration programs [9]. LoongFlow combines multi-island and MAP-Elites memory [10]; EvoScientist "
        "distills successful and unsuccessful interactions into persistent memories [11]. These systems "
        "motivate memory as capability. We instead ask what its recorded search institution does to epistemic diversity, attribution, cost, and yield.",
    ))
    story.append(h2("Evaluation gap"))
    story.append(P(
        "A best-score curve cannot distinguish plural search from a single lucky chain. Conversely, branch count alone can reward ceremonial diversity that the agent never uses. "
        "Construct-validity and benchmark-audit work cautions that labels should not outrun observable mechanisms [16,18]. We therefore require four linked layers: <b>availability</b> "
        "(what branches are shown), <b>selection</b> (which parent is edited), <b>lexical record</b> (whether messages name alternatives), and <b>consequence</b> (recorded evaluation, yield, and cost).",
    ))
    closest = [
        ["Closest line", "Population / memory role", "Unresolved evaluation question"],
        ["AlphaEvolve; OpenEvolve [8,9]", "evaluator-selected code populations", "lineage use, attribution, composed-factor semantics"],
        ["LoongFlow; EvoScientist [10,11]", "islands, MAP-Elites, distilled memory", "controlled memory contrast and token cost"],
        ["AI Scientist; MLAgentBench [12-15]", "trajectory search and experiment loops", "executable branch survival and revival"],
        ["This audit", "K=1, K=4, and native populations", "availability, selection, lexical record, outcome, cost"],
    ]
    story.append(P(
        "Table 1: Closest systems and the trace-level gap addressed here. The contribution is an audit linkage and composition test, not a new search algorithm.",
        CAPTION,
    ))
    story.append(data_table(closest, [105, 128, 163], font_size=6.2))
    story.append(P(
        "<b>RQ.</b> Under the observed one-incumbent and four-lineage controller policies, how do (i) selected branch allocation, (ii) parent use and the lexical attribution record, "
        "(iii) recorded task progress, and (iv) resource cost differ across tasks; and what factor-validity failure appears when the controller is composed with a native population adapter?",
        QUOTE,
    ))
    pagebreak(story)

    # Main page 3: data, systems, measures, inference.
    story.append(h1("3  Data and methods"))
    story.append(h2("3.1  Corpus and task strata"))
    systems_table = [
        ["Stratum", "Runs", "Horizon", "Search memory", "Objective"],
        ["Tiny / greedy", "32", "70", "K=1 or K=4", "fewest params at >=99%"],
        ["Tiny / native", "32", "70", "5-island population", "fewest params at >=99%"],
        ["Fashion / greedy", "20", "200", "K=1 or K=4", "validation score"],
    ]
    story.append(P(
        "Table 2: Analysis strata. Tiny uses the largest common completed prefix across all 64 trajectories; later events are right-censored. Fashion-MNIST is complete.",
        CAPTION,
    ))
    story.append(data_table(systems_table, [80, 42, 48, 105, 121], font_size=6.9))
    story.append(P(
        "All 84 trajectories use GPT-5.6 Sol at xhigh reasoning and protected task evaluators. The Tiny task asks a learned causal transformer to add two four-digit numbers; "
        "the common qualified baseline has 21,952 learned parameters. Fashion-MNIST starts at validation score 8,928.37. C0/C1 have nominal single memory and C2/C3 nominal K=4 "
        "memory; C1/C3 additionally receive assumption challenges every tenth opportunity. Tiny C0/C1 and C2/C3 share literal proposals 1-9 before forking, but K=1 and K=4 do "
        "not share a prefix with each other. We keep prompt policy as a stratum because it changes search behavior and is not the present paper's treatment.",
    ))
    story.append(h2("3.2  Memory systems"))
    story.append(P(
        "The <b>greedy K=1</b> controller exposes and edits only the current incumbent. <b>Greedy K=4</b> maintains four qualified parent-descendant lineages. A frozen selector "
        "fills slots from the seed, then chooses the least-selected lineage with deterministic fitness, age, and identifier tie breaks; a strict improvement replaces the selected "
        "lineage. This is a descriptive whole-system portfolio contrast, including longer prompts and the deterministic fairness selector. The <b>native population</b> delegates parent sampling, valid-candidate admission, "
        "MAP-Elites retention, five islands, inspirations, and migration to the vendored OpenEvolve ProgramDatabase. Because external native selection supplies the actual visible set, "
        "its C0-C3 labels are not treated as K=1/K=4 memory assignments.",
    ))
    story.append(h2("3.3  Measures"))
    story.append(P(
        "For every proposal we link the selected parent, candidate, incumbent before/after, evaluation, retained population, agent-authored mechanism/hypothesis/evidence, token increment, "
        "and source snapshot. A top-level lineage is the first child below the common seed in a parent graph. We report exp(Shannon entropy) over selected lineages after opportunity 9, "
        "the non-incumbent parent rate, exact-parent and lineage reactivations after gaps of at least 10 proposals, and the share of strict global improvements descending from a non-incumbent "
        "parent. Lexical-record proxies are explicit reference-design attribution, comparative-language use, and numeric evidence in the agent's final evidence field. They are observable reporting "
        "behaviors, not private cognition.",
    ))
    story.append(P(
        "Tiny yield is the fraction of baseline parameters eliminated by the qualified incumbent; Fashion yield is score gain divided by the baseline score. We also average this gain across "
        "the trajectory (AUC), record stagnation since the last strict improvement, and sum subject-agent tokens. The trajectory is the unit of analysis. K=4-minus-K=1 contrasts are paired by "
        "block within prompt policy; 95% percentile ranges resample blocks 20,000 times. Because labels were protocol-assigned rather than randomized, ranges are descriptive sensitivity "
        "intervals, not randomization-based confidence intervals or p-values [19].",
        SMALL,
    ))
    pagebreak(story)

    # Main page 4: structural effects and message audit.
    story.append(h1("4  The controllers instantiate different branch allocation"))
    story.extend(figure(
        DERIVED / "fig1_system_summary.png",
        5.35 * inch,
        "Figure 1: Tiny AdderBoard system summaries at the common 70-proposal horizon. Small points are trajectories; dark points and bars are block-bootstrap means and 95% percentile ranges. "
        "Native conditions are pooled by prompt policy because the external population supersedes nominal K labels.",
    ))
    story.append(h2("4.1  Branch allocation is a manipulation check"))
    story.append(P(
        "Across both tasks and prompt policies, K=1 selected exactly one top-level branch. The least-selected-lineage rule mechanically made K=4 allocation nearly balanced: "
        f"{f(summary('tiny_adderboard','greedy_portfolio','ordinary')['effective_top_lineages_mean'],2)}-"
        f"{f(summary('tiny_adderboard','greedy_portfolio','challenge')['effective_top_lineages_mean'],2)} effective selected branches on Tiny and "
        f"{f(summary('fashion_mnist','greedy_portfolio','ordinary')['effective_top_lineages_mean'],2)}-"
        f"{f(summary('fashion_mnist','greedy_portfolio','challenge')['effective_top_lineages_mean'],2)} on Fashion. These values validate controller execution; they are not an independent discovery. K=4 selected a parent other than the global incumbent on "
        f"{pct(summary('tiny_adderboard','greedy_portfolio','ordinary')['alternative_parent_rate_mean'])} of Tiny proposals and "
        f"{pct(summary('fashion_mnist','greedy_portfolio','ordinary')['alternative_parent_rate_mean'])} of Fashion proposals. Thus the portfolio was not a display-only archive: it repeatedly "
        "allocated evaluations to non-leading chains.",
    ))
    story.append(P(
        "The four branch roots were not byte-identical variants. Mean pairwise Jaccard distance between their normalized source-delta sets was "
        f"{f(summary('tiny_adderboard','greedy_portfolio','ordinary')['branch_delta_diversity_mean'],2)} ordinary and "
        f"{f(summary('tiny_adderboard','greedy_portfolio','challenge')['branch_delta_diversity_mean'],2)} challenge on Tiny, and "
        f"{f(summary('fashion_mnist','greedy_portfolio','ordinary')['branch_delta_diversity_mean'],2)} and "
        f"{f(summary('fashion_mnist','greedy_portfolio','challenge')['branch_delta_diversity_mean'],2)} on Fashion. This confirms distinct executable edits, not semantically independent research programs.",
        SMALL,
    ))
    story.append(h2("4.2  Agent messages leave a partial lexical record"))
    story.append(P(
        "We inspected every final evidence message with a strict literal-phrase rule (`reference design`, `alternative design`, or `available design`). Under ordinary prompting, phrase use was "
        f"{pct(summary('tiny_adderboard','greedy_single','ordinary')['reference_attribution_rate_mean'])} to "
        f"{pct(summary('tiny_adderboard','greedy_portfolio','ordinary')['reference_attribution_rate_mean'])} on Tiny and from "
        f"{pct(summary('fashion_mnist','greedy_single','ordinary')['reference_attribution_rate_mean'])} to "
        f"{pct(summary('fashion_mnist','greedy_portfolio','ordinary')['reference_attribution_rate_mean'])} on Fashion. Under challenges the corresponding portfolio rates were "
        f"{pct(summary('tiny_adderboard','greedy_portfolio','challenge')['reference_attribution_rate_mean'])} and "
        f"{pct(summary('fashion_mnist','greedy_portfolio','challenge')['reference_attribution_rate_mean'])}. A broader rule allowing any use of `reference` yields "
        f"{pct(summary('tiny_adderboard','greedy_portfolio','ordinary')['broad_reference_attribution_rate_mean'])}/"
        f"{pct(summary('tiny_adderboard','greedy_portfolio','challenge')['broad_reference_attribution_rate_mean'])} on Tiny and "
        f"{pct(summary('fashion_mnist','greedy_portfolio','ordinary')['broad_reference_attribution_rate_mean'])}/"
        f"{pct(summary('fashion_mnist','greedy_portfolio','challenge')['broad_reference_attribution_rate_mean'])} on Fashion.",
    ))
    story.append(P(
        "K=4 prompts themselves contain reference-design vocabulary, so this is a prompt-conditioned lexical trace, not validated cognitive uptake. Most K=4 messages still did not name a reference. "
        "The source-parent graph is the authoritative process record; the phrase rates only show what the agent chose to mention in its summary.",
        SMALL,
    ))
    pagebreak(story)

    # Main page 5: outcome and cost.
    story.append(h1("5  Pluralism is not a free performance gain"))
    key_rows: list[list[str]] = [["Task / prompt", "Endpoint K4-K1", "95% range", "AUC K4-K1", "Tokens / proposal"]]
    for task, label in [("tiny_adderboard", "Tiny"), ("fashion_mnist", "Fashion")]:
        for prompt in ("ordinary", "challenge"):
            endpoint = contrast(task, prompt, "normalized_gain")
            auc = contrast(task, prompt, "auc_normalized_gain")
            tokens = contrast(task, prompt, "tokens_per_proposal")
            key_rows.append([
                f"{label} / {prompt}",
                signed(endpoint["paired_difference"], 4),
                ci(endpoint, digits=4),
                signed(auc["paired_difference"], 4),
                f"{float(tokens['paired_difference']):+,.0f}",
            ])
    story.append(P(
        "Table 3: Block-paired K=4-minus-K=1 descriptive contrasts. Positive yield favors K=4. Ranges are block-bootstrap sensitivity intervals. Token differences include the whole portfolio context.",
        CAPTION,
    ))
    story.append(data_table(key_rows, [105, 78, 98, 78, 80], font_size=6.9))
    story.extend(figure(
        DERIVED / "fig4_cross_task.png",
        5.25 * inch,
        "Figure 2: Endpoint and trajectory-AUC contrasts do not transport uniformly. All eight 95% ranges cross zero.",
    ))
    story.append(h2("5.1  Tiny AdderBoard"))
    story.append(P(
        "Under ordinary prompting, K=4 eliminated 0.044 more of the baseline parameter count than K=1 "
        f"{ci(contrast('tiny_adderboard','ordinary','normalized_gain'), digits=3)}; six of eight block differences favored K=4. Under assumption challenges, the difference shrank to "
        f"{signed(contrast('tiny_adderboard','challenge','normalized_gain')['paired_difference'],3)} "
        f"{ci(contrast('tiny_adderboard','challenge','normalized_gain'), digits=3)} with a 4-4 split. The endpoint means translate to 4,037 versus 3,070 parameters for ordinary K=1/K=4 and "
        "2,826 versus 2,596 under challenges. Ordinary endpoints favor the K=4 package, but the observed contrast is small under semantic interventions and is not causally identified.",
    ))
    story.append(h2("5.2  Fashion-MNIST and cost"))
    story.append(P(
        "Fashion-MNIST reverses the sign: endpoint normalized gains are -0.00184 "
        f"{ci(contrast('fashion_mnist','ordinary','normalized_gain'), digits=4)} under ordinary prompting and -0.00146 "
        f"{ci(contrast('fashion_mnist','challenge','normalized_gain'), digits=4)} under challenges. Differences are small relative to baseline score, but they reject a task-general story that "
        "keeping four lineages automatically improves the objective. Meanwhile K=4 added "
        f"{float(contrast('tiny_adderboard','ordinary','tokens_per_proposal')['paired_difference']):,.0f}-"
        f"{float(contrast('tiny_adderboard','challenge','tokens_per_proposal')['paired_difference']):,.0f} tokens per Tiny proposal (36-38%) and "
        f"{float(contrast('fashion_mnist','ordinary','tokens_per_proposal')['paired_difference']):,.0f}-"
        f"{float(contrast('fashion_mnist','challenge','tokens_per_proposal')['paired_difference']):,.0f} per Fashion proposal (25-28%). More branches buy option value and auditability; they do not buy free yield.",
    ))
    pagebreak(story)

    # Main page 6: native population and qualitative paths.
    story.append(h1("6  Reachable paths under the native population institution"))
    story.extend(figure(
        DERIVED / "fig2_lineage_raster.png",
        5.3 * inch,
        "Figure 3: Selected top-level lineage for three block-1 trajectories. Open circles mark strict global incumbent improvements. K=1 remains on one chain; K=4 cycles across four; native selection revisits a smaller set of deep lineages and old exact parents.",
    ))
    story.append(h2("6.1  Native memory is deeper, not simply wider"))
    story.append(P(
        "At opportunity 70, native ordinary runs held a mean 29.0 valid programs and challenge runs 26.9; mean occupied-island counts were 4.82 and 5.00. Native selection used non-incumbent "
        f"parents on {pct(summary('tiny_adderboard','native_population','ordinary')['alternative_parent_rate_mean'])} of ordinary and "
        f"{pct(summary('tiny_adderboard','native_population','challenge')['alternative_parent_rate_mean'])} of challenge proposals. It revisited an exact parent after a gap of at least 10 proposals "
        "17.8 and 20.4 times per run, versus 0 for K=1 and 1.0 for the deterministic K=4 controller. Respectively 91.0% and 89.1% of strict incumbent improvements descended from a "
        "non-incumbent parent. Yet native endpoint reduction averaged 0.737 ordinary and 0.849 challenge, below the corresponding greedy K=4 means of 0.860 and 0.882. This framework contrast "
        "is descriptive because run seeds and LLM calls differ; it shows that branch revival and short-horizon objective yield are distinct outputs.",
    ))
    story.append(h2("6.2  Exhaustive alternative-parent audit and trace cases"))
    story.append(P(
        "Across the common Tiny horizon, the exhaustive table contains 1,032 strict global improvements from a non-incumbent parent after warm-up; every parent/candidate source pair is available. "
        "Forty follow a branch gap of at least 10 proposals (29 native, 11 greedy). Three post hoc examples illustrate different executable pathways rather than estimating prevalence. In native block 5 C0, opportunity 16 returned to a lineage after a 15-proposal absence. Its 23-line source edit "
        "bottlenecked attention to 32 internal dimensions while preserving a width-40 residual stream; its recorded qualified evaluation removed 1,288 parameters. In block 4 C1, opportunity 28 selected a "
        "25-opportunity-old parent and made a 45-line edit that removed a 2,304-parameter value projection from query-free positional attention; it is recorded as qualified and became the global incumbent. "
        "In block 6 C1, opportunity 60 made a 27-line edit replacing dense content-derived query/key projections with learned causal routing maps and removed 1,787 parameters. Parent/candidate "
        "source hashes, diffs, hypotheses, and evaluator records are in the artifact.",
    ))
    story.append(P(
        "These cases establish structural reachability, not average benefit: a strict single-incumbent controller cannot select a discarded parent, whereas these recorded population policies can. "
        "Whether preserving that option improves discovery requires a randomized, state-matched design.",
        SMALL,
    ))
    pagebreak(story)

    # Main page 7: composition audit and implications.
    story.append(h1("7  Memory factors can collapse under system composition"))
    native = AGG["native_semantics_validation"]
    story.append(P(
        "The native adapter revealed a construct failure that a condition label would hide. In nominal C0 and C1, the generic controller state says single-incumbent. External ProgramDatabase "
        "selection nevertheless supplied multiple visible designs in "
        f"{pct(native['fraction_multiple_visible_by_condition']['C0'])} and {pct(native['fraction_multiple_visible_by_condition']['C1'])} of opportunities, with means "
        f"{f(native['mean_visible_by_condition']['C0'],2)} and {f(native['mean_visible_by_condition']['C1'],2)} designs. A recorded C0 prompt can therefore state that no reference design is "
        "available and immediately render a `REFERENCE DESIGN 1`. Nominal C2/C3 also see variable native populations rather than the frozen K=4 controller. The memory factor has not become "
        "noisy; it has changed meaning.",
    ))
    story.extend(figure(
        DERIVED / "fig3_diversity_yield.png",
        4.65 * inch,
        "Figure 4: Tiny endpoint parameter reduction versus effective selected top-level branches. Greedy horizontal positions largely reflect controller rules; the outcome scatter shows why branch allocation is a manipulation check, not a sufficient condition for yield.",
    ))
    story.append(h2("7.1  Why this matters for AI science"))
    story.append(bullet("<b>Epistemic agency is distributed.</b> The LLM proposes an edit, but controllers decide which histories exist, samplers decide which parent is salient, evaluators decide survival, and prompt renderers decide what evidence can be seen."))
    story.append(bullet("<b>Pluralism must be functional.</b> Archive size is insufficient. Report selection entropy, lineage age, dormant revival, lexical attribution, and recorded strict improvements from alternative branches."))
    story.append(bullet("<b>Factorial semantics are compositional.</b> A factor defined at the generic controller may be overridden by an adapter. Validate the final subject-visible prompt and parent source, not only configuration files."))
    story.append(bullet("<b>Efficiency is epistemic.</b> A larger evidence envelope consumes 25-38% more subject-agent tokens. Under fixed budgets, that cost changes how many hypotheses can be tested."))
    story.append(bullet("<b>Null yield is informative.</b> Diversity is valuable for resilience, auditability, and option preservation, but capability claims require task-level endpoint and AUC evidence. Do not equate branch count with discovery."))
    story.append(h2("7.2  Recommended reporting card"))
    reporting = [
        ["Layer", "Minimum audit"],
        ["Availability", "visible designs, evidence envelope, prompt contract"],
        ["Selection", "parent IDs, entropy, non-incumbent use, lineage age"],
        ["Retention", "admission rule, branch survival, archive/island occupancy"],
        ["Lexical record", "literal alternative phrases, source parent, actual edit"],
        ["Consequence", "recorded yield, failures, AUC, cost, right-censoring"],
    ]
    story.append(P(
        "Table 4: A minimal population-memory reporting card. All fields should be recorded at the final composed system boundary.",
        CAPTION,
    ))
    story.append(data_table(reporting, [78, 318], font_size=7.1))
    story.append(P(
        "For evaluations, we recommend prospectively choosing between two estimands. A <i>whole-system portfolio</i> intentionally gives K=4 more source and measures the practical package. A "
        "<i>fixed-evidence diversity</i> design gives every condition the same useful token envelope while distributing K=4 evidence across lineages. The current greedy study estimates the former. "
        "Token normalization after the fact cannot recover the latter because extra context can alter the proposal as well as cost.",
    ))
    pagebreak(story)

    # Main page 8: limitations, conclusion, statements.
    story.append(h1("8  Limitations and conclusion"))
    story.append(h2("Limitations"))
    story.append(P(
        "First, memory labels were not randomly assigned and K=1/K=4 Tiny runs do not share a state-matched prefix; block bootstrap ranges quantify trajectory sensitivity, not sampling uncertainty "
        "from a randomized experiment. Second, the K=4 contrast bundles branch diversity, a deterministic fairness selector, and longer prompts. Third, native-versus-greedy comparisons are "
        "descriptive: framework hashes and run seeds differ, and the native engine invalidates the nominal controller-memory factor. Fourth, the common Tiny horizon was mechanically set to 70 "
        "because later progress was uneven; conclusions may change at 100 or 200 proposals. Fifth, only one subject model, one reasoning setting, two ML tasks, and public evaluator outcomes are "
        "studied. Fashion timeouts and shared local compute can affect feasibility. Sixth, a source lineage is not necessarily a semantically independent research program, and explicit reference "
        "phrases are a low-recall lexical measure. We partly address this by combining parent graphs, source deltas, agent messages, and recorded outcomes. Finally, source-integrity checks support trace "
        "reconstruction, but we treat recorded evaluator outcomes as given and do not re-run training or sealed generalization here. This paper evaluates search behavior rather than "
        "the scientific importance of the discovered architectures.",
    ))
    story.append(h2("Conclusion"))
    story.append(P(
        "The observed population-memory policies preserve executable alternatives and spend evaluations on non-leading paths. In 8,480 proposals, K=4 controllers "
        "implemented stable branch allocation and produced more explicit cross-design lexical records, while native populations revived old parents and recorded large source-audited improvements. None of this "
        "guaranteed better endpoint performance, and the additional context was costly. Worse, a population adapter silently changed the meaning of a nominal single-incumbent factor. The appropriate "
        "epistemic standard is therefore not `did the system keep a population?' but `which alternatives remained reachable, which were actually used, at what cost, and under what composed prompt?' "
        "Treating memory as an auditable research institution - rather than a model feature or archive-size hyperparameter - provides a testable basis for narrower AI scientist evaluation claims.",
    ))
    story.append(h2("Reproducibility, ethics, and AI-use statement"))
    story.append(P(
        "The anonymized artifact contains analysis code, frozen event records analyzed under fixed horizons, structured agent final messages, candidate source snapshots used for source audits, prompt examples, derived tables, "
        "figures, hashes, and a clean-room command. It is provided as anonymous supplementary material under an MIT license covering the complete payload. It requires no model or evaluator call. No human-subject data are analyzed. The experiments consume compute and commercial model capacity; we report "
        "tokens rather than converting them to a potentially misleading monetary cost. This manuscript was produced with extensive AI assistance as a research-direction artifact and is not eligible "
        "for submission under AISciK's policy without substantial human authorship and complete manual verification. AI systems are not authors. Anonymous human authors remain responsible for every claim, "
        "citation, and release decision.",
        SMALL,
    ))
    story.append(h2("Track and claim boundary"))
    story.append(P(
        "Intended track: Datasets and Evaluations. The central contribution is an empirical evaluation design and trace audit of epistemic pluralism in AI-mediated research. Results establish behavior "
        "inside these recorded systems; they do not establish a general law that populations help or harm science.",
        SMALL,
    ))
    pagebreak(story)

    # References begin after exactly eight main pages.
    story.append(h1("References"))
    references = [
        "Kitcher, P. (1990). The division of cognitive labor. <i>The Journal of Philosophy</i>, 87(1):5-22. doi:10.2307/2026796.",
        "Zollman, K. J. S. (2010). The epistemic benefit of transient diversity. <i>Erkenntnis</i>, 72:17-35. doi:10.1007/s10670-009-9194-6.",
        "March, J. G. (1991). Exploration and exploitation in organizational learning. <i>Organization Science</i>, 2(1):71-87. doi:10.1287/orsc.2.1.71.",
        "Lazer, D. and Friedman, A. (2007). The network structure of exploration and exploitation. <i>Administrative Science Quarterly</i>, 52(4):667-694. doi:10.2189/asqu.52.4.667.",
        "Lehman, J. and Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. <i>Evolutionary Computation</i>, 19(2):189-223. doi:10.1162/EVCO_a_00025.",
        "Mouret, J.-B. and Clune, J. (2015). Illuminating search spaces by mapping elites. arXiv:1504.04909.",
        "Pugh, J. K., Soros, L. B., and Stanley, K. O. (2016). Quality diversity: A new frontier for evolutionary computation. <i>Frontiers in Robotics and AI</i>, 3:40. doi:10.3389/frobt.2016.00040.",
        "Novikov, A., Vu, N., Eisenberger, M., et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131.",
        "Sharma, A. and OpenEvolve contributors (2025-2026). OpenEvolve: An open-source implementation of AlphaEvolve. Software repository: github.com/algorithmicsuperintelligence/openevolve.",
        "Wan, C., Dai, X., Wang, Z., Li, M., Wang, Y., Mao, Y., Lan, Y., and Xiao, Z. (2025). LoongFlow: Directed evolutionary search via a cognitive plan-execute-summarize paradigm. arXiv:2512.24077.",
        "Lyu, Y., Zhang, X., Yi, X., et al. (2026). EvoScientist: Towards multi-agent evolving AI scientists for end-to-end scientific discovery. arXiv:2603.08127.",
        "Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. (2024). The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv:2408.06292.",
        "Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., and Ha, D. (2025). The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv:2504.08066.",
        "Huang, Q., Vora, J., Liang, P., and Leskovec, J. (2024). MLAgentBench: Evaluating language agents on machine learning experimentation. <i>ICML, PMLR 235</i>:20271-20309.",
        "Toledo, E., Hambardzumyan, K., Josifoski, M., et al. (2025). AI research agents for machine learning: Search, exploration, and generalization in MLE-bench. arXiv:2507.02554.",
        "Cronbach, L. J. and Meehl, P. E. (1955). Construct validity in psychological tests. <i>Psychological Bulletin</i>, 52(4):281-302. doi:10.1037/h0040957.",
        "AI &amp; Scientific Knowledge Workshop (2026). AI &amp; Science: Evolution or Extinction? Call for Papers. aiscik.github.io.",
        "Reuel, A., Hardy, A., Smith, C., Lamparth, M., Hardy, M., and Kochenderfer, M. J. (2024). BetterBench: Assessing AI benchmarks, uncovering issues, and establishing best practices. <i>NeurIPS 37, Datasets and Benchmarks</i>. doi:10.52202/079017-0685.",
        "Imbens, G. W. and Rubin, D. B. (2015). <i>Causal Inference for Statistics, Social, and Biomedical Sciences</i>. Cambridge University Press.",
    ]
    for index, reference in enumerate(references, 1):
        story.append(P(f"[{index}] {reference}", REF))
    pagebreak(story)

    # Appendix A: lineage definitions and complete contrasts.
    story.append(h1("A  Operational definitions and complete contrasts"))
    story.append(h2("A.1  Lineage reconstruction"))
    story.append(P(
        "Each run begins with one baseline candidate. A candidate event records exactly one selected parent. We recursively follow parent IDs until reaching the baseline; the baseline's first child on "
        "that path identifies a top-level branch. We compute branch-allocation statistics from proposals 10 through the analysis horizon so the Tiny shared-prefix period does not dominate entropy. Root selections "
        "are excluded from effective-branch entropy but included in parent-use rates. Exp(Shannon entropy) equals 1 for one selected branch and approaches K under balanced allocation; under the K=4 fairness selector it is a manipulation check, not a discovery outcome. A strict global "
        "improvement requires objective_after &gt; objective_before, not merely a changed candidate ID.",
        SMALL,
    ))
    story.append(h2("A.2  Paired K=4-minus-K=1 results"))
    appendix_rows = [["Task", "Prompt", "Metric", "Difference", "95% range", "Pairs"]]
    labels = {
        "normalized_gain": "endpoint gain",
        "auc_normalized_gain": "trajectory AUC",
        "effective_top_lineages": "effective selected branches",
        "alternative_parent_rate": "non-incumbent rate",
        "productive_branch_reactivations": "productive revivals",
        "reference_attribution_rate": "reference attribution",
        "tokens_per_proposal": "tokens/proposal",
        "tail_stagnation": "tail stagnation",
    }
    for task, task_label in [("tiny_adderboard", "Tiny"), ("fashion_mnist", "Fashion")]:
        for prompt in ("ordinary", "challenge"):
            for metric in labels:
                row = contrast(task, prompt, metric)
                appendix_rows.append([
                    task_label,
                    prompt,
                    labels[metric],
                    f"{float(row['paired_difference']):+.4f}",
                    ci(row, digits=4),
                    row["n_pairs"],
                ])
    story.append(P(
        "Table A1: Every Paper 3 controller-memory contrast reported in the main analysis. For tokens and count outcomes, four-decimal formatting is retained only for machine readability; the main text rounds to useful units.",
        CAPTION,
    ))
    story.append(data_table(appendix_rows, [42, 54, 105, 65, 91, 39], font_size=6.2))
    pagebreak(story)

    # Appendix B: native composition audit and trace examples.
    story.append(h1("B  Native composition audit and trace anchors"))
    story.append(h2("B.1  Subject-visible memory by nominal condition"))
    native_rows = [["Condition", "Nominal memory", "Mean visible", "Multiple visible"]]
    for condition in ("C0", "C1", "C2", "C3"):
        native_rows.append([
            condition,
            "single" if condition in {"C0", "C1"} else "portfolio",
            f(native["mean_visible_by_condition"][condition], 3),
            pct(native["fraction_multiple_visible_by_condition"][condition], 1),
        ])
    story.append(P(
        "Table B1: Generic proposal-start records after native parent sampling. The external population supplies a variable number of visible candidates in every nominal condition.",
        CAPTION,
    ))
    story.append(data_table(native_rows, [68, 100, 105, 110], font_size=7.2))
    story.append(h2("B.2  Concrete prompt contradiction"))
    story.append(P(
        "Run `...native_openevolve-b01-c0`, opportunity 10, contains the sentence `The current editable design is provided. No reference design is available.` Its Available Designs section then contains "
        "both `CURRENT DESIGN` and `REFERENCE DESIGN 1`. The event records two visible candidate IDs and one selected parent. This is direct subject-visible evidence that the nominal C0 memory contract "
        "was superseded after composition. The artifact preserves this prompt, its manifest, and event rows. The analysis never recodes native C0 as K=1.",
        QUOTE,
    ))
    story.append(h2("B.3  Source-audited recorded alternative-parent improvement cases"))
    example_rows = [["Run / opp.", "Age", "Gap", "Param. reduction", "Changed lines", "Mechanism (abridged)"]]
    selected_examples = [
        next(row for row in EXEMPLARS if "b05-c0" in row["run_id"] and row["opportunity"] == "16"),
        next(row for row in EXEMPLARS if "b04-c1" in row["run_id"] and row["opportunity"] == "28"),
        next(row for row in EXEMPLARS if "b06-c1" in row["run_id"] and row["opportunity"] == "60"),
    ]
    abridged = [
        "residual-width attention bottleneck",
        "embedding-native value transport",
        "learned causal routing maps",
    ]
    for row, mechanism in zip(selected_examples, abridged, strict=True):
        match = row["run_id"].split("-")[-2:]
        run_label = "-".join(match)
        example_rows.append([
            f"{run_label} / {row['opportunity']}",
            row["parent_age"],
            row["branch_gap"],
            f"{float(row['parameter_reduction']):,.0f}",
            row["changed_lines"],
            mechanism,
        ])
    story.append(P(
        "Table B2: Selected trace anchors. `Age` is proposals since parent creation; `Gap` is proposals since that top-level lineage was last selected. Every candidate is recorded as passing the >=99% public qualification gate and becoming the global incumbent. SHA-256 source hashes are in `alternative_branch_improvements.csv`.",
        CAPTION,
    ))
    story.append(data_table(example_rows, [75, 35, 35, 70, 61, 120], font_size=6.6))
    pagebreak(story)

    # Appendix C: reproducibility and integrity.
    story.append(h1("C  Reproducibility and integrity checklist"))
    checks = [
        ["Item", "Recorded evidence"],
        ["Primary units", "64 Tiny trajectories; 20 Fashion trajectories"],
        ["Proposal rows", "4,480 Tiny common-horizon; 4,000 Fashion complete"],
        ["Right-censoring", "Tiny fixed at minimum complete horizon 70"],
        ["Parent graph", "selected parent, candidate, incumbent before/after"],
        ["Agent messages", "mechanism, hypothesis, intended edit, evidence"],
        ["Evaluation", "validity, task metrics, failures, elapsed seconds"],
        ["Resource record", "input, cached-input, output, reasoning tokens"],
        ["Native state", "parent samples, inspirations, islands, archive, admission"],
        ["Source verification", "candidate source snapshots and SHA-256 hashes"],
        ["Uncertainty", "trajectory-level block bootstrap, fixed seed 20260901"],
        ["Privacy", "no credentials, host paths, private chain-of-thought, remotes"],
    ]
    story.append(data_table(checks, [110, 286], font_size=7.0))
    story.append(h2("C.1  Reproduction"))
    story.append(P(
        "Unpack `paper3_reproducibility_artifact.zip`, create an environment with the pinned NumPy and Matplotlib versions, and run from the archive root:",
        SMALL,
    ))
    story.append(P(
        "MPLCONFIGDIR=/tmp/aiscik-p3 python3 papers/aiscik2026/paper3/analysis.py --data-root . --verify-input-hashes",
        QUOTE,
    ))
    story.append(P(
        f"The script validates 64 Tiny runs, 20 Fashion runs, contiguous event horizons, 32 native ledgers, and {AGG['input_files_hashed']} raw input hashes before regenerating all tables and figures. "
        "The artifact includes a payload checksum manifest. The release check verifies all 21,205 payload hashes and scans for host paths, the operator name, remotes, credentials, and key patterns. No network, provider, or training call is required.",
        SMALL,
    ))
    story.append(h2("C.2  Analysis boundaries"))
    story.append(P(
        "We did not select examples before estimating aggregate contrasts. The exhaustive alternative-parent table was generated first; trace cases were then chosen post hoc to illustrate different recorded-qualified source changes. "
        "They illustrate pathways but do not enter the mean-yield estimand. Nominal native K contrasts are saved only as negative-control diagnostics; they are excluded from K=1/K=4 claims. Complete "
        "failure rows remain in qualification and cost denominators. No proposal is dropped because it failed, timed out, produced an invalid patch, or was not retained.",
        SMALL,
    ))
    pagebreak(story)

    return story


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = NeuripsDoc(str(OUTPUT))
    story = build_story()
    document.build(story)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(path)
