#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the AISciK-ready Paper 2 PDF in a NeurIPS-like layout."""

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
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
OUTPUT = HERE.parents[2] / "output/pdf/paper2_state_matched_defixation.pdf"
MAIN_PAGES = 8


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


AGG = json.loads((DERIVED / "aggregate.json").read_text(encoding="utf-8"))
FORK_EFFECTS = {
    (row["subset"], row["metric"]): row
    for row in rows(DERIVED / "fork_effects.csv")
}
TRAJECTORY_EFFECTS = {
    (row["subset"], row["metric"]): row
    for row in rows(DERIVED / "trajectory_effects.csv")
}
PHASE1_EFFECTS = {
    (row["subset"], row["metric"]): row
    for row in rows(DERIVED / "phase1_effects.csv")
}
MECHANISMS = rows(DERIVED / "fork_mechanism_summary.csv")
HORIZONS = rows(DERIVED / "horizon_effects.csv")
FASHION_EFFECTS = {
    (row["scope"], row["metric"]): row
    for row in rows(DERIVED / "fashion_effects.csv")
}


def f(value: Any, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def pp(value: Any, digits: int = 1) -> str:
    return f"{100 * float(value):+.{digits}f} pp"


def effect(
    table: dict[tuple[str, str], dict[str, str]], subset: str, metric: str
) -> dict[str, str]:
    return table[(subset, metric)]


def interval(row: dict[str, str], scale: float = 1.0, digits: int = 2) -> str:
    low = scale * float(row["cluster_bootstrap_low"])
    high = scale * float(row["cluster_bootstrap_high"])
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


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
            title="Can a Prompt Defixate an Autonomous Compression Agent?",
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
    fontSize=10,
    leading=11,
    alignment=TA_JUSTIFY,
    spaceAfter=5.2,
    allowWidows=0,
    allowOrphans=0,
)
SMALL = ParagraphStyle(
    "Small",
    parent=BODY,
    fontSize=8.25,
    leading=9.25,
    spaceAfter=3.1,
)
TINY = ParagraphStyle(
    "Tiny",
    parent=BODY,
    fontSize=7.2,
    leading=8.05,
    spaceAfter=2.4,
)
H1 = ParagraphStyle(
    "H1",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=12,
    leading=13,
    alignment=TA_LEFT,
    spaceBefore=5,
    spaceAfter=4,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=10,
    leading=11,
    alignment=TA_LEFT,
    spaceBefore=4,
    spaceAfter=2.5,
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
    fontName=FONT,
    fontSize=10,
    leading=11,
    alignment=TA_CENTER,
    spaceAfter=5,
)
ABSTRACT = ParagraphStyle(
    "Abstract",
    parent=BODY,
    fontSize=9.4,
    leading=10.5,
    leftIndent=0.48 * inch,
    rightIndent=0.48 * inch,
    spaceAfter=5,
)
ABSTRACT_HEAD = ParagraphStyle(
    "AbstractHead",
    parent=H1,
    alignment=TA_CENTER,
    fontSize=11,
    spaceBefore=2,
    spaceAfter=3,
)
CAPTION = ParagraphStyle(
    "Caption",
    parent=SMALL,
    fontSize=7.65,
    leading=8.55,
    alignment=TA_JUSTIFY,
    spaceBefore=2,
    spaceAfter=4,
)
BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    leftIndent=12,
    firstLineIndent=-7,
    bulletIndent=2,
    spaceAfter=2.5,
)
QUOTE = ParagraphStyle(
    "Quote",
    parent=SMALL,
    leftIndent=12,
    rightIndent=10,
    borderColor=colors.HexColor("#9ca3af"),
    borderWidth=0,
    borderPadding=4,
    backColor=colors.HexColor("#f3f4f6"),
    spaceBefore=2,
    spaceAfter=4,
)
REF = ParagraphStyle(
    "Reference",
    parent=SMALL,
    fontSize=8.0,
    leading=8.9,
    leftIndent=10,
    firstLineIndent=-10,
    spaceAfter=3.1,
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
        "Can a Prompt Defixate an Autonomous Compression Agent?<br/>"
        "<font size='13'>State-Matched Forks in Model Search</font>",
        TITLE,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=2, spaceAfter=9))
    story.append(P("Anonymous Authors", AUTHOR))


def data_table(
    data: list[list[Any]],
    widths: list[float],
    header_rows: int = 1,
    font_size: float = 7.2,
) -> Table:
    table = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, header_rows - 1), FONT_BOLD),
        ("FONTNAME", (0, header_rows), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.1),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e5e7eb")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
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

    # Main page 1: title, abstract, introduction.
    ruled_title(story)
    story.append(P("Abstract", ABSTRACT_HEAD))
    story.append(P(
        "Autonomous coding agents often reuse a productive line of attack until local search stalls. Can a short instruction redirect "
        "the same compression agent from the same recorded search state toward a structurally different, workable edit? We study 64 model-compression trajectories "
        "whose control/treatment pairs are event- and provenance-matched through nine proposals, then fork at proposal 10. Treated agents receive an explicit "
        "instruction to identify and challenge load-bearing assumptions; controls receive the ordinary task direction. Across 32 exact forks, "
        "the instruction is associated with 0.0309 more normalized source novelty [0.0216, 0.0428], 13.1 more changed lines [11.6, 14.7], and "
        "a 53.1 percentage-point increase [34.4, 71.9] in a post hoc mechanism-family tag. It lowers qualification by 9.4 points [-28.1, 9.4] but raises "
        "unconditional one-step parameter reduction by 537 [209, 941]. Successful treated edits remove a median 1,765 parameters versus 81 for "
        "controls. Before a second intervention, treated trajectories end 786 parameters smaller [-1,271, -345]; through a common 70-proposal "
        "horizon, a seven-intervention adaptive regime, they end 1,652 smaller [-2,301, -1,083] in 28/32 pairs. An exploratory, condition-aware trace audit "
        "suggests punctuated defixation: the prompt initiates architectural jumps, then ordinary follow-ups exploit them. Yet 17/32 treated forks converge "
        "on token-interface factorization, and a descriptive Fashion-MNIST portability check shows a larger novelty-feasibility penalty without clear "
        "endpoint gain. Semantic interventions can improve search by changing which failures are attempted, but novelty, feasibility, and cost "
        "must be reported jointly.",
        ABSTRACT,
    ))
    story.append(h1("1  Introduction"))
    story.append(P(
        "Language-model agents now design, implement, train, and select machine-learning systems in closed loops [9-13]. Their search policies "
        "condition on an incumbent, experimental history, and evaluator feedback. This memory is useful, but it also creates a scientific version "
        "of design fixation: repeated adherence to the same concepts can narrow the reachable search space [1,8]. Verbal reflection can steer agents "
        "without weight updates [2,6], and diversity prompts broaden one-shot reasoning [3-5]. Those results do not establish whether a prompt changes "
        "the executable mechanism selected by a long-horizon research agent, whether the alternative survives evaluation, or whether later search "
        "benefits from the deviation.",
    ))
    story.append(P(
        "We ask: <b>In an autonomous transformer-compression loop, does an assumption-challenge instruction, from a matched search state, shift the next "
        "proposal away from local pruning; and how do source-level novelty, qualification, compression progress, and cost change immediately and under a "
        "repeated-intervention regime?</b> This question makes the AI research process - not merely its best model - the object of study.",
    ))
    pagebreak(story)

    # Main page 2: contributions and related work.
    story.append(h1("1.1  Contributions and claim boundary"))
    story.append(bullet(
        "<b>State-matched intervention.</b> We construct 32 control/treatment forks across greedy and native OpenEvolve. Each pair shares the exact "
        "first nine proposal events, candidate provenance, and selected proposal-10 parent; the treatment prompt differs only by an inserted direction."
    ))
    story.append(bullet(
        "<b>Triangulated process measurement.</b> We connect all 4,480 recorded final proposal messages to normalized source diffs, AST changes, "
        "evaluator outcomes, retention, token increments, and incumbent parameter counts. A condition-aware analyst also coded the 64 fork messages; "
        "that taxonomy is explicitly exploratory, not an independently validated outcome."
    ))
    story.append(bullet(
        "<b>Immediate, propagated, and repeated regimes.</b> We separate proposal 10, the one-intervention phase 10-19, and a common 70-proposal "
        "seven-intervention adaptive regime. A complete 4,000-proposal Fashion-MNIST corpus is used only as a descriptive portability check."
    ))
    story.append(bullet(
        "<b>Negative result within the positive result.</b> The instruction is followed by a shift from local pruning toward alternative computations and greater compression, "
        "but reduces qualification, increases output and evaluator cost, and concentrates more than half of treated forks on one alternative."
    ))
    story.append(P(
        "We use <i>defixation</i> behaviorally: departure from a trajectory's current mechanism family. We do not infer private reasoning or a human-like "
        "creative faculty. <i>Novel</i> means source-level departure from the selected parent or recorded trajectory history, not globally unprecedented. "
        "<i>Feasible</i> means at least 99% exact-answer accuracy under the fixed evaluator. The design sharply controls observed history at the first fork "
        "but does not randomize labels; all bootstrap ranges are descriptive cluster sensitivity ranges, not confidence intervals or p-values [16-18].",
        SMALL,
    ))
    story.append(h1("2  Related work"))
    story.append(h2("Design fixation and machine creativity"))
    story.append(P(
        "Jansson and Smith define design fixation as blind adherence to limiting ideas and demonstrate it experimentally [1]. An LLM-directed prototyping "
        "study later reports fixation, premature abandonment, and needless complexity, recommending explicit alternatives and feedback loops [8]. "
        "Creativity measurement distinguishes divergent novelty from convergent appropriateness: MacGyver exposes imaginative but physically infeasible "
        "solutions [4], NEOGAUGE combines divergent and convergent code behavior [5], and CDAT shows that novelty alone can reward contextual noise [7]. "
        "Prompted divergence can also trade creativity against response stability [14], while automated scores remain measurement proxies requiring cross-task construct validation "
        "[15]. Construct-validity work warns that benchmark labels can outpace the measurements that support them [17,18]. This motivates source, execution, "
        "qualification, and cost measures together; our post hoc family taxonomy does not carry the primary claim."
    ))
    story.append(h2("Semantic steering of agents"))
    story.append(P(
        "Reflexion stores linguistic feedback as a semantic gradient [2]. DDPrompt generates task-adaptive diverse reasoning paths [3]; denial prompting "
        "forces new code strategies [5]; and DORA diversifies reflection advice after detecting early-stop reflection [6]. These studies show that language "
        "can redirect inference. We instead hold an executable research state fixed, allow exactly one added direction, and follow both evaluation and "
        "downstream architecture search."
    ))
    story.append(h2("Autonomous ML research"))
    story.append(P(
        "MLAgentBench operationalizes experiment design, execution, and iteration [9]. Toledo et al. formalize research agents as search policies plus "
        "operators and show their interaction is critical [10]. AlphaEvolve, OpenEvolve, and CodeEvolve couple LLM code mutation to evolutionary selection "
        "[11,12,21]; ADAS searches over agent architectures [20], while the AI Scientist systems automate broader scientific workflows [13,19]. Our "
        "contribution is narrower: a state-matched prompt contrast inside a code-search loop, with executable edits and downstream trajectories. LoongFlow "
        "and EvoScientist use structured planning or persistent memory to fight long-horizon stagnation [22,23]; we isolate one lightweight semantic operator."
    ))
    pagebreak(story)

    # Main page 3: design and measures.
    story.append(h1("3  Experimental design"))
    story.append(h2("Task and research systems"))
    story.append(P(
        "Tiny AdderBoard asks a learned transformer to add two four-digit numbers exactly. A candidate qualifies with at least 99% exact-answer accuracy; "
        "among qualified candidates, fewer deduplicated learned parameters is better. The common baseline has 21,952 parameters. The evaluator owns "
        "deterministic data generation, training checkpoints, source-integrity checks, and an attention-dependence check. All arms use GPT-5.6 Sol with "
        "xhigh reasoning and default service tier, the same task and prompt bundle, and the same advertised 100-proposal budget. The Codex interface recorded "
        "model, reasoning effort, and service tier but exposed no separate temperature or sampling-seed setting. We stratify by controller architecture: greedy OpenEvolve "
        "retains a controller-selected incumbent/portfolio, whereas native OpenEvolve also samples an island/MAP-Elites population."
    ))
    story.append(h2("Conditions and exact fork"))
    condition_data = [
        ["Condition", "Memory", "Direction at 10,20,...", "Blocks x systems"],
        ["C0", "Single incumbent", "Ordinary", "8 x 2"],
        ["C1", "Single incumbent", "Assumption challenge", "8 x 2"],
        ["C2", "Portfolio", "Ordinary", "8 x 2"],
        ["C3", "Portfolio", "Assumption challenge", "8 x 2"],
    ]
    story.append(data_table(condition_data, [58, 94, 142, 70], font_size=7.2))
    story.append(P(
        "Table 1: Factorial conditions. Within each block/system, C0/C1 and C2/C3 share proposals 1-9 and fork from the same proposal-10 parent.",
        CAPTION,
    ))
    story.append(P(
        "Condition labels were fixed, but their launch positions were permuted across the 16 architecture-block schedules rather than always ordered alike. "
        "Treated arms launched after their matched controls in 9/16 single-memory schedules and 6/16 portfolio schedules; the remaining pairs reversed that "
        "order. This counters a simple universal launch-order explanation but is not random assignment.",
        SMALL,
    ))
    story.append(P(
        "The inserted direction asks the agent to step back, identify load-bearing assumptions in its current line and available designs, challenge them "
        "with a genuinely different learned computational mechanism, avoid unsupported retries, preserve evidence-backed components, state the old "
        "assumption and alternative, and implement a clean feasible test. Controls receive the same task state, candidate sources, evidence, output "
        "contract, and evaluator rules without that paragraph. The complete prompt diff is in Appendix A."
    ))
    story.append(h2("Analysis windows and integrity"))
    story.append(P(
        "Proposal 10 supplies the immediate state-matched contrast. Proposals 10-19 describe propagation after one intervention and before the next. "
        "The 100-proposal budget was an upper bound, not the analysis horizon. At the frozen analysis snapshot, trajectories had unequal later completion; "
        "proposal 70 was selected mechanically as the minimum completed horizon across all 64 runs, without inspecting outcomes. It estimates an adaptive "
        "regime containing seven scheduled directions, not the effect of one prompt. The analysis asserts 288 identical paired-prefix events, 288 matching "
        "source or provenance records, 32 matching fork parents, and "
        "32 insertion-only prompt diffs. All 4,480 horizon events have a recorded final message. Because condition labels were fixed rather than "
        "randomized, we treat matching as design evidence for the fork contrast, not as a substitute for random assignment [16]."
    ))
    story.append(h2("Measures and uncertainty"))
    story.append(P(
        "Source novelty is one minus Jaccard similarity between parent/candidate Python token 3-grams after abstracting identifiers and literals. AST "
        "distance compares node-type multisets. Declared novelty is the minimum word-set distance to earlier proposal summaries in the same trajectory. "
        "A fixed lexical rule detects first-use family labels and explicit assumption/alternative language; the latter is partly a direct manipulation check. "
        "Qualification, retention, tokens, "
        "evaluator seconds, and incumbent parameters come directly from events. We report paired means and 10,000 fixed-seed cluster-bootstrap resamples "
        "of architecture x block clusters, retaining both memory pairs per cluster. These are descriptive sensitivity ranges. Full definitions, missingness "
        "rules, and condition-aware coding limits appear in Appendices B and D."
    ))
    pagebreak(story)

    # Main page 4: immediate effect.
    story.append(h1("4  The next executable proposal differs at the matched fork"))
    story.extend(figure(
        DERIVED / "fig1_fork_effects.png",
        5.35 * inch,
        "Figure 1: Paired treatment-minus-control contrasts at the exact proposal-10 fork, stratified by controller architecture and memory. Error bars are "
        "descriptive 95% architecture-block cluster-bootstrap sensitivity ranges. Output-token contrasts are counts; qualification and retention are proportions.",
    ))
    immediate_data = [
        ["Outcome", "Ordinary", "Challenge", "Paired difference [95% range]"],
        ["Source novelty", "0.0035", "0.0342", "+0.0309 [0.0216, 0.0428]"],
        ["AST distance", "0.0067", "0.0412", "+0.0342 [0.0248, 0.0444]"],
        ["Changed lines", "4.34", "17.47", "+13.13 [11.63, 14.69]"],
        ["New family", "9.4%", "62.5%", "+53.1 pp [34.4, 71.9]"],
        ["Qualified", "50.0%", "40.6%", "-9.4 pp [-28.1, 9.4]"],
        ["Parameter reduction", "67", "604", "+537 [209, 941]"],
        ["Output tokens", "1,418", "3,869", "+2,451 [2,026, 2,907]"],
        ["Evaluator seconds", "87.0", "104.9", "+17.9 [3.0, 33.3]"],
    ]
    story.append(data_table(immediate_data, [103, 62, 63, 168], font_size=6.85))
    story.append(P(
        "Table 2: Exact-fork outcomes, 32 pairs. Parameter reduction is unconditional: failed/unretained proposals contribute zero.",
        CAPTION,
    ))
    story.append(P(
        "The prompt produces larger source and semantic moves, but not free success. Sixteen ordinary and 13 treated proposals improve the incumbent. "
        "Conditional on improvement, treated proposals remove a mean 1,487 (median 1,765) parameters versus 134 (median 81) for controls. Because this "
        "conditional contrast selects on post-treatment success, Table 2's unconditional +537 paired difference is primary. Qualified structural edits rise "
        "from 28.1% to 40.6% (+12.5 points [-3.1, 28.1]): the instruction trades many tiny safe edits for fewer, substantially larger workable jumps."
    ))
    story.append(h2("What the matched contrast supports"))
    story.append(P(
        "The intervention's operative sentence is concrete: <i>'identify the load-bearing assumptions ... [and] test genuinely different learned "
        "computational mechanisms.'</i> It also requests the old assumption, alternative, evidence, and falsifier. Exact sequence diffs verify that this "
        "paragraph is the only prompt insertion at the fork; the task state, selected parent source, evidence capsule, evaluator contract, and output schema "
        "are otherwise matched. The immediate contrast therefore attributes the observed source/evaluation difference to the inserted direction under the "
        "assumption that no unrecorded arm-specific process differed. Fixed labels and nondeterministic generation prevent a randomized causal claim."
    ))
    story.append(P(
        "Three distinctions matter. First, assumption language rises from 0/32 to 28/32, but this is a manipulation check because the prompt requests such "
        "language. Second, output grows by 2,451 tokens, partly because the direction asks for more justification; length is a cost, not independent evidence "
        "of conceptual change. Third, deterministic source novelty, AST distance, and changed lines cannot by themselves identify a useful mechanism. Their "
        "combination with qualification and unconditional parameter reduction is the primary evidence: edits are larger, success is somewhat rarer, and "
        "successful jumps are much larger. The family taxonomy on the next page illustrates what those edits look like but is not needed for this conclusion."
    ))
    pagebreak(story)

    # Main page 5: message mechanisms and cases.
    story.append(h1("5  Exploratory trace taxonomy and matched cases"))
    story.extend(figure(
        DERIVED / "fig5_mechanism_taxonomy.png",
        5.35 * inch,
        "Figure 2: Exploratory, mutually exclusive mechanism-family tags from all 64 fork messages, implemented as a transparent classifier over the declared "
        "mechanism label. Counts sum to 32 per arm. The taxonomy was designed by one condition-aware analyst after seeing the corpus and was not independently coded.",
    ))
    story.append(P(
        "Ordinary agents mostly interpolate a known boundary: 18/32 narrow the feedforward path and 13/32 remove or tie normalization/bias parameters. "
        "Treated agents instead separate token rank from residual width (17), reparameterize attention routing (9), exchange width for iterative/shared "
        "depth (4), reuse projections across sublayers (1), or replace absolute positions with relative-offset attention (1). Manual review finds a "
        "structural alternative in every treated fork. Because this manual read was condition-aware and post hoc, it is qualitative context rather than a "
        "validated outcome. A conservative assumption-language regex detects 28/32, but that phrase is requested by the prompt and therefore verifies uptake, "
        "not independent defixation. The prereproducible evidence is the source/AST departure, qualification, retention, and parameter change in Table 2."
    ))
    story.append(h2("Matched examples expose the exploration policy"))
    story.append(P(
        "<b>Greedy B1, single memory.</b> The control changes FFN width 24 to 22 after width 20 missed 99%: a two-line midpoint test. Its twin states that "
        "prior width tests confounded symbol and residual width, then inserts an 8-dimensional token code projected into the supported 12-dimensional "
        "attention stream. Both fail. The intervention changes the hypothesis, not the fact that alternatives can be infeasible."
    ))
    story.append(P(
        "<b>Native B2, single memory.</b> The treated agent challenges independent FFN matrices by reusing the query and attention-output projections as "
        "nonlinear channel mixers. It qualifies and removes 2,080 parameters. The next ordinary proposal ties those projections and removes another 1,024; "
        "later proposals prune normalization and attention biases. Its matched control remains on FFN-width and bias boundary search."
    ))
    story.append(P(
        "<b>Greedy B8, single memory.</b> The treated agent replaces one 32-wide attention stage with two 16-wide stages. It qualifies and removes 4,080 "
        "parameters at once, then ordinary follow-ups remove another 352 while preserving the new depth-for-width mechanism. The control twin removes 160 "
        "through normalization and bias edits over the same phase. These traces motivate our term <i>punctuated defixation</i>: a semantic jump followed by "
        "local exploitation, not continuously maximal novelty."
    ))
    pagebreak(story)

    # Main page 6: propagated/repeated effects.
    story.append(h1("6  The fork propagates through later ordinary search"))
    story.extend(figure(
        DERIVED / "fig2_parameter_trajectories.png",
        5.25 * inch,
        "Figure 3: Arm-wise median incumbent parameters with interquartile bands (visual summary, not a paired estimator). Paired arms are identical through "
        "proposal 9; lower is better. Vertical lines are scheduled intervention opportunities. Text and tables report paired mean differences. All 64 runs are observed through 70.",
    ))
    story.append(P(
        "Before any second treatment, proposals 10-19 leave treated incumbents 786 parameters smaller [-1,271, -345]: 4,557 versus 5,344. The reduction "
        "fraction rises from 7.72% to 16.03% (+8.31 points [4.37, 12.68]) while qualification differs by only -3.13 points [-9.69, 3.13]. Mean source "
        "novelty across all ten proposals differs by +0.0031 [-0.0027, 0.0089], much less than the immediate +0.0309. Thus the single intervention is followed "
        "by a different architectural state from which ordinary local search proceeds; it does not keep every later edit novel."
    ))
    story.extend(figure(
        DERIVED / "fig4_horizon_effect.png",
        4.25 * inch,
        "Figure 4: Mean paired treatment-minus-control incumbent parameters as the common horizon expands. Shading is the descriptive cluster-bootstrap 95% range; "
        "negative favors treatment. Points at 10, 20, ..., 70 include a new intervention at their endpoint; intervening points show ordinary exploitation.",
    ))
    story.append(P(
        "By proposal 70, the seven-intervention adaptive regime yields 3,015 mean parameters versus 4,667, a paired difference of -1,652 [-2,301, -1,083]. "
        "This contrast combines repeated prompts, altered parents, and path-dependent retention; it is not a sevenfold estimate of the proposal-10 insertion. Treatment is lower "
        "in 28/32 pairs. Qualification falls by 9.84 points [-16.50, -2.87], but total accounted tokens are nearly unchanged (+5,045 on a 1.68M control "
        "mean, interval crossing zero); evaluator time rises by 1,378 seconds [367, 2,371]."
    ))
    pagebreak(story)

    # Main page 7: moderation and replication.
    story.append(h1("6.1  Search architecture and memory stratify the contrast"))
    moderation_data = [
        ["Stratum", "Pairs", "Control final P", "Challenge final P", "Paired difference [95% range]", "Challenge lower"],
        ["Greedy", "16", "3,554", "2,711", "-843 [-1,427, -407]", "14/16"],
        ["Native", "16", "5,780", "3,319", "-2,461 [-3,611, -1,437]", "14/16"],
        ["Single memory", "16", "5,160", "2,983", "-2,177 [-3,489, -1,148]", "16/16"],
        ["Portfolio", "16", "4,174", "3,047", "-1,127 [-1,913, -407]", "12/16"],
    ]
    story.append(data_table(moderation_data, [70, 34, 70, 79, 116, 55], font_size=6.6))
    story.append(P(
        "Table 3: Common-horizon 70 results. P denotes qualified incumbent parameters. Architecture and memory comparisons are exploratory moderation, "
        "not randomized interaction tests.",
        CAPTION,
    ))
    story.append(P(
        "The direction is stable across both research systems and both memory regimes. The larger native contrast is consistent with population search "
        "preserving and recombining a high-impact fork, but framework-specific parents and selection histories prevent isolating that mechanism. Portfolio "
        "controls already end 987 parameters smaller than single-memory controls on average, leaving less marginal room for the prompt. This supports an "
        "institutional interpretation: semantic defixation and population memory may be partially substitutable exploration operators. A future randomized "
        "factorial study should test that interaction directly; here it remains a measured hypothesis, not a causal conclusion."
    ))
    story.append(h1("7  Descriptive portability check: Fashion-MNIST"))
    story.append(P(
        "We reanalyze a complete Fashion-MNIST campaign: five blocks x four conditions x 200 proposals = 4,000 events. It uses the same every-tenth "
        "intervention concept but a different task, score, source program, and evaluator. Same-block arms had already diverged before each checkpoint, so "
        "these are descriptive matched contrasts and checkpoint-minus-prior-step differences, not exact forks. Blocks 1-3 are the original scope and "
        "Blocks 4-5 a recorded extension; Appendix E reports them separately."
    ))
    fashion_data = [
        ["Checkpoint outcome", "Ordinary", "Challenge", "Difference"],
        ["Source novelty", "0.0298", "0.0640", "+0.0346"],
        ["New family", "0.5%", "9.0%", "+8.5 pp"],
        ["Assumption language", "0.5%", "78.0%", "+77.5 pp"],
        ["Qualified", "62.0%", "31.0%", "-31.0 pp"],
        ["Retained", "36.5%", "5.0%", "-31.5 pp"],
        ["Output tokens", "1,271", "4,532", "+3,261"],
    ]
    story.append(data_table(fashion_data, [127, 77, 85, 89], font_size=7.0))
    story.append(P(
        "Table 4: 200 same-block intervention checkpoints on already-diverged paths. Proposal-level dependence precludes treating these as independent replications.",
        CAPTION,
    ))
    story.append(P(
        "Relative to each trajectory's preceding opportunity, the treated-control change is +0.0383 for source novelty (169 source-observed pairs), -31.5 "
        "points for qualification, -29.5 for retention, and +2,765 output tokens. Fashion-MNIST shows a directionally similar descriptive redirection/cost "
        "pattern but does not causally replicate the Tiny AdderBoard contrast or its compression benefit. The result cautions against interpreting more novel code as better research independent of the "
        "task's feasibility landscape and selection rule."
    ))
    story.append(h2("Why the task boundary matters"))
    story.append(P(
        "Tiny AdderBoard has a hard qualification cliff and many algebraic or architectural redundancies: a rare feasible change can remove thousands of "
        "parameters and then become a productive parent. Fashion-MNIST uses a different score and source program; checkpoint proposals on treated paths are "
        "more structurally distant but are retained only 5.0% of the time versus 36.5% for controls. This is consistent with an exploration operator whose "
        "value depends on the density of high-reward alternatives, not a task-general novelty premium. Because the Fashion paths were not state-matched, that "
        "explanation is a hypothesis, not an identified moderator."
    ))
    story.append(P(
        "The original three Fashion blocks and the two-block extension show the same directional source-novelty and qualification pattern (Appendix E), "
        "which reduces concern that the aggregate is created by one extension block. It does not repair the missing exact fork. A proper portability test "
        "would mirror the first nine events and parent source on the new task, predeclare source and feasibility outcomes, and use enough blocks to estimate "
        "task-by-intervention heterogeneity."
    ))
    pagebreak(story)

    # Main page 8: implications, limitations, conclusion.
    story.append(h1("8  Discussion"))
    story.append(h2("A prompt is an exploration operator, not a creativity guarantee"))
    story.append(P(
        "The matched fork shows a behavioral contrast: an inserted direction is followed by larger source departures and more architectural alternatives than "
        "the control direction. The first alternative often fails, but successful jumps are an order of magnitude larger and create new basins for "
        "ordinary exploitation. This resembles a high-variance mutation operator, not a universally better policy. Research-agent evaluations should "
        "therefore report at least four coupled outcomes: departure from prior mechanisms, evaluator qualification, improvement magnitude, and resource "
        "cost. A novelty-only score would reward many Fashion-MNIST failures; an endpoint-only score would hide how the Tiny AdderBoard gains arose [7]."
    ))
    story.append(h2("Prompted alternatives can themselves fixate"))
    story.append(P(
        "The intervention escapes one local rut but creates a population-level attractor: 17/32 treated forks choose low-rank token interfaces. The prompt "
        "does not enumerate that mechanism. Convergence plausibly reflects shared architecture affordances and model priors. Future interventions should "
        "condition on population-level repetition, or pair a defixation direction with selection mechanisms that reward qualified coverage rather than "
        "textual difference. This is testable with the artifact's family labels; it is not resolved by the present sample."
    ))
    story.append(h2("Limitations"))
    story.append(P(
        "<b>One model and compact tasks.</b> The primary evidence uses one subject-model configuration and one synthetic transformer-compression task; "
        "Fashion-MNIST changes several factors at once. <b>Fixed labels.</b> Exact shared prefixes remove observed pre-fork history differences, but condition "
        "labels were not randomized and subject generation may not be deterministic. <b>Proxy validity.</b> Token/AST distance and lexical families are "
        "transparent but incomplete measures of mechanistic novelty. The family taxonomy was constructed post hoc by one condition-aware analyst and has no "
        "independent-coder reliability; it is exploratory. Primary claims instead triangulate deterministic source/AST differences with evaluator and retention "
        "records [17,18]. <b>Demand characteristics.</b> Assumption language and some output-length increase are direct consequences of the requested response; "
        "neither is independent evidence of search quality. <b>Adaptive selection.</b> Horizon-70 contrasts combine repeated prompts with path-dependent selection; only proposal 10 and "
        "phase 10-19 isolate the first insertion. <b>Evaluator scope.</b> Qualification is fixed-task accuracy, not external scientific validity, robustness, "
        "or human usefulness. <b>Finite clusters.</b> Sixteen architecture-block clusters limit uncertainty calibration; ranges are descriptive."
        , SMALL))
    story.append(h2("Reproducibility and ethical scope"))
    story.append(P(
        "The 52 MB anonymized supplement contains protocol and prompt snapshots, events, all analyzed final messages, candidate sources/provenance, deterministic "
        "analysis code, derived tables, figures, environment receipts, an MIT license, and file hashes. It excludes credentials, machine locks, provider "
        "streams, and private chain-of-thought. No human-subject data are used. We avoid attributing intention or understanding to the agent; 'challenged' "
        "describes an instruction and observable proposal change."
        , SMALL))
    story.append(h2("Implications for AI-for-science evaluation"))
    story.append(P(
        "Endpoint leaderboards alone cannot distinguish a genuinely different search move from prolonged local pruning, while novelty-only scores can reward "
        "unexecutable churn. For agentic science, a minimum process report should pair (i) source- or action-level departure, (ii) domain qualification, "
        "(iii) retained improvement magnitude, and (iv) token and evaluator cost. State matching at an intervention boundary makes these quantities more "
        "interpretable, but construct labels should remain proportional to their validation evidence [17,18]."
        , SMALL))
    story.append(h1("9  Conclusion"))
    story.append(P(
        "From event- and provenance-matched search states, one semantic direction is followed by substantially larger executable source changes than the "
        "ordinary direction. On Tiny AdderBoard, that higher-risk exploration is also associated with smaller qualified models after one phase and under a "
        "seven-intervention adaptive regime; on Fashion-MNIST, the corresponding descriptive pattern mostly "
        "consists of costlier failure. The appropriate conclusion is neither that prompts create creativity nor that novelty is waste. Assumption challenges "
        "are measurable search operators whose value depends on qualification, selection, memory, and task landscape. State-matched process traces make "
        "those dependencies visible."
    ))
    pagebreak(story)

    # References begin after exactly eight main-text pages.
    story.append(h1("References"))
    references = [
        "Jansson, D. G. and Smith, S. M. (1991). Design fixation. <i>Design Studies</i>, 12(1):3-11. doi:10.1016/0142-694X(91)90003-F.",
        "Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. <i>NeurIPS 36</i>. doi:10.52202/075280-0377.",
        "Mu, L., Zhang, W., Zhang, Y., and Jin, P. (2024). DDPrompt: Differential diversity prompting in large language models. <i>ACL Short Papers</i>, 168-174. doi:10.18653/v1/2024.acl-short.17.",
        "Tian, Y., Ravichander, A., Qin, L., et al. (2024). MacGyver: Are large language models creative problem solvers? <i>NAACL</i>, 5303-5324. doi:10.18653/v1/2024.naacl-long.297.",
        "Lu, Y., Wang, D., Li, T., et al. (2025). Benchmarking language model creativity: A case study on code generation. <i>NAACL</i>, Long Paper 141.",
        "Li, K., Zhao, T., Zhou, W., and Hu, S. (2025). DORA: Dynamic optimization prompt for continuous reflection of LLM-based agent. <i>COLING</i>, 7546-7557.",
        "Nakajima, K., Zuiderveld, J., and Pezzelle, S. (2026). Beyond divergent creativity: A human-based evaluation of creativity in large language models. <i>Findings of EACL</i>, 2639-2660. doi:10.18653/v1/2026.findings-eacl.138.",
        "Ege, D. N., Ovrebo, H. H., Stubberud, V., Berg, M. F., Elverum, C., Steinert, M., and Vestad, H. (2025). ChatGPT as an inventor: Eliciting strengths and weaknesses against humans in engineering design. <i>Artificial Intelligence for Engineering Design, Analysis and Manufacturing</i>, 39:e6, 1-15. doi:10.1017/S0890060425000010.",
        "Huang, Q., Vora, J., Liang, P., and Leskovec, J. (2024). MLAgentBench: Evaluating language agents on machine learning experimentation. <i>ICML, PMLR 235</i>:20271-20309.",
        "Toledo, E., Hambardzumyan, K., Josifoski, M., et al. (2025). AI research agents for machine learning: Search, exploration, and generalization in MLE-bench. arXiv:2507.02554.",
        "Novikov, A., Vu, N., Eisenberger, M., et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131.",
        "Sharma, A. (2025). OpenEvolve: An open-source evolutionary coding agent. Software repository: github.com/algorithmicsuperintelligence/openevolve.",
        "Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. (2024). The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv:2408.06292.",
        "Chen, H. and Ding, N. (2023). Probing the creativity of large language models: Can models produce divergent semantic association? <i>Findings of EMNLP</i>, 12881-12888. doi:10.18653/v1/2023.findings-emnlp.858.",
        "Sen, T. M., Chun, Z. C. K., Alsagoff, S. A. R., Wangsajaya, N. Y., Mohor, B., Saikia, S. B., and Chan, A. (2026). Automated creativity evaluation of language models across open-ended tasks. <i>ACL</i>, 23139-23173. doi:10.18653/v1/2026.acl-long.1061.",
        "Imbens, G. W. and Rubin, D. B. (2015). <i>Causal Inference for Statistics, Social, and Biomedical Sciences</i>. Cambridge University Press.",
        "Bean, A. M., et al. (2025). Measuring what matters: Construct validity in large language model benchmarks. <i>NeurIPS 38, Datasets and Benchmarks</i>. doi:10.52202/085713-0590.",
        "Reuel, A., Hardy, A., Smith, C., Lamparth, M., Hardy, M., and Kochenderfer, M. J. (2024). BetterBench: Assessing AI benchmarks, uncovering issues, and establishing best practices. <i>NeurIPS 37, Datasets and Benchmarks</i>. doi:10.52202/079017-0685.",
        "Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., and Ha, D. (2025). The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv:2504.08066.",
        "Hu, S., Lu, C., and Clune, J. (2025). Automated design of agentic systems. <i>ICLR</i>.",
        "Assumpcao, H., Ferreira, D., Campos, L., and Murai, F. (2025). CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization. arXiv:2510.14150.",
        "Wan, C., Dai, X., Wang, Z., Li, M., Wang, Y., Mao, Y., Lan, Y., and Xiao, Z. (2025). LoongFlow: Directed evolutionary search via a cognitive plan-execute-summarize paradigm. arXiv:2512.24077.",
        "Lyu, Y., Zhang, X., Yi, X., et al. (2026). EvoScientist: Towards multi-agent evolving AI scientists for end-to-end scientific discovery. arXiv:2603.08127.",
    ]
    for index, reference in enumerate(references, 1):
        story.append(P(f"[{index}] {reference}", REF))
    pagebreak(story)

    # Appendix A: exact prompt insertion.
    story.append(h1("A  Intervention and fork verification"))
    story.append(h2("A.1  Inserted direction at proposal 10"))
    story.append(P(
        "The treated prompt inserts the following direction into the otherwise matched prompt. Line wrapping differs by snapshot, but content is fixed:",
        SMALL,
    ))
    story.append(P(
        "Before choosing the next change, step back from the current line of work. Think very critically about the assumptions made so far and shared by "
        "available designs. Identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions and test genuinely "
        "different learned computational mechanisms. Think critically about how the next change could make more progress and implement that thinking. "
        "The change should alter how the transformer represents or computes the task. Do not revisit a failed type of change unless recent evidence gives "
        "a specific reason it should now behave differently. Prefer a clean test of the alternative; state the old assumption and new approach in the final "
        "summary. Use prior results to explain why the alternative is plausible and informative. Identify the strongest evidence for and against it, make "
        "a decisive feasible test, preserve supported components, and explain what result would falsify the approach.",
        QUOTE,
    ))
    story.append(P(
        "This appendix paraphrases only line wrapping and joins adjacent protocol sentences. The artifact contains the exact opportunity-10 `prompt.md` for "
        "every arm, the subject-facing prompt snapshot, and the prompt manifest. The analysis uses a line-level sequence diff and rejects any pair with deletion or replacement; all 32 "
        "forks pass and contain at least one inserted line mentioning assumptions.",
        SMALL,
    ))
    story.append(h2("A.2  State matching checks"))
    integrity_data = [
        ["Check", "Expected", "Observed"],
        ["Primary runs", "64", str(AGG["primary_runs"])],
        ["Paired forks", "32", str(AGG["primary_fork_pairs"])],
        ["Common horizon", "70", str(AGG["primary_common_horizon"])],
        ["Prefix event matches", "288", str(AGG["primary_integrity"]["prefix_event_matches"])],
        ["Prefix source/provenance matches", "288", str(AGG["primary_integrity"]["prefix_source_or_provenance_matches"])],
        ["Fork parent matches", "32", str(AGG["primary_integrity"]["fork_parent_matches"])],
        ["Insertion-only prompt diffs", "32", str(AGG["primary_integrity"]["fork_prompt_insertion_only"])],
        ["Horizon event/message rows", "4,480 / 4,480", f"{AGG['primary_rows']:,} / {AGG['primary_recorded_messages']:,}"],
    ]
    story.append(data_table(integrity_data, [190, 90, 116], font_size=7.2))
    story.append(h2("A.3  Condition assignment and identification"))
    story.append(P(
        "Conditions were protocol-fixed rather than randomly assigned. Their ordinal launch positions were permuted across blocks: across 16 architecture-block "
        "schedules, C0/C1 appeared in each of positions 1-4 a combined 8, 5, 9, and 10 times; C2/C3 appeared 8, 11, 7, and 6 times. Literal prefix mirroring "
        "produces equal observed histories and selected parents at the "
        "fork, supporting a controlled state-matched contrast under the assumption that no unrecorded arm-specific process affects proposal generation. "
        "It does not justify randomization-based p-values. Exact matching is strongest for opportunity 10. During opportunities 11-19, paths differ because "
        "proposal 10 may change the incumbent; this is the propagated contrast following the first intervention. At 20 and later, contrasts include repeated treatment "
        "and adaptive selection."
    ))
    pagebreak(story)

    # Appendix B: operational definitions and complete fork effects.
    story.append(h1("B  Operational definitions and complete fork effects"))
    story.append(h2("B.1  Source and text measures"))
    story.append(P(
        "Python tokenization removes comments, indentation, and line boundaries; preserves operators and keywords; and maps identifiers, numbers, and strings "
        "to ID, NUM, and STR. Source structural novelty is one minus Jaccard similarity of token-trigram sets. AST distance is one minus multiset overlap of "
        "node types. If either candidate or parent source is absent, both are missing rather than zero. Changed lines count additions plus deletions in a "
        "zero-context unified diff. Declared lexical novelty is one minus the maximum word-set Jaccard similarity to earlier mechanism/hypothesis/edit summaries "
        "in the run after a fixed stop list. Broad family, assumption, and mechanism-shift indicators are fixed regexes in analysis.py.",
        SMALL,
    ))
    story.append(h2("B.2  Outcome and cost measures"))
    story.append(P(
        "Qualification is the evaluator's valid flag, which requires at least 99% exact-answer accuracy plus task integrity checks. Retention is the recorded "
        "controller decision. Became-incumbent requires retention and candidate/incumbent identity. Immediate reduction is the pre-event incumbent's "
        "deduplicated learned parameters minus the post-event incumbent's; failed or unretained proposals therefore contribute zero. Total, input, cached-input, "
        "and output tokens are recorded per-event usage increments. Evaluator seconds are recorded elapsed evaluation increments and include scheduled training "
        "checkpoints."
    ))
    fork_rows = [row for (subset, _metric), row in FORK_EFFECTS.items() if subset == "all"]
    metric_labels = {
        "structural_novelty": "Source novelty",
        "ast_distance": "AST distance",
        "changed_lines": "Changed lines",
        "declared_lexical_novelty": "Declared novelty",
        "new_family_tag": "New family tag",
        "assumption_language": "Assumption language",
        "mechanism_shift_language": "Mechanism-shift language",
        "qualified": "Qualified",
        "qualified_structural": "Qualified structural",
        "retained": "Retained",
        "retained_structural": "Retained structural",
        "became_incumbent": "Became incumbent",
        "immediate_parameter_reduction": "Immediate parameter reduction",
        "total_tokens": "Total tokens",
        "output_tokens": "Output tokens",
        "evaluator_seconds": "Evaluator seconds",
    }
    full_effect_data = [["Outcome", "Control", "Treatment", "Difference", "95% range"]]
    for row in fork_rows:
        metric = row["metric"]
        if metric not in metric_labels:
            continue
        scale = 100 if metric in {
            "new_family_tag", "assumption_language", "mechanism_shift_language",
            "qualified", "qualified_structural", "retained", "retained_structural",
            "became_incumbent",
        } else 1
        digits = 1 if scale == 100 or metric in {"changed_lines", "immediate_parameter_reduction", "total_tokens", "output_tokens", "evaluator_seconds"} else 4
        full_effect_data.append([
            metric_labels[metric],
            f"{scale * float(row['control_mean']):.{digits}f}",
            f"{scale * float(row['treated_mean']):.{digits}f}",
            f"{scale * float(row['paired_difference']):+.{digits}f}",
            interval(row, scale=scale, digits=digits),
        ])
    story.append(data_table(full_effect_data, [115, 57, 60, 69, 95], font_size=6.35))
    story.append(P(
        "Table B1: All exact-fork outcomes. Rates are displayed as percentages but differences are percentage points. Descriptive ranges resample architecture-block "
        "clusters and include both memory pairs within a selected cluster.",
        CAPTION,
    ))
    pagebreak(story)

    # Appendix C: taxonomy and downstream results.
    story.append(h1("C  Exploratory taxonomy and downstream contrasts"))
    mechanism_data = [["Arm", "Mechanism family", "n", "Qualified", "Mean reduction", "Median reduction if success"]]
    labels = {
        "attention_routing_reparameterization": "Attention routing",
        "feedforward_compression": "Feedforward compression",
        "normalization_or_bias_pruning": "Normalization/bias pruning",
        "iterative_or_shared_depth": "Iterative/shared depth",
        "projection_reuse": "Projection reuse",
        "relative_position_attention": "Relative position",
        "token_interface_factorization": "Token interface",
    }
    for row in MECHANISMS:
        median = row["median_successful_parameter_reduction"]
        mechanism_data.append([
            "Challenge" if row["arm"] == "assumption_challenge" else "Ordinary",
            labels[row["mechanism_family"]],
            row["n"],
            pct(row["qualified_rate"]),
            f(row["mean_parameter_reduction"], 0),
            "-" if median == "nan" else f(median, 0),
        ])
    story.append(data_table(mechanism_data, [56, 115, 25, 55, 73, 72], font_size=6.35))
    story.append(P(
        "Table C1: Post hoc, condition-aware mutually exclusive taxonomy of all fork messages. Means include failures as zero; conditional medians are descriptive.",
        CAPTION,
    ))
    story.append(h2("C.1  One-intervention phase and repeated regime"))
    downstream_data = [["Outcome", "Window", "Control", "Treatment", "Paired difference [95% range]"]]
    for label, table, metrics in [
        ("O10-19", PHASE1_EFFECTS, ["final_parameters", "parameter_reduction_fraction", "qualified_rate", "structural_novelty_mean", "evaluator_seconds"]),
        ("O10-70", TRAJECTORY_EFFECTS, ["final_parameters", "parameter_reduction_fraction", "qualified_rate", "structural_novelty_mean", "tokens", "evaluator_seconds"]),
    ]:
        for metric in metrics:
            row = effect(table, "all", metric)
            metric_label = {
                "final_parameters": "Final parameters",
                "parameter_reduction_fraction": "Reduction fraction",
                "qualified_rate": "Qualification rate",
                "structural_novelty_mean": "Mean source novelty",
                "tokens": "Total tokens",
                "evaluator_seconds": "Evaluator seconds",
            }[metric]
            scale = 100 if metric in {"parameter_reduction_fraction", "qualified_rate"} else 1
            digits = 2 if metric == "structural_novelty_mean" else (1 if scale == 100 else 0)
            downstream_data.append([
                metric_label,
                label,
                f"{scale * float(row['control_mean']):.{digits}f}",
                f"{scale * float(row['treated_mean']):.{digits}f}",
                f"{scale * float(row['paired_difference']):+.{digits}f} {interval(row, scale, digits)}",
            ])
    story.append(data_table(downstream_data, [101, 50, 58, 61, 126], font_size=6.35))
    story.append(P(
        "Table C2: O10-19 includes exactly one intervention; O10-70 includes seven. Rate values are percentages and differences percentage points.",
        CAPTION,
    ))
    story.append(h2("C.2  Horizon emergence"))
    horizon_data = [["Horizon", "Interventions", "Control P", "Treatment P", "Difference [95% range]", "Treatment lower"]]
    for row in HORIZONS:
        if int(row["horizon"]) not in {10, 19, 20, 30, 40, 50, 60, 70}:
            continue
        horizon_data.append([
            row["horizon"], row["interventions_received"],
            f(row["control_final_parameters_mean"], 0),
            f(row["treated_final_parameters_mean"], 0),
            f"{float(row['paired_difference']):+.0f} [{float(row['cluster_bootstrap_low']):.0f}, {float(row['cluster_bootstrap_high']):.0f}]",
            f"{row['treated_lower_count']}/32",
        ])
    story.append(data_table(horizon_data, [46, 62, 65, 67, 101, 55], font_size=6.45))
    pagebreak(story)

    # Appendix D: trace anchors.
    story.append(h1("D  Trace-level qualitative audit"))
    story.append(P(
        "One condition-aware analyst read all 64 opportunity-10 final messages before defining the exclusive taxonomy. The following cases were selected to cover success/failure, "
        "greedy/native controllers, single/portfolio memory, and distinct mechanism families. Each claim was cross-checked against the candidate source, "
        "evaluation, retention event, and incumbent parameter sequence.",
        SMALL,
    ))
    trace_data = [
        ["Trace", "Arm", "Declared mechanism", "Qualified", "Immediate reduction", "O19 incumbent"],
        ["Greedy B1 C0", "Ordinary", "FFN midpoint 24 to 22", "No", "0", "2,808"],
        ["Greedy B1 C1", "Challenge", "Rank-8 symbol manifold", "No", "0", "2,494"],
        ["Greedy B6 C0", "Ordinary", "Bias-free normalization", "Yes", "60", "4,240"],
        ["Greedy B6 C1", "Challenge", "Relative-offset attention", "Yes", "176", "3,954"],
        ["Greedy B8 C0", "Ordinary", "Bias-free final norm", "No", "0", "8,160"],
        ["Greedy B8 C1", "Challenge", "Depth-for-width attention", "Yes", "4,080", "3,888"],
        ["Native B2 C0", "Ordinary", "Multi-query plus FFN 24", "No", "0", "8,352"],
        ["Native B2 C1", "Challenge", "Cross-sublayer projection reuse", "Yes", "2,080", "5,664"],
        ["Native B5 C2", "Ordinary", "Bias-free width/FFN combination", "No", "0", "11,169"],
        ["Native B5 C3", "Challenge", "Shared key/value transport", "Yes", "2,904", "6,450"],
        ["Native B7 C0", "Ordinary", "FFN boundary interpolation", "Yes", "50", "2,608"],
        ["Native B7 C1", "Challenge", "Rank-8 symbol representation", "Yes", "360", "2,298"],
    ]
    story.append(data_table(trace_data, [63, 54, 143, 48, 67, 55], font_size=5.95))
    story.append(P("Table D1: Selected matched-fork traces and propagated incumbent at opportunity 19.", CAPTION))
    story.append(h2("D.1  Verbatim proposal fragments"))
    story.append(P(
        "<b>Greedy B8 C1/O10:</b> 'The current design assumes one 32-wide attention stage is the most parameter-efficient computation; two sequential "
        "16-wide attention stages will instead retain at least 99% accuracy.' The patch replaces one block with two learned blocks. It reaches 99.94% and "
        "removes 4,080 parameters.",
        QUOTE,
    ))
    story.append(P(
        "<b>Native B2 C1/O10:</b> 'Reusing the full-width query and attention-output projections as the feedforward channel mixers' tests whether independent "
        "FFN matrices are load-bearing. The patch routes normalized hidden states through the attention query weight, GELU plus learned channel gain, and the "
        "attention output weight. It qualifies and is retained.",
        QUOTE,
    ))
    story.append(P(
        "<b>Greedy B6 C1/O10:</b> 'No result shows that full-width absolute position vectors are necessary'; the agent preserves position information through "
        "44 learned per-head relative-distance biases. The edit qualifies, and later ordinary steps remove normalization and gauge-redundant relative biases.",
        QUOTE,
    ))
    story.append(h2("D.2  Coding limits"))
    story.append(P(
        "The taxonomy was induced after the analyst saw condition labels and outcomes, and it was not checked by independent coders. Category counts should be "
        "treated as auditable hypothesis generation rather than a validated ontology or confirmatory endpoint. The deterministic classifier uses only the declared "
        "mechanism label because evidence and intended-edit fields frequently mention preserved components and prior failures. All final messages and classifier rules "
        "are included for blinded recoding in future work.",
        SMALL,
    ))
    pagebreak(story)

    # Appendix E: Fashion replication and scope.
    story.append(h1("E  Fashion-MNIST portability check and scope sensitivity"))
    story.append(h2("E.1  Matched checkpoint contrasts"))
    fashion_metrics = [
        ("structural_novelty", "Source novelty", 1, 4),
        ("ast_distance", "AST distance", 1, 4),
        ("changed_lines", "Changed lines", 1, 1),
        ("declared_lexical_novelty", "Declared novelty", 1, 3),
        ("new_family_tag", "New family tag", 100, 1),
        ("assumption_language", "Assumption language", 100, 1),
        ("qualified", "Qualified", 100, 1),
        ("retained", "Retained", 100, 1),
        ("output_tokens", "Output tokens", 1, 0),
        ("evaluator_seconds", "Evaluator seconds", 1, 1),
    ]
    replication_data = [["Scope", "Outcome", "Control", "Treatment", "Difference"]]
    for scope in ("original", "extension", "all"):
        for metric, label, scale, digits in fashion_metrics:
            row = FASHION_EFFECTS[(scope, metric)]
            replication_data.append([
                {"original": "B1-3", "extension": "B4-5", "all": "All"}[scope],
                label,
                f"{scale * float(row['control_mean']):.{digits}f}",
                f"{scale * float(row['treated_mean']):.{digits}f}",
                f"{scale * float(row['paired_difference']):+.{digits}f}",
            ])
    story.append(data_table(replication_data, [44, 104, 68, 74, 73], font_size=5.85))
    story.append(P(
        "Table E1: Descriptive contrasts at every tenth opportunity. B1-3 are the original campaign scope; B4-5 are a recorded extension. Rates are percentages "
        "and differences percentage points. No cluster interval is shown because paths are not exact forks and only five blocks are complete.",
        CAPTION,
    ))
    story.append(h2("E.2  Checkpoint-minus-previous-opportunity contrast"))
    did_data = [["Outcome", "Usable pairs", "Treated-control change"]]
    for metric, label, scale, digits in fashion_metrics:
        key = ("all_checkpoint_did", metric)
        if key not in FASHION_EFFECTS:
            continue
        row = FASHION_EFFECTS[key]
        did_data.append([
            label,
            row["n_checkpoint_pairs"],
            f"{scale * float(row['paired_difference']):+.{digits}f}",
        ])
    story.append(data_table(did_data, [155, 95, 146], font_size=6.6))
    story.append(P(
        "Table E2: Within-trajectory checkpoint change minus matched control change. Source-based outcomes omit pairs without both source snapshots. These "
        "differences reduce stable arm-level differences but remain descriptive because the paths and parents differ.",
        CAPTION,
    ))
    pagebreak(story)

    # Appendix F: reproducibility and provenance.
    story.append(h1("F  Reproducibility and provenance manifest"))
    manifest_data = [
        ["Item", "Recorded value"],
        ["Primary protocol", "Unified v3.0"],
        ["Subject configuration", "GPT-5.6 Sol, xhigh reasoning"],
        ["Primary task", "Exact four-digit addition, >=99% qualification"],
        ["Baseline learned parameters", "21,952"],
        ["Primary systems", "Greedy OpenEvolve; native OpenEvolve"],
        ["Blocks / conditions / runs", "8 / 4 / 64"],
        ["Primary analysis horizon", "70 proposals per run"],
        ["Primary event/message rows", "4,480 / 4,480"],
        ["Fashion replication", "20 runs x 200 proposals = 4,000"],
        ["Analysis seed", str(AGG["analysis_seed"])],
        ["Bootstrap repetitions", "10,000 per effect/stratum"],
        ["Artifact checksum", "PAPER2_SHA256SUMS (SHA-256 per payload file)"],
    ]
    story.append(data_table(manifest_data, [155, 241], font_size=7.15))
    story.append(Spacer(1, 8))
    story.append(h2("F.1  Clean reproduction"))
    story.append(P(
        "From the anonymized archive root, install the fixed versions in requirements.txt and run:<br/>"
        "<font name='AISciKTimes'>MPLCONFIGDIR=/tmp/aiscik-mpl python3 papers/aiscik2026/paper2/analysis.py</font><br/>"
        "The script asserts all integrity counts before writing aggregate.json; primary, fork, phase, trajectory, horizon, mechanism, and Fashion CSV tables; "
        "and five publication figures. It uses no network access or model call. Missing candidate snapshots remain missing and are never imputed as zero.",
        SMALL,
    ))
    story.append(h2("F.2  Included evidence"))
    story.append(bullet("Campaign, protocol, task, framework, schedule, validation, prompt-bundle, amendment, and environment-receipt records."))
    story.append(bullet("Per-run manifests, state files, completed-event streams, exact fork prompts, prompt manifests, per-opportunity candidate provenance, and all analyzed final messages."))
    story.append(bullet("All candidate and selected-parent `train.py` snapshots needed for normalized-token, AST, and line-diff measures."))
    story.append(bullet("Deterministic analysis source, fixed dependencies, derived tables/figures, artifact guide, MIT license, and per-file SHA-256 hashes."))
    story.append(h2("F.3  Exclusions and privacy"))
    story.append(P(
        "The compact archive excludes redundant per-opportunity prompt transcripts outside the exact fork, evaluator workspaces, native population checkpoints not read by the analysis, provider JSONL streams, "
        "machine-local locks, caches, credentials, absolute host paths, repository remotes, and private chain-of-thought. Recorded subject final messages "
        "are included because they are the declared scientific proposals analyzed in the paper. Text files are anonymized during packaging; candidate source "
        "and event semantics are preserved."
    ))

    return story


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = NeuripsDoc(str(OUTPUT))
    doc.build(build_story())
    return OUTPUT


if __name__ == "__main__":
    print(build())
