#!/usr/bin/env python3
"""Build the AISciK-ready Paper 1 PDF in a NeurIPS-like layout."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
OUTPUT = HERE.parents[2] / "output/pdf/paper1_construct_validity_audit.pdf"
MAIN_PAGES = 8


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


AGG = json.loads((DERIVED / "aggregate.json").read_text(encoding="utf-8"))
SCOPE = {row["scope"]: row for row in rows(DERIVED / "scope_summary.csv")}
PAIRED = {row["measure"]: row for row in rows(DERIVED / "paired_pre_post_summary.csv")}
RUNS = rows(DERIVED / "run_summary.csv")
EDIT_CLASSES = {row["edit_class"]: row for row in rows(DERIVED / "edit_class_summary.csv")}
TIEBREAK_MAG = {row["statistic"]: row for row in rows(DERIVED / "tiebreak_magnitude_summary.csv")}


def f(value: Any, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


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
            title="What Does an Autonomous Research Benchmark Measure?",
            author="Anonymous Authors",
            subject="AISciK Workshop (NeurIPS 2026) submission",
        )
        # NeurIPS specifies a 5.5 inch by 9 inch text rectangle. Keep the frame
        # exactly 5.5 inches wide and 9 inches high, offset 1.5 inches left.
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
    # Submission-style line numbers. They are a reading aid only and sit in the
    # margin, never in the text rectangle.
    canvas.setFont(FONT, 4.8)
    canvas.setFillColor(colors.HexColor("#9ca3af"))
    line_start = (page - 1) * 55 + 1
    for i in range(55):
        y = 9.47 * inch - i * (9.0 * inch / 54)
        canvas.drawRightString(1.42 * inch, y, str(line_start + i))
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
    spaceAfter=5.5,
    allowWidows=0,
    allowOrphans=0,
)
SMALL = ParagraphStyle(
    "Small",
    parent=BODY,
    fontSize=8.25,
    leading=9.25,
    spaceAfter=3.2,
)
TINY = ParagraphStyle(
    "Tiny",
    parent=BODY,
    fontSize=7.25,
    leading=8.1,
    spaceAfter=2.5,
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
    fontSize=9.5,
    leading=10.6,
    leftIndent=0.5 * inch,
    rightIndent=0.5 * inch,
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
    fontSize=7.7,
    leading=8.6,
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
    fontSize=8.1,
    leading=9.0,
    leftIndent=10,
    firstLineIndent=-10,
    spaceAfter=3.2,
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
        "What Does an Autonomous Research Benchmark Measure?<br/>"
        "<font size='13'>A Trace-Level Construct-Validity Audit of 4,000 Agent Proposals</font>",
        TITLE,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=2, spaceAfter=9))
    story.append(P("Anonymous Authors", AUTHOR))


def data_table(data: list[list[Any]], widths: list[float], header_rows: int = 1, font_size: float = 7.2) -> Table:
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
    image = Image(str(path), width=width, height=width * 0.36)
    image.hAlign = "CENTER"
    return [image, P(caption, CAPTION)]


def build_story() -> list[Flowable]:
    story: list[Flowable] = []

    # Main page 1: title, abstract, and introduction.
    ruled_title(story)
    story.append(P("Abstract", ABSTRACT_HEAD))
    story.append(P(
        "Benchmarks for autonomous machine-learning research commonly rank agents by the best scalar task score they obtain. "
        "We ask what such a score supports about the agent as a research process. We audit 20 completed trajectories in five matched "
        "trajectories, 4,000 proposals, 92.2 million accounted tokens, agent hypotheses and evidence statements, and candidate-parent "
        "source snapshots. The score lexicographically combines validation-correct count with cross-entropy as a continuous tie-break. "
        "Of 1,009 new scalar bests, 762 (75.5%) changed only the tie-break. After each trajectory's final correct-count gain, 38.0% of "
        "proposals, 36.2% of agent tokens, and 37.6% of evaluator time remained. Across trajectories, literal-only edits rose by 0.299 "
        "and source-structural novelty fell by 0.043 after that point; both directions persist in an eight-run extension. Agent records "
        "show increasingly precise searches over temperatures, ensemble weights, and numerical calibration, while source-structural "
        "edits were associated with more timeouts. Endpoint gains had weak and uncertain rank relationships with measured process properties. "
        "The benchmark validly measures adaptive score optimization, but score alone underdetermines mechanistic breadth, evidence-directed "
        "search, and generalization. We propose a process-aware reporting protocol for autonomous-research evaluations.",
        ABSTRACT,
    ))
    story.append(h1("1  Introduction"))
    story.append(P(
        "Language-model agents now write code, run experiments, inspect results, and iterate toward better machine-learning systems. "
        "MLAgentBench, MLE-bench, and RE-Bench operationalize this capability through task environments and scalar outcomes "
        "(Huang et al., 2024; Chan et al., 2025; Wijk et al., 2024). Evolutionary coding systems extend the same feedback loop to "
        "open-ended algorithm search (Novikov et al., 2025), while end-to-end systems place experimentation inside a larger scientific "
        "workflow (Lu et al., 2026). These systems make more experiments observable than conventional benchmark submissions: every "
        "hypothesis, edit, failure, and retention can be logged. Yet evaluations often collapse that record back into one number.",
    ))
    story.append(P(
        "A high task score is direct evidence that an agent optimized the scored task. It is not automatically evidence that the agent "
        "conducted broad, explanatory, or generalizable research. Validity concerns the interpretation supported by a score, not a binary "
        "property of the test (Cronbach and Meehl, 1955; Messick, 1995). This distinction matters when a metric is optimized by an agent "
        "that can observe hundreds of intermediate results and adapt its program accordingly.",
    ))
    story.append(P(
        "<b>Research question.</b> When an autonomous ML-research agent improves a scalar validation score, how often does the "
        "improvement reflect a new correct prediction, and what scientifically relevant search behavior is hidden by the endpoint score?",
        QUOTE,
    ))
    story.append(P(
        "We answer with a trace-level case study rather than a new capability leaderboard. Our contributions are: (i) a reproducible "
        "decomposition of every score improvement in 4,000 proposals; (ii) trajectory-level measures of result-referencing text, source change, "
        "failure, and post-improvement effort; and (iii) design recommendations that preserve task scoring while preventing that score "
        "from standing in for the scientific process. The result is deliberately bounded to one task and scaffold.",
    ))
    pagebreak(story)

    # Main page 2: related work and measurement framework.
    story.append(h1("2  Related work and validity framework"))
    story.append(h2("2.1  Autonomous ML research and coding agents"))
    story.append(P(
        "MLAgentBench defines ML experimentation as designing, running, analyzing, and iterating on experiments, and evaluates agents on "
        "13 tasks while retaining interpretable plans and actions (Huang et al., 2024). MLE-bench broadens engineering coverage to 75 "
        "Kaggle competitions and emphasizes medals and leaderboard performance (Chan et al., 2025). RE-Bench compares agents with human "
        "experts in seven realistic AI R&D environments and explicitly validates feasibility, score ceilings, and environmental issues "
        "(Wijk et al., 2024). Its qualitative audit also documents noisy objectives and high-scoring loopholes, showing why task success "
        "needs trace inspection. AlphaEvolve demonstrates that evaluator-guided code evolution can discover valuable algorithms "
        "(Novikov et al., 2025), and The AI Scientist connects iterative experiment search to manuscript production (Lu et al., 2026).",
    ))
    story.append(P(
        "These benchmarks answer important capability questions. Our question is orthogonal: what does an endpoint score leave unresolved "
        "about the research process that produced it? Recent work finds that AI research-agent ideas can be concentrated near their starting "
        "literature (Tang and Yang, 2026), while Messeri and Crockett (2024) warn that apparent productivity and breadth can produce "
        "illusions of understanding. We study concentration within executable search trajectories rather than across generated ideas.",
    ))
    story.append(h2("2.2  Construct validity and benchmark interpretation"))
    story.append(P(
        "Construct validity asks whether evidence supports the intended interpretation of a measurement (Cronbach and Meehl, 1955). "
        "Messick (1995) centers score meaning and the consequences of score use. In machine learning, Jacobs and Wallach (2021) frame "
        "operationalization as an explicit measurement model. BetterBench finds substantial quality variation across benchmark life cycles "
        "(Reuel et al., 2024). Bean et al. (2025) review 445 LLM benchmarks and identify underspecified phenomena, weak task-to-construct "
        "links, and rare statistical testing. Alaa et al. (2025) show how strong claims can outrun what benchmark tasks represent.",
    ))
    story.append(P(
        "We distinguish two interpretations. The <i>narrow</i> interpretation is that the agent can adapt code to maximize this score under "
        "this evaluator. The <i>broad</i> interpretation is that the score ranks systems by research quality: mechanistic exploration, "
        "evidence-responsive updating, and improvements likely to generalize. Our audit does not dispute the narrow interpretation. It "
        "tests how much evidence the instrument supplies for the broad one.",
    ))
    story.append(h2("2.3  Adaptive leaderboards and metric optimization"))
    story.append(P(
        "Repeated feedback makes validation adaptive. Dwork et al. (2015) show why conventional holdout guarantees fail when analyses are "
        "chosen using previous holdout results. Blum and Hardt (2015) study the same problem for leaderboards. Goodhart effects become more "
        "important as optimization power increases (Manheim and Garrabrant, 2019). Our contribution is empirical and process-level: we "
        "observe how an autonomous agent reallocates proposals after the scored predictions stop improving, without claiming that every "
        "tie-break improvement is harmful or that a retrospective plateau can be known online.",
    ))
    story.append(h2("2.4  Gap"))
    story.append(P(
        "Prior benchmark critiques usually inspect benchmark documents, task coverage, or final solutions. Agent benchmarks often release "
        "traces but use them mainly for failure examples. We connect the full score trajectory to declared hypotheses, evidence statements, "
        "source-level edits, evaluator failures, and resource use. This turns process logs into validity evidence rather than supplementary "
        "anecdotes.",
    ))
    pagebreak(story)

    # Main page 3: instrument and methods.
    story.append(h1("3  Instrument, corpus, and analysis"))
    story.append(h2("3.1  Task and score"))
    story.append(P(
        "Each trajectory begins from the same 105,866-parameter convolutional classifier, which scores 8,928/10,000 on a public Fashion-MNIST "
        "validation split. The agent may edit one Python training program, including architecture, optimization, augmentation, and inference, "
        "subject to at most 250,000 learned parameters and exactly 100,000 training-example exposures. A local Apple-MPS evaluator allows 90 "
        "seconds per proposal. The agent sees validation correct count, accuracy, cross-entropy, parameters, examples, optimizer steps, and time.",
    ))
    story.append(P(
        "The objective is <i>S = C + 0.5/(1+L)</i>, where <i>C</i> is validation-correct count and <i>L</i> is cross-entropy. Since the "
        "tie-break lies in (0, 0.5], one additional correct prediction always dominates any cross-entropy change. At fixed <i>C</i>, however, "
        "arbitrarily small reductions in <i>L</i> strictly increase <i>S</i>. The score therefore encodes a defensible lexicographic preference "
        "and an unbounded-precision local search surface.",
        QUOTE,
    ))
    story.append(h2("3.2  Campaign and amendment"))
    story.append(P(
        "The original campaign configuration defines three matched blocks of four conditions (12 trajectories). Conditions cross single-incumbent "
        "versus four-member portfolio memory with ordinary versus assumption-changing proposals every tenth opportunity. Each trajectory "
        "receives 200 ephemeral proposal sessions from the same configured GPT-5.6 Sol model at xhigh reasoning effort. Strict score "
        "improvement governs the single-incumbent cells; deterministic portfolio rules govern the others. We pool conditions for the validity "
        "audit and do not estimate treatment effects here.",
    ))
    story.append(P(
        "During collection, an operator-authorized amendment added two blocks (eight trajectories) without modifying existing runs. We report "
        "Blocks 1-3 as the original scope and Blocks 4-5 as an extension. All 20 trajectories completed 200 opportunities. The original "
        "12-run validator reports identical task, evaluator, budget, start artifact, model settings, and failure rule; extension manifests "
        "carry the same task, framework, protocol, and runtime hashes.",
    ))
    story.append(h2("3.3  Trace corpus"))
    story.append(P(
        "The corpus contains 4,000 structured completion records, 3,964 raw final agent messages, 3,732 started evaluator calls with "
        "candidate and selected-parent source snapshots, 92,166,218 accounted tokens, and "
        "285,706 evaluator-seconds. We treat a proposal as valid only when the recorded evaluator flag is true; metrics emitted by invalid "
        "candidates are not counted as outcomes. There are 2,313 valid and 1,687 invalid proposals.",
    ))
    story.append(h2("3.4  Measures"))
    story.append(bullet(
        "<b>Outcome decomposition.</b> A new-best event either raises the best correct count or lowers cross-entropy at the same count. "
        "Post-final-gain effort is the proposals, tokens, and evaluator time after the trajectory's last correct-count improvement."
    ))
    story.append(bullet(
        "<b>Source process.</b> We tokenize candidate and parent programs, remove layout/comments, and abstract identifiers, strings, and "
        "numbers. A literal-only edit leaves that normalized sequence unchanged. Source-structural novelty is Jaccard distance between "
        "normalized token-trigram sets. This measures program shape, not semantic novelty."
    ))
    story.append(bullet(
        "<b>Declared process.</b> Declared novelty is one minus the maximum word-set Jaccard similarity to prior mechanism/hypothesis/edit "
        "statements in that trajectory. Transparent lexical indicators mark numerical evidence, calibration language, and hypotheses that "
        "explicitly promise unchanged predictions with lower cross-entropy."
    ))
    story.append(bullet(
        "<b>Inference.</b> Runs, not proposals, are observational units. We report run-paired pre/post means and Spearman associations at "
        "n=20. Five-block bootstrap percentile ranges are sensitivity summaries, not calibrated confidence statements. Original/extension "
        "results are separated. A rule-selected 420-row reader contains 417 nonempty raw messages for close audit."
    ))
    story.append(P(
        "All rules were executed over the full corpus before examples were chosen. Exact definitions, per-run values, hashes, and a one-command "
        "analysis are in the anonymized artifact and Appendix A.",
        SMALL,
    ))
    pagebreak(story)

    # Main page 4: headline results.
    story.append(h1("4  Scalar progress is mostly tie-break progress"))
    story.append(P(
        "Every trajectory improves substantially over the common baseline: final correct counts range from 9,192 to 9,360, with a mean gain "
        "of 362.3 predictions. This demonstrates real task optimization. The event-level decomposition changes its interpretation. Across "
        "1,009 new scalar bests, 247 increase the best correct count and 762 (75.5%) alter only cross-entropy at the same count. Thus one "
        "endpoint trajectory can contain many retained score improvements without additional correct predictions.",
    ))
    headline_data = [["Scope", "Runs", "New bests", "Score-only", "Post-gain props", "Post-gain tokens"]]
    for label, key in [("Original B1-3", "original"), ("Extension B4-5", "extension"), ("All", "all")]:
        row = SCOPE[key]
        headline_data.append([
            label,
            row["n_runs"],
            row["new_best_events"],
            pct(row["tiebreak_share_of_new_best"]),
            pct(row["post_accuracy_plateau_proposal_fraction"]),
            pct(row["post_accuracy_plateau_token_fraction"]),
        ])
    story.append(P("Table 1: Headline decomposition by declared campaign scope.", CAPTION))
    story.append(data_table(headline_data, [92, 35, 56, 57, 75, 76], font_size=7.4))
    story.append(Spacer(1, 5))
    story.extend(figure(
        DERIVED / "fig3_improvement_types.png",
        5.05 * inch,
        "Figure 1: New scalar bests by trajectory. Blue events increase validation-correct count; orange events improve only the "
        "cross-entropy tie-break at the current best count. Score-only events dominate in 18 of 20 trajectories, but the balance varies "
        "considerably, which an endpoint score cannot show.",
    ))
    story.append(h2("4.1  Same-count improvements span numerical and substantive scales"))
    story.append(P(
        "Event count alone also hides magnitude. Across the 762 same-count improvements, the median cross-entropy decrease is "
        "1.32 x 10<super>-7</super> (25th percentile 7.63 x 10<super>-9</super>; 75th percentile 1.63 x 10<super>-5</super>; "
        "90th percentile 8.44 x 10<super>-4</super>; maximum 1.59 x 10<super>-2</super>). Overall, 63.4% are at most "
        "10<super>-6</super>, 83.2% are at most 10<super>-4</super>, and 9.6% exceed 10<super>-3</super>. Thus many events are "
        "numerical micro-improvements, while a minority could represent meaningful probability-quality gains. Without reruns, a calibration "
        "diagnostic, or a sealed holdout, this corpus cannot determine which interpretation generalizes.",
        SMALL,
    ))
    story.append(h2("4.2  Directional consistency across the campaign extension"))
    story.append(P(
        "The core pattern is not created by the amendment. Score-only events are 72.1% of new bests in the original 12 runs and 80.0% in "
        "the eight-run extension. The post-final-gain token share is 29.6% and 46.4%, respectively. Literal-only edits rise and "
        "source-structural novelty falls after the final correct-count gain in both scopes (Appendix B). The extension increases the "
        "estimated magnitude but preserves the direction. Because it was added during collection under the same scaffold, we treat it as "
        "a sensitivity extension, not an independent replication.",
        SMALL,
    ))
    story.append(h2("4.3  Endpoint score does not identify process properties"))
    story.append(P(
        "Across 20 runs, correct-count gain has weak descriptive rank associations with valid-proposal rate (Spearman r_s=0.065), number "
        "of new bests (r_s=-0.021), post-final-gain token share (r_s=-0.068), literal-only edit rate (r_s=-0.170), and mean declared "
        "novelty (r_s=0.215). Block-bootstrap sensitivity ranges are wide and include zero. These estimates do not prove independence. They show "
        "that, in this sample, endpoint gain supplies little information about those recorded process properties and cannot substitute for reporting them.",
        SMALL,
    ))
    pagebreak(story)

    # Main page 5: within-run process shift.
    story.append(h1("5  Search narrows while result vocabulary remains common"))
    story.append(P(
        "The post-final-gain interval contains 1,520 proposals (38.0%), 33.3 million tokens (36.2%), and 107,324 evaluator-seconds (37.6%). "
        "Its median length is 72 proposals, ranging from one to 192. These are retrospective intervals: B1C3 makes its final correct-count "
        "gain at opportunity 199, so no online observer at opportunity 150 could know that its apparent plateau would end. We use the "
        "interval to characterize reward allocation, not to prescribe stopping.",
    ))
    story.extend(figure(
        DERIVED / "fig2_search_dynamics.png",
        5.35 * inch,
        "Figure 2: Pooled process indicators by 20-opportunity bin. Source-structural novelty declines as literal-only edits and explicit "
        "preserve-predictions/lower-cross-entropy hypotheses rise. The lexical indicator is transparent and descriptive, not a semantic judge.",
    ))
    selected = [
        ("Literal-only edit rate", "literal_only_edit_rate"),
        ("Source-structural novelty", "source_structural_novelty"),
        ("Declared-text novelty", "declared_novelty"),
        ("Preserve predictions/lower CE", "preserve_predictions_lower_ce_rate"),
        ("Calibration language", "calibration_language_rate"),
        ("Tie-break improvement rate", "tiebreak_improvement_rate"),
    ]
    shift_data = [["Run-paired measure", "Before", "After", "After-before", "Block-bootstrap range"]]
    for label, key in selected:
        row = PAIRED[key]
        shift_data.append([
            label,
            f(row["pre_mean"], 3),
            f(row["post_mean"], 3),
            f(row["mean_paired_difference"], 3),
            f"[{f(row['block_bootstrap_ci_low'], 3)}, {f(row['block_bootstrap_ci_high'], 3)}]",
        ])
    story.append(P("Table 2: Within-run process shifts around the final correct-count gain (n=20 runs).", CAPTION))
    story.append(data_table(shift_data, [133, 42, 42, 65, 106], font_size=7.0))
    story.append(Spacer(1, 4))
    story.append(P(
        "Numerical-result vocabulary appears in 99.8% of proposals at opportunities 1-20 and 95.0% at 181-200; a broader lexical rule for "
        "result-updating language remains above 94% in every bin. Because that rule includes common words such as 'score,' 'correct,' and "
        "'loss,' it establishes prompt-compliant reference to benchmark results, not evidence quality. The co-occurrence of persistent result "
        "vocabulary with narrower source search is more informative than either measure alone, but it is not a causal effect of the score.",
    ))
    story.append(P(
        "Each trajectory contributes one paired difference regardless of whether its post boundary contains one or 192 proposals. The "
        "displayed percentile ranges resample only five matched blocks and should be read as finite-corpus sensitivity summaries. Appendix B "
        "shows every run and segment boundary.",
        SMALL,
    ))
    pagebreak(story)

    # Main page 6: qualitative cases and runtime selection pressure.
    story.append(h1("6  What the recorded messages reveal"))
    story.append(h2("6.1  From representation changes to numerical boundary search"))
    story.append(P(
        "B1C0 raises validation correct from 8,928 to 9,290 by opportunity 11 through a deeper normalized CNN, matched translation "
        "augmentation, and a wider dense bottleneck. It then uses 189 opportunities and 94.5% of its tokens after that final gain. At "
        "opportunity 50 the hypothesis explicitly promises to preserve all 9,290 argmax predictions while lowering cross-entropy with "
        "temperature sharpening. Opportunity 100 bisects a live/EMA mixture at 50.687890625%; opportunity 192 fits a temperature near "
        "0.717663 at float32 precision. The recorded statements cite neighboring measurements and timeouts.",
    ))
    story.append(P(
        "B5C2 is the extreme case. Its final correct-count gain occurs at opportunity 8. Later proposals fit successively higher-order "
        "confidence corrections; opportunity 194 estimates a 17th-degree coefficient near 2.4 x 10<super>-14</super> while preserving "
        "9,192 predictions. The trajectory spends 96.5% of its tokens after its last correct-count gain. The score records these as "
        "strict improvements, even though the learned classifier's decisions are unchanged.",
    ))
    story.append(h2("6.2  Contrasts prevent a simplistic stagnation account"))
    story.append(P(
        "Not every late trajectory collapses into calibration. At opportunity 10, B1C3 explicitly rejects sequential local features and "
        "a flattened head in favor of parallel multi-scale features and global statistics. Later messages test learned downsampling, "
        "spatial pyramids, and reliability gating; its final correct-count gain arrives at opportunity 199. B5C1 similarly states at "
        "opportunity 200 that repeated calibration cannot change rankings and returns to spatial-moment representations. These contrasts "
        "show why full traces are superior to a blanket label such as 'wasted search.'",
    ))
    story.append(h2("6.3  Edit class is associated with evaluator survival"))
    literal = EDIT_CLASSES["literal_only"]
    structural = EDIT_CLASSES["structural_token_change"]
    runtime_data = [
        ["Edit class", "n", "Valid", "Timeout", "New accuracy best", "New tie-break best"],
        ["Literal-only", literal["n"], pct(literal["valid_rate"]), pct(literal["timeout_rate"]), pct(literal["accuracy_improvement_rate"]), pct(literal["tiebreak_improvement_rate"])],
        ["Structural token change", structural["n"], pct(structural["valid_rate"]), pct(structural["timeout_rate"]), pct(structural["accuracy_improvement_rate"]), pct(structural["tiebreak_improvement_rate"])],
    ]
    story.append(P("Table 3: Evaluator outcomes by source-edit class (3,732 started evaluations).", CAPTION))
    story.append(data_table(runtime_data, [112, 38, 50, 50, 77, 77], font_size=7.15))
    story.append(Spacer(1, 4))
    story.append(P(
        "Structural token changes are 3.8 times as likely to yield a new correct-count best (10.6% versus 2.8%) but also time out more "
        "often (41.4% versus 30.7%). Literal-only edits are more likely to be valid and retained, largely through score tie-break gains. "
        "This is not a controlled causal comparison: harder ideas may both require structural edits and more compute. It identifies an "
        "association between edit class and operational censoring that a scalar endpoint does not disclose.",
    ))
    story.append(h2("6.4  Adaptive validation remains unresolved"))
    story.append(P(
        "All 200 rounds expose results from the same public validation split, and no sealed holdout endpoint was collected. Precise "
        "temperature, view-weight, and polynomial searches therefore optimize repeatedly observed data. The trace proves public-score "
        "improvement, not test generalization; adaptive-data-analysis results explain why those claims differ (Dwork et al., 2015; Blum "
        "and Hardt, 2015).",
    ))
    pagebreak(story)

    # Main page 7: construct-validity diagnosis and recommendations.
    story.append(h1("7  Construct-validity diagnosis"))
    story.append(P(
        "The instrument has strong face validity for adaptive code optimization: the agent proposes executable changes, observes objective "
        "feedback, and improves a classifier under fixed exposure and parameter constraints. The audit identifies four boundaries on "
        "stronger interpretations.",
    ))
    story.append(bullet(
        "<b>Content underrepresentation.</b> If the intended construct is broader research quality, one scalar does not encode question choice, "
        "mechanistic breadth, falsification, explanation, or robustness. Two trajectories with similar endpoints may reach them through broad representation search or extended "
        "numerical calibration."
    ))
    story.append(bullet(
        "<b>Construct-irrelevant variance.</b> Provider failures, execution errors, and especially a 90-second timeout affect which ideas "
        "can be observed. Structural edits are associated with higher timeout rates."
    ))
    story.append(bullet(
        "<b>Score precision.</b> The continuous tie-break makes tiny calibration gains legitimately retainable. Repeated micro-improvements "
        "co-occur with narrower source search, but this observational audit cannot identify the score component as the cause."
    ))
    story.append(bullet(
        "<b>Generalization ambiguity.</b> Repeated public-validation feedback creates adaptive dependence. Without a sealed final result, "
        "the endpoint supports no direct claim about unseen-data performance."
    ))
    story.append(P(
        "These are limits on interpretation, not proof that the score is badly designed. Cross-entropy is a proper predictive score and can "
        "capture probability quality that accuracy omits. Its construct relevance depends on the declared goal, meaningful precision, "
        "reliability, and held-out generalization. The problem appears when an optimization score is treated as a sufficient statistic for "
        "scientific research without that measurement argument.",
    ))
    story.append(h1("8  A process-aware reporting protocol"))
    story.append(P(
        "We recommend augmenting, not replacing, endpoint scoring. The following protocol can be computed from logs that many agent "
        "evaluations already collect.",
    ))
    story.append(bullet(
        "<b>Decompose the objective.</b> Report changes in primary decisions, continuous tie-breaks, constraint satisfaction, and compute "
        "separately. State which components can improve without changing task outputs."
    ))
    story.append(bullet(
        "<b>Publish process trajectories.</b> Alongside the best score, report valid/invalid counts, failure causes, last primary-output "
        "improvement, proposal and token allocation, edit magnitude, and representative agent messages."
    ))
    story.append(bullet(
        "<b>Separate feedback from final evidence.</b> Use public feedback for search and a sealed, one-shot holdout for generalization. "
        "Round or privatize leaderboard feedback when fine precision is not scientifically meaningful."
    ))
    story.append(bullet(
        "<b>Make runtime censoring visible.</b> Report timeout rates by proposal class and, where affordable, use multi-fidelity evaluation "
        "or a declared compute-normalized objective so structural ideas are not silently converted into failures."
    ))
    story.append(bullet(
        "<b>Match claims to constructs.</b> Call the result score optimization unless independent process or outcome evidence supports "
        "claims about research quality. Validate process metrics rather than substituting a new opaque composite score."
    ))
    story.append(P(
        "A vector-valued report is intentionally less convenient than a leaderboard. That inconvenience is epistemically useful: it makes "
        "the trade-off between task gain, calibration, diversity, reliability, and compute available for scientific argument rather than "
        "burying it in an aggregate.",
    ))
    pagebreak(story)

    # Main page 8: limitations, implications, and conclusion.
    story.append(h1("9  Limitations and alternative explanations"))
    story.append(P(
        "<b>Single case.</b> All runs use one image-classification task, one base-model family, one evaluator, one LLM configuration, and a "
        "controlled greedy retention scaffold. Findings may differ for proof-verified mathematics, language modeling, population-based "
        "evolution, continuous conversations, or human researchers. The campaign conditions create useful heterogeneity but do not make "
        "the pooled case representative of autonomous science.",
    ))
    story.append(P(
        "<b>Retrospective boundary.</b> 'After the final correct-count gain' is defined with future information. Long intervals can include "
        "reasonable attempts to escape a plateau; they are not estimates of avoidable cost. We use them to ask what the completed score "
        "trajectory contains. An online stopping policy would require a separate prospective study.",
    ))
    story.append(P(
        "<b>Process proxies.</b> Literal-only edits can change architecture width, batch size, or optimization substantially, while a large "
        "source diff can be conceptually shallow. Lexical indicators can miss paraphrases, and the result-updating rule has low specificity "
        "because it includes common benchmark words. We therefore define these measures narrowly and triangulate them with messages and "
        "outcomes. No blinded human coding or inter-rater validation was collected; the proxies do not measure 'scientific quality.'",
    ))
    story.append(P(
        "<b>Stochastic evaluation and public feedback.</b> Candidate training is stochastic and the same validation set is reused. We do "
        "not rerun candidates, estimate test-retest reliability, or evaluate a sealed holdout because those data were not collected. Some "
        "one-prediction changes may be noise, strengthening rather than weakening the need for uncertainty and holdout reporting.",
    ))
    story.append(P(
        "<b>Amendment and dependence.</b> Blocks 4-5 were added after collection began. We never relabel them as original scope; their role "
        "is a directionally consistent sensitivity extension, not independent replication. Proposals are nested within trajectories and are not treated as 4,000 independent samples. With five "
        "blocks, run-level association ranges remain wide.",
    ))
    story.append(h1("10  Implications for AI and scientific knowledge"))
    story.append(P(
        "The most consequential observation is not that agents optimize metrics; they were asked to do so. It is that a rising leaderboard "
        "coexists with major shifts in what the trace contains. Hypotheses remain explicit, result vocabulary remains numerical, and experiments "
        "remain reproducible, yet many later proposals concern float-adjacent calibration rather than learned representations. This case does "
        "not identify the cause of that shift. Productivity, local score optimization, and epistemic breadth are separable properties.",
    ))
    story.append(P(
        "For evaluators of AI research systems, traces should be treated as primary measurement data. They reveal whether success came from "
        "new outputs, calibration, loopholes, repeated failures, or a different computational mechanism. They can also reveal associations "
        "between infrastructure constraints and what the search attempts or observes. This is the level at which claims about AI's role in "
        "scientific practice should be tested.",
    ))
    story.append(h1("11  Conclusion"))
    story.append(P(
        "In 20 completed autonomous ML-research trajectories, substantial classifier improvement coexists with a descriptive divergence "
        "between scalar progress and process. Three quarters of new scalar bests change only a cross-entropy tie-break; more than a third "
        "of tokens and evaluator time occur after the final correct-count gain; and source search narrows while benchmark-result vocabulary remains common. "
        "The result does not invalidate task scoring. It does not support treating endpoint score as a sufficient account of scientific work. "
        "A process-aware benchmark should preserve the scalar for optimization, add a sealed outcome for generalization, and report the "
        "trajectory as evidence.",
    ))
    story.append(h2("Artifact availability"))
    story.append(P(
        "One anonymized supplementary archive contains protocol snapshots, integrity hashes, complete derived tables, the rule-selected "
        "qualitative reader, all source snapshots and raw final messages required by the analysis, a license, and a one-command reproduction "
        "path. Provider credentials, private model traces, and machine-local controller state are excluded.",
        SMALL,
    ))
    pagebreak(story)

    # References (outside the eight-page main-text limit).
    story.append(h1("References"))
    refs = [
        "Alaa, A., Hartvigsen, T., Golchini, N., Dutta, S., Dean, F., Raji, I. D., and Zack, T. (2025). Position: Medical large language model benchmarks should prioritize construct validity. <i>Proceedings of the 42nd ICML</i>, PMLR 267.",
        "Bean, A. M., Kearns, R. O., Romanou, A., et al. (2025). Measuring what matters: Construct validity in large language model benchmarks. arXiv:2511.04703.",
        "Blum, A. and Hardt, M. (2015). The Ladder: A reliable leaderboard for machine learning competitions. <i>Proceedings of ICML</i>, PMLR 37:1006-1014.",
        "Chan, J. S., Chowdhury, N., Jaffe, O., et al. (2025). MLE-bench: Evaluating machine learning agents on machine learning engineering. <i>ICLR 2025</i>. OpenReview: 6s5uXNWGIh.",
        "Cronbach, L. J. and Meehl, P. E. (1955). Construct validity in psychological tests. <i>Psychological Bulletin</i>, 52(4):281-302. doi:10.1037/h0040957.",
        "Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., and Roth, A. (2015). The reusable holdout: Preserving validity in adaptive data analysis. <i>Science</i>, 349(6248):636-638. doi:10.1126/science.aaa9375.",
        "Huang, Q., Vora, J., Liang, P., and Leskovec, J. (2024). MLAgentBench: Evaluating language agents on machine learning experimentation. <i>Proceedings of ICML</i>, PMLR 235:20271-20309.",
        "Jacobs, A. Z. and Wallach, H. (2021). Measurement and fairness. <i>Proceedings of FAccT</i>, 375-385. doi:10.1145/3442188.3445901.",
        "Lu, C., Lu, C., Lange, R. T., Yamada, Y., Hu, S., Foerster, J., Ha, D., and Clune, J. (2026). Towards end-to-end automation of AI research. <i>Nature</i>, 651:914-919. doi:10.1038/s41586-026-10265-5.",
        "Manheim, D. and Garrabrant, S. (2019). Categorizing variants of Goodhart's Law. arXiv:1803.04585.",
        "Messeri, L. and Crockett, M. J. (2024). Artificial intelligence and illusions of understanding in scientific research. <i>Nature</i>, 627:49-58. doi:10.1038/s41586-024-07146-0.",
        "Messick, S. (1995). Validity of psychological assessment: Validation of inferences from persons' responses and performances as scientific inquiry into score meaning. <i>American Psychologist</i>, 50(9):741-749. doi:10.1037/0003-066X.50.9.741.",
        "Novikov, A., Vu, N., Eisenberger, M., et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131.",
        "Reuel, A., Hardy, A., Smith, C., Lamparth, M., Hardy, M., and Kochenderfer, M. J. (2024). BetterBench: Assessing AI benchmarks, uncovering issues, and establishing best practices. <i>NeurIPS 37, Datasets and Benchmarks</i>. doi:10.52202/079017-0685.",
        "Tang, Y. and Yang, Y. (2026). AI research agents narrow scientific exploration. arXiv:2605.27905.",
        "Wijk, H., Lin, T., Becker, J., et al. (2024). RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. arXiv:2411.15114.",
    ]
    for index, ref in enumerate(refs, 1):
        story.append(P(f"[{index}] {ref}", REF))
    pagebreak(story)

    # Appendix A: exact operational definitions.
    story.append(h1("A  Reproducible operational definitions"))
    story.append(h2("A.1  Event reconstruction"))
    story.append(P(
        "For each run, events are sorted by integer opportunity. The common baseline initializes best score and correct count. A proposal is "
        "a new scalar best iff its recorded valid flag is true and its score strictly exceeds the running maximum. It is an accuracy "
        "improvement iff its correct count also exceeds the running maximum; it is a tie-break improvement iff correct count equals the "
        "running maximum. Invalid events never update either maximum. Final values are cross-checked against the state-record incumbent.",
        SMALL,
    ))
    story.append(h2("A.2  Retrospective process boundary"))
    story.append(P(
        "For trajectory r, let t_r be the last opportunity at which the running best correct count increases. Post-final-gain proposals "
        "have opportunity greater than t_r. Token and evaluator fractions sum recorded per-event increments in that interval and divide by "
        "the corresponding complete-run sum. Every run has at least one post-boundary proposal; t_r ranges from 8 to 199.",
        SMALL,
    ))
    story.append(h2("A.3  Source measures"))
    story.append(P(
        "Python tokenization discards comments, indentation, and line boundaries; preserves operators and keywords; and maps identifiers, "
        "numbers, and strings to ID, NUM, and STR. Literal-only means candidate and selected parent have identical normalized sequences but "
        "different source. Structural novelty is one minus Jaccard similarity of normalized token-trigram sets. If either snapshot is missing, "
        "source measures are missing, not zero. Snapshots are available for 3,732/4,000 proposals.",
        SMALL,
    ))
    story.append(h2("A.4  Text measures"))
    story.append(P(
        "Declared novelty uses lowercase alphanumeric word sets from mechanism, hypothesis, and intended edit after a fixed stop list. It is "
        "one minus the maximum Jaccard similarity to any earlier declaration in the same run (the first proposal is 1). Numeric evidence marks "
        "a number in the evidence field. Evidence updating uses a fixed lexical list (previous/prior/recent, achieved, failed, timeout, result, "
        "score, correct, accuracy, cross-entropy, loss, trial, evidence). Preserve/lower-CE requires both a prediction-preservation phrase and "
        "a lower-cross-entropy/loss phrase in the hypothesis. Calibration language uses temperature, calibration, logit scale, ensemble/mixture/"
        "blend weight, binary search, or probability mixture. Complete regular expressions ship in analysis.py.",
        SMALL,
    ))
    story.append(h2("A.5  Uncertainty"))
    story.append(P(
        "Paired differences are computed per run. Percentile ranges resample the five complete blocks with replacement 10,000 times and "
        "include all four conditions in each sampled block. Spearman ranges use the same block bootstrap. With only five clusters these are "
        "unstable finite-corpus sensitivity summaries, not calibrated confidence intervals or support for binary significance claims.",
        SMALL,
    ))
    pagebreak(story)

    # Appendix B: per-run and scope tables.
    story.append(h1("B  Per-trajectory results"))
    run_data = [["Run", "Final correct", "Gain", "Valid", "New best", "Score-only", "Last correct gain", "Post-gain tokens"]]
    for row in sorted(RUNS, key=lambda r: (int(r["block"]), r["condition"])):
        run_data.append([
            f"B{row['block']}{row['condition']}",
            row["final_correct"],
            row["correct_gain"],
            row["valid_count"],
            row["new_best_count"],
            row["tiebreak_improvement_count"],
            row["last_accuracy_improvement"],
            pct(row["post_accuracy_plateau_token_fraction"]),
        ])
    story.append(data_table(run_data, [43, 63, 38, 37, 48, 55, 78, 72], font_size=6.65))
    story.append(Spacer(1, 7))
    story.append(P(
        "Table B1: All trajectories begin from 8,928 validation-correct. B1-B3 are original scope; B4-B5 are the declared extension. "
        "Post-gain token fractions range from 0.6% (B1C3, final correct gain at opportunity 199) to 96.5% (B5C2, final gain at opportunity 8).",
        CAPTION,
    ))
    story.append(h2("B.1  Original versus extension sensitivity"))
    scope_data = [["Measure", "Original B1-3", "Extension B4-5", "All"]]
    for label, key, formatter in [
        ("Valid proposal rate", "valid_rate", pct),
        ("Score-only share of new bests", "tiebreak_share_of_new_best", pct),
        ("Post-gain proposal share", "post_accuracy_plateau_proposal_fraction", pct),
        ("Post-gain token share", "post_accuracy_plateau_token_fraction", pct),
        ("Literal-only pre", "literal_only_edit_rate_pre_plateau", pct),
        ("Literal-only post", "literal_only_edit_rate_post_plateau", pct),
        ("Structural novelty pre", "structural_novelty_pre_plateau", lambda x: f(x, 4)),
        ("Structural novelty post", "structural_novelty_post_plateau", lambda x: f(x, 4)),
    ]:
        scope_data.append([label, formatter(SCOPE["original"][key]), formatter(SCOPE["extension"][key]), formatter(SCOPE["all"][key])])
    story.append(data_table(scope_data, [150, 82, 90, 74], font_size=7.0))
    pagebreak(story)

    # Appendix C: phase table and examples.
    story.append(h1("C  Phase dynamics and qualitative anchors"))
    phase_rows = rows(DERIVED / "phase_summary.csv")
    phase_data = [["Opp.", "Valid", "Accuracy best", "Score-only best", "Literal-only", "Structural novelty", "Preserve/lower CE"]]
    for row in phase_rows:
        phase_data.append([
            f"{row['opportunity_start']}-{row['opportunity_end']}",
            pct(row["valid_rate"]),
            pct(row["accuracy_improvement_rate"]),
            pct(row["tiebreak_improvement_rate"]),
            pct(row["literal_only_edit_rate"]),
            f(row["mean_source_structural_novelty"], 3),
            pct(row["preserve_predictions_lower_ce_rate"]),
        ])
    story.append(data_table(phase_data, [45, 48, 70, 72, 60, 88, 83], font_size=6.8))
    story.append(Spacer(1, 6))
    story.append(P(
        "Table C1: Pooled 20-opportunity bins (400 proposals each). These are descriptive proposal-level rates; uncertainty and dependence "
        "are assessed at run/block level in the main text.",
        CAPTION,
    ))
    story.append(h2("C.1  Same-count cross-entropy magnitude distribution"))
    quantile_keys = ["minimum", "p25", "median", "p75", "p90", "p95", "p99", "maximum"]
    magnitude_data = [
        ["Statistic"] + ["Min", "P25", "Median", "P75", "P90", "P95", "P99", "Max"],
        ["CE decrease"] + [f"{float(TIEBREAK_MAG[key]['cross_entropy_improvement']):.2e}" for key in quantile_keys],
    ]
    story.append(data_table(magnitude_data, [59] + [48] * 8, font_size=6.65))
    story.append(Spacer(1, 5))
    threshold_data = [["Threshold", "<=1e-8", "<=1e-7", "<=1e-6", "<=1e-5", "<=1e-4", "<=1e-3", "<=1e-2"]]
    threshold_data.append([
        "Fraction",
        *[pct(TIEBREAK_MAG[f"at_or_below_{key}"]["fraction_at_or_below"]) for key in ["1e-08", "1e-07", "1e-06", "1e-05", "1e-04", "1e-03", "1e-02"]],
    ])
    story.append(data_table(threshold_data, [59] + [55] * 7, font_size=6.65))
    story.append(P(
        "Table C2: Decrease in validation cross-entropy at each of 762 retained new-best events whose correct count equals the running "
        "maximum. All values are positive by construction. The range includes both float-adjacent and materially larger changes.",
        CAPTION,
    ))
    story.append(h2("C.2  Exact trace anchors"))
    story.append(P(
        "<b>B1C0/O100.</b> Mechanism: conservative binary search below the unresolved live/EMA boundary. The hypothesis chooses a "
        "50.687890625% live mixture to preserve 9,290 correct while lowering cross-entropy. Evidence contrasts neighboring weights and two "
        "timeouts. The source edit changes literals only and is retained as a new tie-break best.",
        SMALL,
    ))
    story.append(P(
        "<b>B5C2/O194.</b> Mechanism: tight-bracket refitted heptadecic confidence calibration. The hypothesis selects coefficient "
        "+0.0000000000000240253 to preserve 9,192 argmax predictions. Evidence performs quadratic interpolation over three coefficients "
        "whose cross-entropies differ in the ninth decimal place. The event is retained as a tie-break best.",
        SMALL,
    ))
    story.append(P(
        "<b>B1C3/O10.</b> Mechanism: parallel multi-scale context fusion with global statistical pooling. The message explicitly identifies "
        "the old sequential/local and flattened-head assumption, proposes complementary local/contextual features, and raises correct count "
        "from the current 9,112 to 9,202. This is a structural, retained accuracy improvement.",
        SMALL,
    ))
    story.append(P(
        "<b>B5C1/O200.</b> Mechanism: learned spatial-moment residual classification. Evidence states that confidence calibration repeatedly "
        "lowered cross-entropy without changing 9,360 decisions and therefore returns to coarse spatial layout. The structural proposal is "
        "valid but does not beat the incumbent.",
        SMALL,
    ))
    pagebreak(story)

    # Appendix D: integrity and reproducibility manifest.
    story.append(h1("D  Integrity and reproducibility manifest"))
    integrity_data = [
        ["Item", "Recorded value"],
        ["Protocol version", "2.1"],
        ["Task", "Fixed-exposure 10-class grayscale classification"],
        ["Configured subject model", "gpt-5.6-sol, xhigh reasoning"],
        ["Conversation mode", "Ephemeral per opportunity"],
        ["Original / extension blocks", "3 / 2"],
        ["Opportunities per trajectory", "200"],
        ["Training exposure", "100,000 examples"],
        ["Learned-parameter cap", "250,000"],
        ["Evaluator timeout", "90 seconds"],
        ["Protocol SHA-256", "68ca90f25b4e086..."],
        ["Task SHA-256", "aeb39424b7241339..."],
        ["Framework SHA-256", "38dec18995640ec3..."],
        ["Scientific runtime SHA-256", "dc5ef7657528bf91..."],
        ["Collection interval (UTC)", "2026-08-23 23:31 to 2026-08-25 10:16"],
    ]
    story.append(data_table(integrity_data, [155, 241], font_size=7.5))
    story.append(Spacer(1, 8))
    story.append(h2("D.1  Reproduction"))
    story.append(P(
        "From the anonymized archive root, run:<br/><font name='AISciKTimes'>MPLCONFIGDIR=/tmp/aiscik-mpl python3 "
        "papers/aiscik2026/paper1/analysis.py</font><br/>The script asserts 20 runs, 200 completed proposals per run, and 4,000 output rows. "
        "It writes aggregate.json, proposal_table.csv, run_summary.csv, scope/condition/phase/edit-class summaries, block-bootstrap tables, "
        "the qualitative sample manifest, and all figures. No network access or model calls are required.",
        SMALL,
    ))
    story.append(h2("D.2  Artifact contents"))
    story.append(bullet("Protocol, task, framework, campaign, amendment, schedule, and validation snapshots."))
    story.append(bullet("Per-run manifests, state summaries, completed-event rows, all 3,964 raw final messages, and all candidate source snapshots required by the analysis."))
    story.append(bullet("Analysis source with fixed seed 20260901 and only standard-library, NumPy, and Matplotlib dependencies."))
    story.append(bullet("Derived CSV/JSON tables and publication figures; MIT LICENSE; SHA-256 manifest for every included file."))
    story.append(P(
        "The source campaign occupies approximately 303 MB. The archive omits evaluator caches, bytecode, machine-local locks, provider "
        "event streams, and duplicate workspaces; none is read by the analysis. Paths and text records are anonymized. All claims reproduce "
        "from included event logs, final messages, and source snapshots, not dashboard caches.",
        SMALL,
    ))
    story.append(h2("D.3  Ethical scope"))
    story.append(P(
        "The corpus contains outputs of AI systems and no human-subject data. We avoid attributing intention or understanding to the agent: "
        "phrases such as 'the agent uses evidence' describe recorded text and action dependencies. The benchmark could influence resource "
        "allocation and claims about research automation, which is why transparent measurement boundaries matter.",
        SMALL,
    ))

    return story


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = NeuripsDoc(str(OUTPUT))
    doc.build(build_story())
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(path)
