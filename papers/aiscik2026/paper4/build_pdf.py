#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the AISciK-ready Paper 4 PDF."""

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
OUTPUT = HERE.parents[2] / "output/pdf/paper4_history_refresh.pdf"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


SUMMARY = json.loads((DERIVED / "aggregate_summary.json").read_text(encoding="utf-8"))
CONTRASTS = SUMMARY["contrasts"]
TRAJECTORIES = load_rows(DERIVED / "trajectory_metrics.csv")
PAIRS = load_rows(DERIVED / "pair_details.csv")
QUAL = json.loads((DERIVED / "qualitative_examples.json").read_text(encoding="utf-8"))
PHASE_START = json.loads((DERIVED / "phase_start_summary.json").read_text(encoding="utf-8"))
STRATUM_CONTRASTS = load_rows(DERIVED / "stratum_contrasts.csv")
LEAVE_ONE_OUT = load_rows(DERIVED / "leave_one_stratum_out.csv")


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def signed(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def comma(value: Any, digits: int = 0) -> str:
    return f"{float(value):,.{digits}f}"


def ci(metric: str, digits: int = 3) -> str:
    row = CONTRASTS[metric]
    return f"[{float(row['bootstrap_low']):.{digits}f}, {float(row['bootstrap_high']):.{digits}f}]"


def contrast(metric: str) -> dict[str, Any]:
    return CONTRASTS[metric]


def pair_rows(metric: str) -> list[dict[str, str]]:
    return [row for row in PAIRS if row["metric"] == metric]


def trajectory(task: str, architecture: str, arm: str, replicate: int) -> dict[str, str]:
    for row in TRAJECTORIES:
        if row["task"] == task and row["architecture"] == architecture and row["arm"] == arm and int(row["replicate"]) == replicate:
            return row
    raise KeyError((task, architecture, arm, replicate))


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
        return "AISciKTimes", "AISciKTimes-Bold", "AISciKTimes-Italic", "AISciKTimes-BoldItalic"
    return "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"


FONT, FONT_BOLD, FONT_ITALIC, FONT_BOLD_ITALIC = register_fonts()
styles = getSampleStyleSheet()

BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontName=FONT, fontSize=9.55, leading=10.65, alignment=TA_JUSTIFY, spaceAfter=4.6, allowWidows=0, allowOrphans=0)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.05, leading=9.0, spaceAfter=3.0)
TINY = ParagraphStyle("Tiny", parent=BODY, fontSize=7.0, leading=7.8, alignment=TA_LEFT, spaceAfter=2.3)
H1 = ParagraphStyle("H1", parent=BODY, fontName=FONT_BOLD, fontSize=11.7, leading=12.8, alignment=TA_LEFT, spaceBefore=4.2, spaceAfter=3.0, keepWithNext=True)
H2 = ParagraphStyle("H2", parent=BODY, fontName=FONT_BOLD, fontSize=9.7, leading=10.7, alignment=TA_LEFT, spaceBefore=3.2, spaceAfter=2.0, keepWithNext=True)
TITLE = ParagraphStyle("Title", parent=BODY, fontName=FONT_BOLD, fontSize=16.6, leading=18.6, alignment=TA_CENTER, spaceBefore=5, spaceAfter=6)
AUTHOR = ParagraphStyle("Author", parent=BODY, fontSize=9.2, leading=10.2, alignment=TA_CENTER, spaceAfter=7)
ABSTRACT_HEAD = ParagraphStyle("AbstractHead", parent=BODY, fontName=FONT_BOLD, fontSize=9.2, leading=10.2, alignment=TA_CENTER, spaceAfter=2)
ABSTRACT = ParagraphStyle("Abstract", parent=BODY, fontSize=8.7, leading=9.8, leftIndent=0.18 * inch, rightIndent=0.18 * inch, alignment=TA_JUSTIFY, spaceAfter=5)
CAPTION = ParagraphStyle("Caption", parent=SMALL, fontName=FONT_ITALIC, alignment=TA_LEFT, spaceBefore=2.0, spaceAfter=3.2)
REF = ParagraphStyle("Ref", parent=SMALL, leftIndent=0.18 * inch, firstLineIndent=-0.18 * inch, fontSize=7.7, leading=8.6, spaceAfter=2.2)
QUOTE = ParagraphStyle("Quote", parent=SMALL, leftIndent=0.18 * inch, rightIndent=0.18 * inch, borderColor=colors.HexColor("#d7dde5"), borderWidth=0.6, borderPadding=4, backColor=colors.HexColor("#f7f9fb"))


class PaperDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=1.5 * inch,
            rightMargin=1.0 * inch,
            topMargin=0.72 * inch,
            bottomMargin=0.72 * inch,
            title="Forgetting Without Restarting",
            author="Anonymous Authors",
            subject="AISciK Workshop (NeurIPS 2026) submission",
        )
        frame = Frame(1.5 * inch, 0.6 * inch, 5.5 * inch, 9.0 * inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="main")
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
        canvas.drawString(1.5 * inch, 0.28 * inch, "Submitted to the AISciK Workshop (NeurIPS 2026).")
    canvas.restoreState()


def P(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def h1(text: str) -> Paragraph:
    return P(text, H1)


def h2(text: str) -> Paragraph:
    return P(text, H2)


def pagebreak(story: list[Flowable]) -> None:
    story.append(PageBreak())


def ruled_title(story: list[Flowable]) -> None:
    story.append(HRFlowable(width="100%", thickness=4, color=colors.black, spaceAfter=9))
    story.append(P("Forgetting Without Restarting: Periodic History Refresh in Autonomous ML Research Agents", TITLE))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=2, spaceAfter=8))
    story.append(P("Anonymous Authors", AUTHOR))


def bullet(text: str) -> Flowable:
    table = Table([[P("-", SMALL), P(text, SMALL)]], colWidths=[0.16 * inch, 5.14 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return table


def data_table(data: list[list[str]], widths: list[float], font_size: float = 7.0) -> Table:
    converted = [[P(str(cell), TINY) for cell in row] for row in data]
    table = Table(converted, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 0.8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd6df")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def figure(path: Path, width: float, caption: str) -> list[Flowable]:
    image = Image(str(path), width=width, height=width * 0.58)
    image.hAlign = "CENTER"
    return [image, P(caption, CAPTION)]


def design_figure() -> list[Flowable]:
    image = Image(str(DERIVED / "figure0_design.png"), width=5.25 * inch, height=1.96 * inch)
    image.hAlign = "CENTER"
    return [
        image,
        P("Figure 1: Intervention schematic. The treatment is a visible-history refresh, not an incumbent restart; both arms reset provider conversation state at the same cadence.", CAPTION),
    ]


def corpus_table() -> list[list[str]]:
    horizons = SUMMARY["design"]["horizons"]
    return [
        ["Stratum", "Controller", "Horizon", "Runs", "Postfork records"],
        ["Fashion-MNIST / greedy", "greedy", str(horizons["Fashion / greedy"]), "6", str((horizons["Fashion / greedy"] - 5) * 6)],
        ["Fashion-MNIST / native", "native", str(horizons["Fashion / native"]), "6", str((horizons["Fashion / native"] - 5) * 6)],
        ["Tiny Addition / greedy", "greedy", str(horizons["Tiny Addition / greedy"]), "6", str((horizons["Tiny Addition / greedy"] - 5) * 6)],
        ["Tiny Addition / native", "native", str(horizons["Tiny Addition / native"]), "6", str((horizons["Tiny Addition / native"] - 5) * 6)],
    ]


def semantic_context_table() -> list[list[str]]:
    rows = [["Stratum", "Endpoint rank", "Endpoint", "Token rank", "Tokens"]]
    for row in SUMMARY["context_summary"]["periodic_full_refresh_by_stratum"]:
        rows.append([
            row["stratum"],
            f"{int(row['endpoint_rank'])}/23",
            fmt(row["endpoint_progress_mean"], 3),
            f"{int(row['token_rank'])}/23",
            comma(row["tokens_mean"]),
        ])
    return rows


def result_table() -> list[list[str]]:
    return [
        ["Measure", "History retained", "History refreshed", "Refresh - retained", "95% range"],
        ["Endpoint progress", fmt(contrast("endpoint_progress")["passive_mean"]), fmt(contrast("endpoint_progress")["refresh_mean"]), signed(contrast("endpoint_progress")["paired_difference"]), ci("endpoint_progress")],
        ["Trajectory AUC", fmt(contrast("auc_progress")["passive_mean"]), fmt(contrast("auc_progress")["refresh_mean"]), signed(contrast("auc_progress")["paired_difference"]), ci("auc_progress")],
        ["Retention rate", pct(contrast("retained_rate")["passive_mean"]), pct(contrast("retained_rate")["refresh_mean"]), signed(contrast("retained_rate")["paired_difference"]), ci("retained_rate")],
        ["Phase-start retention", pct(contrast("phase_start_retained_rate")["passive_mean"]), pct(contrast("phase_start_retained_rate")["refresh_mean"]), signed(contrast("phase_start_retained_rate")["paired_difference"]), ci("phase_start_retained_rate")],
        ["Mechanism novelty", fmt(contrast("mean_mechanism_lexical_novelty")["passive_mean"]), fmt(contrast("mean_mechanism_lexical_novelty")["refresh_mean"]), signed(contrast("mean_mechanism_lexical_novelty")["paired_difference"]), ci("mean_mechanism_lexical_novelty")],
        ["Tokens", comma(contrast("tokens")["passive_mean"]), comma(contrast("tokens")["refresh_mean"]), comma(contrast("tokens")["paired_difference"]), f"[{comma(contrast('tokens')['bootstrap_low'])}, {comma(contrast('tokens')['bootstrap_high'])}]"],
    ]


def leave_one_table() -> list[list[str]]:
    rows = [["Metric", "Omitted stratum", "Remaining pairs", "Refresh - retained"]]
    for row in LEAVE_ONE_OUT:
        if row["metric"] not in {"endpoint_progress", "auc_progress", "tokens"}:
            continue
        rows.append([
            row["metric"].replace("_", " "),
            row["omitted_stratum"],
            row["remaining_pairs"],
            comma(row["paired_difference_equal_strata"]) if row["metric"] == "tokens" else signed(row["paired_difference_equal_strata"], 3),
        ])
    return rows


def construct_validity_table() -> list[list[str]]:
    return [
        ["Construct proxy", "Supports", "Does not support"],
        ["Endpoint/AUC progress", "Recorded task progress after a matched fork.", "A general claim about scientific creativity."],
        ["Retention and droughts", "Evaluator acceptance dynamics and stagnation.", "Proposal quality independent of evaluator design."],
        ["Message novelty/history terms", "Changes in the written subject-visible record.", "Private reasoning or hidden chain-of-thought."],
        ["Source/AST distance", "Local executable-code edit magnitude.", "Semantic novelty of the learned mechanism."],
        ["Tokens/evaluator seconds", "Resource movement between model context and host compute.", "A universal dollar cost or social cost."],
    ]


def stratum_endpoint_range() -> tuple[float, float]:
    values = [float(row["paired_difference"]) for row in STRATUM_CONTRASTS if row["metric"] == "endpoint_progress"]
    return min(values), max(values)


def loo_range(metric: str) -> tuple[float, float]:
    values = [float(row["paired_difference_equal_strata"]) for row in LEAVE_ONE_OUT if row["metric"] == metric]
    return min(values), max(values)


def endpoint_by_stratum_table() -> list[list[str]]:
    rows = [["Stratum / replicate", "Retained", "Refreshed", "Difference"]]
    for row in pair_rows("endpoint_progress"):
        rows.append([f"{row['stratum']} r{row['replicate']}", fmt(row["passive"], 3), fmt(row["refresh"], 3), signed(row["refresh_minus_passive"], 3)])
    return rows


def qualitative_table() -> list[list[str]]:
    helped = QUAL["most_helped_pair"]
    hurt = QUAL["most_hurt_pair"]
    refresh_help = trajectory(helped["task"], helped["architecture"], "periodic_full_refresh", int(helped["replicate"]))
    passive_help = trajectory(helped["task"], helped["architecture"], "passive_control", int(helped["replicate"]))
    refresh_hurt = trajectory(hurt["task"], hurt["architecture"], "periodic_full_refresh", int(hurt["replicate"]))
    passive_hurt = trajectory(hurt["task"], hurt["architecture"], "passive_control", int(hurt["replicate"]))
    return [
        ["Case", "Passive-control trajectory", "Periodic-refresh trajectory", "Interpretation"],
        [
            "Largest gain",
            f"{helped['stratum']} r{helped['replicate']}: progress {fmt(passive_help['endpoint_progress'])}; final params {fmt(passive_help['final_parameters'],0)}.",
            f"Progress {fmt(refresh_help['endpoint_progress'])}; final params {fmt(refresh_help['final_parameters'],0)}.",
            "Refresh escaped the rank-8 token-interface thread and later accepted a width-reduced attention/MLP bottleneck.",
        ],
        [
            "Only loss",
            f"{hurt['stratum']} r{hurt['replicate']}: progress {fmt(passive_hurt['endpoint_progress'])}; endpoint correct {fmt(passive_hurt['endpoint_value'],0)}.",
            f"Progress {fmt(refresh_hurt['endpoint_progress'])}; endpoint correct {fmt(refresh_hurt['endpoint_value'],0)}.",
            "Refresh forgot useful optimizer/batch evidence and spent its short horizon on augmentation variants.",
        ],
    ]


def build_story() -> list[Flowable]:
    story: list[Flowable] = []
    design = SUMMARY["design"]
    token_drop = -float(contrast("tokens")["paired_difference"]) / float(contrast("tokens")["passive_mean"])

    ruled_title(story)
    story.append(P("Abstract", ABSTRACT_HEAD))
    story.append(P(
        "Autonomous ML-research agents accumulate textual evidence, failed edits, source archives, and evaluator summaries. That history can guide search, but it can also preserve local obsessions. "
        "We ask whether periodically clearing subject-visible research history, while keeping the best verified model, changes autonomous scientific search. We analyze four paused semantic-intervention campaigns "
        "covering Fashion-MNIST and Tiny Addition under greedy and native OpenEvolve-style controllers. Within each of 12 matched replicates, passive control and periodic-full-refresh arms share proposals 1-5 exactly "
        "and fork at proposal 6; both reset provider conversations every five proposals, but only refresh clears the visible archive. Across 1,068 postfork proposals, refresh improves normalized endpoint progress by "
        f"{signed(contrast('endpoint_progress')['paired_difference'])} {ci('endpoint_progress')} and trajectory AUC by {signed(contrast('auc_progress')['paired_difference'])} {ci('auc_progress')}. It has higher endpoint progress in 11 of 12 pairs and uses "
        f"{pct(token_drop)} fewer subject-agent tokens. The pattern is not a simple acceptance-rate effect: first-proposal-after-refresh retention falls from {pct(contrast('phase_start_retained_rate')['passive_mean'])} to {pct(contrast('phase_start_retained_rate')['refresh_mean'])}, while within-phase retention is unchanged. Message novelty rises, prior-history language falls, and local source edit distance barely changes. In this corpus, forgetting appears to work as a search-policy bottleneck: it makes agents re-summarize from the incumbent and sometimes take larger conceptual jumps, but it also causes droughts and can erase useful negative evidence. We propose history-refresh audits for evaluating AI scientist systems.",
        ABSTRACT,
    ))
    story.append(h1("1  Introduction"))
    story.append(P(
        "Research agents such as AlphaEvolve, OpenEvolve, and AI Scientist-style systems are often framed as long-horizon loops: propose a change, evaluate it, remember the result, and continue [1-4]. "
        "The memory part sounds obviously helpful. A record of failed ideas can prevent repetition, and a record of successful ideas can support cumulative improvement. But accumulated history also shapes attention. "
        "In a code-search scientist, the visible archive decides which mechanisms are salient, which local parameter knobs seem worth another attempt, and which failures become part of the agent's self-narrative.",
    ))
    story.append(P(
        "This paper studies a narrow intervention: forget the visible research history but do not restart the task. The incumbent program remains the parent; the evaluator and objective are unchanged. What disappears "
        "is the subject-visible path by which the incumbent was reached. This distinguishes history refresh from ordinary restarts, from population memory, and from assumption-changing prompts. AISciK asks for studies "
        "that make AI systems and their effects on scientific practice the object of study, and that treat benchmark measurement itself as a research question [12]. We therefore analyze recorded agent behavior, not just endpoint scores.",
    ))
    story.append(h2("Contributions"))
    story.append(bullet("A state-matched trace analysis of 24 focal trajectories, using exact shared proposals 1-5 and right-censored common horizons within each task-by-controller stratum."))
    story.append(bullet("Evidence that full history refresh can improve endpoint and AUC progress while lowering token use, even though it lowers first-proposal retention after each refresh."))
    story.append(bullet("A mixed quantitative and qualitative account of how refresh changes messages, source deltas, repetition, droughts, and failure modes in autonomous ML research loops."))
    story.append(P(
        "<b>RQ.</b> Holding the incumbent model and evaluator fixed, how does clearing subject-visible research history every five proposals affect objective progress, edit novelty, agent summaries, resource use, and stagnation?",
        QUOTE,
    ))
    pagebreak(story)

    story.append(h1("2  Related work"))
    story.append(h2("Autonomous research and evolutionary code search"))
    story.append(P(
        "The AI Scientist automates idea generation, experiment execution, paper writing, and review [3], and AI Scientist-v2 extends that loop with agentic tree search [4]. AlphaEvolve casts discovery as LLM-written code improvement under evaluator feedback [1]. "
        "OpenEvolve operationalizes an open-source evolutionary coding pipeline with MAP-Elites, islands, archive sampling, migration, and inspiration programs [2]. These systems make memory and accumulated evidence operational necessities; our contribution is to treat one memory policy as the measured intervention.",
    ))
    story.append(h2("Memory in language-agent learning"))
    story.append(P(
        "Reflexion showed that language agents can improve by writing feedback into an episodic memory buffer rather than updating weights [5]. Many scientific-agent systems similarly rely on self-written summaries, trace files, archives, and previous-result sections. "
        "The expected benefit is sample efficiency. The risk, less often measured, is fixation: the agent may inherit an over-specific local theory of why earlier edits worked. Our refresh condition tests the whole-system consequence of removing that written record while preserving the executable incumbent.",
    ))
    story.append(h2("Restarts, novelty, and quality diversity"))
    story.append(P(
        "Random-restart theory in Las Vegas algorithms formalizes the value of abandoning a run when completion times are heavy-tailed or unknown [6]. Stochastic-local-search work similarly uses restarts to manage heavy-tailed failure modes [7]. "
        "Novelty search and quality-diversity algorithms instead preserve or reward behavioral alternatives to escape deceptive objectives [8,9]. The intervention here sits between those traditions: it is a restart of remembered history, not of the incumbent solution or evaluator state.",
    ))
    story.append(h2("Measurement gap"))
    story.append(P(
        "A best-score curve can miss whether an agent explored a new mechanism or merely repeated the same local tweak with shorter prompts. Conversely, lexical novelty can rise without executable novelty. AISciK's Datasets and Evaluations track explicitly welcomes evaluations of epistemic properties and meta-evaluations of agent benchmarks [12]. We therefore link five layers: design validity, objective progress, retention dynamics, source structure, and recorded agent language.",
    ))
    story.extend(design_figure())
    pagebreak(story)

    story.append(h1("3  Data and methods"))
    story.append(h2("3.1  Corpus and right-censoring"))
    story.append(P(
        f"We analyze {design['focal_trajectories']} focal trajectories from four semantic-interventions-v4 campaigns. The focal arms are <b>passive_control</b> and <b>periodic_full_refresh</b>. "
        f"The campaigns were paused before the advertised 200-proposal endpoint, so each stratum is analyzed only through the largest contiguous common horizon available across its six focal runs. This yields {design['logical_proposal_records']} logical proposal records and "
        f"{design['postfork_proposal_records']} postfork proposal records. No missing suffix is imputed.",
    ))
    story.append(P("Table 1: Analysis corpus. Each row contains three passive and three refresh trajectories.", CAPTION))
    story.append(data_table(corpus_table(), [128, 70, 53, 38, 94], font_size=6.8))
    story.append(h2("3.2  Secondary semantic-intervention context"))
    context_design = SUMMARY["context_design"]
    story.append(P(
        f"The focal contrast is supplemented by a non-causal context table over all {context_design['context_trajectories']} semantic-intervention trajectories at the same refresh-common horizons "
        f"({context_design['context_proposal_records']} proposal records). This context does not replace the matched-pair estimate, but it tests whether periodic refresh is an isolated outlier among the 23 contemporaneous intervention labels.",
        SMALL,
    ))
    story.append(P("Table 2: Periodic full refresh ranked among all 23 semantic-condition labels within each stratum. Lower token rank means fewer subject-agent tokens.", CAPTION))
    story.append(data_table(semantic_context_table(), [125, 72, 68, 62, 72], font_size=6.55))
    story.append(P(
        f"Refresh ranks in the top third for endpoint progress in {SUMMARY['context_summary']['refresh_top_third_endpoint_strata']} of 4 strata and in the lowest-token half in all 4. The paired focal comparison below is therefore not the only evidence that refresh occupies a distinctive part of the intervention landscape.",
        SMALL,
    ))
    story.append(h2("3.3  Matched fork design"))
    story.append(P(
        "Within every replicate and stratum, proposals 1-5 are byte-mirrored from the passive run into the refresh run. The validation checks confirm identical candidate IDs, incumbents, and evaluator records for these prefix proposals, with 120 mirrored-prefix records and 111 scheduled refresh events verified. "
        "Both arms begin a new provider conversation every five proposals. The passive arm keeps the visible result archive, candidate archive, and parent history. The refresh arm clears those history objects at opportunities 6, 11, 16, and so on, then continues from the current incumbent as though that incumbent were the starting design.",
    ))
    story.append(h2("3.4  Measures and inference"))
    story.append(P(
        "For Tiny Addition, progress is the fraction of fork-time model parameters eliminated by the best still-qualified incumbent, where qualified means at least 99% exact public validation accuracy on the addition task. For Fashion-MNIST, progress is the fraction of remaining possible validation-correct improvement achieved after the fork, measured out of 10,000 validation images. "
        "We also compute trajectory AUC, retention and qualification rates, phase-start retention, within-phase retention, drought length, source-token Jaccard distances, AST distance, mechanism lexical novelty, new-family language, prior-history language, numeric-evidence language, evaluator seconds, and subject-agent token increments.",
    ))
    story.append(P(
        "The unit of comparison is the matched trajectory pair. Pooled contrasts first compute refresh-minus-passive within each pair, then average the four task-by-controller stratum means equally. Intervals are percentile sensitivity ranges from 20,000 stratified bootstrap resamples. They are descriptive because the campaigns are paused, some protocol elements were operator-amended before this analysis, and trajectories are not sampled from a superpopulation.",
        SMALL,
    ))
    pagebreak(story)

    story.append(h1("4  The refreshed arm records higher progress"))
    story.extend(figure(DERIVED / "figure1_progress.png", 5.2 * inch, "Figure 2: Mean and replicate-level progress from the matched fork. Vertical lines mark scheduled refresh opportunities. All strata use the largest common contiguous horizon available in that stratum."))
    story.append(P("Table 3: Pooled refresh-minus-passive results, equal-weighted across strata.", CAPTION))
    story.append(data_table(result_table(), [113, 75, 75, 75, 92], font_size=6.45))
    story.append(P(
        f"Endpoint progress is higher under refresh in this corpus in {contrast('endpoint_progress')['refresh_higher']} of 12 pairs, with a pooled difference of {signed(contrast('endpoint_progress')['paired_difference'])} {ci('endpoint_progress')}. "
        f"Trajectory AUC is also higher, by {signed(contrast('auc_progress')['paired_difference'])} {ci('auc_progress')}, in {contrast('auc_progress')['refresh_higher']} of 12 pairs. The largest gains occur in Tiny Addition, where a refreshed native trajectory in replicate 2 reaches "
        f"{fmt(trajectory('tiny_adderboard','native','periodic_full_refresh',2)['final_parameters'],0)} parameters versus {fmt(trajectory('tiny_adderboard','native','passive_control',2)['final_parameters'],0)} for its passive match.",
    ))
    endpoint_min, endpoint_max = stratum_endpoint_range()
    loo_min, loo_max = loo_range("endpoint_progress")
    auc_loo_min, auc_loo_max = loo_range("auc_progress")
    story.append(P(
        f"The sign is not carried by a single task/controller stratum: the endpoint contrast is positive in all four strata, ranging from {signed(endpoint_min)} to {signed(endpoint_max)}. "
        f"It also survives leave-one-stratum-out checks, where endpoint contrasts range from {signed(loo_min)} to {signed(loo_max)} and AUC contrasts from {signed(auc_loo_min)} to {signed(auc_loo_max)}.",
        SMALL,
    ))
    story.append(P(
        "The effect is not monotone and should not be sold as a law. Fashion/native replicate 3 is the only endpoint loss: the passive trajectory improves modestly while the refresh match makes no net progress by horizon 13. Short horizons particularly punish refresh if the first few postrefresh proposals rediscover poor directions.",
        SMALL,
    ))
    pagebreak(story)

    story.append(h1("5  The mechanism is not simple acceptance-rate improvement"))
    story.append(P(
        f"Refresh accepts fewer proposals overall ({pct(contrast('retained_rate')['refresh_mean'])} versus {pct(contrast('retained_rate')['passive_mean'])}), and the interval for retained rate crosses zero. The sharp effect is at the beginning of each five-proposal phase: phase-start retention falls by "
        f"{signed(contrast('phase_start_retained_rate')['paired_difference'])} {ci('phase_start_retained_rate')}, with passive higher in 9 pairs and refresh higher in none. Within phases, however, retention is essentially unchanged: "
        f"{pct(contrast('within_phase_retained_rate')['passive_mean'])} passive versus {pct(contrast('within_phase_retained_rate')['refresh_mean'])} refresh. This is exactly what a forgetting bottleneck predicts. The first postrefresh proposal often lacks the local negative evidence that would make an edit conservative; later proposals can rebuild enough context to exploit the new path.",
    ))
    story.append(P(
        "These rates are equal-stratum paired summaries. The raw phase-start audit in Section 6.1 uses direct event counts, which answers a different descriptive question: how many recorded phase-start proposals were retained.",
        SMALL,
    ))
    story.extend(figure(DERIVED / "figure2_process.png", 5.25 * inch, "Figure 3: Paired process measures. Refresh improves message novelty and reduces direct prior-history language, but does not create larger local source changes on average."))
    story.append(P(
        f"Droughts lengthen: maximum nonretention drought rises from {fmt(contrast('max_drought')['passive_mean'],1)} to {fmt(contrast('max_drought')['refresh_mean'],1)} proposals. But retained proposals become more consequential: progress per postfork retention doubles from "
        f"{fmt(contrast('progress_per_retention')['passive_mean'],3)} to {fmt(contrast('progress_per_retention')['refresh_mean'],3)}, and the largest progress jump is higher by {signed(contrast('largest_progress_jump')['paired_difference'])} {ci('largest_progress_jump')}. "
        "The practical pattern is bursty search: refresh gives up some routine incremental acceptances in exchange for occasional larger incumbent moves.",
    ))
    story.append(P(
        "This distinction matters for AI-science evaluation. If an evaluator reports only accepted proposals or final score, the same run can look either less efficient or more creative depending on which denominator is chosen. The trace reveals both.",
        SMALL,
    ))
    pagebreak(story)

    story.append(h1("6  What actually changes in the agent traces"))
    story.append(h2("6.1  Language changes more than source structure"))
    story.append(P(
        f"Mechanism lexical novelty rises from {fmt(contrast('mean_mechanism_lexical_novelty')['passive_mean'])} to {fmt(contrast('mean_mechanism_lexical_novelty')['refresh_mean'])}, a paired difference of {signed(contrast('mean_mechanism_lexical_novelty')['paired_difference'])} {ci('mean_mechanism_lexical_novelty')}. "
        f"The rate of entering a new coarse mechanism family also rises by {signed(contrast('new_family_rate')['paired_difference'])} {ci('new_family_rate')}. Direct prior-history language falls by {signed(contrast('prior_history_language_rate')['paired_difference'])} {ci('prior_history_language_rate')}, and numeric-evidence language falls slightly. These are manipulation checks on the record the subject writes, not access to hidden reasoning.",
    ))
    passive_start = PHASE_START["passive_control"]
    refresh_start = PHASE_START["periodic_full_refresh"]
    story.append(P(
        f"The phase-start trace audit covers all {passive_start['phase_start_events']} matched passive starts and all {refresh_start['phase_start_events']} refresh starts. Passive starts retain {passive_start['retained']} candidates and mention prior-history terms {passive_start['prior_history_language']} times. Refresh starts retain {refresh_start['retained']} candidates and mention prior-history terms once. "
        "The most frequent refresh-start mechanism families are width/capacity, attention/routing, and token-interface changes; optimizer language is rarer than in passive starts. This supports a specific account: refresh does not simply make the agent report that it is starting over; it reallocates early-phase attention toward architectural bottlenecks and away from previously narrated training evidence.",
        SMALL,
    ))
    story.append(P(
        f"Source structure tells a subtler story. Mean candidate-parent source novelty changes by only {signed(contrast('mean_source_novelty')['paired_difference'])} {ci('mean_source_novelty')}, with seven pairs favoring passive and five favoring refresh. Mean novelty to any prior source is lower under refresh. That initially looks backward, but it is consistent with the intervention: each refresh collapses the visible world back to the incumbent, so the next proposal often edits that incumbent locally while describing the move in less path-dependent terms. Forgetting the archive changes framing and parent context more than it guarantees a distant syntactic jump.",
    ))
    story.append(h2("6.2  Qualitative anchors"))
    story.append(P("Table 4: Source-and-message inspected cases. These are illustrative, not prevalence estimates.", CAPTION))
    story.append(data_table(qualitative_table(), [58, 118, 118, 116], font_size=6.15))
    story.append(P(
        "In the largest gain pair, passive/native Tiny Addition continues a projection-tied rank-8 token-interface thread through several retained refinements. The refresh arm later summarizes the incumbent as having margin above 99% and tries a width-reduced attention-and-MLP bottleneck, retaining a substantially smaller model. "
        "In the loss pair, the passive Fashion/native run retains useful optimizer/batch-size evidence; the refreshed arm spends its short available horizon on image augmentation variants whose failures were less constrained by remembered evidence. The qualitative difference is not intelligence versus error. It is which evidence is available to make a risky idea seem informative.",
    ))
    pagebreak(story)

    story.append(h1("7  Cost, validity, and implications"))
    story.append(h2("7.1  Refresh saves subject-agent tokens"))
    story.append(P(
        f"Periodic refresh reduces accounted subject-agent tokens from {comma(contrast('tokens')['passive_mean'])} to {comma(contrast('tokens')['refresh_mean'])} per trajectory after the fork, a difference of {comma(contrast('tokens')['paired_difference'])} tokens {ci('tokens',0)}. "
        f"Input tokens account for most of the reduction: the input-token difference is {comma(contrast('input_tokens')['paired_difference'])}, including {comma(contrast('cached_input_tokens')['paired_difference'])} cached-input tokens. Output-token differences are smaller and uncertain. This is not merely a billing fact; under fixed model-use budgets, shorter history changes how many hypotheses a system can test.",
    ))
    story.append(P(
        f"Evaluator time moves the other way: refresh adds {comma(contrast('evaluator_seconds')['paired_difference'],1)} evaluator-seconds on average {ci('evaluator_seconds',1)}. The plausible interpretation is that refresh proposes more candidates worth full evaluation, even while accepting fewer at phase starts. Token efficiency and host-compute efficiency therefore point in different directions.",
    ))
    story.append(h2("7.2  Validity threats"))
    story.append(P(
        "The strongest design feature is the exact five-proposal shared prefix inside each pair. The strongest limitation is incompleteness: common horizons range from 13 to 92, not the planned 200. Results are therefore early-to-mid-course behavior. The passive and refresh arms were created inside already-running, operator-managed campaigns rather than preregistered randomized trials. Condition labels are balanced by replicate but not random samples. "
        "Native adapters have their own population semantics, so greedy and native are analyzed as strata. Agent-message fields in native event metadata can be shifted by the adapter, so the analysis reads saved final-message files directly and treats event fields as fallback only.",
        SMALL,
    ))
    story.append(P(
        "Construct validity is the central boundary: endpoint and AUC are objective progress proxies, message novelty is a proxy for recorded framing, source distance is a proxy for local executable edit magnitude, and token/evaluator time are resource proxies. None is treated as direct measurement of private reasoning or scientific creativity.",
        SMALL,
    ))
    story.append(h2("7.3  Implications for AI scientist evaluation"))
    story.append(bullet("<b>Report memory policy as a treatment.</b> Conversation resets, visible-history resets, population archives, and incumbent restarts are different interventions."))
    story.append(bullet("<b>Measure the process, not only the endpoint.</b> The refreshed arm records higher progress here while lowering first-proposal acceptance and increasing droughts. A single scalar would hide the mechanism."))
    story.append(bullet("<b>Separate semantic novelty from executable novelty.</b> The messages become more novel even when local source distance barely changes. Both layers should be audited."))
    story.append(bullet("<b>Price memory in tokens and compute.</b> A history archive can spend tokens; forgetting can spend evaluator time. Either cost can change scientific practice under a budget."))
    pagebreak(story)

    story.append(h1("8  Conclusion"))
    story.append(P(
        "Periodic full refresh is a small intervention with a large interpretive lesson. In these recorded autonomous ML-research traces, clearing visible history while retaining the incumbent is associated with higher right-censored endpoint progress and AUC, lower subject-agent token use, higher recorded mechanism novelty, and less direct reliance on prior-history language. It also lowers first-proposal retention after each refresh, lengthens droughts, and sometimes erases useful negative evidence. "
        "The evidence supports neither a simple pro-memory nor anti-memory slogan. It shows that memory is an experimental variable in AI-mediated science. The same incumbent can become a different research object when the path to it is hidden.",
    ))
    story.append(P(
        "For AISciK's questions about scientific integrity and measurement, the central contribution is methodological: evaluate AI scientist systems by auditing what histories are visible, what histories are forgotten, how that changes proposals, and what costs move between model tokens and host compute. "
        "A future prospective trial should run completed horizons, randomize refresh schedules before launch, vary refresh cadence, and compare full-history, visible-history-refresh, incumbent-restart, and population-memory conditions. The current paper provides the trace evidence and measurement vocabulary needed to make that design precise.",
    ))
    story.append(h2("Reproducibility, ethics, and AI-use statement"))
    story.append(P(
        "The artifact contains the exact analysis script, frozen event records, saved final proposal messages, candidate and parent source snapshots, derived tables, figures, input hashes, and a checksum manifest. It requires no model calls, no network, no evaluator execution, and no training. "
        "No human-subject data are analyzed. Experiments consumed commercial model capacity and local/remote compute; token and evaluator-time accounting are reported rather than converted into a universal monetary cost. This manuscript was produced with extensive AI assistance as a research-direction artifact. It is not eligible for AISciK submission without substantial human authorship, manual verification, and truthful OpenReview disclosure. AI systems are not authors.",
        SMALL,
    ))
    story.append(P(
        "Track: Datasets and Evaluations. Claim boundary: recorded behavior in these right-censored campaigns, not a general theorem about scientific creativity or a benchmark leaderboard result.",
        SMALL,
    ))
    pagebreak(story)

    story.append(h1("References"))
    references = [
        "Novikov, A., Vu, N., Eisenberger, M., et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131.",
        "Sharma, A. and OpenEvolve contributors (2025-2026). OpenEvolve: an open-source implementation of AlphaEvolve. Software repository: github.com/algorithmicsuperintelligence/openevolve.",
        "Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. (2024). The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv:2408.06292.",
        "Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., and Ha, D. (2025). The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv:2504.08066.",
        "Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. arXiv:2303.11366.",
        "Luby, M., Sinclair, A., and Zuckerman, D. (1993). Optimal speedup of Las Vegas algorithms. Information Processing Letters 47(4):173-180. doi:10.1016/0020-0190(93)90029-9.",
        "Gomes, C. P., Selman, B., and Kautz, H. (1998). Boosting combinatorial search through randomization. AAAI 1998.",
        "Lehman, J. and Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. Evolutionary Computation 19(2):189-223. doi:10.1162/EVCO_a_00025.",
        "Mouret, J.-B. and Clune, J. (2015). Illuminating search spaces by mapping elites. arXiv:1504.04909.",
        "Pugh, J. K., Soros, L. B., and Stanley, K. O. (2016). Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI 3:40. doi:10.3389/frobt.2016.00040.",
        "Cronbach, L. J. and Meehl, P. E. (1955). Construct validity in psychological tests. Psychological Bulletin 52(4):281-302. doi:10.1037/h0040957.",
        "AI & Scientific Knowledge Workshop (2026). AI & Science: Evolution or Extinction? Call for Papers. https://aiscik.github.io/call-for-papers/.",
        "Bean, M. et al. (2025). Measuring what matters: Construct validity in large language model benchmarks. arXiv:2511.04703.",
        "Reuel, A., Hardy, A., Smith, C., Lamparth, M., Hardy, M., and Kochenderfer, M. J. (2024). BetterBench: Assessing AI benchmarks, uncovering issues, and establishing best practices. NeurIPS 37 Datasets and Benchmarks.",
    ]
    for index, reference in enumerate(references, 1):
        story.append(P(f"[{index}] {reference}", REF))
    pagebreak(story)

    story.append(h1("A  Complete paired endpoint contrasts"))
    story.append(P("Table A1: Endpoint progress by matched pair. Positive values favor periodic full refresh.", CAPTION))
    story.append(data_table(endpoint_by_stratum_table(), [135, 83, 83, 82], font_size=6.3))
    story.append(h1("B  Additional operational notes"))
    story.append(P("Table B1: Leave-one-stratum-out sensitivity. Token entries are token-count differences, so negative values favor refresh.", CAPTION))
    story.append(data_table(leave_one_table(), [82, 129, 72, 99], font_size=6.2))
    story.append(P("Table B2: Construct-validity boundary for the paper's process measures.", CAPTION))
    story.append(data_table(construct_validity_table(), [82, 147, 154], font_size=6.15))
    story.append(P(
        "Design validation checks in `analysis.py` verify the mirrored proposals 1-5, scheduled refresh events, absence of passive refresh events, contiguous horizons, and input hashes. The event-level table records source-availability flags because some native candidate files are materialized only inside the evaluation workspace; the analysis uses that workspace snapshot as the candidate source when the candidate archive lacks a direct `train.py`.",
        SMALL,
    ))
    story.append(P(
        "Mechanism novelty is computed from normalized words in the saved final proposal message's MECHANISM line. Mechanism family labels are post hoc regular-expression categories used only as a coarse descriptive proxy. They are not a claim that the agent internally used a taxonomy.",
        SMALL,
    ))
    story.append(P(
        "The final archive includes all raw files used by the analysis plus a SHA-256 manifest. Verify `PAPER4_SHA256SUMS` before regenerating outputs; then re-run the script to reproduce the metrics without access to private credentials or machine-local absolute paths.",
        SMALL,
    ))
    return story


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = PaperDoc(str(OUTPUT))
    document.build(build_story())
    return OUTPUT


if __name__ == "__main__":
    print(build())
