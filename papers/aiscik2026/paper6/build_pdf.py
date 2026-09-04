#!/usr/bin/env python3
# ruff: noqa: E501
"""Build Paper 6: assumption challenges in autonomous ML research."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, HRFlowable, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DERIVED = HERE / "derived"
OUTPUT = REPO / "output/pdf/paper6_assumption_challenge_across_tasks.pdf"

OVERVIEW = json.loads((DERIVED / "overview.json").read_text(encoding="utf-8"))


def rows(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


CHECKPOINT = rows("checkpoint_effects.csv")
PLACEBO = rows("preintervention_placebo_effects.csv")
CYCLES = rows("cycle_gain_effects.csv")
ENDPOINT = rows("endpoint_effects.csv")
OUTCOMES = rows("trajectory_outcomes.csv")
DISPERSION = rows("population_dispersion.csv")
THEMES = rows("message_theme_summary.csv")


def register_fonts() -> tuple[str, str, str, str]:
    font_dir = Path("/System/Library/Fonts/Supplemental")
    paths = {
        "PaperTimes": font_dir / "Times New Roman.ttf",
        "PaperTimes-Bold": font_dir / "Times New Roman Bold.ttf",
        "PaperTimes-Italic": font_dir / "Times New Roman Italic.ttf",
        "PaperTimes-BoldItalic": font_dir / "Times New Roman Bold Italic.ttf",
    }
    if all(path.exists() for path in paths.values()):
        for name, path in paths.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily("PaperTimes", normal="PaperTimes", bold="PaperTimes-Bold", italic="PaperTimes-Italic", boldItalic="PaperTimes-BoldItalic")
        return "PaperTimes", "PaperTimes-Bold", "PaperTimes-Italic", "PaperTimes-BoldItalic"
    return "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"


FONT, FONT_BOLD, FONT_ITALIC, FONT_BOLD_ITALIC = register_fonts()
base = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=base["BodyText"], fontName=FONT, fontSize=10.0, leading=11.0, alignment=TA_JUSTIFY, spaceAfter=3.2, allowWidows=0, allowOrphans=0)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.0, leading=8.8, spaceAfter=2.5)
TINY = ParagraphStyle("Tiny", parent=BODY, fontSize=6.8, leading=7.45, alignment=TA_LEFT, spaceAfter=1.2)
H1 = ParagraphStyle("H1", parent=BODY, fontName=FONT_BOLD, fontSize=12.0, leading=13.0, alignment=TA_LEFT, spaceBefore=2.8, spaceAfter=2.1, keepWithNext=True)
H2 = ParagraphStyle("H2", parent=BODY, fontName=FONT_BOLD, fontSize=10.3, leading=11.1, alignment=TA_LEFT, spaceBefore=2.0, spaceAfter=1.3, keepWithNext=True)
TITLE = ParagraphStyle("Title", parent=BODY, fontName=FONT_BOLD, fontSize=15.7, leading=17.2, alignment=TA_CENTER, spaceBefore=2, spaceAfter=5)
AUTHOR = ParagraphStyle("Author", parent=BODY, fontSize=9.0, leading=9.7, alignment=TA_CENTER, spaceAfter=5)
ABSTRACT = ParagraphStyle("Abstract", parent=BODY, fontSize=8.4, leading=9.2, leftIndent=0.17 * inch, rightIndent=0.17 * inch, spaceAfter=4)
CAPTION = ParagraphStyle("Caption", parent=SMALL, fontName=FONT_ITALIC, fontSize=7.25, leading=8.0, spaceBefore=1, spaceAfter=2.5)
REF = ParagraphStyle("Ref", parent=SMALL, leftIndent=0.18 * inch, firstLineIndent=-0.18 * inch, fontSize=7.45, leading=8.1, spaceAfter=2.0)
QUOTE = ParagraphStyle("Quote", parent=SMALL, leftIndent=0.12 * inch, rightIndent=0.12 * inch, borderColor=colors.HexColor("#d1d5db"), borderWidth=0.5, borderPadding=4, backColor=colors.HexColor("#f8fafc"))


class PaperDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=letter, leftMargin=1.5 * inch, rightMargin=1.0 * inch, topMargin=0.69 * inch, bottomMargin=0.68 * inch, title="Question the Premise, Pay the Price: Assumption-Challenge Prompts in Autonomous ML Research", author="Anonymous Authors", subject="AISciK Workshop (NeurIPS 2026) submission")
        frame = Frame(1.5 * inch, 0.58 * inch, 5.5 * inch, 9.08 * inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="main")
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=draw_page))


def draw_page(canvas, doc) -> None:
    page = canvas.getPageNumber()
    canvas.saveState()
    canvas.setFont(FONT, 7)
    canvas.setFillColor(colors.HexColor("#4b5563"))
    canvas.drawCentredString(letter[0] / 2, 0.26 * inch, str(page))
    canvas.setFont(FONT, 4.7)
    canvas.setFillColor(colors.HexColor("#9ca3af"))
    line_start = (page - 1) * 55 + 1
    for index in range(55):
        y = 9.49 * inch - index * (9.02 * inch / 54)
        canvas.drawRightString(1.42 * inch, y, str(line_start + index))
    if page == 1:
        canvas.setFont(FONT, 7.2)
        canvas.setFillColor(colors.black)
        canvas.drawString(1.5 * inch, 0.26 * inch, "Submitted to the AISciK Workshop (NeurIPS 2026).")
    canvas.restoreState()


def P(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def h1(text: str) -> Paragraph:
    return P(text, H1)


def h2(text: str) -> Paragraph:
    return P(text, H2)


def bullet(text: str) -> Table:
    table = Table([[P("•", SMALL), P(text, SMALL)]], colWidths=[0.16 * inch, 5.20 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return table


def table(data: list[list[str]], widths: list[float], font_size: float = 7.4) -> Table:
    cell_style = ParagraphStyle(
        f"TableCell{font_size}",
        parent=TINY,
        fontSize=font_size,
        leading=font_size + 0.8,
    )
    converted = [[P(str(cell), cell_style) for cell in row] for row in data]
    out = Table(converted, colWidths=widths, hAlign="LEFT", repeatRows=1)
    out.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#64748b")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.HexColor("#94a3b8")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#64748b")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return out


def fig(path: str, width: float, ratio: float, caption: str) -> list[Flowable]:
    image = Image(str(DERIVED / path), width=width, height=width * ratio)
    image.hAlign = "CENTER"
    return [image, P(caption, CAPTION)]


def cp(task: str, metric: str, memory: str = "all") -> dict[str, str]:
    return next(row for row in CHECKPOINT if row["task"] == task and row["metric"] == metric and row["memory"] == memory)


def pl(task: str, metric: str) -> dict[str, str]:
    return next(row for row in PLACEBO if row["task"] == task and row["metric"] == metric)


def cy(task: str, metric: str) -> dict[str, str]:
    return next(row for row in CYCLES if row["task"] == task and row["metric"] == metric)


def ep(task: str, metric: str, memory: str = "all") -> dict[str, str]:
    return next(row for row in ENDPOINT if row["task"] == task and row["metric"] == metric and row["memory"] == memory)


def signed(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def ci(row: dict[str, str], center: str, digits: int = 3) -> str:
    return f"{signed(row[center], digits)} [{signed(row['cluster_bootstrap_low'], digits)}, {signed(row['cluster_bootstrap_high'], digits)}]"


def pagenext(story: list[Flowable]) -> None:
    story.append(PageBreak())


def page1(story: list[Flowable]) -> None:
    story.append(HRFlowable(width="100%", thickness=4, color=colors.black, spaceAfter=7))
    story.append(P("Question the Premise, Pay the Price:<br/>Assumption-Challenge Prompts in Autonomous ML Research", TITLE))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=2, spaceAfter=6))
    story.append(P("Anonymous Authors", AUTHOR))
    story.append(P("Abstract", ParagraphStyle("AbstractHead", parent=BODY, fontName=FONT_BOLD, fontSize=9, alignment=TA_CENTER, spaceAfter=2)))
    story.append(P(
        "Long-running research agents accumulate successful edits, failed experiments, and explanations that can turn one line of inquiry into an attractor. We study a lightweight intervention: at every tenth proposal, instruct the agent to identify and challenge its load-bearing assumptions and test a different learned computation. The object of study is the research process, not the benchmark winner. We analyze 6,080 executable code proposals from 52 GPT-5.6 Sol trajectories on a trained transformer for 10-digit addition, Fashion-MNIST classification, and fixed-time language-model pretraining. The 13 block-level replications contain matched but nonrandomized ordinary and challenged trajectories using either one incumbent or four-lineage memory; memory is a moderator. At 304 intervention checkpoints, challenges are associated with messages more novel relative to their own histories and substantially larger source changes on image classification and language modeling. They also require 2.8k--4.8k more output tokens locally and are less likely to be retained. Rare successful challenges seed productive ordinary descendants: in ten-proposal cycles, most of the Fashion-MNIST and language-model gain arrives after the challenge rather than at it. Yet broad mechanism-family dispersion across challenged runs decreases in all three tasks. We call this observed pattern <i>punctuated defixation with fixation displacement</i>. Assumption challenges are neither empty reflection nor free creativity; under this regime they behave as costly search-policy interventions whose value depends on evaluator feedback and descendant credit.",
        ABSTRACT,
    ))
    story.append(h1("1. Introduction"))
    story.append(P(
        "An autonomous ML-research loop does not merely answer a question once. It proposes code, observes an evaluator, retains or rejects the candidate, and constructs the next proposal from an increasingly selective record. This can make the system effective at exploitation while narrowing what it treats as worth trying. Recent work reports declining within-run creativity in ML agents [3], premature reliance on early assumptions in multi-turn models [4], and experience-following in agent memory [5]. The practical question is not whether a model is globally 'creative,' but whether a small intervention can redirect an ongoing executable investigation—and what that redirection costs.",
    ))
    story.append(P(
        "We examine repeated <b>assumption challenges</b>: a direction added at scheduled checkpoints that asks the agent to step back, identify load-bearing assumptions, and implement a genuinely different learned representation or computation. The comparison direction simply asks for the most informative next change. Both arms see the same task contract, source, recent evaluator evidence, and output schema. Every proposal becomes an exact patch and is evaluated from a fresh initialization. This connects natural-language intervention, executable artifact, and measured consequence.",
    ))
    story.append(P(
        "Our central result is a tension. At scheduled checkpoints, challenged trajectories shift their public rationales and, on two tasks, code structure; however, immediate feasibility and retention fall. Successful departures can still be productive because subsequent ordinary proposals exploit the new design. At the same time, independent challenged runs converge on similar broad mechanism families. The observed regime is consistent with leaving one local fixation while concentrating around another population-level attractor.",
    ))
    story.append(P("<b>Contributions.</b>", BODY))
    story.append(bullet("A repeated-intervention study spanning 52 research trajectories, three executable ML tasks, 6,080 proposals, and 304 matched challenge/control checkpoints."))
    story.append(bullet("A process measurement separating within-trajectory departure, between-run dispersion, executability, strict retention, token/evaluator cost, and descendant value."))
    story.append(bullet("A Research-track empirical account of AI-mediated scientific search behavior: trace-grounded evidence for punctuated defixation and fixation displacement, with explicit identification limits."))


def page2(story: list[Flowable]) -> None:
    story.append(h1("2. From fixation to executable research trajectories"))
    story.append(h2("2.1 Fixation is a system property"))
    story.append(P(
        "Design fixation originally described adherence to limiting example features [7]; later synthesis showed that examples can narrow variety while improving depth or quality [8]. That corrective matters here. Repeatedly refining a good mechanism can be rational exploitation, so repetition alone is not failure. We use <b>trajectory fixation</b> for persistent concentration around a mechanism or edit family relative to available alternatives, with consequences for coverage, feasibility, improvement, or cost. It is observable behavior produced jointly by model, prompt, memory, selector, and evaluator—not a claim about private cognition.",
    ))
    story.append(P(
        "Several mechanisms can produce such concentration. Seeing one's own prior answer increases persistence in controlled choice tasks [6]. Multi-turn histories introduce premature solutions and unsupported assumptions on which later answers rely [4]. Retrieved memories elicit similar outputs and can propagate errors [5]. Research-agent ideas can remain close to their seeds [11], and aligned model populations can exhibit low collective diversity despite competent individual outputs [12]. Conversely, a warning can move search away from one example but toward a replacement suggestion—<i>fixation displacement</i> in human-AI design [9]."
    ))
    story.append(h2("2.2 Why an executable intervention study is needed"))
    story.append(P(
        "Textual novelty is not enough. A persuasive alternative may not parse, train, satisfy a hard constraint, or improve an incumbent. Bhushan et al. separate novelty from usefulness and find that within-run novelty declines while task performance rises [3]. Work on self-correction similarly warns that ungrounded reflection is unreliable without external feedback [13]. Our study observes the complete chain: visible prompt → public hypothesis → exact source edit → isolated evaluator → retained lineage → descendants.",
    ))
    story.append(P(
        "The search loop is inspired by evolutionary coding systems such as AlphaEvolve [2], but it is intentionally simpler than official population OpenEvolve [14]. A <b>single-incumbent</b> run expands the current best and retains only strict improvements. A <b>four-lineage</b> run rotates among four retained lineages and replaces the selected lineage only on strict improvement. We call both <b>greedy OpenEvolve-style</b> trajectories: each proposal is an ephemeral model call over a bounded evidence record, not a continuing chat or a full island/MAP-Elites database.",
    ))
    story.append(h2("2.3 Research questions"))
    story.append(bullet("<b>RQ1—Redirection:</b> Does the challenge change the next public mechanism and executable source relative to the trajectory's immediately preceding proposal and a matched ordinary trajectory?"))
    story.append(bullet("<b>RQ2—Price:</b> What happens to validity, strict retention, model tokens, evaluator time, and immediate objective gain?"))
    story.append(bullet("<b>RQ3—Afterlife:</b> Do ordinary descendants exploit successful alternatives, or does search immediately return to its old basin?"))
    story.append(bullet("<b>RQ4—Displacement:</b> Does greater departure from each run's history broaden the population, or synchronize it around another family?"))
    story.append(bullet("<b>RQ5—Memory:</b> Are these effects qualitatively stable with one incumbent versus four-lineage portfolio memory?"))
    story.append(P(
        "These questions match AISciK's concern with how an AI system changes scientific practice and with what an evaluation actually measures [1]. Benchmark scores serve as evidence about the search process; they are not the paper's scientific target.",
    ))


def page3(story: list[Flowable]) -> None:
    story.append(h1("3. Study design"))
    story.append(h2("3.1 Three research environments"))
    story.append(table([
        ["Environment", "Research objective and verifier", "Horizon", "Runs / challenged checkpoints"],
        ["10-digit addition transformer", "Minimize learned parameters subject to ≥99% exact accuracy on 10,010 cases; fresh MPS training; baseline 1,644 parameters.", "80", "20 / 80"],
        ["Fashion-MNIST classifier", "Maximize correct predictions on a fixed 10,000-image validation split, cross-entropy tie-break; exactly 100,000 training examples; ≤250k parameters.", "200", "20 / 200"],
        ["Fixed-time language-model pretraining", "Minimize validation bits per byte after 300 s on one H100; editable training program; task records throughput, VRAM, tokens, depth, and parameter count.", "40", "12 / 24"],
    ], [1.15*inch, 2.55*inch, 0.45*inch, 1.20*inch]))
    story.append(P(
        "The horizons are task-specific and complete by analytic definition: 80 proposals for addition, 200 for Fashion-MNIST, and 40 for language-model pretraining. We do not pool raw objectives across tasks. Fashion-MNIST began with three blocks and was prospectively extended with two additional complete blocks; analyses report all five, with original-block sensitivity. The language-model study is analyzed as a 40-proposal study.", SMALL))
    story.append(h2("3.2 Matched trajectories and intervention"))
    story.append(P(
        "Each block supplies a run seed shared by four independently sampled trajectories. The two intervention arms receive an additional direction every tenth proposal; the two memory policies determine which retained parent and evidence are available."
    ))
    story.append(table([
        ["Direction", "Single retained incumbent", "Four retained lineages"],
        ["Ordinary", "Ordinary direction at every proposal", "Ordinary direction at every proposal"],
        ["Assumption challenge", "Challenge at proposals 10, 20, ...", "Challenge at proposals 10, 20, ..."],
    ], [1.25*inch, 2.05*inch, 2.05*inch], 7.0))
    story.append(P("At a challenged checkpoint, the task-specific direction is:", SMALL))
    story.append(P(
        "“Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far ... identify the load-bearing assumptions ... test genuinely different learned computational mechanisms ... state the old assumption and the new approach ... [and] use prior results to explain why the alternative is plausible and informative.”",
        QUOTE,
    ))
    story.append(P(
        "For image classification the instruction says to alter how the classifier represents images or computes class predictions; for language modeling, how it represents context or computes predictions; for addition, how the transformer represents or computes the task. Ordinary checkpoints receive no substitute direction. Both arms are then told to use available evidence to choose the most informative next change.",
    ))
    story.append(h2("3.3 Controlled interface and evidence"))
    story.append(P(
        "All runs request GPT-5.6 Sol with xhigh reasoning. Each proposal starts a fresh model call. The prompt exposes only editable source, current/reference designs permitted by the memory condition, and a bounded ledger of recent verification evidence. It forbids filesystem exploration, online data, pretrained weights, self-run evaluation, and hidden alternatives. The response must include a mechanism, falsifiable hypothesis, intended edit, evidence citation, and exact SEARCH/REPLACE patch. The controller applies the patch, runs the fixed evaluator, and records source snapshots, outcome, retention, token usage, and public final message.",
    ))
    story.append(P(
        "This design holds the executable interface constant while changing one direction at scheduled checkpoints. It does <i>not</i> hold trajectory state constant: stochastic proposals diverge before proposal 10, and the repeated intervention changes later parents and evidence. Thus each checkpoint is matched by block and memory, not an exact shared-state fork.",
    ))


def page4(story: list[Flowable]) -> None:
    story.append(h1("4. Measurement and identification"))
    story.append(h2("4.1 Process outcomes"))
    story.append(P(
        "We operationalize six layers of observable effect. <b>Public redirection</b> uses lexical distance from all prior mechanism/hypothesis/evidence summaries, explicit assumption/causal language, and appearance of a broad mechanism tag not previously used in that trajectory. <b>Artifact departure</b> compares candidate to selected parent using normalized Python-token 3-gram Jaccard distance, AST-node multiset distance, and changed lines. <b>Feasibility</b> records evaluator validity and strict retention. <b>Cost</b> uses event-level input/output tokens and evaluator seconds. <b>Usefulness</b> records immediate objective gain and gain over the remaining nine ordinary proposals in that cycle. <b>Population dispersion</b> is pairwise lexical and broad-family distance among independent runs at the same checkpoint.",
    ))
    story.append(P(
        "The broad family vocabulary—attention/routing, position/sequence, token/embedding, spatial representation, factorization/sharing, normalization/bias, capacity/width, training procedure, regularization/augmentation, ensembling/calibration, and loss/objective—is transparent, regex-assisted, post hoc, coarse, and multi-label. It is a descriptive lens, not a definitive mechanism taxonomy or claim of historical scientific novelty. Source/AST measures and evaluator outcomes remain primary. Missing source for malformed/provider failures is retained as failure but excluded only from source-distance denominators.",
    ))
    story.append(h2("4.2 Estimand"))
    story.append(P(
        "For checkpoint <i>t</i>, task <i>j</i>, block <i>b</i>, and memory stratum <i>m</i>, the local contrast is a matched difference-in-differences: Δ = (challenged<sub>t</sub> − challenged<sub>t−1</sub>) − (ordinary<sub>t</sub> − ordinary<sub>t−1</sub>). This subtracts each trajectory's preceding-proposal level and the simultaneous transition in its matched ordinary run. We average within blocks and use 10,000 block-cluster bootstrap resamples for sensitivity intervals. The true replication unit is the block: 13 total (five addition, five Fashion-MNIST, three language-model). The 304 checkpoint pairs comprise 80, 200, and 24 pairs respectively; the language-model evidence is therefore exploratory.",
    ))
    story.append(P(
        "As a falsification check, we apply the same Δ estimator at proposals 2--9, before any challenge. These placebo transitions should not show the large intervention pattern. For ten-proposal cycles we report paired raw differences in immediate, follow-up, and total gain. Primary claims concern observable research-process outcomes. Task-performance contrasts are secondary: cycle comparisons are associated outcomes under the repeated policy, and longer-horizon proposal-9-to-endpoint changes are descriptive because paths were already unequal at proposal 9.",
    ))
    story.append(h2("4.3 Qualitative audit"))
    story.append(P(
        "We read all saved checkpoint metadata fields (mechanism, hypothesis, intended edit, evidence, outcome) in a generated 608-opportunity audit corpus. We then inspected complete final messages and patches for the largest source departure, largest retained improvement, and first invalid challenged alternative in every task × memory stratum, plus adjacent proposals around three productive interventions. Coding asks whether (i) the message names an old assumption, (ii) the patch actually instantiates a different computation, (iii) negative evidence constrains the alternative, and (iv) descendants exploit it. Three provider failures have no message; all remain outcome failures. No private chain-of-thought is available or inferred.",
    ))
    story.append(h2("4.4 Integrity and denominator checks"))
    story.append(P(
        "All 52 required block-condition assignments are present and every trajectory covers its analytic horizon, totaling 6,080 proposals. Treatment appears at every scheduled challenged checkpoint and nowhere in matched ordinary prompts. Of 608 checkpoint opportunities, 605 have final messages. Candidate source is reconstructible for 95.4% of all proposals; unavailable cases are primarily patches that could not produce an editable candidate.",
    ))


def page5(story: list[Flowable]) -> None:
    story.append(h1("5. Immediate effects: redirection has a price"))
    story.append(h2("5.1 Public proposals leave the local narrative"))
    story.append(P(
        "The manipulation is behaviorally visible. Relative to the preceding proposal and matched ordinary transition, lexical novelty is higher by 0.193 [0.132, 0.249] for addition, 0.233 [0.203, 0.271] for Fashion-MNIST, and 0.049 [0.018, 0.088] for language modeling. Explicit assumption language is higher by 0.838, 0.775, and 0.875 respectively; mechanism-shift language by 0.438, 0.590, and 0.542. A broad mechanism family not previously named in that trajectory is 7.5, 10.0, and 16.7 percentage points more likely. These are manipulation checks, not evidence of private understanding.",
    ))
    story.append(h2("5.2 Executable source changes—except where controls are already complex"))
    story.append(P(
        "Fashion-MNIST shows the cleanest artifact-level shift: source-token distance increases by 0.038 [0.030, 0.048], AST distance by 0.093 [0.077, 0.105], and changed lines by 30.9 [26.2, 36.0]. Language-model changes show the same pattern: +0.0095 [0.0036, 0.0143] source distance, +0.0177 [0.0149, 0.0204] AST distance, and +17.3 [14.9, 21.8] lines. Addition source distance rises only +0.0056 [−0.0028, 0.0133], while AST distance and changed lines do not increase. Its ordinary controls already implement code-heavy exact symmetry quotienting, so conceptual redirection is not synonymous with a larger patch.",
    ))
    story.extend(fig("figure1_checkpoint_effects.png", 5.35 * inch, 0.30, "Figure 1. Local matched difference-in-differences at scheduled checkpoints. Intervals resample blocks, the design's replication unit. 'Executable / qualified' is task-specific validity; higher source novelty means greater normalized-token distance from the selected parent."))
    story.append(h2("5.3 Novel proposals fail more often and cost more words"))
    story.append(table([
        ["Task", "Validity Δ", "Retention Δ", "Output-token Δ", "Immediate objective-gain Δ"],
        ["Addition", "−0.138 [−0.288, +0.013]", "−0.150 [−0.288, −0.013]", "+3,380 [+2,274, +4,706]", "+29.29 params [+16.05, +41.38]"],
        ["Fashion-MNIST", "−0.315 [−0.410, −0.220]", "−0.295 [−0.385, −0.205]", "+2,765 [+2,374, +3,280]", "+0.015 score [−1.135, +1.095]"],
        ["Language model", "−0.042 [−0.125, +0.125]", "−0.208 [−0.375, −0.125]", "+4,765 [+4,191, +5,344]", "−0.000077 bpb [−0.000274, +0.000117]"],
    ], [1.05*inch, 1.03*inch, 1.03*inch, 1.12*inch, 1.22*inch]))
    story.append(P(
        "Table 1 reports local Δ. In Fashion-MNIST, challenged checkpoints are retained only 5.0% of the time versus 36.5% for ordinary checkpoints; in language modeling, 25.0% versus 41.7%. Addition is different: challenges are also less often retained, but the rare successes remove many parameters, so immediate objective gain is strongly positive. Evaluator time shows no clear local increase on the two Mac tasks; on the H100 task it rises by 581 s [61, 1,244], reflecting some slower proposed programs.", SMALL))


def page6(story: list[Flowable]) -> None:
    story.append(h1("6. What the messages and descendants reveal"))
    story.append(h2("6.1 Addition: a branch-forming token bottleneck"))
    story.append(P(
        "Immediately before one challenged checkpoint, the agent had removed one QKV gauge coordinate at a time. At proposal 70 it instead argued that learned relative-lag attention had separated position from token identity, so the tied token interface might not require all eight residual channels. The patch retained eight-dimensional internal computation but learned only six token channels, removing 227 parameters while qualifying. Ordinary proposals 71 and 72 then reduced the bottleneck to five and four channels; proposal 73 found that three failed; proposals 74--75 resumed safe gauge reductions on the four-channel architecture. Later messages were not permanently more 'creative'; instead, the retained intervention supplied a different object for ordinary exploitation to refine.",
    ))
    story.append(P(
        "Another challenged addition proposal replaced the full absolute-position table with head-specific learned relative-lag logits and retained a 140-parameter reduction. Earlier fixed Fourier positions had failed, so the new message did not merely repeat 'try relative position': it preserved learnability and moved position into attention. More generally, controls emphasize LayerNorm-null directions, bias absorption, and row-by-row gauges; challenged messages test shared/private values, decoupled routing widths, positional codebooks, recurrent reuse, and token bottlenecks. Many fail the 99% threshold, but successful cases are followed by search over the new retained design.",
    ))
    story.append(h2("6.2 Fashion-MNIST: representation shift, then a new fixation"))
    story.append(P(
        "One trajectory spent proposals 2--9 refining translation augmentation and test-time ensembling. At proposal 10, the challenged message diagnosed a raw-pixel representation bottleneck and added fixed Sobel and Laplacian channels. The actual patch registers differential kernels, concatenates their responses with the image, and expands the first convolution. It gains 31 correct classifications and is retained. Ordinary descendants test multiscale filters and weight averaging; a 25% endpoint/EMA blend is later retained. This is a clean path from public diagnosis to code to evaluator to descendants.",
    ))
    story.append(P(
        "Over 200 proposals, however, challenge messages repeatedly return to spatial re-aggregation: attentive pooling, mixed max/mean downsampling, multiscale fusion, spatial gates, class-conditioned evidence heads, and full-grid classifiers. They do use failures to preserve learned constraints—for example, retaining the flattened spatial head after global pooling regresses—but many remain variations inside a new 'spatial representation' basin. The observed record is consistent with displacement from loss/calibration/ensemble tuning into another concentrated family, not unconstrained diversity.",
    ))
    story.append(h2("6.3 Language modeling: a challenged branch and its local program"))
    story.append(P(
        "A language-model trajectory first tunes logit softcaps. Proposal 10 challenges the assumption that attention weights alone control contextual influence and adds query-conditioned per-head output gates. It is retained. Ordinary descendants test full-state, stratified, head-aligned, MLP-branch, and fused variants, using regressions to narrow the gate design. Other challenged runs propose lexical residual paths, head-wise global context, hybrid rotary attention, or parameter-matched SwiGLU. These are different mechanisms in exact wording, yet most occupy attention/context/capacity families.",
    ))
    story.append(P(
        "Across tasks, the better messages treat negative evaluations as constraints: full sharing fails, so share one channel; fixed positions fail, so learn relative lags; global pooling fails, so add a complementary branch while preserving spatial layout. The weaker messages repeatedly rationalize a failed family. This distinction is why external evaluations and descendant behavior are more informative than eloquent self-critique [13].",
    ))


def page7(story: list[Flowable]) -> None:
    story.append(h1("7. Afterlife, displacement, and memory"))
    story.append(h2("7.1 Most useful value can arrive after the challenge"))
    story.append(table([
        ["Task", "Immediate gain: challenged − ordinary", "Next 1–9 proposals", "Whole ten-proposal cycle"],
        ["Addition", "+30.05 parameters", "+16.14 parameters", "+46.19 [27.40, 67.31] parameters"],
        ["Fashion-MNIST", "+0.80 validation score", "+2.90 score", "+3.70 [1.51, 5.89] score"],
        ["Language model", "−0.000120 bpb reduction", "+0.001288 bpb reduction", "+0.001168 [0.000651, 0.001636] bpb reduction"],
    ], [1.05*inch, 1.42*inch, 1.25*inch, 1.68*inch]))
    story.append(P(
        "Table 2 decomposes paired raw objective gain within each cycle; positive values favor challenged trajectories. Addition benefits both immediately and through descendants. Fashion-MNIST obtains 78% of its differential cycle gain during ordinary follow-up proposals. In language modeling the challenged proposal is, on average, worse immediately, but follow-up exploitation reverses the cycle. This pattern is consistent with <b>punctuated defixation</b>: a scheduled semantic shock is sometimes followed by retention of an alternative program that ordinary evidence-driven search then develops.",
    ))
    story.append(h2("7.2 Within-run departure coexists with population convergence"))
    story.append(P(
        "At matched checkpoints, challenged proposals are slightly more lexically dispersed across runs than controls (mean pairwise distance: addition 0.912 vs 0.904; Fashion-MNIST 0.880 vs 0.859; language modeling 0.926 vs 0.919). Broad family dispersion moves in the opposite direction: 0.427 vs 0.494, 0.532 vs 0.628, and 0.403 vs 0.581. Challenge messages in Fashion-MNIST mention spatial representation in 98% of checkpoints and capacity/width in 93.5%; language-model challenges overwhelmingly mention capacity, position/context, and attention. Shared task affordances matter, but the common direction also defines what 'different' should look like. Departure from one's own history and diversity across agents are distinct constructs.",
    ))
    story.append(h2("7.3 Trajectory progress is encouraging but not identified"))
    story.append(P(
        "From proposal 9 to the task-specific endpoint, challenged runs improve more than matched ordinary runs in all 10 addition pairs, 9/10 Fashion-MNIST pairs, and all 6 language-model pairs. Mean normalized progress differences are +23.10%, +0.812%, and +0.468% of the proposal-9 objective. But pre-intervention paths already differ: at proposal 9 challenged addition runs are 20 parameters better; Fashion-MNIST runs are 71.1 score units worse; language-model runs are 0.00628 bpb worse. By the endpoint, challenged addition runs are 389.5 parameters better and Fashion-MNIST runs 2.9 score units better, while language-model runs remain 0.00161 bpb worse. The last case is recovery, not endpoint superiority. These levels rule out a simple victory narrative.",
    ))
    story.append(h2("7.4 Portfolio memory is a moderator, not the finding"))
    story.append(P(
        "The redirection/cost pattern appears in both memory strata, but interactions are inconsistent. Source-novelty Δ is larger with one incumbent for addition (+0.010 vs +0.001), larger with four lineages for Fashion-MNIST (+0.051 vs +0.031), and similar for language modeling (+0.010 vs +0.009). Retention penalties also reverse by task. Proposal-9-to-endpoint progress favors challenge in every memory stratum, but the relative magnitude varies. Five blocks on two tasks and three on the third are insufficient for a stable memory interaction claim. Portfolio memory changes available parents and evidence; it does not reliably substitute for, or amplify, semantic challenge.",
    ))
    story.append(P(
        "Pre-intervention placebo transitions strengthen the immediate process interpretation. Placebo source-novelty Δ is −0.0039 for addition, −0.0011 for Fashion-MNIST, and −0.00023 for language modeling; the large positive structural jumps occur only at intervention checkpoints on the latter two tasks. Placebo output-token differences are near zero or negative, versus +2.8k--4.8k at challenged checkpoints.", SMALL))


def page8(story: list[Flowable]) -> None:
    story.append(h1("8. Implications for AI-supported science"))
    story.append(h2("8.1 An assumption challenge is a search operator"))
    story.append(P(
        "The intervention should not be evaluated as inspirational prose. At its scheduled checkpoints, observed proposals shift away from locally validated edits and toward mechanism-level alternatives. A strict evaluator rejects many of those alternatives, but rare survivors can anchor high-value descendant branches. The appropriate unit of credit is therefore not only the prompted proposal. Research-agent evaluations should report immediate feasibility, descendant value, and cost together. Otherwise challenge prompts look either useless (low immediate retention) or magical (large endpoint progress) depending on the chosen window.",
    ))
    story.append(h2("8.2 Defixation needs two diversity axes"))
    story.append(P(
        "A system can become more novel relative to itself while becoming more homogeneous relative to its peers. In these records, challenged trajectories show exactly that pattern. Future evaluations should pair <b>departure</b> from incumbent history with <b>dispersion</b> across independent trajectories. The same distinction applies to human-AI scientific ideation: a tool may help every researcher leave an initial idea yet direct all of them toward the same fashionable alternative. Fixation displacement [9] and generative monoculture [12] are not side issues; they are failure modes of successful local defixation.",
    ))
    story.append(h2("8.3 Design recommendations"))
    story.append(bullet("Trigger challenges from evidence—stagnation, repeated family failures, or low transition entropy—rather than calendar time alone."))
    story.append(bullet("Generate several alternative assumptions before selecting a patch, or randomize semantically equivalent challenge directions, to reduce the new shared attractor."))
    story.append(bullet("Retain executable alternatives in behaviorally distinct niches; a single greedy incumbent discards informative failures too quickly, while an unstructured portfolio may preserve textual variants of one family."))
    story.append(bullet("Credit descendants and report token/evaluator cost. Here a checkpoint consumes thousands of extra output tokens, and delayed value often dominates immediate value."))
    story.append(bullet("Separate task evidence from the agent's prior narrative. Memory management can propagate both useful experience and its explanation-induced bias [5]."))
    story.append(h2("8.4 Limitations"))
    story.append(P(
        "The condition labels were fixed, not randomized, and paths diverged before the first intervention; local matched changes are more credible than endpoints but remain vulnerable to state imbalance. Repeated challenges alter later prompts and parents, so later checkpoints characterize an observed policy regime, not the first prompt in isolation. Only one model family and one patch interface are studied. Broad family labels are regex-assisted and post hoc; source-token/AST distances measure structural departure, not scientific novelty, and future work should independently human-code mechanisms. We triangulate text, source, and evaluator layers, and pre-intervention placebo transitions do not show the structural/token pattern, but these checks do not convert the design into a randomized experiment. Task evaluators are narrow and can reward benchmark-specific tactics. Fashion-MNIST's last two blocks were an explicit campaign extension; the structural pattern persists in the original three, while immediate objective-gain estimates are more sensitive. We do not observe private reasoning and make no claim that the model 'believes' an assumption. Finally, 304 repeated checkpoints do not create 304 independent replications; uncertainty is clustered by 13 blocks, with only three language-model blocks.",
    ))
    story.append(h2("9. Conclusion"))
    story.append(P(
        "Across three executable research environments, scheduled assumption-challenge checkpoints are consistently associated with shifts in what an agent publicly proposes and, on two tasks, what it implements. The challenged proposals are costlier and less likely to survive immediately. Their scientific value can appear when a rare alternative becomes an artifact that ordinary descendants test and refine. Yet independent challenged trajectories also concentrate around new broad families. Under this regime, assumption challenging is best understood as punctuated, evaluator-grounded defixation with a displacement risk—not as a general creativity switch. Measuring that process requires prompts, code, failures, descendants, memory, and cost, not endpoint scores alone.",
    ))


REFERENCES = [
    "[1] AISciK Workshop. (2026). <i>Call for Papers: AI &amp; Science—Evolution or Extinction?</i> https://aiscik.github.io/call-for-papers/",
    "[2] Novikov, A., et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. <i>arXiv:2506.13131</i>.",
    "[3] Bhushan, S., Zhang, Y., &amp; Wang, L. (2026). Can LLM agents discover? Evaluating creativity on ML engineering tasks. <i>arXiv:2608.30047</i>.",
    "[4] Laban, P., Hayashi, H., Zhou, Y., &amp; Neville, J. (2026). LLMs get lost in multi-turn conversation. <i>ICLR 2026</i>.",
    "[5] Xiong, Z., et al. (2026). How memory management impacts LLM agents: An empirical study of experience-following behavior. <i>ACL 2026</i>, 623–645.",
    "[6] Kumaran, D., et al. (2026). Competing biases underlie overconfidence and underconfidence in LLMs. <i>Nature Machine Intelligence, 8</i>, 614–627.",
    "[7] Jansson, D. G., &amp; Smith, S. M. (1991). Design fixation. <i>Design Studies, 12</i>(1), 3–11.",
    "[8] Sio, U. N., Kotovsky, K., &amp; Cagan, J. (2015). Fixation or inspiration? A meta-analytic review of the role of examples on design processes. <i>Design Studies, 39</i>, 70–99.",
    "[9] Wadinambiarachchi, S., et al. (2024). The effects of generative AI on design fixation and divergent thinking. <i>CHI 2024</i>.",
    "[10] Alavi Naeini, S., et al. (2023). Large language models are fixated by red herrings. <i>NeurIPS Datasets and Benchmarks</i>.",
    "[11] Tang, Y., &amp; Yang, Y. (2026). AI research agents narrow scientific exploration. <i>arXiv:2605.27905</i>.",
    "[12] Wu, F., Black, E., &amp; Chandrasekaran, V. (2025). Generative monoculture in large language models. <i>ICLR 2025</i>.",
    "[13] Kamoi, R., et al. (2024). When can LLMs actually correct their own mistakes? A critical survey of self-correction of LLMs. <i>TACL, 12</i>.",
    "[14] Algorithmic Superintelligence Lab. (2025–2026). <i>OpenEvolve: Open-source evolutionary coding agent</i>. https://github.com/algorithmicsuperintelligence/openevolve",
    "[15] Crilly, N., &amp; Cardoso, C. (2017). Where next for research on fixation, inspiration and creativity in design? <i>Design Studies, 50</i>, 1–38.",
    "[16] Lu, Y., et al. (2025). Benchmarking language model creativity: A case study on code generation. <i>NAACL 2025</i>, 2776–2794.",
]


def references(story: list[Flowable]) -> None:
    story.append(h1("References"))
    for reference in REFERENCES:
        story.append(P(reference, REF))


def appendix_a(story: list[Flowable]) -> None:
    story.append(h1("Appendix A. Full intervention and condition mapping"))
    story.append(P("The task-general intervention text (the one task-specific sentence is shown in brackets) was:", SMALL))
    story.append(P(
        "Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs[, especially if they have resulted in a lack of progress]. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the [transformer / classifier / language model] represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.",
        QUOTE,
    ))
    story.append(P("The bracketed stagnation phrase appears in the Fashion-MNIST template. The task-specific computation sentence is: addition—'how the transformer represents or computes the task'; Fashion-MNIST—'how the classifier represents images or computes class predictions'; language modeling—'how the language model represents context or computes predictions.'", SMALL))
    story.append(table([
        ["Internal label", "Intervention", "Memory / parent policy", "Term used in paper"],
        ["C0", "ordinary at all proposals", "one current strict-improvement incumbent", "ordinary, single-incumbent"],
        ["C1", "assumption challenge at 10, 20, ...", "one current strict-improvement incumbent", "challenged, single-incumbent"],
        ["C2", "ordinary at all proposals", "four lineages; selected lineage replaced only on strict improvement", "ordinary, four-lineage"],
        ["C3", "assumption challenge at 10, 20, ...", "four lineages; selected lineage replaced only on strict improvement", "challenged, four-lineage"],
    ], [0.55*inch, 1.20*inch, 2.25*inch, 1.35*inch]))
    story.append(P(
        "All conditions receive an identical response contract: concise free-form MECHANISM, falsifiable HYPOTHESIS, INTENDED_EDIT, EVIDENCE grounded in the visible ledger, and exact SEARCH/REPLACE blocks. The mechanism name is not chosen from a fixed list. The patch is generated before evaluation; an invalid patch consumes the proposal and is never retained.",
    ))
    story.append(h2("A.1 Why this is not full native OpenEvolve"))
    story.append(P(
        "The adapter uses OpenEvolve-style proposal/evaluate/select iteration and a bounded population option, but does not reproduce the official system's full population database, MAP-Elites-style archive, inspirations, island migration, or asynchronous population dynamics [14]. The paper therefore uses 'greedy OpenEvolve-style' throughout. This distinction is substantively relevant: the strict selector makes feasibility loss at the challenge checkpoint expensive, and four-lineage memory is the only archive-like moderator examined.",
    ))


def appendix_b(story: list[Flowable]) -> None:
    story.append(h1("Appendix B. Complete quantitative summaries"))
    story.append(h2("B.1 Local checkpoint effects"))
    data = [["Metric", "Addition", "Fashion-MNIST", "Language model"]]
    labels = [
        ("Lexical novelty", "lexical_novelty", 3),
        ("New family", "new_family_tag", 3),
        ("Source novelty", "source_novelty", 4),
        ("AST distance", "ast_distance", 4),
        ("Changed lines", "changed_lines", 2),
        ("Valid", "valid", 3),
        ("Retained", "retained", 3),
        ("Output tokens", "output_tokens", 0),
        ("Evaluator seconds", "evaluator_seconds", 1),
    ]
    for label, metric, digits in labels:
        cells = [label]
        for task in ("addition", "fashion", "nanogpt"):
            row = cp(task, metric)
            cells.append(ci(row, "did_effect", digits))
        data.append(cells)
    story.append(table(data, [1.05*inch, 1.43*inch, 1.43*inch, 1.43*inch], 6.2))
    story.append(P("Entries are local matched difference-in-differences with 95% block-cluster bootstrap sensitivity intervals. Binary effects are probability-point differences; token and time effects are per checkpoint.", CAPTION))
    story.append(h2("B.2 Pre-intervention placebo transitions"))
    placebo_data = [["Metric", "Addition", "Fashion-MNIST", "Language model"]]
    for label, metric, digits in [("Source novelty", "source_novelty", 4), ("AST distance", "ast_distance", 4), ("Changed lines", "changed_lines", 2), ("Valid", "valid", 3), ("Retained", "retained", 3), ("Output tokens", "output_tokens", 0)]:
        cells = [label]
        for task in ("addition", "fashion", "nanogpt"):
            row = pl(task, metric)
            cells.append(ci(row, "mean_pseudo_did", digits))
        placebo_data.append(cells)
    story.append(table(placebo_data, [1.05*inch, 1.43*inch, 1.43*inch, 1.43*inch], 6.2))
    story.append(P("The same estimator is applied to proposals 2–9, before any challenged prompt. The large positive source/AST/token effects do not appear in this placebo window.", CAPTION))
    story.append(h2("B.3 Task-progress levels"))
    story.append(table([
        ["Task", "Mean challenged-control fitness at p9", "At endpoint", "Mean p9→endpoint normalized-progress difference"],
        ["Addition", "+20 params", "+389.5 params", "+0.23099 (10/10 pairs favor challenge)"],
        ["Fashion-MNIST", "−71.10 score", "+2.90 score", "+0.00812 (9/10)"],
        ["Language model", "−0.006282 bpb", "−0.001610 bpb", "+0.00468 (6/6)"],
    ], [1.05*inch, 1.35*inch, 1.15*inch, 1.80*inch]))
    story.append(P("Fitness is oriented so positive favors challenged runs. The endpoint progress difference is descriptive because p9 is imbalanced and trajectories are not exact forks.", CAPTION))


def appendix_c(story: list[Flowable]) -> None:
    story.append(h1("Appendix C. Trace excerpts and audit decisions"))
    story.append(h2("C.1 Productive addition sequence"))
    story.append(table([
        ["Proposal", "Direction", "Public mechanism", "Outcome"],
        ["68–69", "ordinary", "Third/fourth-row QKV input-shift quotients", "two one-parameter retained reductions"],
        ["70", "challenge", "Six-dimensional learned token bottleneck; eight-dimensional internal computation", "valid; retained; −227 parameters"],
        ["71", "ordinary", "Five-dimensional tied token bottleneck", "valid; retained; −114 parameters"],
        ["72", "ordinary", "Four-dimensional tied token bottleneck", "valid; retained; −114 parameters"],
        ["73", "ordinary", "Three-dimensional tied token bottleneck", "fails qualification; establishes boundary"],
        ["74–75", "ordinary", "Fifth/sixth-row QKV quotients", "two retained one-parameter reductions"],
    ], [0.58*inch, 0.72*inch, 2.65*inch, 1.40*inch]))
    story.append(h2("C.2 Productive Fashion-MNIST sequence"))
    story.append(P(
        "The challenged message at proposal 10 says the fixed exposure may be insufficient for a shallow stem to learn shape-sensitive primitives from raw pixels. Its patch adds three fixed differential channels (horizontal/vertical Sobel and Laplacian) and expands the first convolution from one to four inputs. It improves from 9,091 to 9,122 correct. Proposals 11–15 then test EMA, broader fixed filters, and interpolation strengths; proposal 13 retains an endpoint/EMA blend at 9,130 correct. This case is coded as genuine representation change, evidence-grounded, valid, retained, and descendant-producing.",
    ))
    story.append(h2("C.3 Productive language-model sequence"))
    story.append(P(
        "The challenged message at proposal 10 contrasts fixed-amplitude residual attention with token-conditioned control and adds per-head gates. It improves validation bits per byte from 0.995200 to 0.993637. Proposals 11–15 test full-state, channel-stratified, head-aligned, MLP-branch, and fused variants. The descendant program is evidence-responsive: full-state conditioning reduces throughput and regresses, so later variants restore a 32-channel conditioning path. This case is coded as representation change, valid, retained, and descendant-producing.",
    ))
    story.append(h2("C.4 Counterexample: repeated Fashion-MNIST spatial alternatives"))
    story.append(P(
        "In one trajectory, challenges at 80, 100, 130, 150, and 170 all propose variants of per-channel max/average downsampling; nearby checkpoints propose spatial attention pooling, multiscale lateral fusion, or spatial gates. The messages correctly cite shift sensitivity and preserve the flattened head after global pooling fails, but none of these listed checkpoints is retained. This is coded as formal compliance plus fixation displacement: the prompt leaves calibration/ensemble tuning but concentrates on a new spatial family.",
    ))


def appendix_d(story: list[Flowable]) -> None:
    story.append(h1("Appendix D. Reproducibility and additional checks"))
    story.append(h2("D.1 Reproduction"))
    story.append(P(
        "The analysis script is <font name='Courier'>papers/aiscik2026/paper6/analysis.py</font>. It discovers runs from the three campaign manifests, verifies the full task-specific horizons and intervention schedule, reconstructs parent/candidate source, generates proposal- and pair-level CSV files, and uses fixed random seed 20260903 for 10,000 block-cluster bootstrap resamples. Running it performs no model calls and no training.",
    ))
    story.append(P(
        "An anonymized supplemental artifact containing the analysis code, derived tables, figures, and trace-audit materials accompanies the manuscript and is released under the MIT License. Raw run records are omitted from the paper PDF and can be supplied in a separately anonymized archive where venue storage permits.",
        SMALL,
    ))
    story.append(P("Command:", SMALL))
    story.append(P("<font name='Courier'>python papers/aiscik2026/paper6/analysis.py</font>", QUOTE))
    story.append(h2("D.2 Generated evidence"))
    for item in [
        "proposal_records.csv: 6,080 proposal-level rows with prompt arm, source distances, message measures, evaluator outcomes, costs, and objective gain.",
        "checkpoint_pairs.csv / checkpoint_effects.csv: 304 matched checkpoints and task/memory summaries.",
        "preintervention_placebos.csv: identical estimand at proposals 2–9.",
        "cycle_gain_pairs.csv: immediate, follow-up, and ten-proposal gain decomposition.",
        "population_dispersion.csv: between-run lexical and broad-family distances at every checkpoint.",
        "checkpoint_message_corpus.md: all saved public mechanism/hypothesis/edit/evidence fields at 608 checkpoint opportunities.",
        "qualitative_sample.md and qualitative_audit.md: deterministic high-information cases and the trace-grounded audit used in the paper.",
    ]:
        story.append(bullet(item))
    story.append(h2("D.3 Original-block sensitivity for Fashion-MNIST"))
    story.append(P(
        "Using only the three originally created blocks, the intervention's local source-novelty Δ is +0.0300, AST-distance Δ +0.0899, changed-lines Δ +34.41, validity Δ −0.358, retention Δ −0.375, and output-token Δ +2,960. These agree with the all-five-block structural/cost findings. The local immediate objective-gain DiD is −0.533 score units in the original blocks versus approximately zero across all five; objective-gain claims are therefore based on cycle decomposition and are described as sensitive. The paired ten-proposal cycle-gain difference remains positive in original blocks (+4.07 score units) and added blocks (+3.15).",
    ))
    story.append(h2("D.4 Construct validity"))
    story.append(P(
        "Lexical novelty can be gamed by terminology; AST distance can overvalue mechanical rewrites; changed lines can reward verbosity; broad family labels can hide meaningful subfamilies; strict retention conflates task value with evaluator noise; and descendant gain can be misattributed when later ordinary proposals introduce independent ideas. The paper therefore triangulates all six layers and treats disagreement—especially within-run departure with lower across-run family dispersion—as a result rather than averaging it away.",
    ))


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story: list[Flowable] = []
    for function in (page1, page2, page3, page4, page5, page6, page7, page8):
        function(story)
        pagenext(story)
    references(story)
    pagenext(story)
    appendix_a(story)
    pagenext(story)
    appendix_b(story)
    pagenext(story)
    appendix_c(story)
    pagenext(story)
    appendix_d(story)
    PaperDoc(str(OUTPUT)).build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
