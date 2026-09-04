#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the AISciK-ready Paper 5 PDF."""

from __future__ import annotations

import csv
import hashlib
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
REPO = HERE.parents[2]
DERIVED = HERE / "derived"
OUTPUT = REPO / "output/pdf/paper5_interface_instrument.pdf"
ARTIFACT = REPO / "output/paper5_reproducibility_artifact.zip"


SUMMARY = json.loads((DERIVED / "aggregate_summary.json").read_text(encoding="utf-8"))
CONTRASTS = SUMMARY["contrasts"]
PROMPT_CHECKS = SUMMARY["prompt_composition_checks"]
QUAL = SUMMARY["qualitative_examples"]
PROTOCOL_METADATA = SUMMARY.get("protocol_metadata", [])


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


TRAJECTORIES = load_rows(DERIVED / "trajectory_metrics.csv")


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

BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontName=FONT, fontSize=10.0, leading=11.0, alignment=TA_JUSTIFY, spaceAfter=4.0, allowWidows=0, allowOrphans=0)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.0, leading=8.9, spaceAfter=2.4)
TINY = ParagraphStyle("Tiny", parent=BODY, fontSize=6.9, leading=7.7, alignment=TA_LEFT, spaceAfter=2.0)
H1 = ParagraphStyle("H1", parent=BODY, fontName=FONT_BOLD, fontSize=11.25, leading=12.2, alignment=TA_LEFT, spaceBefore=3.7, spaceAfter=2.7, keepWithNext=True)
H2 = ParagraphStyle("H2", parent=BODY, fontName=FONT_BOLD, fontSize=9.45, leading=10.25, alignment=TA_LEFT, spaceBefore=2.7, spaceAfter=1.8, keepWithNext=True)
TITLE = ParagraphStyle("Title", parent=BODY, fontName=FONT_BOLD, fontSize=15.8, leading=17.6, alignment=TA_CENTER, spaceBefore=3, spaceAfter=5)
AUTHOR = ParagraphStyle("Author", parent=BODY, fontSize=9.0, leading=9.9, alignment=TA_CENTER, spaceAfter=6)
ABSTRACT_HEAD = ParagraphStyle("AbstractHead", parent=BODY, fontName=FONT_BOLD, fontSize=9.1, leading=10.0, alignment=TA_CENTER, spaceAfter=2)
ABSTRACT = ParagraphStyle("Abstract", parent=BODY, fontSize=8.55, leading=9.55, leftIndent=0.18 * inch, rightIndent=0.18 * inch, alignment=TA_JUSTIFY, spaceAfter=5)
CAPTION = ParagraphStyle("Caption", parent=SMALL, fontName=FONT_ITALIC, alignment=TA_LEFT, spaceBefore=1.3, spaceAfter=2.8)
REF = ParagraphStyle("Ref", parent=SMALL, leftIndent=0.18 * inch, firstLineIndent=-0.18 * inch, fontSize=7.45, leading=8.25, spaceAfter=2.0)
QUOTE = ParagraphStyle("Quote", parent=SMALL, leftIndent=0.15 * inch, rightIndent=0.15 * inch, borderColor=colors.HexColor("#d8dee9"), borderWidth=0.55, borderPadding=4, backColor=colors.HexColor("#f8fafc"))


class PaperDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=1.5 * inch,
            rightMargin=1.0 * inch,
            topMargin=0.72 * inch,
            bottomMargin=0.72 * inch,
            title="The Interface Is the Instrument",
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


def fmt(value: Any, digits: int = 3) -> str:
    value = float(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def signed(value: Any, digits: int = 3) -> str:
    value = float(value)
    return f"{value:+.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def comma(value: Any, digits: int = 0) -> str:
    return f"{float(value):,.{digits}f}"


def compact_tokens(value: Any) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def summary(comparison: str, interface: str) -> dict[str, Any]:
    for row in SUMMARY["summary"]:
        if row["comparison"] == comparison and row["interface"] == interface:
            return row
    raise KeyError((comparison, interface))


def contrast(comparison: str, metric: str) -> dict[str, Any]:
    return CONTRASTS[f"{comparison}::{metric}"]


def ci(comparison: str, metric: str, digits: int = 3) -> str:
    row = contrast(comparison, metric)
    return f"[{float(row['bootstrap_low']):.{digits}f}, {float(row['bootstrap_high']):.{digits}f}]"


def ci_tokens(comparison: str, metric: str) -> str:
    row = contrast(comparison, metric)
    return f"[{compact_tokens(row['bootstrap_low'])}, {compact_tokens(row['bootstrap_high'])}]"


def artifact_hash() -> str:
    if not ARTIFACT.is_file():
        return "artifact not yet built"
    digest = hashlib.sha256()
    with ARTIFACT.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ruled_title(story: list[Flowable]) -> None:
    story.append(HRFlowable(width="100%", thickness=4, color=colors.black, spaceAfter=8))
    story.append(P("The Interface Is the Instrument: How Autonomous-Research Scaffolds Change Scientific Traces", TITLE))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=2, spaceAfter=7))
    story.append(P("Anonymous Authors", AUTHOR))


def bullet(text: str) -> Flowable:
    table = Table([[P("-", SMALL), P(text, SMALL)]], colWidths=[0.16 * inch, 5.14 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def data_table(data: list[list[str]], widths: list[float], font_size: float = 6.75) -> Table:
    converted = [[P(str(cell), TINY) for cell in row] for row in data]
    table = Table(converted, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 0.8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd6df")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def fig(path: Path, width: float, height_ratio: float, caption: str) -> list[Flowable]:
    image = Image(str(path), width=width, height=width * height_ratio)
    image.hAlign = "CENTER"
    return [image, P(caption, CAPTION)]


def page1(story: list[Flowable]) -> None:
    ruled_title(story)
    story.append(P("Abstract", ABSTRACT_HEAD))
    story.append(P(
        "Autonomous-science evaluations often report a result as if the model were the researcher. In practice, the research system is a composed interface: a language model, a memory surface, a patch contract, an evaluator, a parent-selection rule, and a logging schema. We audit already-recorded neural-architecture search traces to ask what changes when the same family of agent is wrapped as a continuous Autoresearch session, as bounded greedy OpenEvolve-style patch calls, or as a native-population OpenEvolve controller. In the main same-task Fashion-MNIST contrast, bounded calls and continuous sessions show no clear endpoint separation at a 44-proposal common horizon, yet they differ by orders of magnitude in token accounting, trace completeness, timeout profile, source novelty, and leakage-like host-path markers. NanoGPT and Tiny Addition extensions show that these are not merely cosmetic logging differences: population context changes the candidate sources visible to the model and can increase novelty while lowering endpoint improvement. The contribution is a measurement warning and a reporting checklist: an AI scientist benchmark measures an interface-induced research process, not a decontextualized model capability.",
        ABSTRACT,
    ))
    story.append(h1("1. Introduction"))
    story.append(P(
        "The central measurement problem for autonomous research agents is easy to miss. When an agent improves a validation score, the published number names the task, model, and final metric; it often leaves the research interface implicit. But interface choices are not neutral plumbing. A continuous chat session can carry latent conversational state and long summaries. A bounded patch call can force every proposal through a structured hypothesis-edit-evidence contract. A population controller can show multiple parents, inspirations, and archive state. These choices reshape both what the agent can attempt and what the evaluator can observe.",
    ))
    story.append(P(
        "AISciK asks whether AI systems are changing scientific practice, and whether our instruments measure what they claim. We therefore study the interface itself as the object of measurement. The research question is: <i>when autonomous ML-research traces are collected under different scaffolds, which apparent scientific behaviors are stable endpoint outcomes and which are artifacts of the interface used to elicit, constrain, and record proposals?</i>",
    ))
    story.append(P(
        "We make three contributions. First, we provide a trace-level comparison of continuous Autoresearch, bounded greedy OpenEvolve-style, and native-population OpenEvolve-style interfaces on existing campaign logs. Second, we separate endpoint progress from measurement properties: token exposure, schema completeness, source-level novelty, timeout rates, and prompt-composition validity. Third, we give a reproducible reporting checklist for future AI-scientist benchmarks.",
    ))
    pagebreak(story)


def page2(story: list[Flowable]) -> None:
    story.append(h1("2. Related work and framing"))
    story.extend(fig(DERIVED / "figure1_interfaces.png", 5.25 * inch, 0.38, "<b>Figure 1:</b> The paper treats controller scaffolds as measurement instruments. Changing the session, patch contract, parent sampler, or visible archive can change both search behavior and trace validity."))
    story.append(P(
        "Construct validity begins with the question of whether an observed score actually represents the intended construct [2,3,4]. Autonomous-research benchmarks such as MLAgentBench, MLE-bench, and RE-Bench evaluate agents on executable ML or research-engineering tasks [5,6,7]. End-to-end AI-scientist systems add idea generation, experiment execution, and manuscript production [8,9]. Evolutionary coding systems such as AlphaEvolve and OpenEvolve combine model-generated programs with external selection and archive mechanisms [10,11]. Autoresearch-style sessions instead expose a long-running research conversation and workspace to a single subject loop [12].",
    ))
    story.append(P(
        "These systems differ in more than implementation details. Reflexion and related agent-memory work show that textual memory can serve as behavioral feedback [13]. Evolutionary-computation work shows that population structure, novelty, and parent selection change search dynamics [14,15]. Measurement critiques of AI science warn that apparent understanding or progress can be created by inappropriate instruments [16]. Our gap is not another benchmark score. It is an empirical audit of how the interface that produces the score changes the scientific trace itself.",
    ))
    story.append(h2("Claim boundary"))
    story.append(P(
        "The contrasts below are descriptive and interface-composed. We use already-collected traces, right-censor each stratum to a common observed horizon, and pair only by task block and condition when that pairing exists. We do not claim a randomized estimate of a latent model capability, nor do we infer private chain-of-thought. Agent-language measures use recorded final summaries and visible prompts only.",
    ))
    pagebreak(story)


def page3(story: list[Flowable]) -> None:
    story.append(h1("3. Data and methods"))
    fa = summary("fashion_same_task", "continuous_autoresearch")
    fg = summary("fashion_same_task", "bounded_greedy_patch")
    na = summary("nanogpt_early", "continuous_autoresearch")
    ng = summary("nanogpt_early", "bounded_greedy_patch")
    tg = summary("tiny_open_evolve", "bounded_greedy_patch")
    tn = summary("tiny_open_evolve", "native_population")
    table = [
        ["Stratum", "Interfaces", "Runs", "Common horizon", "Endpoint"],
        ["Fashion-MNIST", "continuous vs bounded greedy", f"{int(fa['runs'])}+{int(fg['runs'])}", "44", "validation-correct gain"],
        ["nanoGPT", "continuous vs bounded greedy", f"{int(na['runs'])}+{int(ng['runs'])}", "5", "decrease in validation bpb"],
        ["Tiny Addition", "bounded greedy vs native population", f"{int(tg['runs'])}+{int(tn['runs'])}", "70", "qualified parameter reduction"],
    ]
    story.append(P("<b>Table 1:</b> Analyzed strata. Fashion-MNIST is the primary same-task contrast. nanoGPT and Tiny Addition are extensions that stress-test the interface interpretation.", CAPTION))
    story.append(data_table(table, [1.15 * inch, 1.65 * inch, 0.58 * inch, 0.85 * inch, 1.27 * inch]))
    story.append(P(
        "All inputs are stored under campaign directories in the repository and are enumerated by the reproducibility artifact. The analysis parses 6,008 proposal-completed events from 120 trajectories and records a 34,977-file input ledger. For each trajectory we compute endpoint progress at the right-censored horizon, valid and retained rates, timeout rate, accounted token use per proposal, source-structural novelty, changed lines, parent-selection indicators, schema completeness, and host-path marker rate. For bounded and native OpenEvolve, complete-field rate records whether the structured mechanism, evidence, hypothesis, and intended-edit fields are present; Figure 2 abbreviates these as MEHI fields. For Autoresearch, those fields are absent by design, so the rate is zero rather than a missing-data accident.",
    ))
    story.append(P(
        "Fashion-MNIST trajectories are paired by block and condition: B1--B4, C0--C3. Pairwise intervals use fixed-seed nonparametric bootstrap resampling over trajectory pairs. nanoGPT uses the first five proposals because the continuous Autoresearch nanoGPT runs have short, uneven horizons. Tiny Addition compares two OpenEvolve interfaces under a common protocol at 70 proposals and is paired by block and condition across eight blocks.",
    ))
    story.append(P(
        "Source novelty is a normalized Python-token 3-gram Jaccard distance from the selected parent after abstracting identifiers, numbers, and strings. It is a transparent structural proxy, not a semantic novelty oracle. Prompt-composition checks search the visible prompt files for reference-count language and candidate references; they identify whether a controller says the agent has no references while simultaneously showing candidate context.",
    ))
    pagebreak(story)


def page4(story: list[Flowable]) -> None:
    story.append(h1("4. Same-task result: similar endpoint, different instrument"))
    story.extend(fig(DERIVED / "figure2_fashion_pairs.png", 5.28 * inch, 0.36, "<b>Figure 2:</b> Matched Fashion-MNIST pairs at the first 44 proposals. Endpoint gains are balanced, while token accounting and trace schema differ by orders of magnitude."))
    fa = summary("fashion_same_task", "continuous_autoresearch")
    fg = summary("fashion_same_task", "bounded_greedy_patch")
    endpoint = contrast("fashion_same_task", "endpoint_delta")
    token = contrast("fashion_same_task", "tokens_per_proposal")
    table = [
        ["Measure", "Continuous Autoresearch", "Bounded greedy", "Paired difference"],
        ["endpoint correct gain", comma(fa["endpoint_delta_mean"], 1), comma(fg["endpoint_delta_mean"], 1), f"{signed(endpoint['difference_mean'], 1)} {ci('fashion_same_task', 'endpoint_delta', 1)}"],
        ["tokens/proposal", compact_tokens(fa["tokens_per_proposal_mean"]), compact_tokens(fg["tokens_per_proposal_mean"]), f"{compact_tokens(token['difference_mean'])} {ci_tokens('fashion_same_task', 'tokens_per_proposal')}"],
        ["valid rate", pct(fa["valid_rate_mean"]), pct(fg["valid_rate_mean"]), f"{signed(contrast('fashion_same_task', 'valid_rate')['difference_mean'], 3)} {ci('fashion_same_task', 'valid_rate', 3)}"],
        ["timeout rate", pct(fa["timeout_rate_mean"]), pct(fg["timeout_rate_mean"]), f"{signed(contrast('fashion_same_task', 'timeout_rate')['difference_mean'], 3)} {ci('fashion_same_task', 'timeout_rate', 3)}"],
        ["source novelty", fmt(fa["mean_source_novelty_mean"], 3), fmt(fg["mean_source_novelty_mean"], 3), f"{signed(contrast('fashion_same_task', 'mean_source_novelty')['difference_mean'], 3)} {ci('fashion_same_task', 'mean_source_novelty', 3)}"],
    ]
    story.append(P("<b>Table 2:</b> Main Fashion-MNIST measurement contrast. The paired difference is bounded greedy minus continuous Autoresearch.", CAPTION))
    story.append(data_table(table, [1.25 * inch, 1.25 * inch, 1.08 * inch, 1.92 * inch]))
    story.append(P(
        "The endpoint lesson is deliberately modest. Bounded calls gain 5.9 more validation-correct examples on average, with a bootstrap interval from -22.1 to +34.0 and an exact 8--8 split in which interface is ahead. A score-only report would treat the two systems as roughly interchangeable at this horizon.",
    ))
    story.append(P(
        "The trace lesson is not modest. Continuous sessions average 6.86 million accounted tokens per proposal versus 22.8 thousand for bounded calls; every one of the 16 matched pairs has larger continuous token accounting. Bounded calls have complete structured fields by construction, while continuous sessions do not. Bounded calls time out more often and validate less often, but they also make larger and more structurally novel source edits. The same endpoint therefore hides a different research process and a different measurement aperture.",
    ))
    pagebreak(story)


def page5(story: list[Flowable]) -> None:
    story.append(h1("5. What the interface changes"))
    fa = summary("fashion_same_task", "continuous_autoresearch")
    fg = summary("fashion_same_task", "bounded_greedy_patch")
    story.append(h2("5.1 Token exposure and session continuity"))
    story.append(P(
        f"The largest observed difference is context exposure. Continuous Fashion-MNIST sessions average {compact_tokens(fa['tokens_per_proposal_mean'])} accounted tokens/proposal, with output alone near {compact_tokens(fa['output_tokens_per_proposal_mean'])}. Bounded calls average {compact_tokens(fg['tokens_per_proposal_mean'])} tokens/proposal and {compact_tokens(fg['output_tokens_per_proposal_mean'])} output tokens/proposal. This is not simply a cost fact. More prompt history changes what the model can condition on, which failures it can remember, and which local path details or previous measurements can enter its final summaries.",
    ))
    story.append(h2("5.2 Schema as an epistemic instrument"))
    story.append(P(
        f"The bounded controller records a complete hypothesis/edit/mechanism/evidence schema for every analyzed Fashion-MNIST proposal; continuous Autoresearch records none of those exact fields. Conversely, Autoresearch has a persistent session identifier, while bounded calls reconstruct continuity from files and controller state. Neither schema is intrinsically more scientific. The point is that they measure different observables: one captures conversational continuity coarsely, the other captures proposal-level mechanistic claims explicitly.",
    ))
    story.append(h2("5.3 Validity, timeouts, and source novelty"))
    story.append(P(
        f"Bounded Fashion-MNIST calls validate less often ({pct(fg['valid_rate_mean'])} versus {pct(fa['valid_rate_mean'])}) and time out more often ({pct(fg['timeout_rate_mean'])} versus {pct(fa['timeout_rate_mean'])}). They also make larger source edits: mean changed lines are higher by {fmt(contrast('fashion_same_task', 'mean_changed_lines')['difference_mean'], 2)}, and source novelty is higher by {fmt(contrast('fashion_same_task', 'mean_source_novelty')['difference_mean'], 3)}. A benchmark that counts only retained endpoint improvements can miss whether an interface suppresses risky mechanisms or merely fails to finish evaluating them.",
    ))
    story.append(h2("5.4 Leakage-like trace artifacts"))
    story.append(P(
        f"Recorded continuous Fashion-MNIST messages contain host-path markers at a mean rate of {pct(fa['local_path_marker_rate_mean'])}; bounded calls are zero under the same detector. This does not mean the model cheated or saw sealed test data. It means the interface allowed machine-local operational detail to enter the scientific trace. For AI-science measurement, such details matter: a trace can look more richly reasoned because it contains implementation context that another interface prevents or structures differently.",
    ))
    story.append(P(
        "These four differences make the main claim: autonomous-research traces are not portable observations of a model's scientific ability unless the interface is part of the reported instrument.",
    ))
    pagebreak(story)


def page6(story: list[Flowable]) -> None:
    story.append(h1("6. Cross-task and native-population checks"))
    story.extend(fig(DERIVED / "figure3_extensions.png", 5.2 * inch, 0.36, "<b>Figure 3:</b> Extensions. nanoGPT reproduces the token/schema contrast at a five-proposal early horizon. Tiny Addition shows that native population context changes novelty and endpoint progress within OpenEvolve-family controllers."))
    na = summary("nanogpt_early", "continuous_autoresearch")
    ng = summary("nanogpt_early", "bounded_greedy_patch")
    tg = summary("tiny_open_evolve", "bounded_greedy_patch")
    tn = summary("tiny_open_evolve", "native_population")
    tiny_endpoint = contrast("tiny_open_evolve", "endpoint_delta")
    table = [
        ["Stratum", "Main endpoint contrast", "Trace contrast"],
        ["nanoGPT early", f"bounded - continuous bpb gain {signed(contrast('nanogpt_early', 'endpoint_delta')['difference_mean'], 5)} {ci('nanogpt_early', 'endpoint_delta', 5)}", f"tokens/proposal {compact_tokens(na['tokens_per_proposal_mean'])} vs {compact_tokens(ng['tokens_per_proposal_mean'])}; complete fields 0% vs 100%"],
        ["Tiny Addition", f"native - greedy parameter reduction {signed(tiny_endpoint['difference_mean'], 0)} {ci('tiny_open_evolve', 'endpoint_delta', 0)}", f"source novelty {fmt(tn['mean_source_novelty_mean'], 3)} vs {fmt(tg['mean_source_novelty_mean'], 3)}; non-incumbent parent rate +{fmt(contrast('tiny_open_evolve', 'non_incumbent_parent_rate')['difference_mean'], 3)}"],
    ]
    story.append(P("<b>Table 3:</b> Extension checks on nanoGPT and Tiny Addition.", CAPTION))
    story.append(data_table(table, [1.0 * inch, 2.15 * inch, 2.35 * inch]))
    story.append(P(
        "nanoGPT is intentionally right-censored at five proposals because the continuous runs are short. Even there, endpoint gains are small and overlapping, but token accounting and schema completeness diverge strongly: continuous calls average 481.6k tokens/proposal while bounded calls average 31.7k, and the structured-field contrast remains 0% versus 100%.",
    ))
    story.append(P(
        "Tiny Addition isolates a different question: what changes when OpenEvolve-family search uses native population context rather than a bounded greedy incumbent interface? Native runs expose more candidate context and select non-incumbent parents far more often. They also show higher source novelty. But at 70 proposals they reduce qualified parameters less than the bounded greedy runs: native-minus-greedy endpoint delta is -1,418 parameters with interval [-2,202, -624]. The interface creates more population-like exploration, not automatic endpoint superiority.",
    ))
    checks = PROMPT_CHECKS["native_population"]
    story.append(P(
        f"The native prompt audit also caught a compositional artifact: {pct(checks['no_reference_sentence_with_reference_rate'])} of native prompts contained a no-reference sentence while also exposing reference candidates. This is not a fatal flaw for the stored runs, but it is exactly the kind of controller-level contradiction that a score table would not reveal.",
    ))
    pagebreak(story)


def page7(story: list[Flowable]) -> None:
    story.append(h1("7. Qualitative audit"))
    example = QUAL.get("closest_fashion_endpoint_pair", {})
    story.append(P(
        f"The closest Fashion-MNIST endpoint pair occurs at block {example.get('block', 'NA')}, condition {example.get('condition', 'NA')}: bounded-minus-continuous correct-count gain {example.get('bounded_minus_continuous_correct_delta', 'NA')}. This is the paper's most useful negative example. Near-identical endpoint progress coexists with radically different token exposure, field completeness, and source-edit style.",
    ))
    local = QUAL.get("autoresearch_local_marker", {})
    syntax = QUAL.get("nanogpt_syntax_only", {})
    story.append(P("Observed trace examples include:", H2))
    story.append(bullet(f"Continuous Fashion-MNIST {local.get('run_id', 'run')} opportunity {local.get('opportunity', 'NA')} includes an editor-action summary with a machine-local path marker. The analysis counts this as operational context entering the scientific record, not as evidence of sealed-data access."))
    story.append(bullet(f"Continuous nanoGPT {syntax.get('run_id', 'run')} opportunity {syntax.get('opportunity', 'NA')} reports syntax-oriented work in the subject message before harness evaluation. This illustrates why recorded final messages should be triangulated with evaluator outputs and source snapshots."))
    story.append(bullet("Native Tiny Addition prompts frequently expose selected parents, inspiration candidates, and archive context. The source-novelty and non-incumbent-parent metrics are therefore measurements of a different research institution, not just a stronger or weaker base model."))
    story.append(h2("Reporting checklist"))
    checklist = [
        "Report the session model: one continuous conversation, independent bounded calls, or external population controller.",
        "Report what history is visible: prior messages, final summaries, candidate sources, evaluator outputs, archive state, and parent IDs.",
        "Report the edit contract: freeform workspace editing, exact patch, whole-file replacement, or native controller prompt.",
        "Report trace completeness: which fields are mandatory, which are inferred, and which are absent by design.",
        "Report resource exposure: accounted input, cached input, output, wall-clock, evaluator seconds, and timeout denominators.",
        "Audit prompt composition and leakage-like operational artifacts before treating agent text as scientific evidence.",
    ]
    for item in checklist:
        story.append(bullet(item))
    story.append(P(
        "The checklist is intentionally mundane. It makes the scaffold auditable because the scaffold partly determines what is observed.",
    ))
    pagebreak(story)


def page8(story: list[Flowable]) -> None:
    story.append(h1("8. Discussion"))
    story.append(P(
        "These results support a conservative interpretation. Continuous Autoresearch, bounded greedy patch calls, and native OpenEvolve populations can all be sensible ways to run autonomous ML research. But they should not be collapsed into a single variable named model capability. In our main same-task stratum, endpoint progress is nearly exchangeable while measurement properties are not. In the Tiny Addition stratum, adding native population context increases visible exploration yet reduces the endpoint at the common horizon. Interface-induced behavior can therefore change both the process and the meaning of the resulting trace.",
    ))
    story.append(h2("Limitations"))
    story.append(P(
        "The study is observational and uses existing campaigns. Fashion-MNIST is the strongest comparison because task and seed candidate align, but even there the interfaces differ in controller design and session handling. nanoGPT is only an early-prefix check. Tiny Addition compares two OpenEvolve-style controllers but on a different task and protocol. Bootstrap intervals resample matched trajectories and are descriptive sensitivity intervals, not causal confidence intervals. Text analysis uses recorded final summaries and prompt files; it does not inspect hidden reasoning. Source novelty and schema completeness are useful proxies but not validated measures of scientific creativity.",
    ))
    story.append(h2("Conclusion"))
    story.append(P(
        "An autonomous-science benchmark is a scientific instrument. Its readings depend on how hypotheses are elicited, how memory is carried, how candidate parents are selected, how evaluation feedback is returned, and how traces are logged. Reporting only the endpoint score treats the instrument as invisible. A more credible AI-science evaluation reports the interface as part of the measurement apparatus and audits whether that apparatus changes the scientific behavior it claims to observe.",
    ))
    story.append(h2("Reproducibility"))
    story.append(P(
        f"The anonymized artifact packages {SUMMARY['input_file_count']:,} input files and deterministic analysis code under an MIT license. Archive SHA-256: <font name='{FONT_BOLD}'>{artifact_hash()}</font>. The archive is intended to be shareable as a static supplement; it reproduces all tables and figures without calling model providers or evaluators.",
        SMALL,
    ))
    pagebreak(story)


def references(story: list[Flowable]) -> None:
    story.append(h1("References"))
    refs = [
        "AISciK. 2026. AI & Science: Evolution or Extinction? Call for papers. https://aiscik.github.io/call-for-papers/.",
        "Cronbach, L. J., and Meehl, P. E. 1955. Construct validity in psychological tests. Psychological Bulletin 52(4):281--302.",
        "Jacobs, A. Z., and Wallach, H. 2021. Measurement and fairness. Proceedings of ACM FAccT.",
        "Bean, A. M.; Kearns, R. O.; Romanou, A.; Hafner, F. S.; Mayne, H.; Batzner, J.; et al. 2025. Measuring what matters: construct validity in large language model benchmarks. arXiv:2511.04703.",
        "Huang, Q.; Vora, J.; Liang, P.; and Leskovec, J. 2024. MLAgentBench: evaluating language agents on machine learning experimentation. ICML/PMLR 235.",
        "Chan, J. S.; Chowdhery, A.; et al. 2025. MLE-bench: evaluating machine learning agents on machine learning engineering. ICLR 2025.",
        "Wijk, H.; Lin, T.; Becker, J.; et al. 2024. RE-Bench: evaluating frontier AI R&D capabilities of language model agents against human experts. arXiv:2411.15114.",
        "Lu, C.; Lu, C.; Lange, R. T.; et al. 2024. The AI Scientist: towards fully automated open-ended scientific discovery. arXiv:2408.06292.",
        "Lu, C.; Lu, C.; Lange, R. T.; et al. 2026. Towards end-to-end automation of AI research. Nature 651:914--919.",
        "Novikov, A.; Vũ, N.; Eisenberger, M.; Dupont, E.; Huang, P.-S.; et al. 2025. AlphaEvolve: a coding agent for scientific and algorithmic discovery. arXiv:2506.13131.",
        "Sharma, A. 2025--2026. OpenEvolve: an open-source evolutionary coding agent. Software repository, https://github.com/algorithmicsuperintelligence/openevolve.",
        "Karpathy, A. 2025. Autoresearch. Software repository and experimental autonomous-research scaffold.",
        "Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: language agents with verbal reinforcement learning. NeurIPS 36.",
        "Lehman, J., and Stanley, K. O. 2011. Abandoning objectives: evolution through the search for novelty alone. Evolutionary Computation 19(2):189--223.",
        "Mouret, J.-B., and Clune, J. 2015. Illuminating search spaces by mapping elites. arXiv:1504.04909.",
        "Messeri, L., and Crockett, M. J. 2024. Artificial intelligence and illusions of understanding in scientific research. Nature 627:49--58.",
        "Dwork, C.; Feldman, V.; Hardt, M.; Pitassi, T.; Reingold, O.; and Roth, A. 2015. The reusable holdout: preserving validity in adaptive data analysis. Science 349:636--638.",
    ]
    for index, ref in enumerate(refs, 1):
        story.append(P(f"[{index}] {ref}", REF))
    pagebreak(story)


def appendix(story: list[Flowable]) -> None:
    story.append(h1("Appendix A. Additional operational details"))
    story.append(P(
        "The Fashion-MNIST main contrast uses the largest common prefix available for all 32 trajectories in the two same-task interfaces. Bounded OpenEvolve continued to 200 proposals, but only the first 44 proposals are used in the direct paired contrast because that is the common horizon shared with the Autoresearch trajectories. The nanoGPT contrast is similarly truncated to five proposals. Tiny Addition uses 70 proposals, the common horizon available in both OpenEvolve-family campaigns.",
    ))
    story.append(P(
        "Endpoint progress is task-specific: additional public-validation correct predictions for Fashion-MNIST, negative validation bits-per-byte for nanoGPT, and negative parameter count among candidates with at least 99% exact accuracy for Tiny Addition. Token accounting uses stored controller usage records; when provider-side accounting differs across interfaces, the paper treats this difference as an observed property of the research instrument rather than normalizing it away.",
    ))
    story.append(P(
        "All privacy-sensitive local paths in the packaged artifact are anonymized. The path-marker metric is computed from the original local analysis before anonymization, and the artifact preserves the existence of the marker through neutral replacement tokens rather than exposing user-specific filesystem paths.",
    ))
    if PROTOCOL_METADATA:
        models = sorted({str(row.get("model_name")) for row in PROTOCOL_METADATA})
        efforts = sorted({str(row.get("reasoning_effort")) for row in PROTOCOL_METADATA})
        tiers = sorted({str(row.get("service_tier")) for row in PROTOCOL_METADATA})
        modes = sorted({str(row.get("conversation_mode")) for row in PROTOCOL_METADATA})
        story.append(P(
            "Subject configuration from the included campaign protocol files: model "
            + ", ".join(models)
            + "; reasoning effort "
            + ", ".join(efforts)
            + "; service tiers "
            + ", ".join(tiers)
            + "; conversation modes "
            + ", ".join(modes)
            + ". Event files contain realized token usage and service-tier fields; the frozen protocol files provide the requested subject configuration.",
        ))
    story.append(Spacer(1, 0.08 * inch))
    rows = [
        ["Campaign key", "Runs", "Terminal proposals", "Analyzed horizon", "Tokens/proposal"],
    ]
    for row in SUMMARY["summary"]:
        rows.append([
            row["interface"].replace("_", " ") + " / " + row["comparison"].replace("_", " "),
            str(int(row["runs"])),
            comma(row["terminal_proposals_mean"], 1),
            comma(row["horizon_mean"], 0),
            compact_tokens(row["tokens_per_proposal_mean"]),
        ])
    story.append(P("<b>Table A1:</b> Interface-level summary used by the main text.", CAPTION))
    story.append(data_table(rows, [2.25 * inch, 0.45 * inch, 1.0 * inch, 0.85 * inch, 0.95 * inch], font_size=6.5))


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story: list[Flowable] = []
    page1(story)
    page2(story)
    page3(story)
    page4(story)
    page5(story)
    page6(story)
    page7(story)
    page8(story)
    references(story)
    appendix(story)
    doc = PaperDoc(str(OUTPUT))
    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
