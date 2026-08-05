"""Dependency-free SVG figures for the trajectory analysis report."""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Mapping, Sequence

from .annotations import ResolvedAnnotation
from .schemas import TrajectoryEvent
from .storage import write_text


COLORS = {
    "openevolve": "#2563eb",
    "autoresearch": "#dc2626",
    "ttt_discover": "#059669",
    "ontology_preserving": "#64748b",
    "ontology_changing": "#7c3aed",
    "mixed": "#d97706",
    "unclassified": "#cbd5e1",
}


def _svg(body: str, *, width: int = 900, height: int = 520) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<style>text{font-family:ui-sans-serif,system-ui;font-size:13px;fill:#172033}'
        '.title{font-size:20px;font-weight:700}.axis{stroke:#94a3b8;stroke-width:1}'
        '.grid{stroke:#e2e8f0;stroke-width:1}.legend{font-size:12px}</style>'
        f'<rect width="100%" height="100%" fill="white"/>{body}</svg>\n'
    )


def frontier_progression(
    run_summaries: Sequence[Mapping[str, object]], path: str | Path
) -> None:
    points = [
        (summary, point)
        for summary in run_summaries
        for point in summary.get("frontier_progression", [])
    ]
    body = '<text x="50" y="36" class="title">Qualifying parameter frontier</text>'
    body += '<line x1="70" y1="450" x2="860" y2="450" class="axis"/>'
    body += '<line x1="70" y1="70" x2="70" y2="450" class="axis"/>'
    if not points:
        body += '<text x="300" y="250">No ≥threshold candidate with parameter metadata</text>'
        write_text(path, _svg(body))
        return
    max_x = max(int(point["sequence_index"]) for _, point in points) or 1
    values = [int(point["parameter_count"]) for _, point in points]
    minimum, maximum = min(values), max(values)
    span = max(maximum - minimum, 1)
    for summary in run_summaries:
        progression = summary.get("frontier_progression", [])
        coords = [
            (
                70 + 790 * int(point["sequence_index"]) / max_x,
                430 - 340 * (int(point["parameter_count"]) - minimum) / span,
            )
            for point in progression
        ]
        if coords:
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            color = COLORS.get(str(summary["paradigm"]), "#334155")
            body += f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3"/>'
            for x, y in coords:
                body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
    body += f'<text x="75" y="88">{maximum:,} params</text>'
    body += f'<text x="75" y="438">{minimum:,} params</text>'
    body += '<text x="720" y="485">trajectory sequence →</text>'
    write_text(path, _svg(body))


def boundary_mix(
    run_summaries: Sequence[Mapping[str, object]], path: str | Path
) -> None:
    categories = [
        "ontology_preserving", "ontology_changing", "mixed", "unclassified"
    ]
    body = '<text x="50" y="36" class="title">Architecture-boundary edit mix</text>'
    body += '<line x1="70" y1="450" x2="860" y2="450" class="axis"/>'
    bar_width = 660 / max(len(run_summaries), 1)
    for index, summary in enumerate(run_summaries):
        counts = Counter(summary.get("boundary_class_counts", {}))
        total = sum(counts.values())
        x = 90 + index * bar_width
        y = 450.0
        for category in categories:
            height = 330 * counts[category] / total if total else 0
            y -= height
            body += (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * .65:.1f}" '
                f'height="{height:.1f}" fill="{COLORS[category]}"/>'
            )
        label = escape(str(summary["run_id"]))
        body += f'<text x="{x:.1f}" y="475" class="legend">{label}</text>'
    for index, category in enumerate(categories):
        y = 80 + index * 24
        body += f'<rect x="680" y="{y - 12}" width="14" height="14" fill="{COLORS[category]}"/>'
        body += f'<text x="702" y="{y}">{escape(category)}</text>'
    write_text(path, _svg(body))


def rolling_diversity(
    events: Sequence[TrajectoryEvent],
    annotations: Mapping[str, ResolvedAnnotation],
    path: str | Path,
    *,
    window: int = 5,
) -> None:
    body = '<text x="50" y="36" class="title">Rolling architecture-edit diversity</text>'
    body += '<line x1="70" y1="450" x2="860" y2="450" class="axis"/>'
    by_run: dict[str, list[TrajectoryEvent]] = {}
    for event in events:
        if event.event_id in annotations:
            by_run.setdefault(event.run_id, []).append(event)
    max_length = max((len(items) for items in by_run.values()), default=1)
    for run_id, items in sorted(by_run.items()):
        items.sort(key=lambda item: item.sequence_index)
        values: list[float] = []
        for index in range(len(items)):
            window_events = items[max(0, index - window + 1) : index + 1]
            labels = {
                annotations[event.event_id].edit_family.value for event in window_events
            }
            values.append(len(labels) / min(window, index + 1))
        coords = [
            (70 + 790 * index / max(max_length - 1, 1), 430 - value * 330)
            for index, value in enumerate(values)
        ]
        if coords:
            paradigm = items[0].paradigm.value
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            body += f'<polyline points="{points}" fill="none" stroke="{COLORS[paradigm]}" stroke-width="3"/>'
            body += f'<text x="710" y="{65 + 22 * list(sorted(by_run)).index(run_id)}">{escape(run_id)}</text>'
    body += '<text x="75" y="100">1.0</text><text x="75" y="438">0.0</text>'
    write_text(path, _svg(body))
