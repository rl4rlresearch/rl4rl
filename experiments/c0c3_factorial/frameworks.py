"""Codex CLI proposal adapters for controlled Autoresearch and OpenEvolve."""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .codex_cli import CodexCli, CodexResult
from .prompts import RenderedPrompt
from .spec import FrameworkKind, FrameworkSpec, ModelSpec, TaskSpec

_HYPOTHESIS = re.compile(r"^HYPOTHESIS:\s*(.+)$", re.MULTILINE)
_INTENDED_EDIT = re.compile(r"^INTENDED_EDIT:\s*(.+)$", re.MULTILINE)
_FILE_OPEN = re.compile(r"^===== FILE: (?P<path>[^\n]+) =====$")
_FILE_CLOSE = "===== END FILE ====="


@dataclass(frozen=True)
class ProposalExecution:
    codex: CodexResult
    hypothesis: str
    intended_edit: str
    adapter_error: str | None = None


def parse_metadata(message: str) -> tuple[str, str]:
    hypothesis = _HYPOTHESIS.search(message)
    intended_edit = _INTENDED_EDIT.search(message)
    return (
        hypothesis.group(1).strip() if hypothesis else "[missing hypothesis]",
        intended_edit.group(1).strip() if intended_edit else "[missing intended edit]",
    )


def bundle_workspace(workspace: Path, editable_paths: tuple[str, ...]) -> str:
    sections: list[str] = []
    for relative in sorted(editable_paths):
        text = (workspace / relative).read_text(encoding="utf-8")
        if _FILE_CLOSE in text or _FILE_OPEN.search(text):
            raise ValueError(
                f"editable file contains a reserved bundle marker: {relative}"
            )
        sections.append(f"===== FILE: {relative} =====\n{text.rstrip()}\n{_FILE_CLOSE}")
    return "\n\n".join(sections) + "\n"


def unbundle_workspace(
    bundle: str, workspace: Path, editable_paths: tuple[str, ...]
) -> None:
    expected = set(editable_paths)
    files: dict[str, str] = {}
    lines = bundle.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index]:
            index += 1
            continue
        match = _FILE_OPEN.fullmatch(lines[index])
        if match is None:
            raise ValueError(f"invalid bundle line: {lines[index][:100]}")
        relative = match.group("path")
        if relative in files:
            raise ValueError(f"duplicate bundled file: {relative}")
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index] != _FILE_CLOSE:
            content.append(lines[index])
            index += 1
        if index >= len(lines):
            raise ValueError(f"unterminated bundled file: {relative}")
        files[relative] = "\n".join(content) + "\n"
        index += 1
    if set(files) != expected:
        raise ValueError(
            f"bundle changed editable file set; expected={sorted(expected)}, "
            f"actual={sorted(files)}"
        )
    for relative, content in files.items():
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


class AutoresearchAdapter:
    """Karpathy-style direct edit of the selected candidate."""

    def __init__(self, codex: CodexCli) -> None:
        self.codex = codex

    def propose(
        self,
        *,
        rendered: RenderedPrompt,
        workspace: Path,
        model: ModelSpec,
        log_root: Path,
        call_id: str,
        timeout_seconds: int,
        **_unused: object,
    ) -> ProposalExecution:
        result = self.codex.run(
            prompt=rendered.text,
            workspace=workspace,
            model=model,
            log_root=log_root,
            call_id=call_id,
            sandbox="workspace-write",
            timeout_seconds=timeout_seconds,
        )
        hypothesis, intended_edit = parse_metadata(result.last_message)
        return ProposalExecution(result, hypothesis, intended_edit)


class OpenEvolveAdapter:
    """Controlled port of OpenEvolve prompt sampling and SEARCH/REPLACE mutation.

    The shared factorial controller intentionally replaces OpenEvolve's native
    database sampling and retention: those are the randomized factors here.
    OpenEvolve still owns the evolutionary prompt/history representation and
    its mutation parser. This follows FML-bench's strategy/infrastructure split.
    """

    def __init__(self, codex: CodexCli, *, vendor_root: Path) -> None:
        self.codex = codex
        self.vendor_root = vendor_root

    def _imports(self):
        value = str(self.vendor_root)
        if value not in sys.path:
            sys.path.insert(0, value)
        from openevolve.config import PromptConfig
        from openevolve.prompt.sampler import PromptSampler
        from openevolve.utils.code_utils import apply_diff, extract_diffs

        return PromptConfig, PromptSampler, apply_diff, extract_diffs

    def propose(
        self,
        *,
        rendered: RenderedPrompt,
        workspace: Path,
        model: ModelSpec,
        log_root: Path,
        call_id: str,
        timeout_seconds: int,
        task: TaskSpec,
        visible_workspaces: tuple[Path, ...],
        selected_parent_id: str,
        visible_records: tuple[dict[str, object], ...],
        **_unused: object,
    ) -> ProposalExecution:
        PromptConfig, PromptSampler, apply_diff, extract_diffs = self._imports()
        parent_bundle = bundle_workspace(workspace, task.editable_paths)
        programs = []
        for candidate_workspace, record in zip(
            visible_workspaces, visible_records, strict=True
        ):
            programs.append(
                {
                    "id": record["candidate_id"],
                    "code": bundle_workspace(candidate_workspace, task.editable_paths),
                    "metrics": record["metrics"],
                    "changes_description": record.get("hypothesis", ""),
                    "metadata": {},
                }
            )
        prompt_config = PromptConfig(
            system_message=rendered.text,
            num_top_programs=len(programs),
            num_diverse_programs=max(0, len(programs) - 1),
            use_template_stochasticity=False,
            include_artifacts=False,
        )
        sampler = PromptSampler(prompt_config)
        parent_record = next(
            record
            for record in visible_records
            if record["candidate_id"] == selected_parent_id
        )
        sampled = sampler.build_prompt(
            current_program=parent_bundle,
            parent_program=parent_bundle,
            program_metrics=parent_record["metrics"],
            previous_programs=programs,
            top_programs=programs,
            inspirations=[
                program for program in programs if program["id"] != selected_parent_id
            ],
            language="text",
            evolution_round=int(call_id.rsplit("-", 1)[-1]),
            diff_based_evolution=True,
            feature_dimensions=[],
        )
        combined_prompt = (
            f"{sampled['system']}\n\n{sampled['user']}\n\n"
            "Return HYPOTHESIS and INTENDED_EDIT metadata lines, followed by one "
            "or more exact OpenEvolve SEARCH/REPLACE blocks."
        )
        prompt_workspace = log_root / f"{call_id}-prompt-workspace"
        prompt_workspace.mkdir(parents=True, exist_ok=False)
        result = self.codex.run(
            prompt=combined_prompt,
            workspace=prompt_workspace,
            model=model,
            log_root=log_root,
            call_id=call_id,
            sandbox="read-only",
            timeout_seconds=timeout_seconds,
        )
        hypothesis, intended_edit = parse_metadata(result.last_message)
        error: str | None = None
        if result.returncode == 0:
            try:
                diffs = extract_diffs(result.last_message)
                if not diffs:
                    raise ValueError("OpenEvolve response contained no valid diff")
                child_bundle = apply_diff(parent_bundle, result.last_message)
                if child_bundle == parent_bundle:
                    raise ValueError("OpenEvolve diff did not match selected parent")
                unbundle_workspace(child_bundle, workspace, task.editable_paths)
            except (OSError, ValueError) as exception:
                error = str(exception)
        shutil.rmtree(prompt_workspace, ignore_errors=True)
        return ProposalExecution(result, hypothesis, intended_edit, error)


def make_framework_adapter(
    framework: FrameworkSpec, codex: CodexCli, *, repo_root: Path
) -> AutoresearchAdapter | OpenEvolveAdapter:
    if framework.framework_id is FrameworkKind.AUTORESEARCH:
        return AutoresearchAdapter(codex)
    if framework.framework_id is FrameworkKind.OPENEVOLVE:
        return OpenEvolveAdapter(
            codex,
            vendor_root=repo_root / "architecture_discovery/vendor/openevolve",
        )
    raise ValueError(f"unsupported framework {framework.framework_id}")
