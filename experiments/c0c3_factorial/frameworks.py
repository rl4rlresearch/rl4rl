"""Codex CLI proposal adapters for controlled Autoresearch and OpenEvolve."""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .codex_cli import CodexCli, CodexResult
from .prompts import RenderedPrompt
from .spec import FrameworkKind, FrameworkSpec, ModelSpec, TaskSpec

_HYPOTHESIS = re.compile(r"^HYPOTHESIS:\s*(.+)$", re.MULTILINE)
_INTENDED_EDIT = re.compile(r"^INTENDED_EDIT:\s*(.+)$", re.MULTILINE)
_MECHANISM = re.compile(r"^MECHANISM:\s*(.+)$", re.MULTILINE)
_EVIDENCE = re.compile(r"^EVIDENCE:\s*(.+)$", re.MULTILINE)
_FILE_OPEN = re.compile(r"^===== FILE: (?P<path>[^\n]+) =====$")
_FILE_CLOSE = "===== END FILE ====="
_MARKDOWN_FIELD = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*)?(?P<label>[A-Za-z _-]+?)(?:\*\*)?\s*:\s*(?P<value>.+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ProposalExecution:
    codex: CodexResult
    hypothesis: str
    intended_edit: str
    adapter_error: str | None = None
    adapter_failure_kind: str | None = None
    mechanism: str = "[not recorded]"
    evidence: str = "[not recorded]"


def parse_metadata(message: str) -> tuple[str, str]:
    hypothesis = _HYPOTHESIS.search(message)
    intended_edit = _INTENDED_EDIT.search(message)
    return (
        hypothesis.group(1).strip() if hypothesis else "[missing hypothesis]",
        intended_edit.group(1).strip() if intended_edit else "[missing intended edit]",
    )


def parse_flexible_metadata(message: str) -> tuple[str, str]:
    """Recover useful Autoresearch provenance without imposing a wire format."""

    fields: dict[str, str] = {}
    for match in _MARKDOWN_FIELD.finditer(message):
        label = " ".join(match.group("label").casefold().split())
        fields.setdefault(label, match.group("value").strip())

    hypothesis = fields.get("hypothesis")
    intended_edit = next(
        (
            fields[label]
            for label in ("intended edit", "what changed", "change", "changed")
            if label in fields
        ),
        None,
    )
    useful_lines = [
        line.strip(" -*#\t")
        for line in message.splitlines()
        if line.strip(" -*#\t")
    ]
    if hypothesis is None:
        hypothesis = next(
            (
                line
                for line in useful_lines
                if "hypothes" in line.casefold()
            ),
            useful_lines[0] if useful_lines else "No summary was returned.",
        )
    if intended_edit is None:
        intended_edit = next(
            (
                line
                for line in useful_lines
                if any(
                    word in line.casefold()
                    for word in ("changed", "implemented", "modified", "edited")
                )
            ),
            " ".join(useful_lines[:3]) or "No edit summary was returned.",
        )
    return hypothesis[:1000], intended_edit[:1000]


def parse_v2_metadata(message: str) -> tuple[str, str, str, str]:
    hypothesis, intended_edit = parse_metadata(message)
    mechanism = _MECHANISM.search(message)
    evidence = _EVIDENCE.search(message)
    return (
        hypothesis,
        intended_edit,
        mechanism.group(1).strip() if mechanism else "[missing mechanism]",
        evidence.group(1).strip() if evidence else "[missing evidence]",
    )


def _strict_apply_diff(
    original: str,
    response: str,
    extract_diffs,
) -> str:
    """Apply every OpenEvolve block exactly once or fail before evaluation."""

    blocks = extract_diffs(response)
    marker_counts = (
        response.count("<<<<<<< SEARCH"),
        response.count("=======\n"),
        response.count(">>>>>>> REPLACE"),
    )
    if not blocks or marker_counts != (len(blocks), len(blocks), len(blocks)):
        raise ValueError("malformed SEARCH/REPLACE block structure")
    result = original
    for index, (search, replacement) in enumerate(blocks, start=1):
        if not search.strip():
            raise ValueError(f"diff block {index} has an empty SEARCH section")
        occurrences = result.count(search)
        if occurrences == 0:
            raise ValueError(f"diff block {index} SEARCH did not match")
        if occurrences > 1:
            raise ValueError(f"diff block {index} SEARCH matched ambiguously")
        result = result.replace(search, replacement, 1)
    return result


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

    def __init__(self, codex: CodexCli, *, flexible_metadata: bool = False) -> None:
        self.codex = codex
        self.flexible_metadata = flexible_metadata

    def propose(
        self,
        *,
        rendered: RenderedPrompt,
        workspace: Path,
        model: ModelSpec,
        log_root: Path,
        call_id: str,
        timeout_seconds: int,
        run_seed: int,
        resume_session_id: str | None = None,
        persist_session: bool = False,
        neutral_subject: bool = False,
        artifact_clean_subject: bool = False,
        **_unused: object,
    ) -> ProposalExecution:
        result = self.codex.run(
            prompt=rendered.text,
            workspace=workspace,
            model=model,
            log_root=log_root,
            call_id=call_id,
            sandbox="workspace-write",
            run_seed=run_seed,
            timeout_seconds=timeout_seconds,
            resume_session_id=resume_session_id,
            persist_session=persist_session,
            neutral_subject=neutral_subject,
            artifact_clean_subject=artifact_clean_subject,
        )
        metadata_parser = (
            parse_flexible_metadata if self.flexible_metadata else parse_metadata
        )
        hypothesis, intended_edit = metadata_parser(result.last_message)
        return ProposalExecution(result, hypothesis, intended_edit)


class OpenEvolveAdapter:
    """Controlled port of OpenEvolve prompt sampling and SEARCH/REPLACE mutation.

    The shared factorial controller intentionally replaces OpenEvolve's native
    database sampling and retention: those are the randomized factors here.
    OpenEvolve still owns the evolutionary prompt/history representation and
    its mutation parser. This follows FML-bench's strategy/infrastructure split.
    """

    def __init__(
        self,
        codex: CodexCli,
        *,
        vendor_root: Path,
        v2: bool = False,
        v21: bool = False,
        template_root: Path | None = None,
    ) -> None:
        self.codex = codex
        self.vendor_root = vendor_root
        self.v2 = v2
        self.v21 = v21
        self.template_root = template_root

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
        run_seed: int,
        resume_session_id: str | None = None,
        persist_session: bool = False,
        task: TaskSpec,
        visible_workspaces: tuple[Path, ...],
        selected_parent_id: str,
        visible_records: tuple[dict[str, object], ...],
        neutral_subject: bool = False,
        artifact_clean_subject: bool = False,
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
            template_dir=(
                str(self.template_root)
                if self.v2 and self.template_root is not None
                else None
            ),
            system_message=rendered.text,
            num_top_programs=0 if self.v2 else len(programs),
            num_diverse_programs=0 if self.v2 else max(0, len(programs) - 1),
            use_template_stochasticity=False,
            include_artifacts=False,
        )
        sampler = PromptSampler(prompt_config)
        parent_record = next(
            record
            for record in visible_records
            if record["candidate_id"] == selected_parent_id
        )
        design_programs = (
            "" if self.v21 else "No alternative retained design is available."
        )
        if self.v21:
            alternatives = []
            reference_index = 0
            for program in programs:
                if program["id"] == selected_parent_id:
                    continue
                reference_index += 1
                alternatives.append(
                    f"REFERENCE DESIGN {reference_index}\n"
                    f"```text\n{program['code']}\n```"
                )
            if alternatives:
                design_programs = "# Reference source\n\n" + "\n\n".join(
                    alternatives
                )
        elif self.v2:
            alternatives = []
            for program in programs:
                if program["id"] == selected_parent_id:
                    continue
                alternatives.append(
                    "VERIFIED VALUES: "
                    f"{program['metrics']}\n```text\n{program['code']}\n```"
                )
            if alternatives:
                design_programs = "\n\n".join(alternatives)
        sampled = sampler.build_prompt(
            current_program=parent_bundle,
            parent_program=parent_bundle,
            program_metrics=parent_record["metrics"],
            previous_programs=[] if self.v2 else programs,
            top_programs=[] if self.v2 else programs,
            inspirations=(
                []
                if self.v2
                else [
                    program
                    for program in programs
                    if program["id"] != selected_parent_id
                ]
            ),
            language="text",
            evolution_round=int(call_id.rsplit("-", 1)[-1]),
            diff_based_evolution=True,
            feature_dimensions=[],
            template_key="diff_user" if self.v2 else None,
            design_programs=design_programs,
            search_marker="<<<<<<< SEARCH",
            divider_marker="=======",
            replace_marker=">>>>>>> REPLACE",
        )
        required_metadata = (
            "MECHANISM, HYPOTHESIS, INTENDED_EDIT, and EVIDENCE"
            if self.v2
            else "HYPOTHESIS and INTENDED_EDIT"
        )
        combined_prompt = f"{sampled['system']}\n\n{sampled['user']}"
        if not self.v21:
            combined_prompt += (
                "\n\n"
                f"Return {required_metadata} metadata lines, followed by one or "
                "more exact SEARCH/REPLACE blocks."
            )
        if self.v2:
            prompt_workspace = Path(
                tempfile.mkdtemp(prefix="transformer-design-cycle-")
            )
        else:
            prompt_workspace = log_root / f"{call_id}-prompt-workspace"
            prompt_workspace.mkdir(parents=True, exist_ok=False)
        result = self.codex.run(
            prompt=combined_prompt,
            workspace=prompt_workspace,
            model=model,
            log_root=log_root,
            call_id=call_id,
            sandbox="read-only",
            run_seed=run_seed,
            timeout_seconds=timeout_seconds,
            resume_session_id=resume_session_id,
            persist_session=persist_session,
            neutral_subject=neutral_subject,
            artifact_clean_subject=artifact_clean_subject,
        )
        if self.v2:
            hypothesis, intended_edit, mechanism, evidence = parse_v2_metadata(
                result.last_message
            )
        else:
            hypothesis, intended_edit = parse_metadata(result.last_message)
            mechanism, evidence = "[not recorded]", "[not recorded]"
        error: str | None = None
        failure_kind: str | None = None
        if self.v2 and result.returncode == 0 and any(
            value.startswith("[missing")
            for value in (hypothesis, intended_edit, mechanism, evidence)
        ):
            error = "OpenEvolve response omitted required proposal metadata"
            failure_kind = "missing_metadata"
        if result.returncode == 0 and error is None:
            try:
                diffs = extract_diffs(result.last_message)
                if not diffs:
                    raise ValueError("OpenEvolve response contained no valid diff")
                child_bundle = (
                    _strict_apply_diff(
                        parent_bundle,
                        result.last_message,
                        extract_diffs,
                    )
                    if self.v2
                    else apply_diff(parent_bundle, result.last_message)
                )
                if child_bundle == parent_bundle:
                    raise ValueError("OpenEvolve diff did not match selected parent")
                unbundle_workspace(child_bundle, workspace, task.editable_paths)
            except (OSError, ValueError) as exception:
                error = str(exception)
                lowered = error.lower()
                if "malformed" in lowered or "no valid diff" in lowered:
                    failure_kind = "malformed_diff"
                elif "ambiguously" in lowered:
                    failure_kind = "ambiguous_diff"
                elif "did not match" in lowered:
                    failure_kind = "unmatched_diff"
                else:
                    failure_kind = "invalid_diff"
        shutil.rmtree(prompt_workspace, ignore_errors=True)
        return ProposalExecution(
            result,
            hypothesis,
            intended_edit,
            error,
            failure_kind,
            mechanism,
            evidence,
        )


def make_framework_adapter(
    framework: FrameworkSpec, codex: CodexCli, *, repo_root: Path
) -> AutoresearchAdapter | OpenEvolveAdapter:
    if framework.framework_id is FrameworkKind.AUTORESEARCH:
        return AutoresearchAdapter(
            codex,
            flexible_metadata=(
                framework.adapter == "codex_direct_editor_confined_session_resume_v2"
            ),
        )
    if framework.framework_id is FrameworkKind.OPENEVOLVE:
        v21 = framework.adapter == "controlled_openevolve_prompt_diff_v2_1"
        v2 = framework.adapter in {
            "controlled_openevolve_prompt_diff_v2",
            "controlled_openevolve_prompt_diff_v2_1",
        }
        return OpenEvolveAdapter(
            codex,
            vendor_root=repo_root / "architecture_discovery/vendor/openevolve",
            v2=v2,
            v21=v21,
            template_root=(
                repo_root
                / "experiments/c0c3_factorial/templates"
                / ("openevolve_v2_1" if v21 else "openevolve_v2")
                if v2
                else None
            ),
        )
    raise ValueError(f"unsupported framework {framework.framework_id}")
