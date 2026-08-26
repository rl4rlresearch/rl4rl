"""Thin, checkpointed bridge to the vendored OpenEvolve search engine.

The repository runner remains responsible for Codex transport, task evaluation,
exact accounting, and pause/recovery.  Parent selection, inspirations,
MAP-Elites cells, islands, archive management, migration, and population
retention are delegated to OpenEvolve's unmodified ``ProgramDatabase``.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import random
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .frameworks import bundle_workspace, unbundle_workspace
from .spec import FrameworkSpec, TaskSpec
from .state import Candidate, Evaluation, SearchController, atomic_json, utc_now

NATIVE_ROOT = Path("native-openevolve")
CURRENT_STATE = NATIVE_ROOT / "current.json"
CHECKPOINTS = NATIVE_ROOT / "checkpoints"
NATIVE_EVENTS = NATIVE_ROOT / "events.jsonl"
_RANDOM_LOCK = threading.Lock()


@dataclass(frozen=True)
class NativeSelection:
    opportunity: int
    target_island: int
    parent_id: str
    visible_ids: tuple[str, ...]
    previous_ids: tuple[str, ...]
    top_ids: tuple[str, ...]
    inspiration_ids: tuple[str, ...]
    feature_dimensions: tuple[str, ...]

    def prompt_context(self) -> dict[str, object]:
        return asdict(self)


def is_native_openevolve(framework: FrameworkSpec) -> bool:
    return framework.framework_key == "native_openevolve"


def _imports(vendor_root: Path):
    value = str(vendor_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    from openevolve.config import DatabaseConfig
    from openevolve.database import Program, ProgramDatabase

    return DatabaseConfig, Program, ProgramDatabase


def _option(framework: FrameworkSpec, name: str, default: Any) -> Any:
    return framework.adapter_options.get(name, default)


def _database_config(framework: FrameworkSpec, *, run_seed: int, vendor_root: Path):
    DatabaseConfig, _Program, _ProgramDatabase = _imports(vendor_root)
    return DatabaseConfig(
        in_memory=True,
        db_path=None,
        population_size=int(_option(framework, "population_size", 1000)),
        archive_size=int(_option(framework, "archive_size", 100)),
        num_islands=int(_option(framework, "num_islands", 5)),
        elite_selection_ratio=float(_option(framework, "elite_selection_ratio", 0.1)),
        exploration_ratio=float(_option(framework, "exploration_ratio", 0.2)),
        exploitation_ratio=float(_option(framework, "exploitation_ratio", 0.7)),
        feature_dimensions=list(
            _option(framework, "feature_dimensions", ["complexity", "diversity"])
        ),
        feature_bins=_option(framework, "feature_bins", 10),
        diversity_reference_size=int(
            _option(framework, "diversity_reference_size", 20)
        ),
        migration_interval=int(_option(framework, "migration_interval", 50)),
        migration_rate=float(_option(framework, "migration_rate", 0.1)),
        random_seed=run_seed,
        embedding_model=None,
        similarity_threshold=float(_option(framework, "similarity_threshold", 0.99)),
    )


def _encode_random_state(value: object) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=5)).decode("ascii")


def _decode_random_state(value: str) -> object:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _append_event(run_dir: Path, value: dict[str, Any]) -> None:
    from .state import append_jsonl

    append_jsonl(run_dir / NATIVE_EVENTS, value)


def _checkpoint_name(opportunity: int, stage: str) -> str:
    return f"{opportunity:04d}-{stage}"


def _save_database(database, destination: Path, *, iteration: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        database.save(str(temporary), iteration=iteration)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _load_database(
    run_dir: Path,
    *,
    framework: FrameworkSpec,
    vendor_root: Path,
    run_seed: int,
):
    config = _database_config(framework, run_seed=run_seed, vendor_root=vendor_root)
    _DatabaseConfig, _Program, ProgramDatabase = _imports(vendor_root)
    database = ProgramDatabase(config)
    pointer_path = run_dir / CURRENT_STATE
    if pointer_path.is_file():
        pointer = _read_json(pointer_path)
        checkpoint = run_dir / NATIVE_ROOT / str(pointer["checkpoint"])
        database.load(str(checkpoint))
        rng_state = _decode_random_state(str(pointer["random_state"]))
    else:
        pointer = {}
        rng_state = random.Random(run_seed).getstate()
    return database, pointer, rng_state


def _fitness_metrics(candidate: Candidate) -> dict[str, Any]:
    return {**candidate.metrics, "combined_score": candidate.fitness}


def _initialize_database(
    run_dir: Path,
    *,
    controller: SearchController,
    task: TaskSpec,
    framework: FrameworkSpec,
    vendor_root: Path,
    run_seed: int,
):
    database, pointer, rng_state = _load_database(
        run_dir,
        framework=framework,
        vendor_root=vendor_root,
        run_seed=run_seed,
    )
    if pointer:
        return database, pointer, rng_state
    _DatabaseConfig, Program, _ProgramDatabase = _imports(vendor_root)
    seed = controller.state.candidates[controller.state.incumbent_id]
    seed_snapshot = run_dir / seed.artifact_path
    program = Program(
        id=seed.candidate_id,
        code=bundle_workspace(seed_snapshot, task.editable_paths),
        changes_description="starting design",
        language="python",
        metrics=_fitness_metrics(seed),
        iteration_found=0,
    )
    with _RANDOM_LOCK:
        prior = random.getstate()
        random.setstate(rng_state)
        try:
            database.add(program, iteration=0, target_island=0)
            rng_state = random.getstate()
        finally:
            random.setstate(prior)
    checkpoint = run_dir / CHECKPOINTS / _checkpoint_name(0, "initial")
    _save_database(database, checkpoint, iteration=0)
    pointer = {
        "schema_version": "native-openevolve-v1",
        "stage": "initial",
        "opportunity": 0,
        "checkpoint": str(checkpoint.relative_to(run_dir / NATIVE_ROOT)),
        "random_state": _encode_random_state(rng_state),
        "updated_at": utc_now(),
    }
    atomic_json(run_dir / CURRENT_STATE, pointer)
    atomic_json(
        run_dir / NATIVE_ROOT / "config.json",
        {
            "schema_version": "native-openevolve-v1",
            "controller": "vendored_openevolve_program_database",
            "database": asdict(database.config) | {"novelty_llm": None},
            "prompt": {
                "num_top_programs": int(_option(framework, "num_top_programs", 3)),
                "num_diverse_programs": int(
                    _option(framework, "num_diverse_programs", 2)
                ),
                "use_template_stochasticity": bool(
                    _option(framework, "use_template_stochasticity", True)
                ),
            },
        },
    )
    _append_event(
        run_dir,
        {
            "schema_version": "native-openevolve-v1",
            "event": "native_database_initialized",
            "timestamp": utc_now(),
            "seed_program_id": seed.candidate_id,
        },
    )
    return database, pointer, rng_state


def _materialize_program_snapshot(run_dir: Path, *, task: TaskSpec, program) -> Path:
    destination = run_dir / "candidates" / program.id
    if destination.is_dir():
        return destination
    temporary = destination.with_name(f".{destination.name}.native-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        unbundle_workspace(program.code, temporary, task.editable_paths)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def _candidate_from_program(program) -> Candidate:
    metrics = dict(program.metrics)
    fitness = float(metrics.pop("combined_score"))
    return Candidate(
        candidate_id=program.id,
        parent_ids=([program.parent_id] if program.parent_id else []),
        fitness=fitness,
        metrics=metrics,
        artifact_path=f"candidates/{program.id}",
        hypothesis=program.changes_description or "OpenEvolve population member",
        intended_edit=str(program.metadata.get("changes", "population member")),
        created_opportunity=int(program.iteration_found),
        retained_order=int(program.iteration_found),
    )


def _selection_from_pointer(pointer: dict[str, Any]) -> NativeSelection:
    selected = dict(pointer["selection"])
    return NativeSelection(
        opportunity=int(pointer["opportunity"]),
        target_island=int(selected["target_island"]),
        parent_id=str(selected["parent_id"]),
        visible_ids=tuple(str(value) for value in selected["visible_ids"]),
        previous_ids=tuple(str(value) for value in selected["previous_ids"]),
        top_ids=tuple(str(value) for value in selected["top_ids"]),
        inspiration_ids=tuple(str(value) for value in selected["inspiration_ids"]),
        feature_dimensions=tuple(
            str(value) for value in selected["feature_dimensions"]
        ),
    )


def _reconcile_completed_accounting(
    run_dir: Path, *, controller: SearchController
) -> None:
    """Resolve the narrow crash window between the two durable ledgers."""

    pointer_path = run_dir / CURRENT_STATE
    if not pointer_path.is_file():
        return
    pointer = _read_json(pointer_path)
    if pointer.get("stage") != "selection":
        return
    opportunity = int(pointer.get("opportunity", -1))
    if opportunity < 1 or controller.state.proposals_used < opportunity:
        return
    pending_path = run_dir / NATIVE_ROOT / f"pending-{opportunity:04d}.json"
    completed = None
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if (
            event.get("event") == "proposal_completed"
            and int(event.get("opportunity", -1)) == opportunity
        ):
            completed = event
    if completed is None:
        raise RuntimeError("generic opportunity advanced without a completion event")
    if pending_path.is_file():
        pending = _read_json(pending_path)
        evaluation = completed.get("evaluation")
        should_promote = (
            isinstance(evaluation, dict)
            and bool(evaluation.get("valid"))
            and completed.get("candidate_id") == pending.get("candidate_id")
            and bool(pending.get("evaluation_valid"))
        )
        if should_promote:
            finalize_native_outcome(
                run_dir,
                opportunity=opportunity,
                candidate_id=str(pending["candidate_id"]),
            )
            return
    # Interrupted/invalid results never enter OpenEvolve's database. Preserve
    # the already-sampled RNG state and advance only the wrapper pointer.
    atomic_json(
        pointer_path,
        {
            "schema_version": "native-openevolve-v1",
            "stage": "completed_without_database_addition",
            "opportunity": opportunity,
            "checkpoint": pointer["checkpoint"],
            "random_state": pointer["random_state"],
            "updated_at": utc_now(),
        },
    )
    _append_event(
        run_dir,
        {
            "schema_version": "native-openevolve-v1",
            "event": "native_outcome_reconciled_without_addition",
            "timestamp": utc_now(),
            "opportunity": opportunity,
            "generic_failure_kind": (
                dict(completed.get("evaluation", {})).get("failure_kind")
            ),
        },
    )


def prepare_native_selection(
    run_dir: Path,
    *,
    controller: SearchController,
    task: TaskSpec,
    framework: FrameworkSpec,
    vendor_root: Path,
    run_seed: int,
    opportunity: int,
) -> NativeSelection:
    """Ask the official database for one parent and its inspirations."""

    _reconcile_completed_accounting(run_dir, controller=controller)
    database, pointer, rng_state = _initialize_database(
        run_dir,
        controller=controller,
        task=task,
        framework=framework,
        vendor_root=vendor_root,
        run_seed=run_seed,
    )
    if (
        pointer.get("stage") == "selection"
        and int(pointer.get("opportunity", -1)) == opportunity
    ):
        selection = _selection_from_pointer(pointer)
    else:
        target_island = (opportunity - 1) % len(database.islands)
        with _RANDOM_LOCK:
            prior = random.getstate()
            random.setstate(rng_state)
            try:
                parent, inspirations = database.sample_from_island(
                    target_island,
                    num_inspirations=int(_option(framework, "num_diverse_programs", 2)),
                )
                rng_state = random.getstate()
            finally:
                random.setstate(prior)
        island_programs = [
            database.programs[identifier]
            for identifier in database.islands[target_island]
            if identifier in database.programs
        ]
        island_programs.sort(
            key=lambda program: float(program.metrics.get("combined_score", 0.0)),
            reverse=True,
        )
        top_count = int(_option(framework, "num_top_programs", 3))
        diverse_count = int(_option(framework, "num_diverse_programs", 2))
        previous = island_programs[:top_count]
        top = island_programs[: top_count + diverse_count]
        visible_ids = tuple(
            dict.fromkeys(
                [
                    parent.id,
                    *(program.id for program in top),
                    *(program.id for program in inspirations),
                ]
            )
        )
        selection = NativeSelection(
            opportunity=opportunity,
            target_island=target_island,
            parent_id=parent.id,
            visible_ids=visible_ids,
            previous_ids=tuple(program.id for program in previous),
            top_ids=tuple(program.id for program in top),
            inspiration_ids=tuple(program.id for program in inspirations),
            feature_dimensions=tuple(database.config.feature_dimensions),
        )
        checkpoint = run_dir / CHECKPOINTS / _checkpoint_name(opportunity, "selection")
        _save_database(database, checkpoint, iteration=database.last_iteration)
        pointer = {
            "schema_version": "native-openevolve-v1",
            "stage": "selection",
            "opportunity": opportunity,
            "checkpoint": str(checkpoint.relative_to(run_dir / NATIVE_ROOT)),
            "random_state": _encode_random_state(rng_state),
            "selection": selection.prompt_context(),
            "updated_at": utc_now(),
        }
        atomic_json(run_dir / CURRENT_STATE, pointer)
        _append_event(
            run_dir,
            {
                "schema_version": "native-openevolve-v1",
                "event": "native_parent_sampled",
                "timestamp": utc_now(),
                **selection.prompt_context(),
            },
        )

    programs = [database.programs[identifier] for identifier in selection.visible_ids]
    for program in programs:
        _materialize_program_snapshot(run_dir, task=task, program=program)
    controller.register_external_candidates(
        [_candidate_from_program(program) for program in programs]
    )
    return selection


def stage_native_outcome(
    run_dir: Path,
    *,
    controller: SearchController,
    task: TaskSpec,
    framework: FrameworkSpec,
    vendor_root: Path,
    run_seed: int,
    selection: NativeSelection,
    candidate_id: str,
    candidate_snapshot: Path,
    hypothesis: str,
    intended_edit: str,
    changes_summary: str,
    evaluation: Evaluation,
    prompt: dict[str, str] | None,
    response: str | None,
) -> dict[str, Any]:
    """Stage the official database result before generic accounting commits."""

    database, pointer, rng_state = _load_database(
        run_dir,
        framework=framework,
        vendor_root=vendor_root,
        run_seed=run_seed,
    )
    if int(pointer.get("opportunity", -1)) != selection.opportunity:
        raise RuntimeError("native OpenEvolve checkpoint is out of step")
    _DatabaseConfig, Program, _ProgramDatabase = _imports(vendor_root)
    candidate_in_population = False
    checkpoint_relative = str(pointer["checkpoint"])
    if evaluation.valid:
        parent = database.get(selection.parent_id)
        if parent is None:
            raise RuntimeError("native OpenEvolve parent disappeared")
        program = Program(
            id=candidate_id,
            code=bundle_workspace(candidate_snapshot, task.editable_paths),
            changes_description="",
            language="python",
            parent_id=selection.parent_id,
            generation=parent.generation + 1,
            metrics={**evaluation.metrics, "combined_score": evaluation.fitness},
            iteration_found=selection.opportunity,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": selection.target_island,
            },
        )
        with _RANDOM_LOCK:
            prior = random.getstate()
            random.setstate(rng_state)
            try:
                database.add(
                    program,
                    iteration=selection.opportunity,
                    target_island=selection.target_island,
                )
                if prompt is not None:
                    database.log_prompt(
                        template_key="diff_user",
                        program_id=candidate_id,
                        prompt=prompt,
                        responses=[response] if response else [],
                    )
                database.increment_island_generation(selection.target_island)
                if database.should_migrate():
                    database.migrate_programs()
                rng_state = random.getstate()
            finally:
                random.setstate(prior)
        candidate_in_population = (
            any(candidate_id in island for island in database.islands)
            or candidate_id in database.archive
        )
        checkpoint = (
            run_dir / CHECKPOINTS / _checkpoint_name(selection.opportunity, "complete")
        )
        _save_database(database, checkpoint, iteration=selection.opportunity)
        checkpoint_relative = str(checkpoint.relative_to(run_dir / NATIVE_ROOT))

    best = database.get_best_program()
    if best is None:
        raise RuntimeError("native OpenEvolve database has no best program")
    _materialize_program_snapshot(run_dir, task=task, program=best)
    if best.id != candidate_id:
        controller.register_external_candidates([_candidate_from_program(best)])
    search = {
        "engine": "vendored_openevolve_program_database",
        "candidate_in_population": candidate_in_population,
        "retention_decision": (
            "native_population_admission"
            if candidate_in_population
            else (
                "native_population_rejection"
                if evaluation.valid
                else "invalid_candidate_not_submitted_to_native_database"
            )
        ),
        "best_program_id": best.id,
        "population_size": len(database.programs),
        "archive_size": len(database.archive),
        "island_sizes": [len(island) for island in database.islands],
        "target_island": selection.target_island,
        "checkpoint": checkpoint_relative,
    }
    atomic_json(
        run_dir / NATIVE_ROOT / f"pending-{selection.opportunity:04d}.json",
        {
            "schema_version": "native-openevolve-v1",
            "opportunity": selection.opportunity,
            "candidate_id": candidate_id,
            "evaluation_valid": evaluation.valid,
            "random_state": _encode_random_state(rng_state),
            "search": search,
            "created_at": utc_now(),
        },
    )
    return search


def finalize_native_outcome(
    run_dir: Path, *, opportunity: int, candidate_id: str
) -> None:
    pending_path = run_dir / NATIVE_ROOT / f"pending-{opportunity:04d}.json"
    pending = _read_json(pending_path)
    if pending.get("candidate_id") != candidate_id:
        raise RuntimeError("native OpenEvolve pending outcome does not match")
    search = dict(pending["search"])
    pointer = {
        "schema_version": "native-openevolve-v1",
        "stage": "completed",
        "opportunity": opportunity,
        "checkpoint": search["checkpoint"],
        "random_state": pending["random_state"],
        "updated_at": utc_now(),
    }
    atomic_json(run_dir / CURRENT_STATE, pointer)
    _append_event(
        run_dir,
        {
            "schema_version": "native-openevolve-v1",
            "event": "native_outcome_committed",
            "timestamp": utc_now(),
            "opportunity": opportunity,
            "candidate_id": candidate_id,
            **search,
        },
    )


def mirror_native_prefix_state(leader_dir: Path, shadow_dir: Path) -> None:
    """Copy immutable native checkpoints and the current pointer to a shadow."""

    source = leader_dir / NATIVE_ROOT
    if not source.is_dir():
        return
    target = shadow_dir / NATIVE_ROOT
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json",):
        if (source / name).is_file() and not (target / name).exists():
            shutil.copy2(source / name, target / name)
    source_checkpoints = source / "checkpoints"
    target_checkpoints = target / "checkpoints"
    target_checkpoints.mkdir(exist_ok=True)
    for checkpoint in source_checkpoints.iterdir():
        destination = target_checkpoints / checkpoint.name
        if checkpoint.is_dir() and not destination.exists():
            shutil.copytree(checkpoint, destination)
    atomic_json(target / "current.json", _read_json(source / "current.json"))
