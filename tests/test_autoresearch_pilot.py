from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREATE_RUN = ROOT / "experiments/autoresearch_pilot/create_run.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_failed_attempt_is_snapshotted_and_restores_incumbent(tmp_path: Path) -> None:
    runs_root = tmp_path / "raw"
    subprocess.run(
        [
            sys.executable,
            str(CREATE_RUN),
            "--run-id",
            "pilot-test-001",
            "--runs-root",
            str(runs_root),
            "--python-bin",
            sys.executable,
            "--agent-model",
            "gpt-5.6-mini",
            "--agent-reasoning-effort",
            "high",
            "--skip-baseline",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = runs_root / "pilot-test-001"
    workspace = run_dir / "workspace"
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text())
    config = json.loads((run_dir / "RUN_CONFIG.json").read_text())
    launcher = (run_dir / "launch_codex.sh").read_text()
    assert manifest["agent_model"] == "gpt-5.6-mini"
    assert manifest["agent_reasoning_effort"] == "high"
    assert config["agent_model"] == "gpt-5.6-mini"
    assert config["agent_reasoning_effort"] == "high"
    assert "CODEX_MODEL=gpt-5.6-mini" in launcher
    assert "CODEX_REASONING_EFFORT=high" in launcher
    program = (workspace / "PROGRAM.md").read_text()
    assert "## Evidence review before a proposal" in program
    assert "## Evaluation discipline" in program
    assert "## Interpreting errors" in program
    assert "## Bounded automations" in program
    assert "## Final report only" in program
    assert "python ../run_attempt.py --run-dir .. attempt" in program
    assert "Do **not** create a series of automation IDs" in program
    assert "keep going until it is exhausted" in program
    assert "## Multi-trial automation protocol" not in program
    assert "Do not tail, poll, pause, or send a progress message" in program
    assert "/Users/" not in program
    program += "\n<!-- operator-owned prompt update -->\n"
    (workspace / "PROGRAM.md").write_text(program)
    checkpoint = workspace / "checkpoints/best.pt"
    expected_checkpoint_hash = _sha256(checkpoint)
    shutil.copy2(checkpoint, run_dir / "state/incumbent.pt")

    state = json.loads((run_dir / "STATE.json").read_text())
    state["baseline_recorded"] = True
    state["incumbent"]["parameters"] = 6080
    (run_dir / "STATE.json").write_text(json.dumps(state))

    config["training_command"] = [sys.executable, "-c", "raise SystemExit(1)"]
    (run_dir / "RUN_CONFIG.json").write_text(json.dumps(config))

    completed = subprocess.run(
        [
            sys.executable,
            str(run_dir / "run_attempt.py"),
            "--run-dir",
            str(run_dir),
            "attempt",
            "--description",
            "intentional test failure",
            "--proposal",
            "exercise snapshot and rollback behavior",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "DISCARDED" in completed.stdout
    result = json.loads((run_dir / "attempts/attempt-0001/result.json").read_text())
    assert result["event"]["status"] == "error"
    assert (run_dir / "attempts/attempt-0001/candidate/src/model.py").is_file()
    assert _sha256(checkpoint) == expected_checkpoint_hash
    assert "operator-owned prompt update" in (workspace / "PROGRAM.md").read_text()
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "show-ref",
                "--verify",
                "refs/autoresearch/attempts/attempt-0001",
            ],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )

    subprocess.run(
        [
            sys.executable,
            str(run_dir / "run_attempt.py"),
            "--run-dir",
            str(run_dir),
            "automation-start",
            "--automation-id",
            "qkv-prune",
            "--family",
            "scalar pruning",
            "--description",
            "bounded QKV pruning automation",
            "--proposal",
            "test macro/micro logging",
            "--max-micro-trials",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    micro = subprocess.run(
        [
            sys.executable,
            str(run_dir / "run_attempt.py"),
            "--run-dir",
            str(run_dir),
            "automation-attempt",
            "--description",
            "zero one scalar",
            "--proposal",
            "first ranked scalar",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "attempt-0002 micro-0001 ERROR" in micro.stdout
    micro_rows = (run_dir / "AUTOMATION_RESULTS.tsv").read_text().splitlines()
    assert len(micro_rows) == 2
    assert micro_rows[1].startswith("attempt-0002\tqkv-prune\tmicro-0001\t")
    assert (run_dir / "automations/attempt-0002/micro-0001/result.json").is_file()
    assert "operator-owned prompt update" in (workspace / "PROGRAM.md").read_text()
    trigger = json.loads(
        (run_dir / "automations/attempt-0002/AUTOMATION_TRIGGER.json").read_text()
    )
    assert trigger["reason"] == "micro_trial_error"
    assert trigger["micro_attempt_id"] == "micro-0001"

    subprocess.run(
        [
            sys.executable,
            str(run_dir / "run_attempt.py"),
            "--run-dir",
            str(run_dir),
            "automation-end",
            "--summary",
            "one failed micro-trial reached the test boundary",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result_rows = (run_dir / "RESULTS.tsv").read_text().splitlines()
    assert result_rows[-1].startswith("attempt-0002\t")
    assert "[automation:qkv-prune]" in result_rows[-1]
    state = json.loads((run_dir / "STATE.json").read_text())
    assert state["attempts_used"] == 2
    assert "active_automation" not in state

    with (run_dir / ".runner.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        overlapping = subprocess.run(
            [
                sys.executable,
                str(run_dir / "run_attempt.py"),
                "--run-dir",
                str(run_dir),
                "attempt",
                "--description",
                "must not begin while another runner owns the lock",
                "--proposal",
                "exercise exclusive-run protection",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert overlapping.returncode == 3
    assert "Runner busy" in overlapping.stderr
    assert not (run_dir / "attempts/attempt-0003").exists()

    partial_attempt = run_dir / "attempts/attempt-0003"
    partial_attempt.mkdir()
    (partial_attempt / "train.stdout.log").write_text("interrupted before result")
    recovered = subprocess.run(
        [
            sys.executable,
            str(run_dir / "run_attempt.py"),
            "--run-dir",
            str(run_dir),
            "attempt",
            "--description",
            "recover from an interrupted partial attempt",
            "--proposal",
            "exercise stale-attempt archival before retrying the same ID",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Archived interrupted partial attempt" in recovered.stdout
    assert (run_dir / "attempts/attempt-0003/result.json").is_file()
    archived = list((run_dir / "interrupted-attempts").glob("attempt-0003-*"))
    assert len(archived) == 1
    assert (archived[0] / "train.stdout.log").read_text() == "interrupted before result"
