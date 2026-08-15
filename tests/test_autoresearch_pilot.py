from __future__ import annotations

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
    assert 'CODEX_MODEL=gpt-5.6-mini' in launcher
    assert 'CODEX_REASONING_EFFORT=high' in launcher
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
