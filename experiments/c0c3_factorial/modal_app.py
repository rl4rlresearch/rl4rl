"""Optional Modal execution of the exact same campaign runner used locally.

The operator uploads a created campaign to the named Volume. A remote function
then invokes ``cli run-one`` against that Volume; prompts, state transitions,
retention, logging, and evaluator commands are therefore shared with local
Codex CLI execution rather than reimplemented for Modal.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

try:
    import modal
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    modal = None


APP_NAME = "rl4rl-c0c3-factorial"
VOLUME_NAME = "rl4rl-c0c3-campaigns"
AUTORESEARCH_CACHE_VOLUME_NAME = "rl4rl-autoresearch-cache"
SECRET_NAME = "rl4rl-codex"
REMOTE_REPO = Path("/opt/rl4rl")
REMOTE_CAMPAIGNS = Path("/campaigns")
LOCAL_REPO = Path(__file__).resolve().parents[2]


def safe_campaign_path(value: str) -> Path:
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise ValueError("campaign path must be a safe relative POSIX path")
    return REMOTE_CAMPAIGNS / logical.as_posix()


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "nodejs", "npm")
        .pip_install(
            "dacite",
            "flask",
            "kernels>=0.11.7",
            "matplotlib>=3.10.8",
            "numpy>=2.2.6",
            "openai>=1.0.0",
            "pandas>=2.3.3",
            "pyarrow>=21.0.0",
            "pyyaml",
            "requests>=2.32.0",
            "rustbpe>=0.1.0",
            "tiktoken>=0.11.0",
            "torch==2.9.1",
            "tqdm",
            "uv",
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        .run_commands("npm install --global @openai/codex")
        .add_local_dir(
            str(LOCAL_REPO / "experiments"),
            remote_path=str(REMOTE_REPO / "experiments"),
            copy=True,
            ignore=["**/__pycache__", "**/.pytest_cache", "**/data"],
        )
        .add_local_dir(
            str(LOCAL_REPO / "architecture_discovery/vendor/openevolve/openevolve"),
            remote_path=str(
                REMOTE_REPO / "architecture_discovery/vendor/openevolve/openevolve"
            ),
            copy=True,
            ignore=["**/__pycache__"],
        )
        .add_local_dir(
            str(LOCAL_REPO / "architecture_discovery/vendor/AdderBoard"),
            remote_path=str(REMOTE_REPO / "architecture_discovery/vendor/AdderBoard"),
            copy=True,
            ignore=["**/__pycache__"],
        )
        .workdir(str(REMOTE_REPO))
    )
    app = modal.App(APP_NAME)
    campaign_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    autoresearch_cache = modal.Volume.from_name(
        AUTORESEARCH_CACHE_VOLUME_NAME, create_if_missing=True
    )
    codex_secret = modal.Secret.from_name(SECRET_NAME)

    @app.function(
        image=image,
        gpu="H100",
        cpu=8,
        memory=65536,
        timeout=24 * 60 * 60,
        max_containers=1,
        retries=0,
        include_source=False,
        secrets=[codex_secret],
        volumes={
            str(REMOTE_CAMPAIGNS): campaign_volume,
            "/root/.cache/autoresearch": autoresearch_cache,
        },
    )
    def run_remote(
        campaign: str, run_id: str = "", opportunities: int = 1
    ) -> dict[str, object]:
        if opportunities < 1:
            raise ValueError("opportunities must be positive")
        if run_id and opportunities != 1:
            raise ValueError(
                "explicit run_id is diagnostic-only and requires opportunities=1"
            )
        campaign_volume.reload()
        autoresearch_cache.reload()
        campaign_path = safe_campaign_path(campaign)
        outputs = []
        for _ in range(opportunities):
            command = [
                "python",
                "-m",
                "experiments.c0c3_factorial.cli",
                "run-one" if run_id else "run-next",
                "--campaign",
                str(campaign_path),
                "--python-bin",
                "python",
            ]
            if run_id:
                command.extend(("--run-id", run_id))
            completed = subprocess.run(
                command,
                cwd=REMOTE_REPO,
                text=True,
                capture_output=True,
                check=False,
            )
            campaign_volume.commit()
            if completed.returncode:
                raise RuntimeError(
                    f"remote opportunity failed ({completed.returncode}): "
                    f"{completed.stderr[-4000:]}"
                )
            outputs.append(completed.stdout)
            if completed.stdout.strip() == "campaign completed":
                break
        return {
            "returncode": 0,
            "stdout": "\n".join(outputs),
            "campaign": campaign,
            "run_id": run_id or "[frozen campaign order]",
            "opportunities": opportunities,
        }

    @app.function(
        image=image,
        cpu=8,
        memory=32768,
        timeout=2 * 60 * 60,
        max_containers=1,
        retries=0,
        include_source=False,
        volumes={
            str(REMOTE_CAMPAIGNS): campaign_volume.with_mount_options(read_only=True),
            "/root/.cache/autoresearch": autoresearch_cache,
        },
    )
    def prepare_autoresearch_remote(campaign: str, num_shards: int = 10) -> str:
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        campaign_volume.reload()
        autoresearch_cache.reload()
        campaign_path = safe_campaign_path(campaign)
        schedule = json.loads(
            (campaign_path / "schedule.json").read_text(encoding="utf-8")
        )
        if not schedule:
            raise ValueError("campaign has no runs")
        support = campaign_path / "runs" / str(schedule[0]["run_id"]) / "task-support"
        completed = subprocess.run(
            ["python", "prepare.py", "--num-shards", str(num_shards)],
            cwd=support,
            text=True,
            capture_output=True,
            check=False,
        )
        autoresearch_cache.commit()
        if completed.returncode:
            raise RuntimeError(
                f"autoresearch preparation failed ({completed.returncode}): "
                f"{completed.stderr[-4000:]}"
            )
        return completed.stdout

    @app.local_entrypoint()
    def main(
        campaign: str,
        run_id: str = "",
        opportunities: int = 1,
        prepare_autoresearch: bool = False,
        prepare_only: bool = False,
        num_shards: int = 10,
    ) -> None:
        if prepare_autoresearch:
            print(prepare_autoresearch_remote.remote(campaign, num_shards))
        if prepare_only:
            if not prepare_autoresearch:
                raise ValueError("prepare_only requires prepare_autoresearch=true")
            return
        result = run_remote.remote(campaign, run_id, opportunities)
        print(result["stdout"])

else:
    app = None
    image = None
    campaign_volume = None
    autoresearch_cache = None
    codex_secret = None
