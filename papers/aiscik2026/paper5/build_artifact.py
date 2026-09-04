#!/usr/bin/env python3
"""Build an anonymized Paper 5 reproducibility archive."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTPUT = REPO / "output/paper5_reproducibility_artifact.zip"
ARCHIVE_ROOT = "paper5_interface_instrument_artifact"

LICENSE = """MIT License

Copyright (c) 2026 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this artifact and its included software, trace records, candidate source
snapshots, data, and documentation files (the "Materials"), to deal in the
Materials without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Materials, and to permit persons to whom the Materials are furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Materials.

THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS IN THE
MATERIALS.
"""


def load_analysis() -> ModuleType:
    spec = importlib.util.spec_from_file_location("paper5_analysis", HERE / "analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load paper5 analysis module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sanitize_text(text: str) -> str:
    home = Path.home()
    replacements = {
        str(REPO): "REPOSITORY_ROOT",
        str(home): "/home/anonymous",
        home.name: "anonymous",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"/private/var/folders/[^\s\"')\]}<>]+", "LOCAL_TEMP_PATH", text)
    text = re.sub(r"/var/folders/[^\s\"')\]}<>]+", "LOCAL_TEMP_PATH", text)
    text = re.sub(r"file:///Users/[^\\s\"')\\]}<>]+", "file://LOCAL_USER_PATH", text)
    return text


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = source.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        shutil.copy2(source, destination)
    else:
        destination.write_text(sanitize_text(text), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def privacy_scan(root: Path) -> None:
    forbidden = [
        str(Path.home()).encode(),
        b"/Users/utshaho",
        b"/private/var/folders",
        b"OPENAI_API_KEY",
        b"MODAL_TOKEN",
    ]
    suspicious_key = re.compile(rb"sk-[A-Za-z0-9_-]{16,}")
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for needle in forbidden:
            if needle and needle in data:
                hits.append(f"{path.relative_to(root)} contains {needle.decode(errors='replace')}")
        if suspicious_key.search(data):
            hits.append(f"{path.relative_to(root)} contains a possible API key")
    if hits:
        raise RuntimeError("Privacy scan failed:\n" + "\n".join(hits[:20]))


def write_checksums(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "PAPER5_SHA256SUMS":
            continue
        relative = path.relative_to(root)
        lines.append(f"{sha256(path)}  {relative.as_posix()}")
    (root / "PAPER5_SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_tree(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(ARCHIVE_ROOT) / path.relative_to(root))


def main() -> None:
    analysis = load_analysis()
    runs = analysis.load_runs()
    input_files = analysis.input_files(runs)
    if len(input_files) < 1000:
        raise RuntimeError(f"Suspiciously small input set: {len(input_files)} files")
    with tempfile.TemporaryDirectory(prefix="paper5-artifact-") as temp:
        staging = Path(temp) / ARCHIVE_ROOT
        for source in sorted(input_files):
            copy_file(source, staging / source.relative_to(REPO))
        for relative in [
            "papers/aiscik2026/paper5/analysis.py",
            "papers/aiscik2026/paper5/requirements.txt",
            "papers/aiscik2026/paper5/ARTIFACT_README.md",
        ]:
            copy_file(REPO / relative, staging / relative)
        (staging / "LICENSE").write_text(LICENSE, encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(Path(temp) / "mplconfig"))
        script = staging / "papers/aiscik2026/paper5/analysis.py"
        derived = staging / "papers/aiscik2026/paper5/derived"
        subprocess.run([sys.executable, str(script), "--data-root", str(staging), "--output", str(derived)], check=True, env=env)
        subprocess.run(
            [sys.executable, str(script), "--data-root", str(staging), "--output", str(derived), "--verify-input-hashes"],
            check=True,
            env=env,
        )

        privacy_scan(staging)
        write_checksums(staging)
        zip_tree(staging, OUTPUT)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / (1024 * 1024):.2f} MiB)")
    print(f"SHA256 {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
