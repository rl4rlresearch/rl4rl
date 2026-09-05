"""Process leases must work on Windows without hiding diagnostic metadata."""

import subprocess
import sys

import pytest

from experiments.c0c3_factorial import file_lock


def test_lock_contends_across_processes_and_preserves_metadata(tmp_path):
    path = tmp_path / "lease.json"
    with path.open("a+") as handle:
        file_lock.flock(handle.fileno(), file_lock.LOCK_EX | file_lock.LOCK_NB)
        handle.write('{"owner": 1}')
        handle.flush()
        assert path.read_text() == '{"owner": 1}'
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from experiments.c0c3_factorial import file_lock as f; "
                "import sys; h=open(sys.argv[1], 'a+'); "
                "f.flock(h.fileno(), f.LOCK_EX | f.LOCK_NB)",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "BlockingIOError" in result.stderr
        file_lock.flock(handle.fileno(), file_lock.LOCK_UN)
    with path.open("a+") as handle:
        file_lock.flock(handle.fileno(), file_lock.LOCK_EX | file_lock.LOCK_NB)
        file_lock.flock(handle.fileno(), file_lock.LOCK_UN)


def test_shared_locks_coexist_but_exclude_writer(tmp_path):
    path = tmp_path / "shared"
    with path.open("a+") as one, path.open("a+") as two:
        file_lock.flock(one.fileno(), file_lock.LOCK_SH | file_lock.LOCK_NB)
        file_lock.flock(two.fileno(), file_lock.LOCK_SH | file_lock.LOCK_NB)
        with path.open("a+") as writer, pytest.raises(BlockingIOError):
            file_lock.flock(writer.fileno(), file_lock.LOCK_EX | file_lock.LOCK_NB)
        file_lock.flock(two.fileno(), file_lock.LOCK_UN)
        file_lock.flock(one.fileno(), file_lock.LOCK_UN)
