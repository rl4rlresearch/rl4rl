# Viewing restored campaigns on Windows

From the repository root, run:

```powershell
.\experiments\start_dashboard.ps1
```

Open http://127.0.0.1:8765 in the Codex browser. The launcher uses the repository
virtual environment if present, otherwise the Python runtime bundled with Codex,
and writes server logs to `outputs/dashboard/`. No third-party Python packages
are needed to serve the dashboard.

The trajectory explorer, scientific-process page, and run-transcript page read
campaigns from `data/c0c3`. Windows uses extended paths so deeply nested artifacts
copied from macOS remain readable. The existing command-line campaign options
can also point at another location.

On Windows the dashboard serves saved research data in viewing mode. Statuses
reflect the saved logs. Unix campaign controls, host resource locks, Mac telemetry,
and the background Codex quota sampler are disabled. Opening the dashboard does
not resume campaigns or modify their scientific artifacts. Continuing research
runs on Windows requires separate work on the execution environment.

The September 2026 restoration keeps the original backup folders intact and
copies all nine folders into `data/c0c3`. Four are campaigns, four contain
overnight-supervisor records, and one is a standalone copy of an addition run.
That standalone run's research files already match the campaign copy; it must
not be counted as another trajectory. Finder metadata is the only difference.
The restore receipt and file-by-file SHA-256 checks are in `outputs/restore/`.
Both the restored campaign artifacts and restoration receipts remain gitignored.
