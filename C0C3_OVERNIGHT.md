# Durable C0–C3 overnight runner

The overnight supervisor is operational infrastructure. It lives outside the
hashed `experiments/c0c3_factorial/` runtime and invokes every campaign through
the exact worktree whose scientific-runtime hash matches that campaign.

## Frozen roster

The default roster contains the unfinished runs that remain part of the current
research plan:

- OpenEvolve workshop campaign: all three blocks, C0–C3 and N0 (15 runs).
- subject-neutral Autoresearch v1.5 with the 6,080-parameter parent: Blocks 1
  and 2, C0–C3 only (8 runs).
- subject-neutral Autoresearch v1.5 with the 1,644-parameter parent: Block 1
  C0–C3 only (4 runs).

The superseded one-turn parallel Autoresearch campaign and the continuous v1.4
Block 1 campaign are intentionally not in this roster. Their stale `running`
states do not make them live overnight runs. Dormant v1.5 N0 and extension
blocks are not part of the primary profile.

The separately frozen 1,644-parent Blocks 2 and 3 extension uses its own
supervisor profile. It has an independent process namespace and control files,
so it can run alongside the primary roster without restarting or interrupting
any primary controller:

```bash
RL4RL_OVERNIGHT_PROFILE=1644-extension \
  architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py \
  start --recover-interrupted --all-running
```

Its status and control commands use the same profile prefix, for example:

```bash
RL4RL_OVERNIGHT_PROFILE=1644-extension \
  architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py status
RL4RL_OVERNIGHT_PROFILE=1644-extension \
  architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py pause \
  --reason 'operator-requested pause'
```

Extension control artifacts are under
`data/c0c3/overnight-control-1644-extension/`. The profile deliberately starts
only C0-C3 from Blocks 2 and 3; it does not start the duplicate dormant Block 1
assignments or any N0 assignments in the three-block campaign.

## Start

From the repository root in a normal VS Code terminal:

```bash
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py \
  start --recover-interrupted --all-running
```

The command performs the runtime-hash and dependency preflight, starts a
detached `screen` session, wraps it in `caffeinate -dimsu`, and returns to the
shell. Closing VS Code or its terminal does not terminate the detached session.

`--recover-interrupted` authorizes only the protocol's existing
`recover-active` operation. It never deletes or retries an interrupted
opportunity. Available Codex usage is charged, an infrastructure-interruption
record is appended, and the next opportunity is started. This is currently
required only for the interrupted OpenEvolve N0 opportunity.

Keep the Mac connected to power, keep the lid open, and do not log out or
reboot. `caffeinate` prevents idle sleep, but macOS can still sleep when a
laptop lid is closed and a detached `screen` session does not survive a reboot.

## Status and logs

```bash
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py status
```

The table distinguishes desired state from real process state and includes
PID, proposal range, recorded tokens, lowest qualifying parameter count, and
active-opportunity count. Runtime files are under:

```text
data/c0c3/overnight-control/
```

Each controller has a separate log under `logs/`. The detached terminal output
is `screenlog.0`, and supervisor recovery/restart decisions are in
`supervisor.log` and `recovery.log`.

## Pause, stop, and resume

With no target, a control command applies to the whole roster:

```bash
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py pause \
  --reason 'operator-requested pause'
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py resume
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py stop \
  --reason 'operator-requested stop'
```

A group or exact job can be controlled independently:

```bash
# Groups: openevolve, autoresearch-v1.5-6080,
#         autoresearch-v1.5-1644
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py pause \
  autoresearch-v1.5-6080 --reason 'operator-requested pause'

# Exact v1.5 job example
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py resume \
  autoresearch-v1.5-6080:b01-c2
```

Protocol v1.5 has native cooperative pause: the in-flight opportunity finishes
and commits, then no next opportunity begins. The older frozen runtimes have no
cooperative pause API. For those groups the supervisor interrupts the process
group, preserves the active opportunity, and leaves it untouched while paused;
resume formally recovers that interruption before continuing. This can consume
the interrupted proposal but does not silently retry or erase it.

`stop` keeps the supervisor alive and the selected jobs stopped, so they can be
resumed later. To terminate the outer supervisor itself:

```bash
architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py shutdown
```

An unexpected controller exit is handled the same way: any active opportunity
is formally recovered, then the controller restarts after exponential backoff.
The maximum backoff is 30 minutes, so provider/rate-limit trouble slows retries
without turning into a permanent unobserved pause.
