# Modal execution runbook

Modal runs the same `SearchController`, prompt renderer, framework adapter,
evaluator, event logger, runtime-hash gate, recovery rule, and `run-next`
orchestrator used locally. `modal_app.py` is transport and compute provisioning,
not a second scientific implementation.

Protocols 2.0 and 2.1 also provide a separate **hybrid evaluator-only** path documented
in [OPENEVOLVE_V2.md](OPENEVOLVE_V2.md). In that path Codex and campaign state
remain local; a deployed Modal L4 function receives only one immutable
candidate/support bundle and returns evaluator artifacts. Do not confuse it
with the full-campaign H100 transport below, and never mix the two in one
campaign.

The current remote worker is deliberately fixed to one H100, 8 CPUs, and 64
GiB RAM for target-backend comparability. Changing GPU type creates a different
hardware stratum and requires new calibration and campaigns.

## 1. Why execution is serialized

Modal documents that Volume updates made by different containers use
last-write-wins semantics and that Volumes do not provide distributed file
locking. A campaign contains shared `state.json` and `events.jsonl`, so parallel
writers to one campaign would be unsafe. The mutation function therefore sets
`max_containers=1`, performs an explicit Volume reload before work and commit
after every opportunity, and disables retries. Do not launch another app,
local runner, or manual Volume writer against the same campaign.

This serialization is global to this app’s mutation function. It favors valid
state over throughput. Parallelism across scientifically independent campaigns
should use separate Modal environments/apps/volumes or a future server-side
transactional coordinator—not concurrent writes to this Volume.

References: [Modal Volume consistency and commits](https://modal.com/docs/guide/volumes),
[Modal Function autoscaling limits](https://modal.com/docs/guide/scale).

## 2. Install and authenticate Modal locally

The repository pins `modal==1.5.3` in the architecture project:

```bash
cd architecture_discovery
uv sync --group modal
cd ..
MODAL=architecture_discovery/.venv/bin/modal
$MODAL setup
```

Confirm the app’s generated CLI without launching compute:

```bash
PYTHONPATH=. $MODAL run -m experiments.c0c3_factorial.modal_app --help
```

Official reference: [Modal CLI](https://modal.com/docs/cli/latest).

## 3. Create the Codex secret and persistent Volumes

The worker expects a Modal Secret named `rl4rl-codex` containing
`OPENAI_API_KEY`. Never put the key in a campaign, command log, image, or git.

```bash
$MODAL secret create rl4rl-codex OPENAI_API_KEY="$OPENAI_API_KEY"
$MODAL volume create rl4rl-c0c3-campaigns
$MODAL volume create rl4rl-autoresearch-cache
```

If an object already exists, inspect and reuse it rather than deleting it.
Deletion is destructive and can invalidate deployed apps. The campaign Volume
holds calibration/campaign state; the cache Volume is mounted at
`/root/.cache/autoresearch` and holds official data shards/tokenizer output.

References: [Modal Secrets](https://modal.com/docs/guide/secrets),
[Modal Volume CLI](https://modal.com/docs/cli/latest/volume).

## 4. Prepare a GPU calibration bundle locally

Pin the official task checkout as described in `FRAMEWORKS_AND_TASKS.md`, then:

```bash
export AUTORESEARCH_ROOT='/absolute/path/to/pinned/autoresearch'
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli
C0C3=experiments/c0c3_factorial
PROTOCOL=$C0C3/configs/protocols/dev.toml
TASK=$C0C3/configs/tasks/karpathy_nanogpt.toml
CAL_LOCAL=data/c0c3/nanogpt-dev-calibration
CAL_REMOTE=c0c3/nanogpt-dev-calibration

$PY -m $CLI prepare-calibration \
  --protocol "$PROTOCOL" \
  --task "$TASK" \
  --output "$CAL_LOCAL"
```

This copies and hashes the task but performs no training. Confirm there is no
`baseline.json` yet.

Upload it:

```bash
$MODAL volume put rl4rl-c0c3-campaigns "$CAL_LOCAL" "$CAL_REMOTE"
$MODAL volume ls rl4rl-c0c3-campaigns "$CAL_REMOTE"
```

Remote parent directories are created automatically. If testing upload naming
for the first time, use a disposable path and inspect `volume ls`; do not use
`--force` on scientific state without first confirming the exact target.

## 5. Prepare the persistent Autoresearch dataset/tokenizer

Run this once per pinned task/data version. The official default downloads ten
training shards plus its pinned validation shard and trains the tokenizer:

```bash
PYTHONPATH=. $MODAL run -m experiments.c0c3_factorial.modal_app \
  --campaign "$CAL_REMOTE" \
  --prepare-autoresearch \
  --prepare-only \
  --num-shards 10
```

Do not run cache preparation concurrently with training. Cache contents are
task infrastructure: record shard count, upstream commit, and preparation logs.

## 6. Execute calibration on the H100

```bash
PYTHONPATH=. $MODAL run -m experiments.c0c3_factorial.modal_app \
  --campaign "$CAL_REMOTE" \
  --execute-calibration
```

The same serialized mutation function used for search executes the prepared
bundle, writes `baseline.json`, and commits it.

Download only the result into the original local bundle:

```bash
$MODAL volume get \
  rl4rl-c0c3-campaigns \
  "$CAL_REMOTE/baseline.json" \
  "$CAL_LOCAL/baseline.json"
```

Inspect it before campaign creation. Its `calibration_kind` must be
`executed_on_target_backend` and the support/protocol hashes must match.

## 7. Create, validate, and upload the campaign

Choose one framework:

```bash
FRAMEWORK=$C0C3/configs/frameworks/autoresearch.toml
CAMPAIGN_LOCAL=data/c0c3/nanogpt-dev-autoresearch
CAMPAIGN_REMOTE=c0c3/nanogpt-dev-autoresearch

$PY -m $CLI create \
  --protocol "$PROTOCOL" \
  --task "$TASK" \
  --framework "$FRAMEWORK" \
  --baseline "$CAL_LOCAL/baseline.json" \
  --output "$CAMPAIGN_LOCAL"

$PY -m $CLI validate --campaign "$CAMPAIGN_LOCAL"
$MODAL volume put \
  rl4rl-c0c3-campaigns \
  "$CAMPAIGN_LOCAL" \
  "$CAMPAIGN_REMOTE"
$MODAL volume ls rl4rl-c0c3-campaigns "$CAMPAIGN_REMOTE"
```

Create a separate remote path for OpenEvolve. Never overwrite one framework’s
campaign with another.

Campaign creation and `modal run` must use the same committed controller source.
The runtime hash is checked inside the H100 worker before every opportunity; a
source mismatch fails before Codex is called.

## 8. Run scientific opportunities

One opportunity in frozen blocked order:

```bash
PYTHONPATH=. $MODAL run -m experiments.c0c3_factorial.modal_app \
  --campaign "$CAMPAIGN_REMOTE"
```

A sequential batch of ten, still using `run-next` for every selection:

```bash
PYTHONPATH=. $MODAL run -m experiments.c0c3_factorial.modal_app \
  --campaign "$CAMPAIGN_REMOTE" \
  --opportunities 10
```

An explicit `--run-id` is accepted only with one opportunity and is diagnostic;
it bypasses campaign ordering. Do not use it for paper data.

The image installs Codex CLI, OpenEvolve dependencies, official Autoresearch
dependencies, PyTorch 2.9.1/CUDA 12.8 wheels, and copies only the controlled
experiment source plus required vendored runtimes. It does not upload root
`data/`, prior run artifacts, `.git`, or local virtual environments.

Official image API reference: [Modal `Image.add_local_dir` and package
installation](https://modal.com/docs/reference/modal.Image).

## 9. Inspect or retrieve remote state

List without mutation:

```bash
$MODAL volume ls rl4rl-c0c3-campaigns "$CAMPAIGN_REMOTE"
```

Download to a new local directory, not over a locally active campaign:

```bash
$MODAL volume get \
  rl4rl-c0c3-campaigns \
  "$CAMPAIGN_REMOTE" \
  data/c0c3/retrieved-nanogpt-dev-autoresearch
```

Treat remote state as authoritative once uploaded. Do not alternate local and
Modal mutations. For analysis, stop remote execution, retrieve a snapshot, hash
it, and perform sealed Layer B/C operations against one authoritative copy.

## 10. Failures and recovery on Modal

The worker commits after each completed opportunity. Modal retries are disabled
because replaying a research opportunity is invalid. If a container disappears
after `state.active` is written, the next invocation refuses to continue.

Retrieve/inspect logs and state, make certain no invocation is alive, then run
the same `recover-active` CLI against an authoritative mounted/downloaded copy.
Recovery must remain visible and charged. Never delete the opportunity directory
or edit `state.json` to make a preemption disappear.

## 11. Pre-paper Modal receipt

Record:

- Modal SDK version and environment/workspace;
- image build ID/log and package versions;
- GPU type and Modal function resource configuration;
- Codex CLI version inside the image;
- task upstream commit and cache-preparation log;
- calibration baseline and elapsed time;
- Volume names and remote campaign path;
- local validation receipt and all campaign hashes;
- confirmation that only one mutation writer was active.

## 12. Protocol-2.0/2.1 evaluator-only deployment

The lab account owner should create a dedicated environment and hard budget,
then deploy once:

```bash
MODAL=architecture_discovery/.venv/bin/modal
$MODAL deploy -m experiments.c0c3_factorial.modal_hybrid_app
```

The deployed app is `rl4rl-c0c3-hybrid-evaluator-v2`; it permits at most three
L4 containers, uses no Codex/OpenAI secret, disables Modal retries, and has a
five-minute maximum idle scaledown window. Local calls use Modal API tokens from
the active profile or `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`.

Use the matching protocol-2.0 or source-only protocol-2.1 Modal task TOML for
both calibration and campaign creation. Never reuse calibration across those
protocol/task hashes.
Each remote call appends `modal-usage.jsonl` to the local campaign. Summarize it
with `c0c3_factorial.cli modal-usage`; use `modal billing` or the Usage & Billing
dashboard for authoritative spend, credits, and remaining budget.

These facts belong in the reproducibility appendix and artifact archive.
