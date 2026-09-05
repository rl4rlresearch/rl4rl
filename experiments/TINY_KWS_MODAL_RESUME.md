# TinyKWS Windows-to-Modal continuation

The operator requested resumption of unfinished C0-C3 TinyKWS trajectories on
Modal using two CPU cores and 4 GiB per evaluator. The continuation selects only
the six trajectories below 200 proposals. C4 and completed trajectories are not
launched or rewritten.

`modal_tiny_kws_app.py` deploys `rl4rl-tiny-kws-cpu`, with no application evaluator
concurrency cap. It embeds the checksummed Mini Speech Commands archive and the
protected feature cache. The existing Tiny AdderBoard transport supplies Modal
dispatch, result identity checks, and Windows fallback. Both tasks share its
three-slot Windows fallback namespace; queued work retries Modal every 30 seconds.
The launcher sets two numerical-library threads per local evaluator. Subject
worker leases are disabled through the task extension options.

The original speech exposure (50,000 examples), qualification (85%), integer MAC
objective, seeds, prompts, and 240-second evaluator timeout remain unchanged.
Moving from Mac CPU to Modal CPU and rebuilding the missing protected feature
cache are recorded as an operational amendment, including hashes, dataset split,
calibration result, and exact starting indices. Original campaign inputs,
manifests, states, events, and quarantined attempts remain preserved. Continuation
uses the already-cleaned state, including its previously cleared unfinished tails.

Prepare data with the existing `tiny_kws_rnn prepare --repo-root .` command, then
deploy with `python -m modal deploy -m experiments.modal_tiny_kws_app`. After the
campaign amendment and isolated Modal smoke evaluation have been verified, run
`./experiments/resume_tiny_kws_modal.ps1`. Inspect its timestamped worker receipt,
per-worker logs, live controller locks, and new `remote-dispatch.json` receipts.
The launcher rejects active proposals; never invoke it on a live trajectory.

The service and launcher live outside the hashed controller package so preparing
speech continuation does not invalidate an already-running Tiny AdderBoard
campaign. The deployed app source hash is included in the amended task options.
