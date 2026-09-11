# Improve fixed-time language-model pretraining

You are an autonomous ML engineer improving the source code for single-GPU
language-model pretraining.

## Goal

Minimize validation bits per byte (`val_bpb`) after a fixed five-minute training
window on the supplied H100 worker. Lower is better. Startup, compilation, and
final validation are outside the measured training window, and every submitted
version starts from a fresh initialization.

You may change the architecture, optimizer, schedules, batching, numerical
implementation, or other contents of `train.py`. The fixed data preparation,
tokenizer, validation procedure, hardware class, and time accounting are not
editable. A useful change must produce a complete trainable implementation and
finish with the required summary metrics.

## Work boundaries

Minimize val_bpb. No additional accuracy threshold.
Editable source files: train.py.
Results reported after each verification: val_bpb, training_seconds, peak_vram_mb, mfu_percent, total_tokens_M, num_steps, num_params_M, depth.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or validation yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"depth": 8.0, "mfu_percent": 37.61, "num_params_M": 50.3, "num_steps": 977.0, "peak_vram_mb": 45060.2, "total_tokens_M": 512.2, "training_seconds": 300.0, "val_bpb": 0.993365}
prior_hypothesis: Reducing short attention windows from 1024 to 512 tokens will increase training throughput enough to lower val_bpb below 0.995558, while the full-context layers at indices 3 and 7 preserve long-range information.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing short attention windows from 1024 to 512 tokens will increase training throughput enough to lower val_bpb below 0.995558, while the full-context layers at indices 3 and 7 preserve long-range information.
change: Use quarter-context windows for “S” layers while retaining the existing SSSL pattern and full-context final layer.
mechanism: Denser periodic global attention with cheaper local layers
evidence_used: The baseline reaches val_bpb 0.995558 after 497.0M tokens at 39.58% MFU; attention remains a substantial compute cost, and six of eight layers currently use relatively expensive half-context windows.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.61, "num_params_M": 50.3, "num_steps": 977.0, "peak_vram_mb": 45060.2, "total_tokens_M": 512.2, "training_seconds": 300.0, "val_bpb": 0.993365}



Use the available technical evidence to choose the most informative next
change. Treat unsuccessful or malformed work as evidence when a useful
subject-level reason is provided. Do not invent missing evidence.

## Response

Return these short metadata lines followed by one or more exact
`SEARCH`/`REPLACE` blocks that together produce one implementation:

`MECHANISM: <a concise free-form name for the computational idea>`

`HYPOTHESIS: <a falsifiable claim grounded in the evidence above>`

`INTENDED_EDIT: <what this patch changes>`

`EVIDENCE: <the most relevant prior result and why it motivates this patch>`

Start each block with `<<<<<<< SEARCH`, put the exact existing lines next, use a
line containing `=======` as the divider, put the replacement lines after it,
and finish the block with `>>>>>>> REPLACE`.

Every `SEARCH` section must be nonempty and match exactly once after earlier
blocks have been applied. All blocks must apply. Together they must describe
one implementation ready for verification. The mechanism name is descriptive,
not chosen from a fixed list. Do not paste whole files, lengthy logs, or routine
progress reports outside the patch.
