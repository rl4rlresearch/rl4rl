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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"depth": 8.0, "mfu_percent": 39.58, "num_params_M": 50.3, "num_steps": 948.0, "peak_vram_mb": 45060.2, "total_tokens_M": 497.0, "training_seconds": 300.2, "val_bpb": 0.995558}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 28.16, "num_params_M": 50.3, "num_steps": 735.0, "peak_vram_mb": 45060.2, "total_tokens_M": 385.4, "training_seconds": 300.4, "val_bpb": 1.012584}
prior_hypothesis: Reducing short-window attention from 1024 to 512 tokens will lower val_bpb below 0.995558 by increasing training throughput while preserving full-context mixing in layers 4 and 8.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing short-window attention from 1024 to 512 tokens will lower val_bpb below 0.995558 by increasing training throughput while preserving full-context mixing in layers 4 and 8.
change: Change the six short-attention layers to quarter-context windows; the two long-attention layers remain unchanged.
mechanism: Hierarchical 512-token local attention
evidence_used: The baseline processes 497.0M tokens at 39.58% MFU with six of eight layers using 1024-token attention, so local attention remains a substantial fixed-time compute cost.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 28.16, "num_params_M": 50.3, "num_steps": 735.0, "peak_vram_mb": 45060.2, "total_tokens_M": 385.4, "training_seconds": 300.4, "val_bpb": 1.012584}



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
