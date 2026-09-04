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
verified_results: {"depth": 8.0, "mfu_percent": 32.4, "num_params_M": 50.9, "num_steps": 768.0, "peak_vram_mb": 45586.1, "total_tokens_M": 402.7, "training_seconds": 300.3, "val_bpb": 1.008305}
prior_hypothesis: Replacing ReLU-squared MLPs with near-parameter-matched SwiGLU MLPs will improve per-token modeling enough to beat 0.995558 val_bpb while retaining the winning SSSL attention layout and similar throughput.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.58, "num_params_M": 50.3, "num_steps": 948.0, "peak_vram_mb": 45060.2, "total_tokens_M": 497.0, "training_seconds": 300.2, "val_bpb": 0.995558}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 28.09, "num_params_M": 50.3, "num_steps": 733.0, "peak_vram_mb": 45060.2, "total_tokens_M": 384.3, "training_seconds": 300.3, "val_bpb": 1.012775}
prior_hypothesis: Reducing short-layer attention windows from 1024 to 512 tokens will increase tokens processed within five minutes enough to beat the current 0.995558 val_bpb, while the two full-context layers preserve long-range modeling.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 32.0, "num_params_M": 50.3, "num_steps": 789.0, "peak_vram_mb": 45060.2, "total_tokens_M": 413.7, "training_seconds": 300.3, "val_bpb": 1.008991}
prior_hypothesis: Replacing the intermediate full-context layer with half-context attention will process more than 497M tokens while preserving effective long-range propagation through stacked 1024-token windows, lowering val_bpb below 0.995558.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing short-layer attention windows from 1024 to 512 tokens will increase tokens processed within five minutes enough to beat the current 0.995558 val_bpb, while the two full-context layers preserve long-range modeling.
change: Change the six short-window layers to use one-quarter context; retain the existing SSSL pattern and forced full-context final layer.
mechanism: Quarter-context sliding attention with periodic full-context layers
evidence_used: The current depth-8 design reaches 0.995558 val_bpb on 497.0M tokens at 39.58% MFU, so reducing attention work is a targeted way to train on more data without reducing model parameters or depth.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 28.09, "num_params_M": 50.3, "num_steps": 733.0, "peak_vram_mb": 45060.2, "total_tokens_M": 384.3, "training_seconds": 300.3, "val_bpb": 1.012775}

RECENT RESULT
hypothesis: Replacing the intermediate full-context layer with half-context attention will process more than 497M tokens while preserving effective long-range propagation through stacked 1024-token windows, lowering val_bpb below 0.995558.
change: Use half-context attention in the first seven layers while retaining the forced full-context final layer.
mechanism: Single full-context anchor over stacked half-context attention
evidence_used: The current 1024-token short-window design achieved 0.995558 val_bpb on 497.0M tokens, whereas reducing windows to 512 unexpectedly cut throughput to 384.3M tokens and worsened val_bpb to 1.012775; this preserves the faster window size and removes only one full-attention layer.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 32.0, "num_params_M": 50.3, "num_steps": 789.0, "peak_vram_mb": 45060.2, "total_tokens_M": 413.7, "training_seconds": 300.3, "val_bpb": 1.008991}

RECENT RESULT
hypothesis: Replacing ReLU-squared MLPs with near-parameter-matched SwiGLU MLPs will improve per-token modeling enough to beat 0.995558 val_bpb while retaining the winning SSSL attention layout and similar throughput.
change: Use a tensor-core-aligned SwiGLU hidden width near 8/3 of model dimension, keeping MLP parameter count and compute close to the current 4× ReLU-squared design.
mechanism: Parameter-matched SwiGLU feed-forward layers
evidence_used: The current SSSL design achieved the best result, 0.995558 val_bpb on 497.0M tokens; both attention-window changes reduced throughput and worsened validation, motivating an independent capacity-quality change that leaves attention unchanged.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 32.4, "num_params_M": 50.9, "num_steps": 768.0, "peak_vram_mb": 45586.1, "total_tokens_M": 402.7, "training_seconds": 300.3, "val_bpb": 1.008305}



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
