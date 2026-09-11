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
verified_results: {"depth": 8.0, "mfu_percent": 37.75, "num_params_M": 50.3, "num_steps": 981.0, "peak_vram_mb": 45060.2, "total_tokens_M": 514.3, "training_seconds": 300.1, "val_bpb": 0.993324}
prior_hypothesis: Reducing short attention windows from 1024 to 512 tokens will beat 0.995558 val_bpb by increasing token throughput while the full-attention layers at depths 4 and 8 preserve long-context integration.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing short attention windows from 1024 to 512 tokens will beat 0.995558 val_bpb by increasing token throughput while the full-attention layers at depths 4 and 8 preserve long-context integration.
change: Set short-window layers to one quarter of the 2048-token context; retain the existing SSSL pattern and final full-attention layer.
mechanism: Quarter-context local attention with periodic global mixing
evidence_used: The starting design reached 0.995558 val_bpb on 497.0M tokens at 39.58% MFU; six of eight layers currently spend attention compute on 1024-token windows despite global mixing occurring every fourth layer.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.75, "num_params_M": 50.3, "num_steps": 981.0, "peak_vram_mb": 45060.2, "total_tokens_M": 514.3, "training_seconds": 300.1, "val_bpb": 0.993324}

RECENT RESULT
hypothesis: Halving short-attention windows from 512 to 256 tokens will beat 0.993324 val_bpb by increasing token throughput, while full-attention layers 4 and 8 preserve long-range integration.
change: Set S layers to one eighth of the 2048-token context while retaining the SSSL pattern and final full-attention layer.
mechanism: Eighth-context local attention with periodic global mixing
evidence_used: Reducing short windows from 1024 to 512 increased training tokens from 497.0M to 514.3M and improved val_bpb from 0.995558 to 0.993324, motivating the next adjacent window-size ablation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 33.92, "num_params_M": 50.3, "num_steps": 922.0, "peak_vram_mb": 45060.2, "total_tokens_M": 483.4, "training_seconds": 300.2, "val_bpb": 0.998036}

RECENT RESULT
hypothesis: Retaining 512-token local windows but reducing full-attention layers from two to one will beat 0.993324 val_bpb by increasing throughput while seven stacked local layers and the final global layer preserve full-context access.
change: Change the attention pattern so layers 1–7 use 512-token windows and only layer 8 uses full attention.
mechanism: Single terminal global-attention layer
evidence_used: The 512-token design improved val_bpb to 0.993324 with 514.3M tokens, while shrinking windows to 256 reduced throughput and regressed to 0.998036; this tests a different way to remove attention compute without leaving the demonstrated 512-token operating point.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.85, "num_params_M": 50.3, "num_steps": 866.0, "peak_vram_mb": 45060.2, "total_tokens_M": 454.0, "training_seconds": 300.1, "val_bpb": 1.002549}

RECENT RESULT
hypothesis: Using full attention at layers 3, 6, and 8 while retaining 512-token windows elsewhere will beat 0.993324 val_bpb by improving long-range mixing without entering the inefficient 256-token regime.
change: Change the repeating attention pattern from SSSL to SSL, increasing full-attention layers from two to three.
mechanism: Three-stage periodic global attention
evidence_used: Reducing full-attention layers from two to one regressed val_bpb from 0.993324 to 1.002549 and reduced training tokens from 514.3M to 454.0M, motivating the adjacent test in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 30.88, "num_params_M": 50.3, "num_steps": 772.0, "peak_vram_mb": 45060.2, "total_tokens_M": 404.8, "training_seconds": 300.3, "val_bpb": 1.009019}



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
