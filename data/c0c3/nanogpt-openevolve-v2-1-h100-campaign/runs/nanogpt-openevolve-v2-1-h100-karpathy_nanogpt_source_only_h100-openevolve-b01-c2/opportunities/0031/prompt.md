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
verified_results: {"depth": 8.0, "mfu_percent": 34.65, "num_params_M": 50.3, "num_steps": 999.0, "peak_vram_mb": 45060.2, "total_tokens_M": 523.8, "training_seconds": 300.0, "val_bpb": 0.99293}
prior_hypothesis: Seven 224-token local layers followed by the full-context final anchor will retain at least 525M training tokens while improving val_bpb below 0.992854.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 34.75, "num_params_M": 50.3, "num_steps": 1009.0, "peak_vram_mb": 45060.2, "total_tokens_M": 529.0, "training_seconds": 300.1, "val_bpb": 0.993246}
prior_hypothesis: Seven 192-token local-attention layers followed by one full-context layer will preserve the throughput benefit of short windows while avoiding the quality loss observed at 128 tokens, lowering val_bpb below 0.992854.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 34.99, "num_params_M": 50.3, "num_steps": 1002.0, "peak_vram_mb": 45060.2, "total_tokens_M": 525.3, "training_seconds": 300.1, "val_bpb": 0.992854}
prior_hypothesis: Seven 256-token local-attention layers followed by one full-context layer will process at least 516.9M tokens and lower val_bpb below 0.993518.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 35.2, "num_params_M": 50.3, "num_steps": 1008.0, "peak_vram_mb": 45060.2, "total_tokens_M": 528.5, "training_seconds": 300.1, "val_bpb": 0.992358}
prior_hypothesis: Restoring seven 256-token local layers and starting linear warmdown at 40% of training will retain roughly 525M tokens while lowering val_bpb below 0.992854.

## Recent verification evidence

RECENT RESULT
hypothesis: Seven 384-token local-attention layers followed by one full-context layer will process more than 518.5M tokens while retaining full-sequence integration, lowering val_bpb below 0.993870.
change: Reduce local windows from 1024 to 384 tokens and change the layout to seven local layers plus the forced full-context final layer.
mechanism: Sub-quarter local attention with a final global anchor
evidence_used: The 512-token single-anchor design achieved the best result at 0.993870 and 518.5M tokens; removing its final global anchor worsened val_bpb to 1.001821, so this patch preserves that anchor while testing whether cheaper local layers improve fixed-time training.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.36, "num_params_M": 50.3, "num_steps": 986.0, "peak_vram_mb": 45060.2, "total_tokens_M": 516.9, "training_seconds": 300.2, "val_bpb": 0.993518}

RECENT RESULT
hypothesis: Seven 256-token local-attention layers followed by one full-context layer will process at least 516.9M tokens and lower val_bpb below 0.993518.
change: Restore the proven 524K-token batch, use a single forced final full-context anchor, and reduce all preceding local windows from 1024 to 256 tokens.
mechanism: Eighth-context local attention with a final global anchor
evidence_used: Moving from seven 512-token local layers to seven 384-token local layers improved val_bpb from 0.993870 to 0.993518 at comparable throughput, while removing the final global anchor worsened val_bpb to 1.001821; this motivates continuing the locality sweep while preserving that anchor.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 34.99, "num_params_M": 50.3, "num_steps": 1002.0, "peak_vram_mb": 45060.2, "total_tokens_M": 525.3, "training_seconds": 300.1, "val_bpb": 0.992854}

RECENT RESULT
hypothesis: Seven 128-token local-attention layers followed by one full-context layer will process more than 525.3M tokens and lower val_bpb below 0.992854.
change: Halve the seven local-attention windows from 256 to 128 tokens while preserving the proven final full-context anchor, architecture, batch, optimizer, and schedule.
mechanism: Sixteenth-context local attention with a final global anchor
evidence_used: Reducing local windows from 512 to 384 and then 256 tokens improved val_bpb from 0.993870 to 0.993518 to 0.992854; the 256-token design also reached the highest throughput at 525.3M tokens, while removing the final global anchor worsened val_bpb to 1.001821.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.02, "num_params_M": 50.3, "num_steps": 1002.0, "peak_vram_mb": 45060.2, "total_tokens_M": 525.3, "training_seconds": 300.1, "val_bpb": 0.995793}

RECENT RESULT
hypothesis: Seven 192-token local-attention layers followed by one full-context layer will preserve the throughput benefit of short windows while avoiding the quality loss observed at 128 tokens, lowering val_bpb below 0.992854.
change: Reduce the seven local-attention windows from 384 to 192 tokens while preserving the final full-context anchor and all other architecture, optimizer, batching, and schedule settings.
mechanism: Bracketed local-attention window interpolation
evidence_used: Validation improved as local windows shrank from 512 to 384 to 256 tokens (0.993870 to 0.993518 to 0.992854), but regressed at 128 tokens to 0.995793; testing 192 tokens directly probes the bracketed optimum between 128 and 256.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 34.75, "num_params_M": 50.3, "num_steps": 1009.0, "peak_vram_mb": 45060.2, "total_tokens_M": 529.0, "training_seconds": 300.1, "val_bpb": 0.993246}

RECENT RESULT
hypothesis: Seven 224-token local layers followed by the full-context final anchor will retain at least 525M training tokens while improving val_bpb below 0.992854.
change: Reduce the current seven 512-token local-attention windows to 224 tokens, preserving the architecture, optimizer, batch size, schedule, and final global anchor.
mechanism: Bracketed 224-token local-attention refinement
evidence_used: The 256-token design achieved the best observed val_bpb of 0.992854, while 192 tokens regressed to 0.993246 despite higher throughput; 224 tokens directly probes the bracketed optimum between them.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 34.65, "num_params_M": 50.3, "num_steps": 999.0, "peak_vram_mb": 45060.2, "total_tokens_M": 523.8, "training_seconds": 300.0, "val_bpb": 0.99293}

RECENT RESULT
hypothesis: Seven 240-token local layers followed by one full-context anchor will retain at least 523.8M training tokens and lower val_bpb below 0.992854.
change: Use a single final global anchor and set the preceding seven local-attention windows to 240 tokens.
mechanism: Bracketed 240-token local-attention refinement
evidence_used: The 256-token design achieved the best val_bpb of 0.992854, while 224 tokens reached 0.992930; testing their midpoint is the most direct refinement of the bracketed optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 34.72, "num_params_M": 50.3, "num_steps": 998.0, "peak_vram_mb": 45060.2, "total_tokens_M": 523.2, "training_seconds": 300.1, "val_bpb": 0.993456}

RECENT RESULT
hypothesis: Redistributing the seven local windows from uniform 256-token attention to progressively expanding 128–512-token windows, while preserving the same 1,792-token aggregate local-attention budget and final global anchor, will retain roughly 525M training tokens and lower val_bpb below 0.992854.
change: Replace seven uniform 256-token windows with 128, 160, 192, 224, 256, 320, and 512-token windows before the unchanged full-context final layer.
mechanism: Compute-neutral progressive local-attention funnel
evidence_used: The 256-token single-anchor design is best at 0.992854, while uniform 128-token attention loses quality and a 512-token pre-anchor bridge remains viable; reallocating rather than increasing attention compute tests whether narrow early layers and broader late layers combine their benefits.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.48, "num_params_M": 50.3, "num_steps": 988.0, "peak_vram_mb": 45060.2, "total_tokens_M": 518.0, "training_seconds": 300.2, "val_bpb": 0.99453}

RECENT RESULT
hypothesis: Seven 272-token local-attention layers followed by the full-context final anchor will retain at least 523M training tokens and lower val_bpb below 0.992854.
change: Increase the seven local-attention windows from 224 to 272 tokens while preserving the architecture, optimizer, batching, schedule, and final global anchor.
mechanism: Above-optimum local-attention window refinement
evidence_used: The 256-token design achieved the best val_bpb of 0.992854, while 224 tokens reached 0.992930 and 384 tokens regressed to 0.993518; 272 tokens is the nearest untested refinement above the observed optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.84, "num_params_M": 50.3, "num_steps": 995.0, "peak_vram_mb": 45060.2, "total_tokens_M": 521.7, "training_seconds": 300.3, "val_bpb": 0.993043}

RECENT RESULT
hypothesis: Restoring seven 256-token local layers and using cosine rather than linear warmdown will retain roughly 525M tokens while lowering val_bpb below 0.992854.
change: Restore the best-performing 256-token windows and replace the linear 50%-window LR decay with an equal-duration cosine decay.
mechanism: Cosine warmdown on the proven 256-token local-attention baseline
evidence_used: The 256-token design achieved the best observed val_bpb, 0.992854; nearby uniform windows and progressive allocation did not improve it, motivating an orthogonal schedule refinement without changing its architecture or batch.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.16, "num_params_M": 50.3, "num_steps": 1007.0, "peak_vram_mb": 45060.2, "total_tokens_M": 528.0, "training_seconds": 300.1, "val_bpb": 0.995509}

RECENT RESULT
hypothesis: Restoring seven 256-token local layers and starting linear warmdown at 40% of training will retain roughly 525M tokens while lowering val_bpb below 0.992854.
change: Restore the best-performing 256-token local windows and extend linear warmdown from 50% to 60% of the training window.
mechanism: Earlier linear warmdown on the proven 256-token attention baseline
evidence_used: The 256-token design achieved the best val_bpb, 0.992854; equal-duration cosine warmdown regressed to 0.995509, motivating a conservative refinement that preserves linear decay but reduces learning rates earlier.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.2, "num_params_M": 50.3, "num_steps": 1008.0, "peak_vram_mb": 45060.2, "total_tokens_M": 528.5, "training_seconds": 300.1, "val_bpb": 0.992358}

RECENT RESULT
hypothesis: Extending linear warmdown from 60% to 70% of training will retain roughly 528M tokens while lowering val_bpb below 0.992358.
change: Preserve the current architecture, optimizer, and batching while starting linear LR decay at 30% instead of 40% of the training window.
mechanism: Earlier linear warmdown on the proven 256-token attention baseline
evidence_used: Extending linear warmdown from 50% to 60% improved val_bpb from 0.992854 to 0.992358 without reducing throughput, whereas cosine decay regressed to 0.995509; this directly tests whether the beneficial earlier linear decay trend continues.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.11, "num_params_M": 50.3, "num_steps": 1006.0, "peak_vram_mb": 45060.2, "total_tokens_M": 527.4, "training_seconds": 300.2, "val_bpb": 0.993193}

RECENT RESULT
hypothesis: Starting linear warmdown at 35% of training will retain roughly 528M tokens and lower val_bpb below 0.992358.
change: Extend linear LR warmdown from 50% to 65% of the training window while preserving the proven 256-token attention architecture, optimizer, and batching.
mechanism: Bracketed linear warmdown refinement
evidence_used: A 60% linear warmdown achieved the best observed val_bpb of 0.992358, improving over 50% at 0.992854, while 70% regressed to 0.993193; 65% directly probes the bracketed schedule optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.11, "num_params_M": 50.3, "num_steps": 1006.0, "peak_vram_mb": 45060.2, "total_tokens_M": 527.4, "training_seconds": 300.3, "val_bpb": 0.992872}



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
