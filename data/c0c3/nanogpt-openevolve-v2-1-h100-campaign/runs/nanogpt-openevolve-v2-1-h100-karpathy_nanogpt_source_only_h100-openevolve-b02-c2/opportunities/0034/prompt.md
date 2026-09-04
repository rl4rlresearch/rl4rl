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
verified_results: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983317}
prior_hypothesis: Using 136-token local windows will lower val_bpb below 0.983758 by moving toward the approximately 138-token minimum implied by the measured 128/144/192-token bracket while slightly reducing attention compute.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 35.64, "num_params_M": 50.3, "num_steps": 1967.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.6, "training_seconds": 300.1, "val_bpb": 0.983641}
prior_hypothesis: Using 128-token windows in the first three local layers and 144-token windows in the final three local layers will lower val_bpb below 0.983317 by allocating more context to higher-level representations while matching the total local-attention compute of the best homogeneous 136-token design.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 35.62, "num_params_M": 50.3, "num_steps": 1965.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.1, "training_seconds": 300.1, "val_bpb": 0.983933}
prior_hypothesis: Using 152-token windows in the first three local layers and 120-token windows in the final three will lower val_bpb below 0.983275 by extending the observed advantage of allocating more local context to earlier layers while preserving the six-layer average of 136 tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983275}
prior_hypothesis: Using 144-token windows in the first three local layers and 128-token windows in the final three will lower val_bpb below 0.983317 by favoring early context formation while preserving the aggregate attention compute of uniform 136-token windows.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing short attention windows from 1024 to 768 tokens while retaining full-context layers 4 and 8 will increase throughput enough to lower val_bpb below 0.985746 without materially reducing usable context.
change: Restore the best verified 50% linear warmdown and reduce only the six local-attention layers to three-eighths context.
mechanism: Narrower local-attention windows with preserved global integration
evidence_used: Two full-context layers achieved 0.985746, outperforming one full layer at 0.986118 and three full layers at 0.986305; this motivates preserving the proven global-attention count while reducing local-attention compute.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.16, "num_params_M": 50.3, "num_steps": 1892.0, "peak_vram_mb": 44908.2, "total_tokens_M": 496.0, "training_seconds": 300.1, "val_bpb": 0.984868}

RECENT RESULT
hypothesis: Reducing the six local-attention layers from 768 to 512 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.984868 by increasing token throughput without removing the proven two-layer global-context path.
change: Restore the best two-global-layer SSSL pattern and halve local attention windows to one-quarter context.
mechanism: Deeper local-window compression with preserved global integration
evidence_used: Reducing local windows from 1024 to 768 while preserving full-context layers 4 and 8 improved val_bpb from 0.985746 to 0.984868 and raised processed tokens from 491.3M to 496.0M; this motivates testing the next 256-token reduction.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.33, "num_params_M": 50.3, "num_steps": 1930.0, "peak_vram_mb": 44908.2, "total_tokens_M": 505.9, "training_seconds": 300.0, "val_bpb": 0.984125}

RECENT RESULT
hypothesis: Reducing the six local-attention layers from 512 to 256 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.984125 by increasing token throughput without eliminating the proven global-context path.
change: Restore the best verified 50% linear warmdown and change short attention windows to one-eighth of the 2048-token context.
mechanism: Further local-window compression with preserved global integration
evidence_used: Successive local-window reductions from 1024 to 768 and then 512 tokens improved val_bpb from 0.985746 to 0.984868 and 0.984125 while increasing processed tokens from 491.3M to 496.0M and 505.9M.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.43, "num_params_M": 50.3, "num_steps": 1968.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.9, "training_seconds": 300.1, "val_bpb": 0.983993}

RECENT RESULT
hypothesis: Reducing the six local-attention layers from 256 to 128 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.983993 by increasing token throughput enough to offset the reduced local receptive field.
change: Change short attention windows from one-eighth to one-sixteenth of the 2048-token context, preserving the SSSL pattern and all optimizer settings.
mechanism: Aggressive local-window compression with preserved global integration
evidence_used: Successive reductions from 1024 to 768, 512, and 256 tokens improved val_bpb from 0.985746 to 0.984868, 0.984125, and 0.983993 while increasing throughput; testing 128 tokens extends this monotonic compression series and probes its likely boundary.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.63, "num_params_M": 50.3, "num_steps": 1969.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.2, "training_seconds": 300.1, "val_bpb": 0.983766}

RECENT RESULT
hypothesis: Reducing the six local-attention layers from 128 to 64 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.983766 by increasing throughput enough to offset the smaller local receptive field.
change: Set short attention windows to one-thirty-second of the 2048-token context while preserving the proven SSSL pattern and optimizer settings.
mechanism: Extreme local-window compression with preserved global integration
evidence_used: Successive reductions from 1024 to 768, 512, 256, and 128 tokens monotonically improved val_bpb from 0.985746 to 0.984868, 0.984125, 0.983993, and 0.983766; testing 64 tokens directly probes whether that trend continues beyond the current best design.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.62, "num_params_M": 50.3, "num_steps": 1991.0, "peak_vram_mb": 44908.2, "total_tokens_M": 521.9, "training_seconds": 300.1, "val_bpb": 0.984784}

RECENT RESULT
hypothesis: Using 192-token local windows will lower val_bpb below 0.983766 by retaining more useful local context than the unsuccessful 64-token design while remaining faster than the 256-token design.
change: Set the six short-attention layers to 192 tokens while preserving full-context layers 4 and 8 and all optimizer settings.
mechanism: Bracketed local-attention window refinement
evidence_used: The best result used 128-token windows at 0.983766; 64 tokens regressed sharply to 0.984784, while 256 tokens was only slightly worse at 0.983993, suggesting the optimum lies above 128 and motivating a hardware-aligned 192-token test.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.87, "num_params_M": 50.3, "num_steps": 1960.0, "peak_vram_mb": 44908.2, "total_tokens_M": 513.8, "training_seconds": 300.1, "val_bpb": 0.984182}

RECENT RESULT
hypothesis: Using 144-token local windows will lower val_bpb below 0.983766 by adding context to the best 128-token design while remaining near its throughput, consistent with the measured minimum implied by the 64/128/192-token bracket.
change: Set the six short-attention layers to 144 tokens while preserving full-context layers 4 and 8 and all optimizer settings.
mechanism: Quadratic-bracket local-window refinement
evidence_used: Local windows of 64, 128, and 192 tokens produced val_bpb values of 0.984784, 0.983766, and 0.984182; quadratic interpolation of this asymmetric bracket places the estimated minimum near 141 tokens, motivating an aligned 144-token test.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.77, "num_params_M": 50.3, "num_steps": 1971.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.7, "training_seconds": 300.1, "val_bpb": 0.983758}

RECENT RESULT
hypothesis: Using 136-token local windows will lower val_bpb below 0.983758 by moving toward the approximately 138-token minimum implied by the measured 128/144/192-token bracket while slightly reducing attention compute.
change: Reduce all six short-attention layers from 144 to 136 tokens, preserving full-context layers 4 and 8 and all optimizer settings.
mechanism: Fine-grained local-window interpolation
evidence_used: Windows of 128, 144, and 192 tokens achieved val_bpb values of 0.983766, 0.983758, and 0.984182; quadratic interpolation of this tighter bracket places its minimum near 138 tokens, motivating the nearest 8-token-aligned setting.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983317}

RECENT RESULT
hypothesis: Using 128-token windows in the first three local layers and 144-token windows in the final three local layers will lower val_bpb below 0.983317 by allocating more context to higher-level representations while matching the total local-attention compute of the best homogeneous 136-token design.
change: Replace uniform 128-token local attention with 128-token early windows and 144-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.
mechanism: Depth-progressive local-context allocation
evidence_used: Uniform 136-token windows achieved the best val_bpb of 0.983317, outperforming both 128 tokens at 0.983766 and 144 tokens at 0.983758; a 3×128 plus 3×144 schedule preserves exactly the same aggregate local-window budget as 6×136 while testing whether context is more valuable at greater depth.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.64, "num_params_M": 50.3, "num_steps": 1967.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.6, "training_seconds": 300.1, "val_bpb": 0.983641}

RECENT RESULT
hypothesis: Using 144-token windows in the first three local layers and 128-token windows in the final three will lower val_bpb below 0.983317 by favoring early context formation while preserving the aggregate attention compute of uniform 136-token windows.
change: Replace uniform 512-token local attention with 144-token early and 128-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.
mechanism: Reverse depth-progressive local-context allocation
evidence_used: Uniform 136-token windows achieved 0.983317, while the compute-matched 128-early/144-late allocation regressed to 0.983641; reversing that allocation directly tests whether local context is more valuable in earlier layers.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983275}

RECENT RESULT
hypothesis: Using 152-token windows in the first three local layers and 120-token windows in the final three will lower val_bpb below 0.983275 by extending the observed advantage of allocating more local context to earlier layers while preserving the six-layer average of 136 tokens.
change: Replace uniform 192-token local attention with 152-token early and 120-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.
mechanism: Compute-matched early-context reallocation
evidence_used: The compute-matched 144-early/128-late design achieved the best result, 0.983275, outperforming uniform 136 at 0.983317 and the reversed 128-early/144-late allocation at 0.983641; increasing the same early-layer bias tests whether that directional improvement continues.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.62, "num_params_M": 50.3, "num_steps": 1965.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.1, "training_seconds": 300.1, "val_bpb": 0.983933}

RECENT RESULT
hypothesis: Compute-matched 140-token early and 132-token late windows will lower val_bpb below 0.983275 by placing the depth bias near the quadratic minimum implied by the uniform, moderate-bias, and strong-bias results.
change: Reduce the early/late local-window contrast from 144/128 to 140/132 while preserving the 136-token average, full-context layers 4 and 8, and all optimizer settings.
mechanism: Quadratic refinement of depth-biased local context
evidence_used: Uniform 136/136 achieved 0.983317, moderate 144/128 improved to 0.983275, and stronger 152/120 regressed to 0.983933; quadratic interpolation places the estimated optimum near 140.5/131.5.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.53, "num_params_M": 50.3, "num_steps": 1960.0, "peak_vram_mb": 44908.2, "total_tokens_M": 513.8, "training_seconds": 300.1, "val_bpb": 0.984113}



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
