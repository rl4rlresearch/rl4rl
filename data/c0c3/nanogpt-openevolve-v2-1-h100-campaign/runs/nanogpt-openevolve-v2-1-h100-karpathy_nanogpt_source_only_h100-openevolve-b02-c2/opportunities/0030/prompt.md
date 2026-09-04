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
verified_results: {"depth": 8.0, "mfu_percent": 35.63, "num_params_M": 50.3, "num_steps": 1969.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.2, "training_seconds": 300.1, "val_bpb": 0.983766}
prior_hypothesis: Reducing the six local-attention layers from 256 to 128 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.983993 by increasing token throughput enough to offset the reduced local receptive field.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 35.87, "num_params_M": 50.3, "num_steps": 1960.0, "peak_vram_mb": 44908.2, "total_tokens_M": 513.8, "training_seconds": 300.1, "val_bpb": 0.984182}
prior_hypothesis: Using 192-token local windows will lower val_bpb below 0.983766 by retaining more useful local context than the unsuccessful 64-token design while remaining faster than the 256-token design.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983317}
prior_hypothesis: Using 136-token local windows will lower val_bpb below 0.983758 by moving toward the approximately 138-token minimum implied by the measured 128/144/192-token bracket while slightly reducing attention compute.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 37.33, "num_params_M": 50.3, "num_steps": 1930.0, "peak_vram_mb": 44908.2, "total_tokens_M": 505.9, "training_seconds": 300.0, "val_bpb": 0.984125}
prior_hypothesis: Reducing the six local-attention layers from 768 to 512 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.984868 by increasing token throughput without removing the proven two-layer global-context path.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the best 50% linear warmdown and shortening Muon’s momentum ramp from 300 to 150 updates will lower val_bpb below 0.985746 by reaching the proven 0.95 momentum earlier in the short run.
change: Restore the best verified LR schedule and halve only the Muon momentum-ramp duration.
mechanism: Faster Muon momentum stabilization
evidence_used: Extending the momentum ramp from 300 to 600 updates worsened val_bpb to 0.988827, while the 300-update configuration was better; warmdown experiments also identify 50% linear decay as the best verified schedule at 0.985746.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.06, "num_params_M": 50.3, "num_steps": 1861.0, "peak_vram_mb": 44908.2, "total_tokens_M": 487.8, "training_seconds": 300.2, "val_bpb": 0.986329}

RECENT RESULT
hypothesis: A reverse-cosine 50% warmdown will lower val_bpb below 0.985746 by preserving the linear schedule’s total learning-rate budget while shifting updates opposite to the unsuccessful cosine schedule—smaller early-warmdown updates and larger late-warmdown refinement updates.
change: Restore the best verified 50% warmdown start and replace linear interpolation with a monotonic reverse-cosine curve.
mechanism: Reverse-cosine learning-rate redistribution
evidence_used: The 50% linear warmdown achieved 0.985746, while moving its start to 40%, 55%, or 60% did not improve it and a same-window cosine warmdown worsened val_bpb to 0.989121; this motivates holding the best window and testing the opposite within-window redistribution.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.09, "num_params_M": 50.3, "num_steps": 1862.0, "peak_vram_mb": 44908.2, "total_tokens_M": 488.1, "training_seconds": 300.1, "val_bpb": 0.986351}

RECENT RESULT
hypothesis: Keeping only the final layer full-context will lower val_bpb below 0.985746 by increasing token throughput while preserving one layer that can integrate the entire sequence.
change: Restore the best verified Muon weight decay of 0.1 and change the attention pattern from two full-context layers to one final full-context layer.
mechanism: Single global-attention consolidation
evidence_used: Weight decay 0.1 with the 50% linear warmdown achieved the best val_bpb of 0.985746, while all tested decay and warmdown refinements were worse; this motivates restoring that baseline and testing compute reallocation through the untested window pattern.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.38, "num_params_M": 50.3, "num_steps": 1877.0, "peak_vram_mb": 44908.2, "total_tokens_M": 492.0, "training_seconds": 300.0, "val_bpb": 0.986118}

RECENT RESULT
hypothesis: Adding a third full-context layer at layer 6 will lower val_bpb below 0.985746 because removing one of the baseline’s two global layers worsened validation to 0.986118 despite slightly higher throughput.
change: Preserve the best optimizer and schedule while changing the eight-layer attention pattern from two to three full-context layers, located at layers 4, 6, and 8.
mechanism: Additional mid-depth global-attention integration
evidence_used: The two-global-layer baseline achieved 0.985746, while the otherwise matched single-global-layer design achieved 0.986118; this indicates global-context capacity was more valuable than its small throughput cost and motivates testing one additional global layer.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.76, "num_params_M": 50.3, "num_steps": 1846.0, "peak_vram_mb": 44908.2, "total_tokens_M": 483.9, "training_seconds": 300.1, "val_bpb": 0.986305}

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
