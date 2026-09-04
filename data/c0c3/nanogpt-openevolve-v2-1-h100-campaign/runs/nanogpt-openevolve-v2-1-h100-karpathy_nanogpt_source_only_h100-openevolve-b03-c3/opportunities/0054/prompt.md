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
verified_results: {"depth": 8.0, "mfu_percent": 38.08, "num_params_M": 50.3, "num_steps": 912.0, "peak_vram_mb": 46348.5, "total_tokens_M": 478.2, "training_seconds": 300.0, "val_bpb": 0.991333}
prior_hypothesis: Restoring the output-gate AdamW learning rate to 0.0115 will lower val_bpb below 0.991889 while retaining at least 470M trained tokens.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991835}
prior_hypothesis: Reallocating the uniform short-layer attention budget from 1024/1024/1024 to 768/1024/1280 tokens before each full-context layer will lower val_bpb below 0.992110 while retaining at least 470M training tokens.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991177}
prior_hypothesis: AdamW at 0.0115 for only the four output-gate matrices will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 37.74, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 46348.5, "total_tokens_M": 474.0, "training_seconds": 300.1, "val_bpb": 0.991917}
prior_hypothesis: Using AdamW at 0.0115 for full-context output gates and 0.0100 for pre-handoff short-context gates will lower val_bpb below 0.991177 while retaining at least 470M training tokens.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the successful AdamW treatment from output gates to value-residual gates will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.
change: Restore the best 704/1024/1344 staged-attention backbone and optimize all eight sigmoid gate matrices with AdamW at 0.01 instead of Muon.
mechanism: Role-aware adaptive optimization for all multiplicative gates
evidence_used: On the 704/1024/1344 design, moving output-gate matrices from Muon to AdamW improved val_bpb from 0.991729 to 0.991514; value-residual gates are equally small multiplicative controllers and remain optimized by Muon.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.95, "num_params_M": 50.3, "num_steps": 909.0, "peak_vram_mb": 46348.5, "total_tokens_M": 476.6, "training_seconds": 300.1, "val_bpb": 0.992169}

RECENT RESULT
hypothesis: Optimizing only the four output-gate matrices with AdamW at 0.015 on the proven 704/1024/1344 backbone will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.
change: Restore the best staged-attention schedule, exclude output gates from Muon, and raise their dedicated AdamW learning rate from the previously successful 0.01 to 0.015.
mechanism: Faster adaptive learning for handoff output gates
evidence_used: Moving output gates to AdamW at 0.01 improved val_bpb from 0.991729 to 0.991514, whereas moving all gate matrices to AdamW regressed to 0.992169; this motivates a focused output-gate learning-rate refinement without changing value-gate optimization.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.93, "num_params_M": 50.3, "num_steps": 909.0, "peak_vram_mb": 46348.5, "total_tokens_M": 476.6, "training_seconds": 300.2, "val_bpb": 0.991889}

RECENT RESULT
hypothesis: Optimizing only the four output-gate matrices with AdamW at 0.0075 on the proven 704/1024/1344 backbone will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.
change: Restore the best staged short-attention schedule, exclude output gates from Muon, and assign those gates a dedicated AdamW learning rate of 0.0075.
mechanism: Conservative adaptive optimization for handoff output gates
evidence_used: AdamW at 0.01 improved output-gated 704/1024/1344 from 0.991729 to the best observed 0.991514, while increasing it to 0.015 regressed to 0.991889; testing 0.0075 probes the more promising lower-learning-rate side while preserving the proven focus on output gates alone.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.91, "num_params_M": 50.3, "num_steps": 908.0, "peak_vram_mb": 46348.5, "total_tokens_M": 476.1, "training_seconds": 300.0, "val_bpb": 0.992015}

RECENT RESULT
hypothesis: AdamW at 0.0115 for only the four output-gate matrices will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.
change: Raise the dedicated output-gate AdamW learning rate from 0.0100 to 0.0115 without changing the proven 704/1024/1344 backbone or other optimizer groups.
mechanism: Quadratically refined output-gate learning rate
evidence_used: Output-gate AdamW achieved 0.991514 at 0.0100, versus 0.992015 at 0.0075 and 0.991889 at 0.0150; quadratic interpolation of these bracketing results places the estimated minimum near 0.0115.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991177}

RECENT RESULT
hypothesis: AdamW at 0.012 for only the four output-gate matrices on the proven 704/1024/1344 backbone will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.
change: Restore the best staged-attention schedule, exclude output-gate matrices from Muon, and optimize them with dedicated AdamW at 0.012.
mechanism: Locally refined adaptive output-gate optimization
evidence_used: Output-gate AdamW at 0.0115 achieved the best observed val_bpb of 0.991177, versus 0.991514 at 0.010 and 0.991889 at 0.015; local quadratic interpolation of these three results places the estimated minimum near 0.0121.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.72, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 46348.5, "total_tokens_M": 474.0, "training_seconds": 300.2, "val_bpb": 0.992081}

RECENT RESULT
hypothesis: Restoring the output-gate AdamW learning rate to 0.0115 will lower val_bpb below 0.991889 while retaining at least 470M trained tokens.
change: Reduce the dedicated learning rate for the four output-gate matrices from 0.015 to the best-verified 0.0115, preserving the 704/1024/1344 backbone and all other optimizer groups.
mechanism: Best-verified adaptive output-gate optimization
evidence_used: Output-gate AdamW at 0.0115 achieved the best observed val_bpb of 0.991177 over 480.2M tokens, outperforming both the current 0.015 setting at 0.991889 and 0.010 at 0.991514.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.08, "num_params_M": 50.3, "num_steps": 912.0, "peak_vram_mb": 46348.5, "total_tokens_M": 478.2, "training_seconds": 300.0, "val_bpb": 0.991333}

RECENT RESULT
hypothesis: AdamW at 0.01125 for only the four output-gate matrices will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.
change: Raise the dedicated output-gate AdamW learning rate from 0.0075 to the untested 0.01125 midpoint between 0.0100 and the best-verified 0.0115, preserving the 704/1024/1344 backbone and all other optimizer groups.
mechanism: Lower-flank output-gate learning-rate refinement
evidence_used: Output-gate AdamW at 0.0115 produced the two best results, 0.991177 and 0.991333, while 0.0100 reached 0.991514 and 0.0120 regressed; testing 0.01125 probes the narrow lower side of the apparent optimum without repeating the unsuccessful higher rate.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.7, "num_params_M": 50.3, "num_steps": 903.0, "peak_vram_mb": 46348.5, "total_tokens_M": 473.4, "training_seconds": 300.0, "val_bpb": 0.991926}

RECENT RESULT
hypothesis: Adding zero-initialized per-head biases to the four output gates will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.
change: Give each output gate an input-independent headwise scale parameter and optimize its bias alongside its weight with the best-verified 0.0115 AdamW learning rate.
mechanism: AdamW-controlled headwise attention-branch calibration
evidence_used: Moving only output-gate weights to AdamW at 0.0115 achieved the best val_bpb, 0.991177, while moving all gate matrices to AdamW regressed; this motivates expanding adaptive control specifically within the successful output gates.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.86, "num_params_M": 50.3, "num_steps": 907.0, "peak_vram_mb": 46348.5, "total_tokens_M": 475.5, "training_seconds": 300.2, "val_bpb": 0.991861}

RECENT RESULT
hypothesis: Replacing the single-branch ReLU² MLPs with 64-aligned, nearly parameter-neutral SwiGLU blocks will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.
change: Keep the proven 704/1024/1344 attention backbone and 0.0115 output-gate AdamW rate, but factor every MLP expansion into learned gate and value streams whose multiplicative interaction controls features at each token.
mechanism: Parameter-neutral SwiGLU feature-selective prediction blocks
evidence_used: Window-allocation and output-gate learning-rate refinements plateaued around the 0.991177 best result, while input-dependent multiplicative output gating was the clearest successful architectural lever. This challenges the shared assumption that an ungated ReLU² feature expansion is sufficient, without increasing the attention budget; the 1344-wide hidden dimension is within 1.6% of iso-parameter SwiGLU sizing and remains tensor-core aligned.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.04, "num_params_M": 50.1, "num_steps": 870.0, "peak_vram_mb": 46093.5, "total_tokens_M": 456.1, "training_seconds": 300.3, "val_bpb": 0.997007}

RECENT RESULT
hypothesis: Conditioning the four output gates on all residual channels with best-verified AdamW at 0.0115 will lower val_bpb below 0.991177 while retaining at least 470M training tokens.
change: Restore the proven 704/1024/1344 attention schedule and dedicated output-gate AdamW group, while widening each output gate’s input from the first 32 channels to the full 512-channel residual stream.
mechanism: Full-residual input-dependent output gating
evidence_used: Output-gate AdamW at 0.0115 on the 704/1024/1344 backbone achieved the best val_bpb, 0.991177; adding input-independent gate biases regressed to 0.991861, motivating richer input-dependent control instead.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.71, "num_params_M": 50.3, "num_steps": 903.0, "peak_vram_mb": 46348.6, "total_tokens_M": 473.4, "training_seconds": 300.0, "val_bpb": 0.996952}

RECENT RESULT
hypothesis: Using AdamW at 0.0115 for full-context output gates and 0.0100 for pre-handoff short-context gates will lower val_bpb below 0.991177 while retaining at least 470M training tokens.
change: Split the four output-gate matrices into full-context and pre-handoff optimizer groups, preserving the best verified rate for global branches while applying the proven conservative rate to short-context branches.
mechanism: Context-scale-specific output-gate optimization
evidence_used: Joint output-gate AdamW improved val_bpb to 0.991514 at 0.0100 and the best 0.991177 at 0.0115, while 0.0150 regressed to 0.991889; separating gates by their different attention scales tests whether the shared optimum masks role-specific sensitivity without changing compute.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.74, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 46348.5, "total_tokens_M": 474.0, "training_seconds": 300.1, "val_bpb": 0.991917}

RECENT RESULT
hypothesis: AdamW at 0.0100 for full-context output gates and 0.0115 for pre-handoff gates will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.
change: Split output gates by attention role, lowering only the two full-context gates to 0.0100 while preserving 0.0115 for the two pre-handoff gates.
mechanism: Context-scale-specific output-gate learning rates
evidence_used: Joint 0.0115 achieved 0.991177, while lowering only pre-handoff gates to 0.0100 regressed to 0.991917; the complementary split directly tests whether the full-context gates are instead responsible for the shared-rate compromise.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.04, "num_params_M": 50.3, "num_steps": 911.0, "peak_vram_mb": 46348.5, "total_tokens_M": 477.6, "training_seconds": 300.0, "val_bpb": 0.991382}



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
