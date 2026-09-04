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
verified_results: {"depth": 8.0, "mfu_percent": 37.91, "num_params_M": 50.3, "num_steps": 908.0, "peak_vram_mb": 46348.5, "total_tokens_M": 476.1, "training_seconds": 300.0, "val_bpb": 0.992015}
prior_hypothesis: Optimizing only the four output-gate matrices with AdamW at 0.0075 on the proven 704/1024/1344 backbone will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991835}
prior_hypothesis: Reallocating the uniform short-layer attention budget from 1024/1024/1024 to 768/1024/1280 tokens before each full-context layer will lower val_bpb below 0.992110 while retaining at least 470M training tokens.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991177}
prior_hypothesis: AdamW at 0.0115 for only the four output-gate matrices will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 38.08, "num_params_M": 50.3, "num_steps": 912.0, "peak_vram_mb": 46348.5, "total_tokens_M": 478.2, "training_seconds": 300.0, "val_bpb": 0.991333}
prior_hypothesis: Restoring the output-gate AdamW learning rate to 0.0115 will lower val_bpb below 0.991889 while retaining at least 470M trained tokens.

## Recent verification evidence

RECENT RESULT
hypothesis: A 704/1024/1344-token progression will lower val_bpb below 0.991835 while retaining at least 475M trained tokens.
change: Restore the proven fixed two-lag embedding FIR and four handoff gates, then use the untested midpoint between the 768/1024/1280 and 640/1024/1408 short-window schedules.
mechanism: Fine-grained compute-neutral staged attention
evidence_used: The 768/1024/1280 schedule achieved the best val_bpb of 0.991835, while nearby 640/1024/1408 reached 0.991994 and more distant schedules regressed; their compute-neutral midpoint provides the most informative local refinement.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.1, "val_bpb": 0.991729}

RECENT RESULT
hypothesis: A 720/1024/1328-token progression will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.
change: Shift 16 tokens of context from the final short layer to the first short layer, preserving the total short-attention FLOPs and all other proven components.
mechanism: Quadratically refined staged attention
evidence_used: The current 704/1024/1344 schedule achieved 0.991729, outperforming the equally spaced neighboring schedules at 640/1024/1408 (0.991994) and 768/1024/1280 (0.991835); quadratic interpolation of these results places the local optimum near 718 tokens for the first window.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.62, "num_params_M": 50.3, "num_steps": 901.0, "peak_vram_mb": 46348.5, "total_tokens_M": 472.4, "training_seconds": 300.0, "val_bpb": 0.992742}

RECENT RESULT
hypothesis: Using the steeper 640/1024/1408 schedule for the first short-to-long handoff and the milder 768/1024/1280 schedule for the second will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.
change: Replace the repeated short-window schedule with complementary, compute-neutral schedules that allocate more predecessor context to the empirically more informative first handoff.
mechanism: First-handoff-biased depth-adaptive staged attention
evidence_used: Repeating the midpoint 704/1024/1344 schedule achieved the best val_bpb of 0.991729, while its aligned neighbors 640/1024/1408 and 768/1024/1280 reached 0.991994 and 0.991835; using those neighbors in separate depth cycles preserves the midpoint’s aggregate attention budget while testing the layer dependence suggested by the stronger first-handoff gating evidence.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.95, "num_params_M": 50.3, "num_steps": 909.0, "peak_vram_mb": 46348.5, "total_tokens_M": 476.6, "training_seconds": 300.1, "val_bpb": 0.992249}

RECENT RESULT
hypothesis: Using the milder 768/1024/1280 schedule for the first handoff and the steeper 640/1024/1408 schedule for the second will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.
change: Replace the repeated 640/1024/1408 schedule with complementary compute-neutral schedules, assigning greater early-layer context to the first cycle and greater immediate-predecessor context to the second.
mechanism: Second-handoff-biased depth-adaptive staged attention
evidence_used: Repeated 704/1024/1344 achieved 0.991729, while assigning the steeper neighboring schedule to the first cycle regressed to 0.992249; reversing that depth allocation directly tests whether the second handoff benefits more from predecessor-focused context.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.06, "num_params_M": 50.3, "num_steps": 912.0, "peak_vram_mb": 46348.5, "total_tokens_M": 478.2, "training_seconds": 300.2, "val_bpb": 0.992112}

RECENT RESULT
hypothesis: Leaving half of each full-context layer’s query/key channels unrotated, while restoring the proven 704/1024/1344 short-window progression, will lower val_bpb below 0.991729 while retaining at least 475M training tokens.
change: Use full RoPE in short-context layers but a mixed positional/content subspace in full-context layers, allowing their learned projections to perform position-independent semantic retrieval without weakening local order modeling.
mechanism: Scale-selective hybrid positional attention
evidence_used: The 704/1024/1344 schedule achieved the best result, 0.991729, while further window refinements regressed. The low-rank prefix-memory path reached only 0.994246 and reduced throughput, motivating global-context representation inside the existing full-attention computation instead of another compressed context path.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.53, "num_params_M": 50.3, "num_steps": 899.0, "peak_vram_mb": 46348.5, "total_tokens_M": 471.3, "training_seconds": 300.1, "val_bpb": 0.992264}

RECENT RESULT
hypothesis: Optimizing the four sigmoid output-gate matrices with AdamW at 0.01 instead of Muon will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.
change: Separate handoff output-gate weights from transformer matrices and optimize them with the same adaptive learning rate used by the successful embedding-context coefficients.
mechanism: Role-aware adaptive optimization for multiplicative gates
evidence_used: Four handoff output gates improved val_bpb to 0.992110, while subsequent context-window refinements plateaued at 0.991729 or regressed; these small multiplicative controllers are a proven lever but are currently orthogonalized as ordinary feature matrices by Muon.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.93, "num_params_M": 50.3, "num_steps": 909.0, "peak_vram_mb": 46348.5, "total_tokens_M": 476.6, "training_seconds": 300.2, "val_bpb": 0.991514}

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
