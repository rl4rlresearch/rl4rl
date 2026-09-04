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
verified_results: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.1, "val_bpb": 0.991729}
prior_hypothesis: A 704/1024/1344-token progression will lower val_bpb below 0.991835 while retaining at least 475M trained tokens.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991835}
prior_hypothesis: Reallocating the uniform short-layer attention budget from 1024/1024/1024 to 768/1024/1280 tokens before each full-context layer will lower val_bpb below 0.992110 while retaining at least 470M training tokens.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 38.0, "num_params_M": 50.3, "num_steps": 910.0, "peak_vram_mb": 46348.5, "total_tokens_M": 477.1, "training_seconds": 300.0, "val_bpb": 0.99211}
prior_hypothesis: Combining the learned two-lag embedding convolution with output gates on each full-context layer and its immediately preceding short-context layer will lower val_bpb below 0.992628 while processing more than 467.1M tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 38.19, "num_params_M": 50.3, "num_steps": 915.0, "peak_vram_mb": 46348.5, "total_tokens_M": 479.7, "training_seconds": 300.2, "val_bpb": 0.991994}
prior_hypothesis: A 640/1024/1408-token progression will lower val_bpb below 0.991835 while retaining at least 475M training tokens.

## Recent verification evidence

RECENT RESULT
hypothesis: Initializing the proven two-lag FIR with small decaying coefficients will lower val_bpb below 0.992110 while retaining at least 470M tokens by exposing useful local context immediately instead of learning both lag paths from zero.
change: Preserve the best four handoff gates and two-lag architecture, but initialize the one-token and two-token embedding coefficients to 0.10 and 0.05 respectively.
mechanism: Decayed causal-context warm start
evidence_used: Fixed two-lag mixing contributed to the best 0.992110 result, whereas a third lag regressed to 0.992455 and token-conditioned scaling reached only 0.992360; this motivates accelerating the established fixed two-lag path without adding computation or parameters.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.2, "num_params_M": 50.3, "num_steps": 915.0, "peak_vram_mb": 46348.5, "total_tokens_M": 479.7, "training_seconds": 300.1, "val_bpb": 0.992459}

RECENT RESULT
hypothesis: Replacing token-conditioned lag scaling with a learned 32-dimensional summary of all preceding embeddings, while restoring the best fixed two-lag FIR and four handoff gates, will lower val_bpb below 0.992110 while retaining at least 465M trained tokens.
change: Restore fixed two-lag embedding mixing and add a neutral-initialized low-rank nonlinear projection of the causal running-mean embedding history before the transformer.
mechanism: Low-rank causal prefix memory
evidence_used: Fixed two-lag mixing with four handoff gates achieved 0.992110, while a third lag and token-conditioned lag gates regressed to 0.992455 and 0.992360. This challenges the shared assumption that additional cheap context must be another local lag: the quarter-context regression shows broader history matters, so a compressed global prefix memory tests a distinct recurrent-style context path.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.36, "num_params_M": 50.4, "num_steps": 895.0, "peak_vram_mb": 47406.0, "total_tokens_M": 469.2, "training_seconds": 300.3, "val_bpb": 0.994246}

RECENT RESULT
hypothesis: Reallocating the uniform short-layer attention budget from 1024/1024/1024 to 768/1024/1280 tokens before each full-context layer will lower val_bpb below 0.992110 while retaining at least 470M training tokens.
change: Restore the best four handoff-layer output gates and progressively expand short-context windows toward each full-context transition without changing total attention-window FLOPs.
mechanism: Compute-neutral staged short-context receptive fields
evidence_used: Four handoff gates achieved the best 0.992110 on 477.1M tokens, and their gain over long-only gating shows predecessor layers matter; the severe regression from uniformly quartering context motivates reallocating rather than reducing the total context budget.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991835}

RECENT RESULT
hypothesis: Using 512/1024/1536-token short windows before each full-context layer, together with the proven four handoff gates, will lower val_bpb below 0.991835 while retaining at least 480M training tokens.
change: Restore output gating on full-context layers and their immediate predecessors, then redistribute each three-layer short-attention budget from 1024/1024/1024 to 512/1024/1536 tokens.
mechanism: Steeper compute-neutral staged short-context attention
evidence_used: The milder 768/1024/1280 progression with four handoff gates achieved the best result, 0.991835 on 480.2M tokens, improving over uniform windows at 0.992110; a steeper progression tests whether concentrating still more context at the handoff extends that gain without reducing total window budget.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.05, "num_params_M": 50.3, "num_steps": 912.0, "peak_vram_mb": 46348.5, "total_tokens_M": 478.2, "training_seconds": 300.3, "val_bpb": 0.992198}

RECENT RESULT
hypothesis: Redistributing the middle short layer’s context to the immediate pre-long layer, while preserving the successful 768-token first window, will lower val_bpb below 0.991835 while retaining at least 475M training tokens.
change: Change each compute-neutral short-window schedule from 768/1024/1280 to 768/896/1408 tokens.
mechanism: Asymmetric predecessor-focused context allocation
evidence_used: The 768/1024/1280 schedule achieved the best val_bpb of 0.991835, while 512/1024/1536 regressed to 0.992198; holding the first window at 768 isolates whether additional pre-long context helps when the earliest layer is not starved.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.73, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 46348.5, "total_tokens_M": 474.0, "training_seconds": 300.2, "val_bpb": 0.992861}

RECENT RESULT
hypothesis: A 832/1024/1216-token progression will lower val_bpb below 0.991835 while retaining at least 475M training tokens.
change: Redistribute each three-layer short-attention budget from uniform 1024-token windows to a milder staged progression without changing total attention-window FLOPs.
mechanism: Refined compute-neutral staged short-context attention
evidence_used: The 768/1024/1280 schedule achieved the best val_bpb, 0.991835, while uniform windows reached 0.992110 and the steeper 512/1024/1536 schedule regressed to 0.992198; refining the successful slope tests the apparent optimum between uniform and overly steep staging.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.91, "num_params_M": 50.3, "num_steps": 908.0, "peak_vram_mb": 46348.5, "total_tokens_M": 476.1, "training_seconds": 300.1, "val_bpb": 0.992309}

RECENT RESULT
hypothesis: A 640/1024/1408-token progression will lower val_bpb below 0.991835 while retaining at least 475M training tokens.
change: Replace the overly steep 512/1024/1536 schedule with the midpoint between it and the best 768/1024/1280 schedule, preserving total short-attention FLOPs.
mechanism: Mid-slope compute-neutral staged attention
evidence_used: The 768/1024/1280 progression achieved the best val_bpb of 0.991835, while 512/1024/1536 regressed to 0.992198; their untested midpoint isolates whether the optimum lies between those slopes while keeping the successful 1024-token middle window.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.19, "num_params_M": 50.3, "num_steps": 915.0, "peak_vram_mb": 46348.5, "total_tokens_M": 479.7, "training_seconds": 300.2, "val_bpb": 0.991994}

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
