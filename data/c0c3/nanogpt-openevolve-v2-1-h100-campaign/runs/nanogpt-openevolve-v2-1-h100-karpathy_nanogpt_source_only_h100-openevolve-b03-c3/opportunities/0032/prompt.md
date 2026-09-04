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
verified_results: {"depth": 8.0, "mfu_percent": 37.84, "num_params_M": 50.3, "num_steps": 907.0, "peak_vram_mb": 45832.5, "total_tokens_M": 475.5, "training_seconds": 300.3, "val_bpb": 0.992949}
prior_hypothesis: Applying the learned two-lag embedding convolution while retaining output gates only in full-context layers will lower val_bpb below 0.992628 and restore training volume to roughly 480M tokens or more.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 38.24, "num_params_M": 50.3, "num_steps": 916.0, "peak_vram_mb": 46348.5, "total_tokens_M": 480.2, "training_seconds": 300.2, "val_bpb": 0.991835}
prior_hypothesis: Reallocating the uniform short-layer attention budget from 1024/1024/1024 to 768/1024/1280 tokens before each full-context layer will lower val_bpb below 0.992110 while retaining at least 470M training tokens.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 37.74, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 46349.5, "total_tokens_M": 474.0, "training_seconds": 300.1, "val_bpb": 0.99236}
prior_hypothesis: Restoring the best two-lag receptive field and dynamically scaling each lag from the current token will lower val_bpb below 0.992110 while retaining at least 470M trained tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 38.0, "num_params_M": 50.3, "num_steps": 910.0, "peak_vram_mb": 46348.5, "total_tokens_M": 477.1, "training_seconds": 300.0, "val_bpb": 0.99211}
prior_hypothesis: Combining the learned two-lag embedding convolution with output gates on each full-context layer and its immediately preceding short-context layer will lower val_bpb below 0.992628 while processing more than 467.1M tokens.

## Recent verification evidence

RECENT RESULT
hypothesis: A learned two-lag embedding convolution will lower val_bpb below 0.993637 while retaining at least 490M training tokens by supplying exact short-range context without reducing attention windows or adding per-layer projections.
change: Replace the shared assumption that token representations remain context-free until attention with a zero-initialized, channel-wise causal FIR over the preceding two token embeddings; optimize its coefficients with AdamW and use the proven 11.75 logit softcap.
mechanism: Learned causal embedding convolution
evidence_used: Quarter-context attention regressed to 1.009577, showing that removing context is harmful, while attention-output gating improved val_bpb to 0.993637 but reduced volume to 482.3M tokens. A single embedding-stage local-context path tests different context computation while preserving full attention and near-baseline throughput.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.66, "num_params_M": 50.3, "num_steps": 926.0, "peak_vram_mb": 45316.5, "total_tokens_M": 485.5, "training_seconds": 300.1, "val_bpb": 0.993485}

RECENT RESULT
hypothesis: Combining the best learned two-lag embedding convolution with proven all-layer attention-output gates will lower val_bpb below 0.993485 while retaining roughly 470M or more training tokens because the mechanisms control complementary local and retrieved context paths.
change: Preserve the current causal embedding convolution and add neutral-initialized, first-32-channel per-head output gates to every attention layer.
mechanism: Embedding-local context plus query-conditioned attention-head gating
evidence_used: The embedding convolution achieved the best result, 0.993485 on 485.5M tokens, while unfused all-layer output gating independently improved val_bpb to 0.993637 on 482.3M tokens; their distinct context pathways motivate testing their combination.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.17, "num_params_M": 50.3, "num_steps": 891.0, "peak_vram_mb": 47380.5, "total_tokens_M": 467.1, "training_seconds": 300.2, "val_bpb": 0.992628}

RECENT RESULT
hypothesis: Applying the learned two-lag embedding convolution while retaining output gates only in full-context layers will lower val_bpb below 0.992628 and restore training volume to roughly 480M tokens or more.
change: Add the proven channel-wise two-lag causal embedding convolution and remove attention-output gates from the six short-context layers.
mechanism: Long-context-selective gating with causal embedding mixing
evidence_used: The full convolution-plus-gating design achieved the best val_bpb, 0.992628, but processed only 467.1M tokens; long-context-only gating was just 0.000228 worse than all-layer gating while increasing training volume, suggesting short-layer gates may not justify their fixed-time cost.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.84, "num_params_M": 50.3, "num_steps": 907.0, "peak_vram_mb": 45832.5, "total_tokens_M": 475.5, "training_seconds": 300.3, "val_bpb": 0.992949}

RECENT RESULT
hypothesis: Combining the learned two-lag embedding convolution with output gates on each full-context layer and its immediately preceding short-context layer will lower val_bpb below 0.992628 while processing more than 467.1M tokens.
change: Add the proven causal embedding convolution and expand long-layer-only gating from two layers to the four S-L handoff layers, targeting the midpoint between the accurate all-layer design and faster long-only design.
mechanism: Causal embedding mixing with pre-long-context attention gating
evidence_used: All-layer convolution-plus-gating reached 0.992628 on 467.1M tokens, while long-only convolution-plus-gating reached 0.992949 on 475.5M; recovering selected short-layer gates may regain the 0.000321 accuracy difference at a fraction of the 8.4M-token throughput cost.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.0, "num_params_M": 50.3, "num_steps": 910.0, "peak_vram_mb": 46348.5, "total_tokens_M": 477.1, "training_seconds": 300.0, "val_bpb": 0.99211}

RECENT RESULT
hypothesis: Extending the best two-lag embedding FIR to three learned lags will lower val_bpb below 0.992110 while preserving at least 470M trained tokens.
change: Restore the best four-layer handoff-gated design, add a neutral-initialized third causal embedding lag, and optimize all three channel-wise coefficients with AdamW.
mechanism: Three-lag causal embedding mixing with pre-long-context attention gating
evidence_used: Two-lag embedding mixing with gates on full-context layers and their immediate predecessors achieved the best result, 0.992110 on 477.1M tokens; the convolution alone also improved to 0.993485, motivating a minimal receptive-field extension without adding per-layer computation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.94, "num_params_M": 50.3, "num_steps": 909.0, "peak_vram_mb": 46348.6, "total_tokens_M": 476.6, "training_seconds": 300.2, "val_bpb": 0.992455}

RECENT RESULT
hypothesis: Restricting output gates to the two short-context layers immediately preceding full-context layers will lower val_bpb below 0.992110 while processing more than 477.1M tokens.
change: Remove output gates from the full-context layers while retaining the proven two-lag embedding FIR and predecessor-layer gates.
mechanism: Pre-long short-context attention gating
evidence_used: Adding predecessor gates to the long-only design improved val_bpb from 0.992949 to 0.992110, whereas expanding to all layers reached only 0.992628; the missing predecessor-only ablation tests whether full-context gates are redundant with their existing value gates.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.08, "num_params_M": 50.3, "num_steps": 912.0, "peak_vram_mb": 45832.5, "total_tokens_M": 478.2, "training_seconds": 300.0, "val_bpb": 0.993768}

RECENT RESULT
hypothesis: Restoring the best two-lag receptive field and dynamically scaling each lag from the current token will lower val_bpb below 0.992110 while retaining at least 470M trained tokens.
change: Remove the regressive third lag and add a neutral-initialized, first-32-channel gate that independently modulates the two channel-wise causal embedding paths.
mechanism: Token-conditioned two-lag embedding mixing
evidence_used: Two-lag mixing with four handoff-layer attention gates achieved the best val_bpb, 0.992110 on 477.1M tokens, while adding a third lag regressed to 0.992455; successful first-32-channel attention gating suggests token-dependent selection is more promising than extending the fixed receptive field.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.74, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 46349.5, "total_tokens_M": 474.0, "training_seconds": 300.1, "val_bpb": 0.99236}

RECENT RESULT
hypothesis: Retaining output gates only on the final short-context predecessor and full-context layer will preserve the complementary S-L gating benefit while processing more than 477.1M tokens, lowering val_bpb below 0.992110.
change: Restore selective gating around only the final S-L handoff, removing gates and their computation from the first six layers while preserving the fixed two-lag embedding FIR.
mechanism: Final short-to-long attention handoff gating
evidence_used: Four handoff-layer gates achieved the best val_bpb of 0.992110, while predecessor-only gating regressed to 0.993768 and long-only gating reached 0.992949; this motivates preserving a complete predecessor-plus-long pair and testing whether the final pair supplies the useful synergy more efficiently.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.18, "num_params_M": 50.3, "num_steps": 915.0, "peak_vram_mb": 45832.5, "total_tokens_M": 479.7, "training_seconds": 300.3, "val_bpb": 0.993029}

RECENT RESULT
hypothesis: Adding the first short-context predecessor gate to both full-context gates will lower val_bpb below 0.992110 while retaining more than 477M training tokens.
change: Gate layers 2, 3, and 7: both full-context layers plus only the short-context layer preceding the first full-context transition.
mechanism: First-handoff predecessor gating
evidence_used: Four handoff gates achieved 0.992110, versus 0.992949 for long-only gating; the final handoff pair alone regressed to 0.993029, indicating the first predecessor is the more informative missing gate to restore.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.79, "num_params_M": 50.3, "num_steps": 906.0, "peak_vram_mb": 46090.5, "total_tokens_M": 475.0, "training_seconds": 300.3, "val_bpb": 0.993023}

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
