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
verified_results: {"depth": 8.0, "mfu_percent": 34.99, "num_params_M": 50.3, "num_steps": 1002.0, "peak_vram_mb": 45060.2, "total_tokens_M": 525.3, "training_seconds": 300.1, "val_bpb": 0.992854}
prior_hypothesis: Seven 256-token local-attention layers followed by one full-context layer will process at least 516.9M tokens and lower val_bpb below 0.993518.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 34.75, "num_params_M": 50.3, "num_steps": 1009.0, "peak_vram_mb": 45060.2, "total_tokens_M": 529.0, "training_seconds": 300.1, "val_bpb": 0.993246}
prior_hypothesis: Seven 192-token local-attention layers followed by one full-context layer will preserve the throughput benefit of short windows while avoiding the quality loss observed at 128 tokens, lowering val_bpb below 0.992854.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 34.72, "num_params_M": 50.3, "num_steps": 998.0, "peak_vram_mb": 45060.2, "total_tokens_M": 523.2, "training_seconds": 300.1, "val_bpb": 0.993456}
prior_hypothesis: Seven 240-token local layers followed by one full-context anchor will retain at least 523.8M training tokens and lower val_bpb below 0.992854.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 34.65, "num_params_M": 50.3, "num_steps": 999.0, "peak_vram_mb": 45060.2, "total_tokens_M": 523.8, "training_seconds": 300.0, "val_bpb": 0.99293}
prior_hypothesis: Seven 224-token local layers followed by the full-context final anchor will retain at least 525M training tokens while improving val_bpb below 0.992854.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing Muon orthogonalization from five iterations to four will preserve update quality while cutting optimizer overhead, process more than 497M tokens, and lower val_bpb below 0.995558.
change: Keep the proven SSSL model, 524K-token batch, and learning-rate schedule unchanged; remove one Muon polar iteration per optimizer step.
mechanism: Four-step Muon polar orthogonalization
evidence_used: The 524K-token design achieved 0.995558 on 497.0M tokens, outperforming the one-microbatch design’s 0.996687 on 408.9M tokens despite fewer updates, indicating that additional throughput is more valuable than update frequency. Muon iteration count is an untested throughput lever that does not reduce model capacity.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.49, "num_params_M": 50.3, "num_steps": 946.0, "peak_vram_mb": 45060.2, "total_tokens_M": 496.0, "training_seconds": 300.2, "val_bpb": 0.995855}

RECENT RESULT
hypothesis: Removing the vocabulary-wide tanh softcap while restoring the verified 524K-token batch will process more than 497M tokens and lower val_bpb below 0.995558, because logits below the cap retain nearly identical behavior while avoiding a large elementwise operation.
change: Restore two-microbatch gradient accumulation and train/evaluate directly on float logits without the softcap transform.
mechanism: Uncapped vocabulary loss on the proven high-throughput batch
evidence_used: The 524K-token SSSL design achieved the best val_bpb, 0.995558, on 497.0M tokens, while reducing Muon polar iterations did not materially increase throughput; this motivates optimizing the per-token vocabulary path instead of further reducing optimizer work.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.56, "num_params_M": 50.3, "num_steps": 947.0, "peak_vram_mb": 45060.2, "total_tokens_M": 496.5, "training_seconds": 300.0, "val_bpb": 1.001142}

RECENT RESULT
hypothesis: Reducing S layers from 1024-token to 512-token windows while restoring the proven 524K-token batch will process more than 497M tokens, and the two full-context anchor layers will preserve enough global information to lower val_bpb below 0.995558.
change: Use quarter-context windows for the six local-attention layers and restore Reference Design 1’s 128-sequence microbatch with two-step gradient accumulation.
mechanism: Quarter-context local attention with full-context anchors
evidence_used: The SSSL baseline achieved 0.995558 on 497.0M tokens, whereas making every layer full-context reduced throughput to 403.7M tokens and worsened val_bpb to 1.011302; this motivates testing more aggressive locality while retaining full-context anchors.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.92, "num_params_M": 50.3, "num_steps": 960.0, "peak_vram_mb": 45060.2, "total_tokens_M": 503.3, "training_seconds": 300.2, "val_bpb": 0.99477}

RECENT RESULT
hypothesis: Replacing the intermediate full-context layer with quarter-context attention will exceed 503.3M tokens, while seven stacked 512-token local layers provide a full-sequence receptive field before the final global anchor, lowering val_bpb below 0.994770.
change: Restore the proven ReLU-squared MLP, use 512-token local windows, and change the eight-layer layout from SSS L SSS L to SSS S SSS L.
mechanism: Single final full-context anchor with stacked quarter-context local attention
evidence_used: Reference Design 3 achieved the best val_bpb, 0.994770, with quarter-context local layers and two full-context anchors; the all-full result fell to 403.7M tokens and 1.011302, showing that full-context layers are expensive, while stacked local layers can propagate information beyond their individual windows.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.41, "num_params_M": 50.3, "num_steps": 989.0, "peak_vram_mb": 45060.2, "total_tokens_M": 518.5, "training_seconds": 300.2, "val_bpb": 0.99387}

RECENT RESULT
hypothesis: Making all eight layers use 512-token attention will exceed 518.5M training tokens, while the stacked receptive field still spans the full 2,048-token sequence, lowering val_bpb below 0.993870.
change: Remove the forced full-context final layer so the existing SSSS pattern produces eight quarter-context attention layers.
mechanism: Fully local stacked receptive-field attention
evidence_used: Replacing the intermediate full-context layer with a 512-token local layer improved val_bpb from 0.994770 to 0.993870 and throughput from 503.3M to 518.5M tokens; seven stacked local layers already provide a full-sequence receptive field before the current final anchor.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.42, "num_params_M": 50.3, "num_steps": 978.0, "peak_vram_mb": 45060.2, "total_tokens_M": 512.8, "training_seconds": 300.0, "val_bpb": 1.001821}

RECENT RESULT
hypothesis: Six 512-token layers followed by a 1024-token penultimate layer and one full-context final layer will retain more than 503.3M training tokens while improving val_bpb below 0.993870.
change: Replace the current two-anchor layout with six quarter-context layers, one half-context bridge layer, and one final full-context anchor.
mechanism: Local-to-global multiscale attention funnel
evidence_used: Reference Design 3’s single final full-context anchor achieved the best val_bpb, 0.993870, while removing that anchor worsened val_bpb to 1.001821; adding a cheaper half-context bridge tests richer pre-anchor mixing without restoring the costly intermediate full-context layer.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.47, "num_params_M": 50.3, "num_steps": 976.0, "peak_vram_mb": 45060.2, "total_tokens_M": 511.7, "training_seconds": 300.2, "val_bpb": 0.99527}

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
