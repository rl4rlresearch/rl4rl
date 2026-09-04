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
verified_results: {"depth": 8.0, "mfu_percent": 36.92, "num_params_M": 50.3, "num_steps": 960.0, "peak_vram_mb": 45060.2, "total_tokens_M": 503.3, "training_seconds": 300.2, "val_bpb": 0.99477}
prior_hypothesis: Reducing S layers from 1024-token to 512-token windows while restoring the proven 524K-token batch will process more than 497M tokens, and the two full-context anchor layers will preserve enough global information to lower val_bpb below 0.995558.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.58, "num_params_M": 50.3, "num_steps": 948.0, "peak_vram_mb": 45060.2, "total_tokens_M": 497.0, "training_seconds": 300.2, "val_bpb": 0.995558}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 32.71, "num_params_M": 50.3, "num_steps": 1560.0, "peak_vram_mb": 44908.2, "total_tokens_M": 408.9, "training_seconds": 300.1, "val_bpb": 0.996687}
prior_hypothesis: Restoring the proven 1024-token short windows while reducing the optimizer batch to one 128-sequence microbatch will retain high token throughput and provide substantially more updates, lowering val_bpb below 0.995558.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 36.41, "num_params_M": 50.3, "num_steps": 989.0, "peak_vram_mb": 45060.2, "total_tokens_M": 518.5, "training_seconds": 300.2, "val_bpb": 0.99387}
prior_hypothesis: Replacing the intermediate full-context layer with quarter-context attention will exceed 503.3M tokens, while seven stacked 512-token local layers provide a full-sequence receptive field before the final global anchor, lowering val_bpb below 0.994770.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the proven 1024-token short windows while reducing the optimizer batch to one 128-sequence microbatch will retain high token throughput and provide substantially more updates, lowering val_bpb below 0.995558.
change: Restore half-context attention and use a 262,144-token batch with no gradient accumulation.
mechanism: Single-microbatch half-window SSSL training
evidence_used: Reference Design 1 achieved 0.995558 with half-context SSSL and 128-sequence microbatches. Reference Design 2 suggests more frequent single-microbatch updates preserve quality despite fewer tokens, but its 192-sequence microbatch reduced throughput; retaining the efficient 128-sequence microbatch isolates the update-frequency benefit.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 32.71, "num_params_M": 50.3, "num_steps": 1560.0, "peak_vram_mb": 44908.2, "total_tokens_M": 408.9, "training_seconds": 300.1, "val_bpb": 0.996687}

RECENT RESULT
hypothesis: Alternating 262K- and 524K-token updates will combine the current design’s higher update frequency with Reference Design 1’s higher throughput, processing roughly 460M tokens in about 1,200 steps and lowering val_bpb below 0.995558.
change: Keep the proven SSSL architecture and 128-sequence microbatch, but alternate gradient accumulation between one and two microbatches while accurately tracking tokens and MFU.
mechanism: Alternating one- and two-microbatch optimizer steps
evidence_used: One-microbatch training reached 0.996687 with 1,560 steps and 408.9M tokens, while two-microbatch training reached 0.995558 with 948 steps and 497.0M tokens; an alternating batch schedule directly tests the unobserved throughput/update-frequency midpoint.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.3, "num_params_M": 50.3, "num_steps": 1000.0, "peak_vram_mb": 45060.2, "total_tokens_M": 393.2, "training_seconds": 300.3, "val_bpb": 1.009763}

RECENT RESULT
hypothesis: Restoring the proven 524K-token, 128-sequence configuration while delaying warmdown from 50% to 60% elapsed time will retain roughly 497M-token throughput and lower val_bpb below 0.995558 by applying peak learning rates to more training tokens while preserving a 120-second cooldown.
change: Restore Reference Design 1’s batch configuration and shorten LR warmdown from half to 40% of the training window.
mechanism: Delayed linear learning-rate warmdown
evidence_used: Reference Design 1 achieved the best val_bpb, 0.995558, with 497.0M tokens and 128-sequence microbatches; smaller or irregular batches produced fewer tokens without improving validation, motivating an optimization-schedule test on the proven configuration.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.74, "num_params_M": 50.3, "num_steps": 762.0, "peak_vram_mb": 45060.2, "total_tokens_M": 399.5, "training_seconds": 300.1, "val_bpb": 1.010374}

RECENT RESULT
hypothesis: Reducing the proven ReLU-squared MLP expansion from 4× to 3× will process more than 497M tokens while retaining eight-layer SSSL capacity, lowering val_bpb below 0.995558.
change: Replace the slower parameter-matched SwiGLU with a tensor-core-aligned 3× ReLU-squared MLP; preserve the proven attention, batching, optimizer, and schedule.
mechanism: Compute-rebalanced 3× ReLU-squared feed-forward layers
evidence_used: Reference Design 1 reached 0.995558 on 497.0M tokens, while near-parameter-matched SwiGLU processed only 402.7M tokens and worsened to 1.008305. A narrower ReLU-squared MLP directly targets the demonstrated throughput sensitivity with a modest capacity reduction.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.03, "num_params_M": 46.1, "num_steps": 832.0, "peak_vram_mb": 40915.7, "total_tokens_M": 436.2, "training_seconds": 300.3, "val_bpb": 1.011834}

RECENT RESULT
hypothesis: Scaling all optimizer learning rates by \(1/\sqrt{2}\) for the halved 262K-token batch will retain its useful 1,560-update regime while reducing update noise and overshoot, lowering val_bpb below 0.995558.
change: Keep the current SSSL architecture and single-microbatch training, but reduce Adam and Muon learning rates by \(1/\sqrt{2}\).
mechanism: Square-root batch-aware learning-rate scaling
evidence_used: The current 262K-token design nearly matched the best result (0.996687 versus 0.995558) despite processing only 408.9M versus 497.0M tokens, but made 1,560 versus 948 optimizer updates using unchanged per-step learning rates; this motivates retuning update magnitude rather than abandoning the higher-frequency regime.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 30.18, "num_params_M": 50.3, "num_steps": 1440.0, "peak_vram_mb": 44908.2, "total_tokens_M": 377.5, "training_seconds": 300.1, "val_bpb": 1.00116}

RECENT RESULT
hypothesis: Restoring the 128-sequence, 524K-token SSSL baseline and replacing its final-half linear decay with an equal-area cosine decay will retain roughly 497M-token throughput while allowing smaller late-training updates, lowering val_bpb below 0.995558.
change: Restore Reference Design 1’s batching and change only the shape of the final-half learning-rate warmdown from linear to cosine.
mechanism: Cosine learning-rate warmdown on the proven high-throughput configuration
evidence_used: Reference Design 1 achieved the best val_bpb, 0.995558, at 497.0M tokens; every tested batching change reduced throughput or worsened validation, while delaying warmdown also worsened validation, motivating a schedule-shape test that preserves the proven batch and cooldown duration.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.4, "num_params_M": 50.3, "num_steps": 754.0, "peak_vram_mb": 45060.2, "total_tokens_M": 395.3, "training_seconds": 300.2, "val_bpb": 1.014183}

RECENT RESULT
hypothesis: Restoring ReLU-squared MLPs and increasing the fixed batch to 786,432 tokens will amortize optimizer overhead enough to process more than 497M tokens while retaining sufficient updates, lowering val_bpb below 0.995558.
change: Restore the proven 4× ReLU-squared MLP and accumulate three 128-sequence microbatches per optimizer step.
mechanism: Fixed three-microbatch optimizer amortization
evidence_used: The fixed two-microbatch design achieved 0.995558 on 497.0M tokens, outperforming the fixed one-microbatch design’s 0.996687 on 408.9M tokens; this motivates testing one additional fixed microbatch without the irregular schedule that fell to 393.2M tokens.
result: the implementation could not be verified

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
