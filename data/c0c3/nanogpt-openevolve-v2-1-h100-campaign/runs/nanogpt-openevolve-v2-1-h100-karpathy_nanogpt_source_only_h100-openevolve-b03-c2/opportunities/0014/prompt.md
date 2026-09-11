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
verified_results: {"depth": 8.0, "mfu_percent": 38.57, "num_params_M": 50.3, "num_steps": 1838.0, "peak_vram_mb": 44908.2, "total_tokens_M": 481.8, "training_seconds": 300.2, "val_bpb": 0.987466}
prior_hypothesis: Extending linear warmdown from 50% to 60% of training will beat val_bpb 0.98713 by shifting learning-rate exposure from the plateau into later refinement.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.58, "num_params_M": 50.3, "num_steps": 948.0, "peak_vram_mb": 45060.2, "total_tokens_M": 497.0, "training_seconds": 300.2, "val_bpb": 0.995558}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 38.57, "num_params_M": 50.3, "num_steps": 1837.0, "peak_vram_mb": 44908.2, "total_tokens_M": 481.6, "training_seconds": 300.0, "val_bpb": 0.989243}
prior_hypothesis: On the proven 128-sequence, 262K-token update geometry, shortening linear warmdown from 50% to 40% will beat val_bpb 0.98713 by retaining larger learning rates later in training.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 39.36, "num_params_M": 50.3, "num_steps": 1875.0, "peak_vram_mb": 44908.2, "total_tokens_M": 491.5, "training_seconds": 300.1, "val_bpb": 0.986676}
prior_hypothesis: A 55% linear warmdown will beat val_bpb 0.98713 because the 60% schedule nearly matched the 50% optimum, while shortening warmdown to 40% regressed substantially, suggesting the optimum lies slightly above 50%.

## Recent verification evidence

RECENT RESULT
hypothesis: Using 96-sequence microbatches with two-way accumulation will retain more of the baseline’s throughput while increasing update frequency; this will beat the baseline val_bpb of 0.995558.
change: Reduce the global batch to 393K tokens and device batch to 96, preserving two microbatches per optimizer step.
mechanism: Two-microbatch 393K-token updates
evidence_used: The 192-sequence single-microbatch design fell to 416.8M tokens and 33.24% MFU versus the baseline’s 497.0M tokens and 39.58% MFU, so its 1.000243 val_bpb does not isolate the benefit of smaller, more frequent updates.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.18, "num_params_M": 50.3, "num_steps": 1248.0, "peak_vram_mb": 33958.2, "total_tokens_M": 490.7, "training_seconds": 300.2, "val_bpb": 0.990599}

RECENT RESULT
hypothesis: Halving the baseline global batch while retaining its efficient 128-sequence microbatch will increase update frequency enough to beat the best observed val_bpb of 0.990599 without the throughput collapse of the 192-sequence design.
change: Reduce each optimizer update from 524K tokens over two microbatches to 262K tokens in one 128-sequence microbatch.
mechanism: Single-microbatch 262K-token updates
evidence_used: The 393K-token, two-microbatch design improved val_bpb from 0.995558 to 0.990599 while preserving 39.18% MFU and 490.7M tokens; the 192-sequence single-microbatch result shows that increasing device batch size harms throughput, motivating smaller updates using the baseline’s efficient device batch.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.23, "num_params_M": 50.3, "num_steps": 1868.0, "peak_vram_mb": 44908.2, "total_tokens_M": 489.7, "training_seconds": 300.0, "val_bpb": 0.98713}

RECENT RESULT
hypothesis: Reducing the global batch from 262K to 197K tokens while using the already efficient 96-sequence microbatch will increase optimizer updates by roughly one-third and lower val_bpb below 0.98713 without materially reducing token throughput.
change: Set both the global batch and device batch so each optimizer step processes one 96-sequence microbatch.
mechanism: Single-microbatch 197K-token updates
evidence_used: The 96-sequence design sustained 490.7M tokens and 39.18% MFU, while reducing update size from 393K to 262K tokens improved val_bpb from 0.990599 to 0.98713 at similar throughput.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.53, "num_params_M": 50.3, "num_steps": 2002.0, "peak_vram_mb": 33807.2, "total_tokens_M": 393.6, "training_seconds": 300.1, "val_bpb": 0.997301}

RECENT RESULT
hypothesis: On the efficient 128-sequence, 262K-token update design, halving cautious weight decay will beat val_bpb 0.98713 while retaining roughly 490M-token throughput, because twice as many optimizer steps otherwise apply approximately twice as many decay opportunities.
change: Restore the best verified single-microbatch configuration and reduce Muon weight decay from 0.2 to 0.1.
mechanism: Update-count-normalized cautious weight decay
evidence_used: Reference Design 3 achieved the best val_bpb, 0.98713, with 1,868 steps and 489.7M tokens; compared with the 948-step baseline, its unchanged per-step decay substantially increases cumulative regularization.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 29.62, "num_params_M": 50.3, "num_steps": 1413.0, "peak_vram_mb": 44908.2, "total_tokens_M": 370.4, "training_seconds": 300.1, "val_bpb": 1.001598}

RECENT RESULT
hypothesis: Restoring the efficient 128-sequence, 262K-token update and extending the Muon momentum ramp from 300 to 600 steps will beat val_bpb 0.98713 by preserving the best throughput while keeping the momentum transition at approximately the baseline’s token horizon.
change: Restore Reference Design 3’s single-microbatch configuration and double the momentum-ramp step count to compensate for its roughly doubled optimizer-update frequency.
mechanism: Token-normalized Muon momentum ramp
evidence_used: Reference Design 3 achieved the best val_bpb, 0.98713, with 1,868 steps and 489.7M tokens; its unchanged 300-step momentum ramp reaches 0.95 after half as many training tokens as the 524K-token baseline, while the 96- and 192-sequence designs show that changing away from the efficient 128-sequence microbatch sharply reduces throughput.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 31.64, "num_params_M": 50.3, "num_steps": 1509.0, "peak_vram_mb": 44908.2, "total_tokens_M": 395.6, "training_seconds": 300.1, "val_bpb": 0.999525}

RECENT RESULT
hypothesis: Replacing the redundant mid-stack full-attention layer with half-context attention will preserve long-range modeling through the final full-context layer while increasing token throughput enough to beat the current 0.98713 val_bpb.
change: Change the eight-layer attention pattern from six short and two full-context layers to seven short and one final full-context layer, retaining the verified 128-sequence, 262K-token update configuration.
mechanism: Single full-context anchor layer
evidence_used: The best design processed 489.7M tokens at 39.23% MFU, while lower-throughput designs processing 416.8M and 393.6M tokens regressed to 1.000243 and 0.997301 val_bpb; this motivates reducing attention compute without disturbing the efficient batch geometry.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.12, "num_params_M": 50.3, "num_steps": 1524.0, "peak_vram_mb": 44908.2, "total_tokens_M": 399.5, "training_seconds": 300.1, "val_bpb": 0.9984}

RECENT RESULT
hypothesis: Using four query heads but only two shared KV heads on the proven 128-sequence, 262K-token update geometry will beat val_bpb 0.98713 by reducing projection, value-embedding, and optimizer overhead while retaining query capacity and full token throughput.
change: Restore the best verified single-microbatch configuration and replace full multi-head KV attention with two-way grouped-query attention.
mechanism: Two-way grouped-query attention with efficient single-microbatch updates
evidence_used: Reference Design 3 achieved the best val_bpb, 0.98713, at 489.7M tokens and 39.23% MFU; lower-throughput variants regressed, motivating an architecture reduction that preserves its efficient batch geometry.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.51, "num_params_M": 39.8, "num_steps": 1936.0, "peak_vram_mb": 41236.2, "total_tokens_M": 507.5, "training_seconds": 300.1, "val_bpb": 0.992471}

RECENT RESULT
hypothesis: On Reference Design 3’s efficient geometry and 300-step momentum ramp, replacing linear warmdown with an equal-area cosine warmdown will beat val_bpb 0.98713 by preserving larger updates early in cooldown and providing gentler refinement near the end without changing throughput or total learning-rate exposure.
change: Restore the best verified 300-step Muon momentum ramp and change only the warmdown curve from linear to cosine while retaining its duration and endpoints.
mechanism: Endpoint-weighted cosine learning-rate cooldown
evidence_used: Reference Design 3 achieved the best val_bpb, 0.98713, at 489.7M tokens; extending its momentum ramp to 600 steps regressed to 0.999525, motivating restoration of the verified ramp and an isolated schedule-shape test that leaves batch geometry and model capacity unchanged.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.26, "num_params_M": 50.3, "num_steps": 1870.0, "peak_vram_mb": 44908.2, "total_tokens_M": 490.2, "training_seconds": 300.0, "val_bpb": 0.990027}

RECENT RESULT
hypothesis: Redistributing cooldown learning rate from its early half to its late half while preserving endpoints and total exposure will beat the linear schedule’s 0.98713 val_bpb.
change: Replace linear warmdown with a monotonic equal-area curve that decays faster initially and retains larger learning rates during late refinement.
mechanism: Late-weighted equal-area cooldown
evidence_used: The equal-area cosine schedule shifted learning rate earlier and regressed from 0.98713 to 0.990027 at essentially identical throughput, motivating the opposite temporal redistribution as an isolated schedule test.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.51, "num_params_M": 50.3, "num_steps": 1882.0, "peak_vram_mb": 44908.2, "total_tokens_M": 493.4, "training_seconds": 300.1, "val_bpb": 0.987466}

RECENT RESULT
hypothesis: Extending linear warmdown from 50% to 60% of training will beat val_bpb 0.98713 by shifting learning-rate exposure from the plateau into later refinement.
change: Restore the best design’s linear cooldown and begin it at 40% training progress instead of 50%.
mechanism: Longer linear refinement tail
evidence_used: The early-weighted cosine cooldown regressed to 0.990027, while the late-weighted schedule recovered to 0.987466; this indicates that preserving relatively larger late-training updates is preferable.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.57, "num_params_M": 50.3, "num_steps": 1838.0, "peak_vram_mb": 44908.2, "total_tokens_M": 481.8, "training_seconds": 300.2, "val_bpb": 0.987466}

RECENT RESULT
hypothesis: On the proven 128-sequence, 262K-token update geometry, shortening linear warmdown from 50% to 40% will beat val_bpb 0.98713 by retaining larger learning rates later in training.
change: Restore Reference Design 3’s efficient single-microbatch geometry and begin linear cooldown at 60% training progress.
mechanism: Shorter linear cooldown with greater late-training learning-rate exposure
evidence_used: Reference Design 3 achieved 0.98713 at 489.7M tokens; extending warmdown to 60% produced 0.987466, while the early-weighted cosine schedule regressed to 0.990027, motivating the opposite adjustment toward greater late-training learning-rate exposure.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.57, "num_params_M": 50.3, "num_steps": 1837.0, "peak_vram_mb": 44908.2, "total_tokens_M": 481.6, "training_seconds": 300.0, "val_bpb": 0.989243}

RECENT RESULT
hypothesis: A 55% linear warmdown will beat val_bpb 0.98713 because the 60% schedule nearly matched the 50% optimum, while shortening warmdown to 40% regressed substantially, suggesting the optimum lies slightly above 50%.
change: Begin the unchanged linear cooldown at 45% training progress, midway between the best 50% and competitive 60% warmdown designs.
mechanism: Interpolated linear refinement tail
evidence_used: Warmdown ratios of 40%, 50%, and 60% produced val_bpb values of 0.989243, 0.98713, and 0.987466 respectively; the asymmetric degradation around 50% motivates testing a modest shift toward the stronger side.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.36, "num_params_M": 50.3, "num_steps": 1875.0, "peak_vram_mb": 44908.2, "total_tokens_M": 491.5, "training_seconds": 300.1, "val_bpb": 0.986676}



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
