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
verified_results: {"depth": 8.0, "mfu_percent": 36.88, "num_params_M": 50.3, "num_steps": 1949.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.9, "training_seconds": 300.1, "val_bpb": 0.983618}
prior_hypothesis: Using 384-token short-layer attention will reduce `val_bpb` below 0.984156 by preserving more local context than the regressing 256-token design while exposing more training tokens than the best 512-token design.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 36.75, "num_params_M": 50.3, "num_steps": 1942.0, "peak_vram_mb": 44908.2, "total_tokens_M": 509.1, "training_seconds": 300.1, "val_bpb": 0.984213}
prior_hypothesis: Compute-neutral 256-token early local attention and 512-token late local attention, with full context at layers 4 and 8, will reduce val_bpb below 0.983618 by using early layers for cheap feature extraction and allocating wider context to later refinement.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 37.23, "num_params_M": 50.3, "num_steps": 1925.0, "peak_vram_mb": 44908.2, "total_tokens_M": 504.6, "training_seconds": 300.0, "val_bpb": 0.984156}
prior_hypothesis: Reducing short-layer attention from 1024 to 512 tokens while retaining two full-context layers and the best verified schedules will increase training throughput enough to achieve val_bpb below 0.985708.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 35.16, "num_params_M": 50.3, "num_steps": 1950.0, "peak_vram_mb": 44908.2, "total_tokens_M": 511.2, "training_seconds": 300.1, "val_bpb": 0.985116}
prior_hypothesis: Using 384-token local attention in the first seven layers and retaining full context only in the final layer will reduce `val_bpb` below 0.983618 by increasing token exposure while preserving end-to-end access to the entire sequence.

## Recent verification evidence

RECENT RESULT
hypothesis: Cooling only the lm_head over 50% of training while retaining the verified 35% cooldown for other AdamW parameters and 59% for Muon will achieve val_bpb below 0.985708.
change: Give the unembedding parameter group an independent 50% cooldown and select AdamW schedules per optimizer group.
mechanism: Parameter-group-specific AdamW cooldown
evidence_used: The shared AdamW sweep found the best result at 35% (0.985708) but also a local best at 50% (0.986065); separating the low-LR lm_head tests whether one shared schedule is masking complementary group optima.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.77, "num_params_M": 50.3, "num_steps": 1847.0, "peak_vram_mb": 44908.2, "total_tokens_M": 484.2, "training_seconds": 300.1, "val_bpb": 0.986361}

RECENT RESULT
hypothesis: A 30% cooldown for token and value embeddings, while retaining the verified 35% cooldown for the lm_head/scalars and 59% for Muon, will reduce val_bpb below 0.985708.
change: Restore the best shared AdamW cooldown, then independently shorten only the representation-embedding cooldown.
mechanism: Reciprocal representation-embedding cooldown
evidence_used: Shared 35% achieved the best val_bpb of 0.985708, shared 30% regressed to 0.986112, and lengthening only the lm_head cooldown regressed to 0.986361; the reciprocal split tests whether embeddings benefit from 30% while the lm_head requires 35%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.79, "num_params_M": 50.3, "num_steps": 1848.0, "peak_vram_mb": 44908.2, "total_tokens_M": 484.4, "training_seconds": 300.2, "val_bpb": 0.986217}

RECENT RESULT
hypothesis: Keeping embeddings and lm_head on the best verified 35% AdamW cooldown while cooling residual-mixing scalars over 59% will reduce val_bpb below 0.985708.
change: Restore the verified 35% AdamW cooldown, tag both scalar parameter groups, and schedule those scalars with the 59% Muon cooldown.
mechanism: Muon-aligned residual-scalar cooldown
evidence_used: Shared 35% AdamW cooldown achieved the best val_bpb of 0.985708, while independently changing the lm_head or representation-embedding cooldown regressed; the remaining untested AdamW split is the residual scalars, whose direct control of block mixing motivates synchronizing them with Muon matrices.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.01, "num_params_M": 50.3, "num_steps": 1858.0, "peak_vram_mb": 44908.2, "total_tokens_M": 487.1, "training_seconds": 300.1, "val_bpb": 0.986025}

RECENT RESULT
hypothesis: A cosine 35% AdamW cooldown paired with the verified linear 59% Muon cooldown will reduce val_bpb below 0.985708 by preserving stronger early refinement while tapering updates more gently near completion.
change: Restore AdamW’s best verified 35% cooldown and change only its tail from linear to cosine; retain Muon’s linear 59% schedule.
mechanism: Cosine-shaped AdamW refinement tail
evidence_used: Linear AdamW cooldowns of 30%, 33%, 34.5%, and 36% all underperformed the 35% result of 0.985708, so schedule duration is locally well explored; a cosine tail tests schedule shape while preserving the best cooldown onset and the same average tail multiplier.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.78, "num_params_M": 50.3, "num_steps": 1847.0, "peak_vram_mb": 44908.2, "total_tokens_M": 484.2, "training_seconds": 300.0, "val_bpb": 0.986841}

RECENT RESULT
hypothesis: Cooling only `resid_lambdas` over 59% while retaining the verified 35% cooldown for `x0_lambdas` and other AdamW parameters will reduce `val_bpb` below 0.985708.
change: Tag only the residual-gain parameter group and give it an independent 59% linear cooldown; leave `x0_lambdas`, embeddings, lm_head, and Muon settings unchanged.
mechanism: Residual-gain-only Muon-aligned cooldown
evidence_used: Cooling both scalar groups over 59% regressed to 0.986025, but the groups differ by 100× in learning rate, use different beta1 values, and control distinct residual paths; isolating the low-LR residual gains tests whether the combined result was degraded by the high-LR `x0_lambdas` schedule.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.77, "num_params_M": 50.3, "num_steps": 1847.0, "peak_vram_mb": 44908.2, "total_tokens_M": 484.2, "training_seconds": 300.1, "val_bpb": 0.986415}

RECENT RESULT
hypothesis: Cooling only `x0_lambdas` over 59% while restoring the verified 35% cooldown for all other AdamW parameters will reduce `val_bpb` below 0.985708.
change: Tag the `x0_lambdas` optimizer group, restore the best shared AdamW cooldown, and apply Muon’s 59% cooldown only to that tagged group.
mechanism: X0-path-only Muon-aligned cooldown
evidence_used: Cooling both scalar groups over 59% reached 0.986025, whereas cooling only `resid_lambdas` reached 0.986415; the relative advantage of the combined split suggests the high-LR `x0_lambdas` schedule may be beneficial and should be isolated against the 0.985708 baseline.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.84, "num_params_M": 50.3, "num_steps": 1850.0, "peak_vram_mb": 44908.2, "total_tokens_M": 485.0, "training_seconds": 300.1, "val_bpb": 0.986302}

RECENT RESULT
hypothesis: Retaining 5% of the initial AdamW learning rate at the end, while preserving the verified 35% AdamW and 59% Muon cooldowns, will reduce val_bpb below 0.985708 by allowing embeddings, unembedding, and residual scalars to track late Muon updates.
change: Restore all AdamW groups to the best shared 35% schedule, add a 5% terminal floor only to AdamW, and keep Muon’s linear cooldown ending at zero.
mechanism: AdamW terminal learning-rate floor
evidence_used: The 35% linear AdamW cooldown achieved the best val_bpb of 0.985708; independently rescheduling parameter groups and replacing the tail with cosine both regressed, motivating a shared schedule whose only new variable is a small amount of continued terminal refinement.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.75, "num_params_M": 50.3, "num_steps": 1846.0, "peak_vram_mb": 44908.2, "total_tokens_M": 483.9, "training_seconds": 300.1, "val_bpb": 0.986117}

RECENT RESULT
hypothesis: Reducing short-layer attention from 1024 to 512 tokens while retaining two full-context layers and the best verified schedules will increase training throughput enough to achieve val_bpb below 0.985708.
change: Restore the verified 35% AdamW cooldown and halve only the short attention window; keep the 59% Muon cooldown and all other settings unchanged.
mechanism: Quarter-context local attention for greater token exposure
evidence_used: The 35% AdamW design is best at 0.985708, while numerous subsequent schedule refinements regressed; this motivates testing compute allocation instead, preserving periodic full-context attention while spending less time on six local-attention layers.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.23, "num_params_M": 50.3, "num_steps": 1925.0, "peak_vram_mb": 44908.2, "total_tokens_M": 504.6, "training_seconds": 300.0, "val_bpb": 0.984156}

RECENT RESULT
hypothesis: Halving short-layer attention from 512 to 256 tokens while retaining two full-context layers and the verified schedules will reduce val_bpb below 0.984156 through greater token exposure within five minutes.
change: Reduce only the six short attention windows from one-quarter to one-eighth of the 2048-token sequence length.
mechanism: Eighth-context local attention
evidence_used: Reducing short attention from 1024 to 512 tokens increased exposure from 489.2M to 504.6M tokens and improved val_bpb from 0.985708 to 0.984156, motivating one further controlled halving.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.15, "num_params_M": 50.3, "num_steps": 1953.0, "peak_vram_mb": 44908.2, "total_tokens_M": 512.0, "training_seconds": 300.1, "val_bpb": 0.984221}

RECENT RESULT
hypothesis: Using 384-token short-layer attention will reduce `val_bpb` below 0.984156 by preserving more local context than the regressing 256-token design while exposing more training tokens than the best 512-token design.
change: Set the six short attention windows to 3/16 of the 2048-token sequence length, retaining two full-context layers and all verified optimizer schedules.
mechanism: Intermediate local-attention compute allocation
evidence_used: Reducing short attention from 1024 to 512 improved `val_bpb` from 0.985708 to 0.984156, while a further reduction to 256 slightly regressed to 0.984221 despite higher throughput; this brackets a likely context-throughput optimum between 256 and 512 tokens.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.88, "num_params_M": 50.3, "num_steps": 1949.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.9, "training_seconds": 300.1, "val_bpb": 0.983618}

RECENT RESULT
hypothesis: Using 384-token local attention in the first seven layers and retaining full context only in the final layer will reduce `val_bpb` below 0.983618 by increasing token exposure while preserving end-to-end access to the entire sequence.
change: Restore the best verified 35% AdamW cooldown and 384-token local window, then remove the intermediate full-context layer by changing the window pattern to all-short; the existing final-layer override remains full-context.
mechanism: Single global-attention consolidation layer
evidence_used: The 384-token design achieved the best result, `val_bpb` 0.983618 at 510.9M tokens, outperforming both 512-token and 256-token windows. Because seven stacked 384-token causal layers have a receptive field exceeding 2048 tokens and the final layer remains globally attentive, the intermediate global layer may be redundant compute.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.16, "num_params_M": 50.3, "num_steps": 1950.0, "peak_vram_mb": 44908.2, "total_tokens_M": 511.2, "training_seconds": 300.1, "val_bpb": 0.985116}

RECENT RESULT
hypothesis: Compute-neutral 256-token early local attention and 512-token late local attention, with full context at layers 4 and 8, will reduce val_bpb below 0.983618 by using early layers for cheap feature extraction and allocating wider context to later refinement.
change: Restore the best shared 35% AdamW schedule and replace uniform short attention with a 256/512-token stagewise split whose average local window remains 384 tokens.
mechanism: Stagewise context reallocation
evidence_used: Uniform 384-token attention achieved the best val_bpb of 0.983618, while uniform 256 and 512 produced similar regressions of 0.984221 and 0.984156; combining those endpoints preserves the winning design’s attention compute while testing whether context is more valuable after the intermediate global layer. Removing that global layer regressed to 0.985116, so both verified full-context layers are retained.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.75, "num_params_M": 50.3, "num_steps": 1942.0, "peak_vram_mb": 44908.2, "total_tokens_M": 509.1, "training_seconds": 300.1, "val_bpb": 0.984213}



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
