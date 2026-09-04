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
verified_results: {"depth": 8.0, "mfu_percent": 37.23, "num_params_M": 50.3, "num_steps": 1925.0, "peak_vram_mb": 44908.2, "total_tokens_M": 504.6, "training_seconds": 300.0, "val_bpb": 0.984156}
prior_hypothesis: Reducing short-layer attention from 1024 to 512 tokens while retaining two full-context layers and the best verified schedules will increase training throughput enough to achieve val_bpb below 0.985708.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.17, "num_params_M": 50.3, "num_steps": 1866.0, "peak_vram_mb": 44908.2, "total_tokens_M": 489.2, "training_seconds": 300.1, "val_bpb": 0.985708}
prior_hypothesis: Pairing the verified 59% Muon cooldown with a 35% AdamW cooldown will beat val_bpb 0.985875 by extending the productive late embedding and unembedding updates indicated by the improvement from 45% to 40%.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 39.01, "num_params_M": 50.3, "num_steps": 1858.0, "peak_vram_mb": 44908.2, "total_tokens_M": 487.1, "training_seconds": 300.1, "val_bpb": 0.986025}
prior_hypothesis: Keeping embeddings and lm_head on the best verified 35% AdamW cooldown while cooling residual-mixing scalars over 59% will reduce val_bpb below 0.985708.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 39.1, "num_params_M": 50.3, "num_steps": 1863.0, "peak_vram_mb": 44908.2, "total_tokens_M": 488.4, "training_seconds": 300.1, "val_bpb": 0.985875}
prior_hypothesis: Pairing the verified 59% Muon cooldown with a 40% AdamW cooldown will beat val_bpb 0.986065 by preserving productive late embedding and unembedding updates.

## Recent verification evidence

RECENT RESULT
hypothesis: Pairing the verified 59% Muon cooldown with a 30% AdamW cooldown will beat val_bpb 0.985708 by extending the productive late embedding and unembedding updates.
change: Shorten only AdamW’s linear cooldown from 35% to 30%, retaining Muon’s best verified 59% cooldown and all other settings.
mechanism: Further-shortened AdamW refinement tail
evidence_used: With Muon fixed at 59%, shortening AdamW cooldown from 45% to 40% improved val_bpb from 0.986100 to 0.985875, and shortening it again to 35% improved val_bpb to 0.985708; testing 30% directly probes whether that trend continues.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.76, "num_params_M": 50.3, "num_steps": 1847.0, "peak_vram_mb": 44908.2, "total_tokens_M": 484.2, "training_seconds": 300.2, "val_bpb": 0.986112}

RECENT RESULT
hypothesis: A 36% AdamW cooldown paired with the verified 59% Muon cooldown will beat val_bpb 0.985708.
change: Shorten AdamW’s linear cooldown from 40% to 36% while retaining Muon’s 59% cooldown and all other settings.
mechanism: Quadratically refined AdamW cooldown
evidence_used: With Muon fixed at 59%, AdamW cooldowns of 30%, 35%, and 40% produced val_bpb values of 0.986112, 0.985708, and 0.985875; quadratic interpolation places the estimated minimum near 36%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.82, "num_params_M": 50.3, "num_steps": 1849.0, "peak_vram_mb": 44908.2, "total_tokens_M": 484.7, "training_seconds": 300.0, "val_bpb": 0.986141}

RECENT RESULT
hypothesis: A 33% AdamW cooldown paired with the verified 59% Muon cooldown will beat the best observed val_bpb of 0.985708.
change: Shorten only AdamW’s linear cooldown from 50% to 33%, retaining Muon’s 59% cooldown and all other settings.
mechanism: Quadratically refined AdamW cooldown
evidence_used: With Muon fixed at 59%, AdamW cooldowns of 30%, 35%, and 36% produced val_bpb values of 0.986112, 0.985708, and 0.986141; quadratic interpolation of these latest observations places the estimated minimum near 33%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.76, "num_params_M": 50.3, "num_steps": 1846.0, "peak_vram_mb": 44908.2, "total_tokens_M": 483.9, "training_seconds": 300.0, "val_bpb": 0.986113}

RECENT RESULT
hypothesis: A 34.5% AdamW cooldown paired with the verified 59% Muon cooldown will beat the best observed val_bpb of 0.985708.
change: Shorten AdamW’s linear cooldown from 55% to 34.5%, retaining Muon’s 59% cooldown and all other settings.
mechanism: Locally interpolated AdamW cooldown
evidence_used: With Muon fixed at 59%, AdamW cooldowns of 33%, 35%, and 36% produced val_bpb values of 0.986113, 0.985708, and 0.986141; quadratic interpolation of these nearest observations places the estimated minimum near 34.5%.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.85, "num_params_M": 50.3, "num_steps": 1851.0, "peak_vram_mb": 44908.2, "total_tokens_M": 485.2, "training_seconds": 300.1, "val_bpb": 0.986221}

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
