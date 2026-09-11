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
verified_results: {"depth": 8.0, "mfu_percent": 38.85, "num_params_M": 50.3, "num_steps": 1851.0, "peak_vram_mb": 44908.2, "total_tokens_M": 485.2, "training_seconds": 300.1, "val_bpb": 0.986221}
prior_hypothesis: A 34.5% AdamW cooldown paired with the verified 59% Muon cooldown will beat the best observed val_bpb of 0.985708.

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
hypothesis: Pairing the best verified 59% Muon cooldown with a 50% AdamW cooldown will beat val_bpb 0.986424 by preserving useful late embedding and unembedding updates while still suppressing late matrix updates.
change: Restore Muon’s best verified 59% cooldown and shorten only AdamW’s cooldown from 55% to 50%.
mechanism: Independently shortened AdamW refinement tail
evidence_used: With Muon fixed at a 60% cooldown, shortening AdamW’s cooldown from 60% to 55% improved val_bpb from 0.987466 to 0.986574; this motivates testing another five-point AdamW reduction while using the stronger 59% Muon schedule.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.27, "num_params_M": 50.3, "num_steps": 1870.0, "peak_vram_mb": 44908.2, "total_tokens_M": 490.2, "training_seconds": 300.0, "val_bpb": 0.986065}

RECENT RESULT
hypothesis: Pairing the best verified 59% Muon cooldown with a 45% AdamW cooldown will beat val_bpb 0.986065 by further preserving productive late embedding and unembedding updates.
change: Shorten only AdamW’s linear cooldown from 50% to 45%, leaving Muon’s 59% cooldown and all other settings unchanged.
mechanism: Independently shortened AdamW refinement tail
evidence_used: With Muon near its optimum, shortening AdamW cooldown from 55% to 50% improved val_bpb from 0.986424 to 0.986065; an isolated five-point continuation directly tests whether that trend persists.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.19, "num_params_M": 50.3, "num_steps": 1867.0, "peak_vram_mb": 44908.2, "total_tokens_M": 489.4, "training_seconds": 300.1, "val_bpb": 0.9861}

RECENT RESULT
hypothesis: Pairing the verified 59% Muon cooldown with a 48% AdamW cooldown will beat val_bpb 0.986065.
change: Shorten AdamW’s cooldown from 55% to 48% while preserving Muon’s 59% cooldown and all other settings.
mechanism: Quadratically refined AdamW-only cooldown
evidence_used: With Muon fixed at 59%, AdamW cooldowns of 45%, 50%, and 55% produced val_bpb 0.986100, 0.986065, and 0.986424; quadratic interpolation places the estimated minimum near 48%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.76, "num_params_M": 50.3, "num_steps": 1846.0, "peak_vram_mb": 44908.2, "total_tokens_M": 483.9, "training_seconds": 300.0, "val_bpb": 0.986576}

RECENT RESULT
hypothesis: Pairing the verified 59% Muon cooldown with a 40% AdamW cooldown will beat val_bpb 0.986065 by preserving productive late embedding and unembedding updates.
change: Shorten AdamW’s linear cooldown from 55% to 40% and restore Muon’s best verified 59% cooldown.
mechanism: Further-shortened AdamW refinement tail
evidence_used: With Muon at 59%, shortening AdamW cooldown from 55% to 50% improved val_bpb from 0.986424 to 0.986065, while 45% remained nearly tied at 0.986100; testing 40% determines whether the apparent late-update plateau extends further.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.1, "num_params_M": 50.3, "num_steps": 1863.0, "peak_vram_mb": 44908.2, "total_tokens_M": 488.4, "training_seconds": 300.1, "val_bpb": 0.985875}

RECENT RESULT
hypothesis: Pairing the verified 59% Muon cooldown with a 35% AdamW cooldown will beat val_bpb 0.985875 by extending the productive late embedding and unembedding updates indicated by the improvement from 45% to 40%.
change: Decouple optimizer-group schedules, set AdamW cooldown to 35%, and retain Muon’s best verified 59% cooldown.
mechanism: Further-shortened AdamW refinement tail with independently cooled Muon matrices
evidence_used: With Muon fixed at 59%, shortening AdamW cooldown from 45% to 40% improved val_bpb from 0.986100 to the best observed 0.985875, leaving 40% as the boundary point worth extending.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.17, "num_params_M": 50.3, "num_steps": 1866.0, "peak_vram_mb": 44908.2, "total_tokens_M": 489.2, "training_seconds": 300.1, "val_bpb": 0.985708}

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
