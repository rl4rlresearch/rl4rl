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
verified_results: {"depth": 8.0, "mfu_percent": 39.33, "num_params_M": 50.3, "num_steps": 1874.0, "peak_vram_mb": 44908.2, "total_tokens_M": 491.3, "training_seconds": 300.2, "val_bpb": 0.986663}
prior_hypothesis: Restoring the jointly verified 262K-token update geometry and 55% linear warmdown will reduce val_bpb below the current design’s 0.995558 and reproduce the strongest available result near 0.986676.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.27, "num_params_M": 50.3, "num_steps": 1870.0, "peak_vram_mb": 44908.2, "total_tokens_M": 490.2, "training_seconds": 300.0, "val_bpb": 0.986065}
prior_hypothesis: Pairing the best verified 59% Muon cooldown with a 50% AdamW cooldown will beat val_bpb 0.986424 by preserving useful late embedding and unembedding updates while still suppressing late matrix updates.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 39.05, "num_params_M": 50.3, "num_steps": 1860.0, "peak_vram_mb": 44908.2, "total_tokens_M": 487.6, "training_seconds": 300.1, "val_bpb": 0.986424}
prior_hypothesis: A 59% Muon cooldown paired with the verified 55% AdamW cooldown will beat val_bpb 0.986574, because the observed 55%, 60%, and 65% Muon cooldown results imply a local optimum slightly below 60%.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 39.1, "num_params_M": 50.3, "num_steps": 1863.0, "peak_vram_mb": 44908.2, "total_tokens_M": 488.4, "training_seconds": 300.1, "val_bpb": 0.985875}
prior_hypothesis: Pairing the verified 59% Muon cooldown with a 40% AdamW cooldown will beat val_bpb 0.986065 by preserving productive late embedding and unembedding updates.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the jointly verified 262K-token update geometry and 55% linear warmdown will reduce val_bpb below the current design’s 0.995558 and reproduce the strongest available result near 0.986676.
change: Halve the global batch so every optimizer step uses one 128-sequence microbatch, and begin linear cooldown after 45% of training.
mechanism: Single-microbatch updates with a 55% linear refinement tail
evidence_used: Reference Design 3 achieved the best available val_bpb of 0.986676 with 491.5M tokens, while the current 524K-token batch reached only 0.995558; adjacent 54% and 56% cooldowns were also worse than 55%.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.33, "num_params_M": 50.3, "num_steps": 1874.0, "peak_vram_mb": 44908.2, "total_tokens_M": 491.3, "training_seconds": 300.2, "val_bpb": 0.986663}

RECENT RESULT
hypothesis: Retaining 5% of the initial learning rate at the end of the proven 55% cooldown will beat val_bpb 0.986663 by making the final updates effective without materially changing the successful early-training trajectory or throughput.
change: Raise the linear cooldown endpoint from zero to 5% of each parameter group’s initial learning rate.
mechanism: Nonzero terminal learning-rate floor
evidence_used: The late-weighted equal-area cooldown recovered to 0.987466 from the early-weighted cosine result of 0.990027 at comparable throughput, indicating that allocating more update magnitude to late refinement is beneficial; a small terminal floor tests that signal while preserving the best verified 55% cooldown duration.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.87, "num_params_M": 50.3, "num_steps": 1804.0, "peak_vram_mb": 44908.2, "total_tokens_M": 472.9, "training_seconds": 300.0, "val_bpb": 0.988551}

RECENT RESULT
hypothesis: Retaining 5% learning rate only for Muon matrix updates will beat val_bpb 0.986663 by enabling late matrix refinement without the aggressive terminal Adam embedding updates introduced by the unsuccessful all-group floor.
change: Preserve the verified 55% linear cooldown for AdamW groups while giving only Muon groups a 5% terminal learning-rate floor.
mechanism: Muon-only terminal learning-rate floor
evidence_used: The all-group 5% floor regressed to 0.988551; because the embedding Adam LR is 0.6 versus 0.04 for Muon matrices, that experiment left a much larger 0.03 terminal embedding LR, motivating a group-isolated test.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.27, "num_params_M": 50.3, "num_steps": 1823.0, "peak_vram_mb": 44908.2, "total_tokens_M": 477.9, "training_seconds": 300.1, "val_bpb": 0.988413}

RECENT RESULT
hypothesis: A 60% Muon cooldown paired with the verified 55% AdamW cooldown will beat val_bpb 0.986663 by reducing late matrix-update magnitude without weakening late embedding refinement.
change: Restore the best 55% cooldown for AdamW groups while beginning Muon’s linear cooldown five percentage points earlier.
mechanism: Group-decoupled linear cooldown
evidence_used: The Muon-only 5% terminal floor regressed from 0.986663 to 0.988413, showing that additional late Muon update magnitude is harmful; the globally applied 60% cooldown also reached a competitive 0.987466, motivating an isolated longer cooldown for Muon.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.96, "num_params_M": 50.3, "num_steps": 1856.0, "peak_vram_mb": 44908.2, "total_tokens_M": 486.5, "training_seconds": 300.1, "val_bpb": 0.986574}

RECENT RESULT
hypothesis: Extending Muon cooldown from 60% to 65% while restoring AdamW’s verified 55% cooldown will beat val_bpb 0.986574 by further suppressing harmful late matrix updates without reducing late embedding refinement.
change: Apply a 55% linear cooldown to AdamW groups and a 65% linear cooldown to Muon groups, retaining the proven architecture and 262K-token update geometry.
mechanism: Earlier Muon-only linear cooldown
evidence_used: Muon-only terminal LR increased val_bpb from 0.986663 to 0.988413, while extending only Muon’s cooldown to 60% improved it to 0.986574; testing 65% determines whether earlier matrix decay continues that improvement.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.57, "num_params_M": 50.3, "num_steps": 1837.0, "peak_vram_mb": 44908.2, "total_tokens_M": 481.6, "training_seconds": 300.0, "val_bpb": 0.98673}

RECENT RESULT
hypothesis: A 59% Muon cooldown paired with the verified 55% AdamW cooldown will beat val_bpb 0.986574, because the observed 55%, 60%, and 65% Muon cooldown results imply a local optimum slightly below 60%.
change: Begin Muon’s linear cooldown at 41% training progress instead of 40%, leaving architecture, batching, AdamW scheduling, and optimizer settings unchanged.
mechanism: Locally refined Muon-only cooldown
evidence_used: Muon cooldowns of 55%, 60%, and 65% produced val_bpb values of 0.986663, 0.986574, and 0.986730 respectively; quadratic interpolation of these flanking results places the estimated minimum near 59%.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.05, "num_params_M": 50.3, "num_steps": 1860.0, "peak_vram_mb": 44908.2, "total_tokens_M": 487.6, "training_seconds": 300.1, "val_bpb": 0.986424}

RECENT RESULT
hypothesis: A 58% Muon cooldown paired with the verified 55% AdamW cooldown will beat the best observed val_bpb of 0.986424.
change: Decouple optimizer-group schedules, preserving AdamW’s 55% cooldown while beginning Muon’s linear cooldown at 42% training progress.
mechanism: Locally refined Muon-only cooldown
evidence_used: Muon cooldowns of 55%, 59%, and 60% produced val_bpb values of 0.986663, 0.986424, and 0.986574; local quadratic interpolation places the estimated minimum near 58%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.72, "num_params_M": 50.3, "num_steps": 1845.0, "peak_vram_mb": 44908.2, "total_tokens_M": 483.7, "training_seconds": 300.2, "val_bpb": 0.986991}

RECENT RESULT
hypothesis: A 59.3% Muon cooldown paired with the verified 55% AdamW cooldown will beat the best observed val_bpb of 0.986424.
change: Decouple optimizer-group schedules, preserving AdamW’s 55% cooldown while beginning Muon’s linear cooldown at 40.7% training progress.
mechanism: Quadratically refined Muon-only cooldown
evidence_used: Muon cooldowns of 58%, 59%, and 60% produced val_bpb values of 0.986991, 0.986424, and 0.986574; quadratic interpolation of these closest observations estimates the minimum near 59.3%.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.17, "num_params_M": 50.3, "num_steps": 1866.0, "peak_vram_mb": 44908.2, "total_tokens_M": 489.2, "training_seconds": 300.1, "val_bpb": 0.98647}

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
