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
verified_results: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.98278}
prior_hypothesis: Restoring the verified-best 5× MLP while compensating its Muon learning rate for aspect-ratio scaling will retain at least 445M training tokens and reduce val_bpb below 0.982905.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.982852}
prior_hypothesis: Applying a fourth-root 7/8 rate reduction to the 512×2560 MLP projections will retain at least 445M training tokens and reduce val_bpb below 0.982763.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 39.6, "num_params_M": 54.5, "num_steps": 1708.0, "peak_vram_mb": 49047.2, "total_tokens_M": 447.7, "training_seconds": 300.1, "val_bpb": 0.983097}
prior_hypothesis: Restoring the best 5× MLP and `sqrt(7/8)` contraction learning rate while applying a second `sqrt(7/8)` reduction to contraction weight decay will retain at least 445M tokens and reduce `val_bpb` below 0.982763.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 40.3, "num_params_M": 54.5, "num_steps": 1738.0, "peak_vram_mb": 49047.2, "total_tokens_M": 455.6, "training_seconds": 300.1, "val_bpb": 0.982713}
prior_hypothesis: Applying the verified `sqrt(7/8)` Muon-rate reduction to both zero-initialized MLP and attention output projections will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.

## Recent verification evidence

RECENT RESULT
hypothesis: Setting only the 512×2560 MLP contraction rate to `(7/8)^0.75` will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.
change: Restore the expansion and attention matrices to the base Muon rate, while moving the contraction rate halfway in log space between the best verified `sqrt(7/8)` rate and the over-reduced `7/8` rate.
mechanism: Stronger-side projection Muon-rate interpolation
evidence_used: Projection-only `sqrt(7/8)` achieved the best observed `val_bpb` of 0.982763, while `7/8` regressed to 0.983602; less-reduced fourth-root and three-eighths rates also failed to improve, so the untested stronger-side midpoint is the most informative local refinement.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.81, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.2, "val_bpb": 0.9831}

RECENT RESULT
hypothesis: Restoring the best verified `sqrt(7/8)` MLP contraction learning rate while preserving the baseline contraction weight-decay magnitude will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.
change: Restore projection-only `sqrt(7/8)` Muon-rate compensation and inversely scale that group’s weight decay so the reduced learning rate does not also weaken cautious decay.
mechanism: Decoupled projection learning rate and weight decay
evidence_used: Projection-only `sqrt(7/8)` achieved the best observed `val_bpb` of 0.982763, while nearby weaker and stronger reductions regressed; because Muon multiplies both gradient and decay updates by the group learning rate, those trials unintentionally coupled two variables, motivating this controlled separation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.74, "num_params_M": 54.5, "num_steps": 1714.0, "peak_vram_mb": 49047.2, "total_tokens_M": 449.3, "training_seconds": 300.1, "val_bpb": 0.982957}

RECENT RESULT
hypothesis: Restoring the best 5× MLP and `sqrt(7/8)` contraction learning rate while applying a second `sqrt(7/8)` reduction to contraction weight decay will retain at least 445M tokens and reduce `val_bpb` below 0.982763.
change: Restore 2560-channel MLPs, apply the verified-best projection-only learning-rate compensation, and reduce only those projections’ scheduled weight decay so their effective decay update is 7/8 of baseline.
mechanism: Stronger projection-only cautious-decay reduction
evidence_used: Projection-only `sqrt(7/8)` learning-rate compensation achieved the best `val_bpb` of 0.982763, while restoring the projection’s baseline effective decay regressed to 0.982957, motivating a controlled move toward weaker projection decay.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.6, "num_params_M": 54.5, "num_steps": 1708.0, "peak_vram_mb": 49047.2, "total_tokens_M": 447.7, "training_seconds": 300.1, "val_bpb": 0.983097}

RECENT RESULT
hypothesis: Increasing only the 512×2560 MLP projection weight decay by `(8/7)^0.25` while retaining its verified-best `sqrt(7/8)` learning rate will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.
change: Move the projection’s effective cautious-decay magnitude halfway in log space from the current best toward baseline, without changing its learning rate or other parameter groups.
mechanism: Projection cautious-decay log-space interpolation
evidence_used: The current effective projection-decay factor achieved 0.982763, while reducing it to 7/8 regressed to 0.983097 and restoring it to 1.0 regressed less severely to 0.982957; this brackets the optimum and motivates refinement on the better upward side.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.5, "num_params_M": 54.5, "num_steps": 1703.0, "peak_vram_mb": 49047.2, "total_tokens_M": 446.4, "training_seconds": 300.1, "val_bpb": 0.98361}

RECENT RESULT
hypothesis: Starting from the best verified projection-only Muon compensation, neutral-initialized per-token head gates will retain at least 445M training tokens and reduce `val_bpb` below 0.982763 by letting attention selectively suppress or amplify retrieved information before it enters the residual stream.
change: Replace the assumption that every attention head must always write with a learned query-dependent no-op/amplification gate, while restoring the verified-best contraction-only learning-rate adjustment.
mechanism: Query-conditioned attention write gates
evidence_used: Projection-only `sqrt(7/8)` compensation achieved 0.982763. Changing global-layer counts and making a whole head positionless regressed, motivating a mechanism that improves how the proven attention pattern writes context without changing its receptive fields, rotary representation, or number of attention calls.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.65, "num_params_M": 54.5, "num_steps": 1667.0, "peak_vram_mb": 51114.2, "total_tokens_M": 437.0, "training_seconds": 300.2, "val_bpb": 0.983601}

RECENT RESULT
hypothesis: Restoring the best `sqrt(7/8)` contraction rate and doubling only contraction matrices’ variance-estimation horizon will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.
change: Use the verified-best learning rate for 512×2560 MLP projections and set their NorMuon `beta2` to `sqrt(0.95)`, leaving all other groups unchanged.
mechanism: Projection-specific NorMuon variance smoothing
evidence_used: Projection-only `sqrt(7/8)` achieved the best observed `val_bpb` of 0.982763, while nearby learning-rate and weight-decay refinements regressed; projection-specific variance smoothing tests an independent stabilization axis without adding model computation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.47, "num_params_M": 54.5, "num_steps": 1702.0, "peak_vram_mb": 49047.2, "total_tokens_M": 446.2, "training_seconds": 300.1, "val_bpb": 0.98334}

RECENT RESULT
hypothesis: On the restored best projection-only learning-rate and decay configuration, reducing contraction-matrix NorMuon `beta2` to 0.90 will retain at least 445M tokens and reduce `val_bpb` below 0.982763 by adapting variance normalization faster.
change: Restore baseline weight decay for all Muon groups, retain the verified-best `sqrt(7/8)` contraction learning rate, and halve only contraction matrices’ variance-estimation horizon.
mechanism: Projection-specific short-horizon NorMuon adaptation
evidence_used: Projection-only `sqrt(7/8)` reached 0.982763, while doubling its variance-estimation horizon with `beta2=sqrt(0.95)` regressed to 0.983340; testing the opposite horizon direction is the clearest remaining projection-specific optimizer ablation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.66, "num_params_M": 54.5, "num_steps": 1710.0, "peak_vram_mb": 49047.2, "total_tokens_M": 448.3, "training_seconds": 300.1, "val_bpb": 0.983289}

RECENT RESULT
hypothesis: Using four orthogonalization steps for non-projection matrices while preserving five steps and the verified-best rate for MLP contractions will retain at least 450M training tokens and reduce `val_bpb` below 0.982763.
change: Keep the sensitive 512×2560 MLP projections unchanged, but reduce Muon’s Polar Express iterations from five to four for attention and MLP expansion matrices to trade marginal orthogonalization accuracy for higher fixed-time throughput.
mechanism: Selective Polar Express step reduction
evidence_used: Projection-only `sqrt(7/8)` compensation achieved the best result at 0.982763 and 450.6M tokens, while subsequent projection learning-rate, decay, and variance refinements regressed; this preserves that projection configuration and tests a separate throughput axis.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.92, "num_params_M": 54.5, "num_steps": 1721.0, "peak_vram_mb": 49047.2, "total_tokens_M": 451.1, "training_seconds": 300.1, "val_bpb": 0.985231}

RECENT RESULT
hypothesis: Restoring the verified-best projection-only `sqrt(7/8)` rate and using an equal-area cosine warmdown will retain at least 445M tokens and reduce `val_bpb` below 0.982763 by reducing disruptive terminal updates without lowering the total scheduled learning-rate budget.
change: Leave MLP expansion and attention rates unchanged, compensate only MLP contractions, and replace linear LR warmdown with a cosine curve having identical endpoints and mean multiplier.
mechanism: Equal-area cosine warmdown with projection-only Muon compensation
evidence_used: Projection-only compensation achieved the best result, 0.982763 at 450.6M tokens, while subsequent rate, decay, variance, and orthogonalization refinements regressed; this preserves that configuration and tests an independent, throughput-neutral scheduling axis.
result: the implementation could not be verified

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Restoring the best projection-only `sqrt(7/8)` Muon rate and replacing linear warmdown with an equal-area cosine curve will retain at least 445M tokens and reduce `val_bpb` below 0.982763 by suppressing disruptive terminal updates without reducing the total learning-rate budget.
change: Restore baseline scheduled weight decay for every Muon group, retain contraction-only learning-rate compensation, and use cosine rather than linear decay over the final half of training.
mechanism: Equal-area cosine warmdown
evidence_used: Projection-only `sqrt(7/8)` compensation achieved the best verified `val_bpb` of 0.982763 at 450.6M tokens; later projection optimizer refinements regressed, while the independent equal-area cosine schedule attempt produced no verifiable implementation and therefore remains untested.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.35, "num_params_M": 54.5, "num_steps": 1697.0, "peak_vram_mb": 49047.2, "total_tokens_M": 444.9, "training_seconds": 300.1, "val_bpb": 0.986445}

RECENT RESULT
hypothesis: Applying the verified `sqrt(7/8)` Muon-rate reduction to both zero-initialized MLP and attention output projections will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.
change: Split transformer matrices by residual-output role; retain base rates for attention input projections and MLP expansions while reducing both `c_proj` families by `sqrt(7/8)`.
mechanism: Residual-output projection Muon-rate compensation
evidence_used: Contraction-only compensation produced the best result, 0.982763 at 450.6M tokens, while nearby rate, decay, and variance refinements regressed. Attention `c_proj` shares the contraction’s zero initialization and residual-write role, making it the closest untested extension without adding forward computation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 40.3, "num_params_M": 54.5, "num_steps": 1738.0, "peak_vram_mb": 49047.2, "total_tokens_M": 455.6, "training_seconds": 300.1, "val_bpb": 0.982713}



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
