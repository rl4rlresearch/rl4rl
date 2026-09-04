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
verified_results: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.982852}
prior_hypothesis: Applying a fourth-root 7/8 rate reduction to the 512×2560 MLP projections will retain at least 445M training tokens and reduce val_bpb below 0.982763.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.98278}
prior_hypothesis: Restoring the verified-best 5× MLP while compensating its Muon learning rate for aspect-ratio scaling will retain at least 445M training tokens and reduce val_bpb below 0.982905.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 39.6, "num_params_M": 54.5, "num_steps": 1708.0, "peak_vram_mb": 49047.2, "total_tokens_M": 447.7, "training_seconds": 300.1, "val_bpb": 0.983097}
prior_hypothesis: Restoring the best 5× MLP and `sqrt(7/8)` contraction learning rate while applying a second `sqrt(7/8)` reduction to contraction weight decay will retain at least 445M tokens and reduce `val_bpb` below 0.982763.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 39.86, "num_params_M": 54.5, "num_steps": 1719.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.6, "training_seconds": 300.1, "val_bpb": 0.982763}
prior_hypothesis: Reducing only the 512×2560 MLP projection learning rate by sqrt(4.375/5) will retain at least 445M training tokens and reduce val_bpb below 0.982780.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the verified-best 5× MLP while compensating its Muon learning rate for aspect-ratio scaling will retain at least 445M training tokens and reduce val_bpb below 0.982905.
change: Restore the 2560-channel MLP and scale only its Muon parameter groups by sqrt(4.375/5), preserving the effective matrix update magnitude previously tuned at 4.375× while leaving attention and gate rates unchanged.
mechanism: Aspect-compensated MLP Muon rate
evidence_used: The 5× MLP achieved the best val_bpb, 0.982905, while 4.875×, 5.125×, 5.25×, and 5.625× were worse. The optimizer additionally multiplies Muon rates by the square root of matrix aspect ratio, so widening the MLP silently raised its effective rate despite the 93.75% rate having been selected on the 4.375× design.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.98278}

RECENT RESULT
hypothesis: On the restored 0.982780 SSSL/aspect-compensated baseline, making one of four heads in each full-context layer positionless will retain at least 445M training tokens and reduce val_bpb below 0.982780 by enabling content-addressed long-range retrieval without distance-dependent rotary phase, while the other three heads preserve ordered context.
change: Restore the verified-best SSSL attention pattern and compensated 5×-MLP Muon rate, then omit RoPE from the final head only in the two full-context layers.
mechanism: Positionless long-range retrieval head
evidence_used: The compensated SSSL design achieved the best result, 0.982780. Removing its intermediate global layer regressed to 0.983844, while adding another global layer regressed to 0.985077, suggesting the informative alternative is improving how the proven two global layers retrieve context rather than changing their count.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.24, "num_params_M": 54.5, "num_steps": 1692.0, "peak_vram_mb": 49046.7, "total_tokens_M": 443.5, "training_seconds": 300.1, "val_bpb": 0.984526}

RECENT RESULT
hypothesis: Compensating only the expanding MLP matrices will retain at least 445M training tokens and reduce val_bpb below 0.982780 by preserving the beneficial aspect correction without unnecessarily lowering the contraction matrices’ learning rate.
change: Apply the sqrt(4.375/5) Muon-rate adjustment only to 2560×512 MLP expansion weights; leave 512×2560 projection weights at the proven base rate.
mechanism: Direction-aware MLP Muon compensation
evidence_used: Aspect compensation improved the 5× MLP from 0.982905 to 0.982780, but the fused optimizer applies its aspect-ratio multiplier only when rows exceed columns. The current symmetric shape test therefore also reduces the projection rate even though that matrix receives no corresponding optimizer-side increase.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.49, "num_params_M": 54.5, "num_steps": 1703.0, "peak_vram_mb": 49047.2, "total_tokens_M": 446.4, "training_seconds": 300.2, "val_bpb": 0.983257}

RECENT RESULT
hypothesis: Reducing only the 512×2560 MLP projection learning rate by sqrt(4.375/5) will retain at least 445M training tokens and reduce val_bpb below 0.982780.
change: Apply aspect compensation exclusively to MLP contraction matrices while leaving expansion and attention rates unchanged.
mechanism: Projection-only MLP Muon rate compensation
evidence_used: Symmetric MLP compensation improved val_bpb from 0.982905 to 0.982780, whereas expansion-only compensation regressed to 0.983257, indicating the reduced projection rate may be the beneficial component.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.86, "num_params_M": 54.5, "num_steps": 1719.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.6, "training_seconds": 300.1, "val_bpb": 0.982763}

RECENT RESULT
hypothesis: Reducing only the 512×2560 MLP projection learning rate to 7/8 of the base rate will retain at least 445M training tokens and reduce val_bpb below 0.982763.
change: Restore the throughput-optimal 5× MLP and apply a second equal log-space reduction to only its contraction matrices, leaving expansion and attention rates unchanged.
mechanism: Stronger projection-only Muon-rate compensation
evidence_used: Projection-only sqrt(7/8) compensation achieved the best observed val_bpb, 0.982763, outperforming both the uncompensated 5× design at 0.982905 and expansion-only compensation at 0.983257; this tests whether the favorable projection-rate trend continues.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.48, "num_params_M": 54.5, "num_steps": 1702.0, "peak_vram_mb": 49047.2, "total_tokens_M": 446.2, "training_seconds": 300.0, "val_bpb": 0.983602}

RECENT RESULT
hypothesis: Applying a fourth-root 7/8 rate reduction to the 512×2560 MLP projections will retain at least 445M training tokens and reduce val_bpb below 0.982763.
change: Restore the proven SSSL attention pattern and reduce only MLP contraction learning rates by `(7/8)^0.25`, halfway in log space between the uncompensated and best verified rates.
mechanism: Projection-only Muon-rate interpolation
evidence_used: Projection-only `sqrt(7/8)` achieved the best val_bpb, 0.982763; no reduction reached 0.982905 and the stronger 7/8 reduction regressed to 0.983602, motivating interpolation between the first two rates.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.982852}

RECENT RESULT
hypothesis: Setting the MLP contraction rate to `(7/8)^(3/8)` will retain at least 445M training tokens and reduce `val_bpb` below 0.982763.
change: Interpolate the 512×2560 MLP projection learning rate halfway in log space between the verified square-root and fourth-root reductions.
mechanism: Projection-only Muon-rate refinement
evidence_used: The square-root reduction achieved the best `val_bpb` of 0.982763, while the nearby fourth-root reduction reached 0.982852 and the stronger 7/8 reduction regressed to 0.983602, motivating a finer search near the best rate on its less-reduced side.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.49, "num_params_M": 54.5, "num_steps": 1703.0, "peak_vram_mb": 49047.2, "total_tokens_M": 446.4, "training_seconds": 300.1, "val_bpb": 0.983407}

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
