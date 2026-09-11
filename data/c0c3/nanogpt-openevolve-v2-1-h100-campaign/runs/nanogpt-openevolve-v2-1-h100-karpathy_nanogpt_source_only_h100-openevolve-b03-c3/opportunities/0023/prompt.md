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
verified_results: {"depth": 8.0, "mfu_percent": 38.71, "num_params_M": 50.3, "num_steps": 927.0, "peak_vram_mb": 45576.2, "total_tokens_M": 486.0, "training_seconds": 300.1, "val_bpb": 0.993865}
prior_hypothesis: Restricting neutral per-head output gates to the two full-context layers will lower val_bpb below 0.993637 while restoring training volume toward 490M tokens.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 37.17, "num_params_M": 50.3, "num_steps": 891.0, "peak_vram_mb": 47380.5, "total_tokens_M": 467.1, "training_seconds": 300.2, "val_bpb": 0.992628}
prior_hypothesis: Combining the best learned two-lag embedding convolution with proven all-layer attention-output gates will lower val_bpb below 0.993485 while retaining roughly 470M or more training tokens because the mechanisms control complementary local and retrieved context paths.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 37.74, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 47124.2, "total_tokens_M": 474.0, "training_seconds": 300.1, "val_bpb": 0.994532}
prior_hypothesis: Fusing the proven first-32-channel attention-output gate with the existing value gate will lower val_bpb below 0.993637 while retaining at least roughly 482M trained tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 37.84, "num_params_M": 50.3, "num_steps": 907.0, "peak_vram_mb": 45832.5, "total_tokens_M": 475.5, "training_seconds": 300.3, "val_bpb": 0.992949}
prior_hypothesis: Applying the learned two-lag embedding convolution while retaining output gates only in full-context layers will lower val_bpb below 0.992628 and restore training volume to roughly 480M tokens or more.

## Recent verification evidence

RECENT RESULT
hypothesis: Conditioning the proven per-head attention-output gates on all 512 hidden channels instead of an arbitrary 32-channel slice will lower val_bpb below 0.993637 while retaining roughly 480M trained tokens.
change: Add neutral-initialized per-head attention-output gates using the full normalized token state, and use the 11.75 softcap from the strongest gated reference design.
mechanism: Full-state query-conditioned attention-head gating
evidence_used: Reference Design 3’s 32-channel attention-output gates improved val_bpb from 0.995200 to 0.993637 while still training 482.3M tokens; this motivates preserving the gate and testing whether complete-state conditioning improves its context-selection signal.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.24, "num_params_M": 50.3, "num_steps": 892.0, "peak_vram_mb": 47124.4, "total_tokens_M": 467.7, "training_seconds": 300.1, "val_bpb": 0.998184}

RECENT RESULT
hypothesis: Sampling 32 channels across the full normalized token state will preserve the proven gate’s throughput while lowering val_bpb below 0.993637 by providing broader conditioning than the first-channel slice.
change: Add neutral-initialized per-head attention-output gates conditioned on 32 evenly spaced hidden channels and use the strongest gated design’s 11.75 softcap.
mechanism: Channel-stratified query-conditioned attention-head gating
evidence_used: The 32-channel output gate achieved 0.993637 on 482.3M tokens, while full-state conditioning regressed to 0.998184 and 467.7M tokens; stratified sampling retains the efficient 32-channel projection while incorporating information across the full state.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.03, "num_params_M": 50.3, "num_steps": 911.0, "peak_vram_mb": 47124.2, "total_tokens_M": 477.6, "training_seconds": 300.1, "val_bpb": 0.994195}

RECENT RESULT
hypothesis: Conditioning each attention-output gate exclusively on 32 channels from its corresponding head will lower val_bpb below 0.993637 while retaining roughly 480M trained tokens.
change: Reuse the existing gate weights and compute each head’s gate from its own normalized hidden-state slice instead of applying a dense gate to the shared first 32 channels.
mechanism: Head-aligned local attention-output gating
evidence_used: Shared first-slice gating achieved 0.993637, while full-state conditioning regressed to 0.998184 and globally stratified conditioning reached 0.994195; this motivates distributing gate inputs across the state while avoiding cross-head global mixing.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.63, "num_params_M": 50.3, "num_steps": 902.0, "peak_vram_mb": 47140.2, "total_tokens_M": 472.9, "training_seconds": 300.3, "val_bpb": 0.995123}

RECENT RESULT
hypothesis: Extending the proven first-slice attention gate with a neutral token-dependent MLP-branch gate will lower val_bpb below 0.993637 while retaining roughly 470M or more trained tokens.
change: Restore first-32-channel attention gating and add a zero-initialized scalar sigmoid gate to each MLP residual branch.
mechanism: Token-conditioned MLP residual gating
evidence_used: First-slice attention gating achieved the best result, 0.993637 on 482.3M tokens, whereas stratified, full-state, and head-aligned conditioning regressed; this preserves that conditioning and tests the same adaptive residual control on the complementary MLP branch.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.98, "num_params_M": 50.3, "num_steps": 910.0, "peak_vram_mb": 49176.2, "total_tokens_M": 477.1, "training_seconds": 300.2, "val_bpb": 0.994155}

RECENT RESULT
hypothesis: Fusing the proven first-32-channel attention-output gate with the existing value gate will lower val_bpb below 0.993637 while retaining at least roughly 482M trained tokens.
change: Add neutral per-head attention-output gates, jointly compute value and output gate logits on value-embedding layers to reduce tiny-projection overhead, and use the strongest gated design’s 11.75 softcap.
mechanism: Fused query-conditioned attention gating
evidence_used: First-slice attention-output gating achieved the best observed val_bpb, 0.993637 on 482.3M tokens; adding a separate MLP gate reduced volume to 477.1M and regressed to 0.994155, motivating a cheaper implementation restricted to the proven attention mechanism.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.74, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 47124.2, "total_tokens_M": 474.0, "training_seconds": 300.1, "val_bpb": 0.994532}

RECENT RESULT
hypothesis: Increasing the proven first-32-channel output gate’s neutral slope by 50% will improve head specialization within the fixed training window and lower val_bpb below 0.993637 while retaining roughly 482M trained tokens.
change: Restore per-head attention-output gating and the 11.75 softcap, but replace the 2×sigmoid gate with an equal-range, neutral tanh gate whose slope is 0.75 instead of 0.5.
mechanism: Faster-adapting neutral attention-head gating
evidence_used: First-slice gating achieved the best result, 0.993637 on 482.3M tokens, while changing its conditioning source or adding another gated branch regressed; this isolates gate responsiveness while preserving the successful conditioning signal and output range.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.79, "num_params_M": 50.3, "num_steps": 905.0, "peak_vram_mb": 47124.2, "total_tokens_M": 474.5, "training_seconds": 300.0, "val_bpb": 0.996193}

RECENT RESULT
hypothesis: Preserving the proven gate’s neutral slope while limiting each head’s multiplier to 0.5–1.5 will prevent excessive head suppression or amplification and lower val_bpb below 0.993637 without reducing throughput.
change: Replace the 0–2 sigmoid output gate with a neutral, equal-slope 0.5–1.5 tanh gate; batching, conditioning channels, softcap, and optimization remain unchanged.
mechanism: Bounded-amplitude attention-head gating
evidence_used: The original first-32-channel sigmoid gate achieved the best val_bpb of 0.993637, while increasing its neutral slope to 0.75 regressed to 0.996193. Matching the successful 0.5 initial slope while narrowing only the attainable amplitude isolates whether large gate excursions caused that regression.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.88, "num_params_M": 50.3, "num_steps": 907.0, "peak_vram_mb": 47124.2, "total_tokens_M": 475.5, "training_seconds": 300.0, "val_bpb": 0.994766}

RECENT RESULT
hypothesis: Restricting neutral per-head output gates to the two full-context layers will lower val_bpb below 0.993637 while restoring training volume toward 490M tokens.
change: Remove the unsuccessful MLP gates and apply the proven first-32-channel attention-output gate only on layers whose effective attention window is full context.
mechanism: Long-context-selective attention-head gating
evidence_used: Per-head attention-output gating achieved the best val_bpb, 0.993637, but reduced volume from 497.0M to 482.3M tokens; alternative gate conditioning and added MLP gating regressed, motivating preservation of the proven gate only where contextual selection is most consequential.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.71, "num_params_M": 50.3, "num_steps": 927.0, "peak_vram_mb": 45576.2, "total_tokens_M": 486.0, "training_seconds": 300.1, "val_bpb": 0.993865}

RECENT RESULT
hypothesis: Restoring the proven unfused first-32-channel output gates and optimizing only those tiny gate matrices with AdamW at 0.01 will lower val_bpb below 0.993637 while retaining at least 480M trained tokens.
change: Undo the unsuccessful value/output forward fusion, exclude output-gate matrices from Muon, and place them in a dedicated AdamW parameter group.
mechanism: Adam-optimized attention-output gating
evidence_used: Unfused all-layer output gating achieved the best val_bpb, 0.993637 on 482.3M tokens, while the fused implementation regressed to 0.994532 on 474.0M; conditioning and amplitude changes also regressed, motivating preservation of the successful gate computation while testing whether Muon is ill-suited to its tiny 4x32 matrices.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.76, "num_params_M": 50.3, "num_steps": 905.0, "peak_vram_mb": 47124.2, "total_tokens_M": 474.5, "training_seconds": 300.2, "val_bpb": 0.994766}

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
