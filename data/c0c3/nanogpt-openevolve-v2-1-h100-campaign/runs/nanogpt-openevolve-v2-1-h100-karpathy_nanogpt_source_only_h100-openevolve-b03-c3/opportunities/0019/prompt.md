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
verified_results: {"depth": 8.0, "mfu_percent": 37.74, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 47124.2, "total_tokens_M": 474.0, "training_seconds": 300.1, "val_bpb": 0.994532}
prior_hypothesis: Fusing the proven first-32-channel attention-output gate with the existing value gate will lower val_bpb below 0.993637 while retaining at least roughly 482M trained tokens.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 39.58, "num_params_M": 50.3, "num_steps": 948.0, "peak_vram_mb": 45060.2, "total_tokens_M": 497.0, "training_seconds": 300.2, "val_bpb": 0.995558}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 38.71, "num_params_M": 50.3, "num_steps": 927.0, "peak_vram_mb": 45576.2, "total_tokens_M": 486.0, "training_seconds": 300.1, "val_bpb": 0.993865}
prior_hypothesis: Restricting neutral per-head output gates to the two full-context layers will lower val_bpb below 0.993637 while restoring training volume toward 490M tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 38.41, "num_params_M": 50.3, "num_steps": 920.0, "peak_vram_mb": 47124.2, "total_tokens_M": 482.3, "training_seconds": 300.1, "val_bpb": 0.993637}
prior_hypothesis: Replacing fixed-amplitude attention outputs with lightweight, token-dependent per-head gates will improve context selection and lower val_bpb below 0.995200 without materially reducing the roughly 493M-token training volume.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising the softcap from 10 to 11 will preserve beneficial confidence control while slightly reducing saturation, lowering val_bpb below 0.995334.
change: Change the FP32 training-and-validation logit softcap from 10 to 11, leaving architecture, batching, and optimization unchanged.
mechanism: Intermediate finite-logit confidence regularization
evidence_used: Cap 10 achieved the best observed val_bpb of 0.995334, outperforming cap 9 at 0.995704 and cap 15 at 0.995558; cap 11 probes the narrower, more promising side of the apparent optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.34, "num_params_M": 50.3, "num_steps": 753.0, "peak_vram_mb": 45060.2, "total_tokens_M": 394.8, "training_seconds": 300.3, "val_bpb": 1.009914}

RECENT RESULT
hypothesis: A softcap of 12 will lower val_bpb below 0.995334 at comparable throughput by relaxing cap 10 slightly without approaching the weaker regularization of cap 15.
change: Change the FP32 training-and-validation logit softcap from 9 to 12, leaving architecture, batching, and optimization unchanged.
mechanism: Intermediate tanh confidence regularization
evidence_used: Cap 10 achieved 0.995334, while full-throughput cap 15 reached 0.995558 versus 0.995704 for cap 9, suggesting the more promising side of cap 10 is upward; the cap-11 run trained only 394.8M tokens versus 493.4M and therefore did not isolate that region.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.26, "num_params_M": 50.3, "num_steps": 940.0, "peak_vram_mb": 45060.2, "total_tokens_M": 492.8, "training_seconds": 300.1, "val_bpb": 0.99523}

RECENT RESULT
hypothesis: A softcap of 11.75 on the proven 524,288-token batch will lower val_bpb below 0.995230 by refining the apparent optimum just below cap 12.
change: Restore two-microbatch gradient accumulation and change the training-and-validation FP32 logit softcap from 15 to 11.75.
mechanism: Full-batch near-optimal logit confidence regularization
evidence_used: Cap 12 achieved the best val_bpb of 0.995230, outperforming cap 10 at 0.995334 and cap 15 at 0.995558; meanwhile, halving the batch reduced trained tokens to 376.2M and regressed val_bpb to 1.001276.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.26, "num_params_M": 50.3, "num_steps": 940.0, "peak_vram_mb": 45060.2, "total_tokens_M": 492.8, "training_seconds": 300.0, "val_bpb": 0.9952}

RECENT RESULT
hypothesis: Replacing fixed-amplitude attention outputs with lightweight, token-dependent per-head gates will improve context selection and lower val_bpb below 0.995200 without materially reducing the roughly 493M-token training volume.
change: Add a zero-initialized sigmoid gate to every attention layer that conditionally attenuates or amplifies each retrieved head. The old assumption is that attention weights alone adequately control contextual influence; the new approach separately learns whether each head’s retrieved context should enter the residual stream.
mechanism: Query-conditioned attention head gating
evidence_used: Quarter-context attention reduced training volume from 497.0M to 401.1M tokens and regressed val_bpb to 1.009577, so discarding context is unpromising. Meanwhile, softcap refinements have produced only marginal gains around 0.9952. This patch preserves the proven context windows, batching, and softcap while testing a distinct learned context-selection mechanism with negligible parameter and compute cost.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.41, "num_params_M": 50.3, "num_steps": 920.0, "peak_vram_mb": 47124.2, "total_tokens_M": 482.3, "training_seconds": 300.1, "val_bpb": 0.993637}

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
