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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"depth": 8.0, "mfu_percent": 38.33, "num_params_M": 50.3, "num_steps": 2651.0, "peak_vram_mb": 31031.5, "total_tokens_M": 477.8, "training_seconds": 300.0, "val_bpb": 0.982576}
prior_hypothesis: Further concentrating Muon weight decay early with a quartic schedule while preserving LR-weighted cumulative exposure will reduce val_bpb below 0.983066.

## Recent verification evidence

RECENT RESULT
hypothesis: Shortening the Muon momentum ramp to 206 steps will reach the noise-suppressing 0.95 momentum sooner and lower val_bpb below 0.984068.
change: Reduce only the Muon momentum-ramp duration from 300 to 206 steps.
mechanism: Earlier Muon momentum smoothing
evidence_used: Extending the ramp from 300 to approximately 436 steps regressed val_bpb from 0.984068 to 0.984170; testing the inverse batch-ratio scaling, 300 × 176/256 ≈ 206, probes the more-promising shorter-ramp direction while preserving every other verified setting.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.47, "num_params_M": 50.3, "num_steps": 2592.0, "peak_vram_mb": 31031.5, "total_tokens_M": 467.1, "training_seconds": 300.1, "val_bpb": 0.985398}

RECENT RESULT
hypothesis: Extending the Muon momentum ramp from 300 to 362 steps will lower val_bpb below 0.984068 by approaching the empirical minimum bracketed by the verified 206-, 300-, and 436-step ramps.
change: Change only the Muon momentum-ramp duration to 362 optimizer steps, preserving the best architecture, batch size, learning rates, and weight decay.
mechanism: Three-point interpolated Muon momentum ramp
evidence_used: The 300-step ramp achieved 0.984068, while 436 steps remained close at 0.984170 and 206 steps regressed substantially to 0.985398; quadratic interpolation of this asymmetric bracket places the estimated minimum near 362 steps.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.34, "num_params_M": 50.3, "num_steps": 2652.0, "peak_vram_mb": 31031.5, "total_tokens_M": 478.0, "training_seconds": 300.1, "val_bpb": 0.984152}

RECENT RESULT
hypothesis: Reducing short-layer context from 1,024 to 768 tokens while retaining two full-context layers will increase token throughput enough to lower val_bpb below 0.984068.
change: Set each `S` attention window to three-eighths of the 2,048-token sequence length.
mechanism: Throughput-biased three-eighths sliding attention
evidence_used: The best design processed 478.5M tokens at 0.984068, while the slower delay-line design processed 442.1M and regressed to 0.986236; shortening only the six local-attention windows targets throughput without removing full-context attention.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.11, "num_params_M": 50.3, "num_steps": 2672.0, "peak_vram_mb": 31031.5, "total_tokens_M": 481.6, "training_seconds": 300.0, "val_bpb": 0.984407}

RECENT RESULT
hypothesis: Increasing local-attention windows from 1,024 to 1,280 tokens will recover more useful context than the modest throughput reduction costs, lowering val_bpb below 0.984068.
change: Set every `S` attention window to five-eighths of the 2,048-token sequence length.
mechanism: Context-favoring five-eighths sliding attention
evidence_used: Reducing `S` windows to 768 increased total tokens from 478.5M to 481.6M but worsened val_bpb from 0.984068 to 0.984407, indicating that additional context was more valuable than the measured throughput gain and motivating a test in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.02, "num_params_M": 50.3, "num_steps": 2597.0, "peak_vram_mb": 31031.5, "total_tokens_M": 468.0, "training_seconds": 300.1, "val_bpb": 0.985631}

RECENT RESULT
hypothesis: Moving the first full-context attention layer one block earlier while retaining six half-context and two full-context layers will give global information one additional nonlinear transformation and lower val_bpb below 0.984068 without reducing throughput.
change: Replace the repeated four-layer window pattern with an explicit eight-layer pattern that shifts the first full-context layer from index 3 to index 2 while keeping the final layer full-context.
mechanism: Compute-neutral earlier global-context injection
evidence_used: Half-context windows outperformed both 768-token windows at 0.984407 and 1,280-token windows at 0.985631, suggesting the 1,024-token width is already near the useful tradeoff; changing full-context placement while preserving window widths and counts isolates whether earlier global integration is more effective.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.04, "num_params_M": 50.3, "num_steps": 2632.0, "peak_vram_mb": 31031.5, "total_tokens_M": 474.3, "training_seconds": 300.1, "val_bpb": 0.985581}

RECENT RESULT
hypothesis: Replacing the fixed additive bigram shortcut with an initially equivalent context-conditioned lexical scale will lower val_bpb below 0.984068 by letting the full contextual state determine which current-token features should influence each prediction channel.
change: Add a zero-initialized readout matrix that computes a bounded channel-wise lexical adjustment from the final contextual state, producing an explicit bilinear interaction between context and the current-token value embedding while preserving baseline behavior at initialization.
mechanism: Context-conditioned bilinear lexical readout
evidence_used: The direct lexical expert improved val_bpb from 0.995511 to 0.994364, but the lexical-only multiplicative trigram readout failed at 0.987766 versus 0.987386. This tests a different load-bearing assumption—fixed additive separation of lexical and contextual evidence—by making lexical influence conditional on the fully contextualized representation rather than multiplying lexical embeddings alone.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.95, "num_params_M": 50.6, "num_steps": 2608.0, "peak_vram_mb": 31737.7, "total_tokens_M": 470.0, "training_seconds": 300.1, "val_bpb": 0.985305}

RECENT RESULT
hypothesis: Redistributing the verified decay exposure uniformly across training will preserve useful late-stage regularization and lower val_bpb below 0.984068.
change: Replace the linearly vanishing 0.126 Muon weight decay with a constant 0.077 coefficient, exactly matching its LR-weighted schedule integral under the existing half-window warmdown.
mechanism: Integral-matched constant cautious Muon decay
evidence_used: Weight decay materially affected val_bpb, with 0.126 reaching 0.984068; neighboring coefficient refinements were noisy, motivating an exposure-matched test of decay timing instead of another amplitude interpolation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.1, "num_params_M": 50.3, "num_steps": 2636.0, "peak_vram_mb": 31031.5, "total_tokens_M": 475.1, "training_seconds": 300.1, "val_bpb": 0.986934}

RECENT RESULT
hypothesis: Concentrating the verified decay exposure earlier while reducing late-stage regularization will lower val_bpb below 0.984068.
change: Replace linear Muon decay with a quadratic schedule and raise its initial coefficient to 0.17884, preserving the original LR-weighted cumulative decay exposure.
mechanism: Integral-matched front-loaded quadratic cautious decay
evidence_used: Redistributing the same exposure uniformly with constant 0.077 decay worsened val_bpb from 0.984068 to 0.986934, indicating that stronger early decay and vanishing late decay are beneficial; quadratic decay tests that direction without changing total exposure.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.26, "num_params_M": 50.3, "num_steps": 2647.0, "peak_vram_mb": 31031.5, "total_tokens_M": 477.1, "training_seconds": 300.1, "val_bpb": 0.983444}

RECENT RESULT
hypothesis: Further concentrating Muon weight decay early with a cubic schedule while preserving cumulative LR-weighted exposure will reduce val_bpb below 0.983444.
change: Replace quadratic cautious decay with cubic decay and raise its peak coefficient from 0.17884 to 0.233924 to preserve the verified schedule’s LR-weighted integral.
mechanism: Integral-matched cubic front-loaded cautious decay
evidence_used: Constant exposure-matched decay regressed to 0.986934, whereas front-loaded quadratic decay improved val_bpb from 0.984068 to 0.983444, directly motivating a controlled test of stronger front-loading.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.1, "num_params_M": 50.3, "num_steps": 2635.0, "peak_vram_mb": 31031.5, "total_tokens_M": 474.9, "training_seconds": 300.0, "val_bpb": 0.983066}

RECENT RESULT
hypothesis: Further concentrating Muon weight decay early with a quartic schedule while preserving LR-weighted cumulative exposure will reduce val_bpb below 0.983066.
change: Replace cubic cautious decay with quartic decay and raise its peak coefficient from 0.233924 to 0.290262 to preserve cumulative LR-weighted decay exposure.
mechanism: Integral-matched quartic front-loaded cautious decay
evidence_used: Exposure-matched quadratic decay improved val_bpb from 0.984068 to 0.983444, and cubic decay further improved it to 0.983066, motivating another controlled step toward stronger early concentration.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.33, "num_params_M": 50.3, "num_steps": 2651.0, "peak_vram_mb": 31031.5, "total_tokens_M": 477.8, "training_seconds": 300.0, "val_bpb": 0.982576}

RECENT RESULT
hypothesis: Further concentrating Muon weight decay early with a quintic schedule while preserving LR-weighted cumulative exposure will reduce val_bpb below 0.982576.
change: Replace quartic cautious decay with quintic decay and raise its peak coefficient from 0.290262 to 0.347275.
mechanism: Integral-matched quintic front-loaded cautious decay
evidence_used: Exposure-matched quadratic, cubic, and quartic schedules successively improved val_bpb from 0.984068 to 0.983444, 0.983066, and 0.982576, supporting one more controlled increase in early decay concentration.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.76, "num_params_M": 50.3, "num_steps": 2612.0, "peak_vram_mb": 31031.5, "total_tokens_M": 470.7, "training_seconds": 300.1, "val_bpb": 0.983299}

RECENT RESULT
hypothesis: Setting the decay exponent to 3.9 with matched cumulative exposure will lower val_bpb below 0.982576.
change: Replace quartic Muon decay with a 3.9-power schedule and reduce its peak coefficient to 0.284591, preserving LR-weighted cumulative decay exposure.
mechanism: Integral-matched interpolated 3.9-power cautious decay
evidence_used: Cubic decay achieved 0.983066, quartic improved to 0.982576, and quintic regressed to 0.983299; quadratic interpolation of this bracket estimates the optimum near exponent 3.9.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.08, "num_params_M": 50.3, "num_steps": 2634.0, "peak_vram_mb": 31031.5, "total_tokens_M": 474.7, "training_seconds": 300.1, "val_bpb": 0.982956}



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
