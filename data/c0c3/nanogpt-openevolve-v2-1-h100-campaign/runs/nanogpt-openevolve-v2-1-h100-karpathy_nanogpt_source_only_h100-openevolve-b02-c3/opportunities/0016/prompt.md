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
verified_results: {"depth": 8.0, "mfu_percent": 39.58, "num_params_M": 50.3, "num_steps": 948.0, "peak_vram_mb": 45060.2, "total_tokens_M": 497.0, "training_seconds": 300.2, "val_bpb": 0.995558}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 37.67, "num_params_M": 50.3, "num_steps": 979.0, "peak_vram_mb": 45060.2, "total_tokens_M": 513.3, "training_seconds": 300.2, "val_bpb": 0.993287}
prior_hypothesis: Reducing the six short-attention layers from 1024 to 512 tokens will increase throughput beyond 497M tokens while the two full-context layers preserve enough long-range modeling to beat 0.995558 val_bpb.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 37.52, "num_params_M": 50.3, "num_steps": 975.0, "peak_vram_mb": 45060.2, "total_tokens_M": 511.2, "training_seconds": 300.1, "val_bpb": 0.993714}
prior_hypothesis: Restoring full KV heads and beginning warmdown at 40% of the training window will preserve roughly 513M-token throughput while improving late-stage convergence below 0.993287 val_bpb.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 37.51, "num_params_M": 50.3, "num_steps": 975.0, "peak_vram_mb": 45060.2, "total_tokens_M": 511.2, "training_seconds": 300.2, "val_bpb": 0.993995}
prior_hypothesis: Restoring the 524K-token batch and shortening warmdown from 50% to 40% will preserve roughly 513M-token throughput while increasing useful high-learning-rate training, reducing val_bpb below 0.993287.

## Recent verification evidence

RECENT RESULT
hypothesis: Using 512-token attention in seven layers and full-context attention only in the final layer will process more than 513.3M tokens while preserving sequence-wide integration, reducing val_bpb below 0.993287.
change: Change the depth-8 window pattern from SSSLSSSL to SSSSSSSL, retaining the proven 512-token short window and forced full-context final layer.
mechanism: Receptive-field-stacked local attention with a single final global layer
evidence_used: The 512-token design achieved the best result, 0.993287 val_bpb at 513.3M tokens; three consecutive 512-token layers already span much of the 2048-token sequence, so removing the intermediate global layer tests whether its compute is better spent on additional training tokens.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 26.58, "num_params_M": 50.3, "num_steps": 725.0, "peak_vram_mb": 45060.2, "total_tokens_M": 380.1, "training_seconds": 300.4, "val_bpb": 1.014831}

RECENT RESULT
hypothesis: Halving the optimizer batch to 262K tokens while restoring the best 512-token SSSL attention design will nearly double update frequency with only modest throughput loss, reducing val_bpb below 0.993287.
change: Restore quarter-context short attention and reduce gradient accumulation from two microbatches to one by halving TOTAL_BATCH_SIZE.
mechanism: Higher-frequency optimizer updates at the proven attention geometry
evidence_used: The 512-token SSSL design achieved the best result, 0.993287 val_bpb at 513.3M tokens, but made only 979 optimizer updates; the 768-token and 256-token window changes both caused throughput cliffs and worse validation, motivating an optimization-frequency test at the proven window size.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 28.25, "num_params_M": 50.3, "num_steps": 1463.0, "peak_vram_mb": 44908.2, "total_tokens_M": 383.5, "training_seconds": 300.1, "val_bpb": 1.000127}

RECENT RESULT
hypothesis: Halving KV heads while restoring 512-token SSSL attention will preserve modeling quality, process more than 513.3M tokens, and reduce val_bpb below 0.993287.
change: Restore quarter-context short windows and use two KV heads shared across four query heads.
mechanism: Two-to-one grouped-query attention at the proven local-attention geometry
evidence_used: The 512-token SSSL design achieved the best val_bpb, 0.993287 at 513.3M tokens, while changing its windows, global-layer pattern, or optimizer batch regressed; reducing redundant KV projections tests an orthogonal efficiency lever without disturbing the proven receptive field or optimizer batch.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 30.91, "num_params_M": 39.8, "num_steps": 853.0, "peak_vram_mb": 41364.2, "total_tokens_M": 447.2, "training_seconds": 300.0, "val_bpb": 1.00841}

RECENT RESULT
hypothesis: Reducing Muon from five to three polar iterations will preserve useful gradient orthogonalization while lowering optimizer overhead, processing more than 513.3M tokens and reducing val_bpb below 0.993287.
change: Retain the proven 512-token SSSL architecture and 524K-token batch, but use three Newton-Schulz-style Polar Express iterations per Muon update.
mechanism: Truncated Polar-Express Muon orthogonalization
evidence_used: The best design reached 0.993287 val_bpb at 513.3M tokens, while halving the batch reduced throughput to 383.5M tokens despite more updates; this implicates per-update overhead and motivates accelerating the optimizer without changing the proven model geometry.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 29.34, "num_params_M": 50.3, "num_steps": 765.0, "peak_vram_mb": 45060.2, "total_tokens_M": 401.1, "training_seconds": 300.2, "val_bpb": 1.013096}

RECENT RESULT
hypothesis: Restoring the 524K-token batch and shortening warmdown from 50% to 40% will preserve roughly 513M-token throughput while increasing useful high-learning-rate training, reducing val_bpb below 0.993287.
change: Restore two-microbatch gradient accumulation and delay linear LR decay until 60% of the training window.
mechanism: Time-compressed learning-rate decay at the proven optimizer batch
evidence_used: The 512-token SSSL design with a 524K-token batch achieved the best result, 0.993287 val_bpb at 513.3M tokens; halving the batch reduced throughput to 383.5M tokens and regressed to 1.000127, so the proven batch should be restored while testing an untried schedule lever.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.51, "num_params_M": 50.3, "num_steps": 975.0, "peak_vram_mb": 45060.2, "total_tokens_M": 511.2, "training_seconds": 300.2, "val_bpb": 0.993995}

RECENT RESULT
hypothesis: Restoring full KV heads and beginning warmdown at 40% of the training window will preserve roughly 513M-token throughput while improving late-stage convergence below 0.993287 val_bpb.
change: Restore the proven full multi-head 512-token SSSL architecture and increase LR warmdown from 50% to 60%.
mechanism: Extended linear learning-rate annealing
evidence_used: Full KV heads with 50% warmdown achieved the best result, 0.993287 at 513.3M tokens; shortening warmdown to 40% slightly regressed to 0.993995 at similar throughput, motivating a test in the opposite direction.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.52, "num_params_M": 50.3, "num_steps": 975.0, "peak_vram_mb": 45060.2, "total_tokens_M": 511.2, "training_seconds": 300.1, "val_bpb": 0.993714}

RECENT RESULT
hypothesis: Replacing the positionwise-MLP assumption with a learned per-channel blend of current and previous-token states will improve local transition modeling, achieving val_bpb below 0.993287 while processing at least 480M tokens.
change: Preserve the proven 512-token SSSL attention geometry, but give every MLP a learned causal one-token context path initialized to retain 88% of the current state; optimize its mixing vectors with AdamW.
mechanism: Learned causal token-shift MLP
evidence_used: The 512-token SSSL design reached 0.993287, while shrinking attention further or removing a global layer caused severe throughput and quality regressions. This motivates retaining proven attention and testing a different context mechanism: cheap temporal mixing outside attention.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 28.54, "num_params_M": 50.3, "num_steps": 744.0, "peak_vram_mb": 47108.3, "total_tokens_M": 390.1, "training_seconds": 300.1, "val_bpb": 1.010413}

RECENT RESULT
hypothesis: A 52% warmdown will preserve roughly 511M-token throughput and reduce val_bpb below 0.993287 by placing the decay length near the bracketed optimum between the tested 50% and 60% schedules.
change: Shorten the current linear warmdown from 60% to 52% of the training window while preserving the proven architecture, batch size, and optimizer.
mechanism: Fine-grained linear warmdown interpolation
evidence_used: The 50% warmdown achieved 0.993287 val_bpb, while 40% and 60% regressed to 0.993995 and 0.993714 at nearly identical throughput; the asymmetric results motivate a narrow interpolation slightly above 50%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.93, "num_params_M": 50.3, "num_steps": 960.0, "peak_vram_mb": 45060.2, "total_tokens_M": 503.3, "training_seconds": 300.2, "val_bpb": 0.994459}

RECENT RESULT
hypothesis: Replacing the best 50% linear warmdown with an equal-duration cosine decay will preserve roughly 513M-token throughput while improving late-stage convergence enough to reduce val_bpb below 0.993287.
change: Restore the proven 50% warmdown and change only its interpolation from linear to cosine, retaining the established architecture, batch size, optimizer, and zero final learning rate.
mechanism: Half-window cosine learning-rate annealing
evidence_used: Linear warmdowns of 40%, 52%, and 60% all underperformed the 50% result of 0.993287 at similar throughput, indicating that 50% is the strongest duration and motivating a schedule-shape test at that fixed duration.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.36, "num_params_M": 50.3, "num_steps": 971.0, "peak_vram_mb": 45060.2, "total_tokens_M": 509.1, "training_seconds": 300.2, "val_bpb": 0.996815}

RECENT RESULT
hypothesis: Doubling the optimizer batch to 1.05M tokens will amortize Muon overhead, process more than 513.3M tokens in five minutes, and reduce val_bpb below 0.993287 despite fewer parameter updates.
change: Double TOTAL_BATCH_SIZE while preserving the proven 512-token SSSL architecture, learning rates, and 50% linear warmdown.
mechanism: Large-batch optimizer-step amortization
evidence_used: Halving the batch increased updates from 979 to 1463 but reduced training volume from 513.3M to 383.5M tokens and regressed val_bpb from 0.993287 to 1.000127, indicating that optimizer overhead and token throughput outweigh additional update frequency.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.65, "num_params_M": 50.3, "num_steps": 495.0, "peak_vram_mb": 45060.2, "total_tokens_M": 519.0, "training_seconds": 300.6, "val_bpb": 1.015936}

RECENT RESULT
hypothesis: Narrowing each MLP from 4× to 3× while restoring the proven 50% linear warmdown will preserve the established attention geometry, exceed 513.3M training tokens without reducing optimizer-step frequency, and lower val_bpb below 0.993287.
change: Reduce MLP hidden width from 4× to 3× model dimension and restore WARMDOWN_RATIO from 0.6 to the best verified value of 0.5.
mechanism: Three-times-width ReLU² feed-forward compute rebalancing
evidence_used: The 512-token SSSL design with 50% linear warmdown achieved the best val_bpb, 0.993287 at 513.3M tokens; doubling the batch reached 519.0M but regressed to 1.015936 because it halved update frequency, motivating throughput gains from cheaper dense layers instead.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.18, "num_params_M": 46.1, "num_steps": 1061.0, "peak_vram_mb": 40913.2, "total_tokens_M": 556.3, "training_seconds": 300.2, "val_bpb": 0.996996}

RECENT RESULT
hypothesis: A 3.5× MLP with the best verified 50% linear warmdown will retain more capacity than the 3× design while exceeding the 4× design’s 513.3M-token throughput, reducing val_bpb below 0.993287.
change: Narrow every MLP from 4× to 3.5× model width and restore WARMDOWN_RATIO from 0.4 to the best verified value of 0.5.
mechanism: Seven-halves-width ReLU² feed-forward compute rebalancing
evidence_used: The 4× design achieved 0.993287 at 513.3M tokens, while 3× increased throughput to 556.3M tokens but regressed modestly to 0.996996; 3.5× directly tests the capacity-throughput midpoint under the strongest verified schedule.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.65, "num_params_M": 48.2, "num_steps": 1010.0, "peak_vram_mb": 42992.7, "total_tokens_M": 529.5, "training_seconds": 300.2, "val_bpb": 0.995582}



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
