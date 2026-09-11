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
verified_results: {"depth": 8.0, "mfu_percent": 37.25, "num_params_M": 52.4, "num_steps": 916.0, "peak_vram_mb": 47138.0, "total_tokens_M": 480.2, "training_seconds": 300.0, "val_bpb": 0.994453}
prior_hypothesis: A 4.5× MLP at the proven 512-token SSSL attention geometry will retain at least 490M training tokens while extending the monotonic quality gains from 3× through 4×, reducing val_bpb below 0.993287.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 36.6, "num_params_M": 50.3, "num_steps": 952.0, "peak_vram_mb": 45066.6, "total_tokens_M": 499.1, "training_seconds": 300.3, "val_bpb": 0.992854}
prior_hypothesis: A 2.25×/5.75× lower/upper MLP taper will preserve the 4× average width and roughly 500M-token throughput while reducing val_bpb below the best observed 0.991816.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 37.1, "num_params_M": 50.3, "num_steps": 964.0, "peak_vram_mb": 45060.2, "total_tokens_M": 505.4, "training_seconds": 300.0, "val_bpb": 0.9919}
prior_hypothesis: Moving from 2.5×/5.5× to 2×/6× lower/upper MLPs will preserve aggregate parameters and at least 500M-token throughput while reducing val_bpb below 0.991816.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 37.36, "num_params_M": 50.3, "num_steps": 971.0, "peak_vram_mb": 45066.0, "total_tokens_M": 509.1, "training_seconds": 300.1, "val_bpb": 0.991816}
prior_hypothesis: Moving from 3×/5× to 2.5×/5.5× lower/upper MLPs will preserve aggregate parameters and roughly 500M-token throughput while reducing val_bpb below 0.992431.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 4.5× MLP at the proven 512-token SSSL attention geometry will retain at least 490M training tokens while extending the monotonic quality gains from 3× through 4×, reducing val_bpb below 0.993287.
change: Restore quarter-context short attention and widen every MLP from 4× to 4.5× model dimension while preserving the best 50% linear warmdown and 524K-token optimizer batch.
mechanism: Four-and-a-half-times-width ReLU² feed-forward capacity scaling
evidence_used: Increasing MLP width from 3× to 3.5× improved val_bpb from 0.996996 to 0.995582, and 4× improved it further to 0.993287; this monotonic trend motivates testing the next tensor-core-friendly width above 4×.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.25, "num_params_M": 52.4, "num_steps": 916.0, "peak_vram_mb": 47138.0, "total_tokens_M": 480.2, "training_seconds": 300.0, "val_bpb": 0.994453}

RECENT RESULT
hypothesis: A 4.25× MLP will retain at least 495M training tokens while balancing the 4× model’s superior throughput against the 4.5× model’s added capacity, reducing val_bpb below 0.993287.
change: Widen every MLP from 4× to 4.25× model dimension while preserving the proven attention geometry, optimizer batch, and 50% linear warmdown.
mechanism: Four-and-a-quarter-width ReLU² feed-forward capacity scaling
evidence_used: The 4× design achieved 0.993287 val_bpb at 513.3M tokens, whereas 4.5× regressed to 0.994453 at 480.2M; 4.25× tests the tensor-core-aligned midpoint within this newly bracketed capacity optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.8, "num_params_M": 51.4, "num_steps": 930.0, "peak_vram_mb": 46101.3, "total_tokens_M": 487.6, "training_seconds": 300.0, "val_bpb": 0.994781}

RECENT RESULT
hypothesis: Shifting MLP capacity from the lower four layers to the upper four while preserving total parameters and compute will retain roughly 513M-token throughput and reduce val_bpb below 0.993287.
change: Use 3.5× MLPs in the lower half and 4.5× MLPs in the upper half, preserving the 4× average width, and restore the best verified 50% linear warmdown.
mechanism: Depth-tapered feed-forward capacity allocation
evidence_used: Uniform MLP widths improved monotonically from 3× to 4×, showing that feed-forward capacity matters, while uniform 4.5× lost throughput and regressed; reallocating the same aggregate 4× capacity toward later representations tests a new capacity lever without increasing compute, under the best verified 50% schedule.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.07, "num_params_M": 50.3, "num_steps": 964.0, "peak_vram_mb": 45064.7, "total_tokens_M": 505.4, "training_seconds": 300.3, "val_bpb": 0.992809}

RECENT RESULT
hypothesis: Moving from 3.5×/4.5× to 3×/5× lower/upper MLPs while preserving average width, total parameters, and the best 50% linear warmdown will reduce val_bpb below 0.992809 without materially reducing throughput.
change: Allocate 3× MLP width to the lower four layers and 5× width to the upper four layers, and restore the best verified 50% warmdown.
mechanism: Stronger depth-tapered feed-forward capacity allocation
evidence_used: The 3.5×/4.5× depth taper improved val_bpb from the uniform 4× design’s 0.993287 to 0.992809 at equal aggregate MLP capacity, showing that shifting feed-forward capacity upward is beneficial; this patch tests a stronger shift along the same axis.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.96, "num_params_M": 50.3, "num_steps": 961.0, "peak_vram_mb": 45066.2, "total_tokens_M": 503.8, "training_seconds": 300.2, "val_bpb": 0.992431}

RECENT RESULT
hypothesis: Replacing additive ReLU² features with parameter-matched multiplicative SwiGLU features, while retaining the successful upper-layer capacity bias, will process at least 490M tokens and reduce val_bpb below 0.992431.
change: Challenge the assumption that wider ungated ReLU² MLPs are the best context representation: use 2× lower-layer and 3.25× upper-layer SwiGLU widths, approximately matching the proven 3×/5× model’s parameter count and compute.
mechanism: Depth-tapered SwiGLU context feature gating
evidence_used: The 3×/5× depth taper achieved the best val_bpb, 0.992431, while further uniform widening regressed; this suggests allocation is useful but ungated width scaling is exhausted. SwiGLU tests learned multiplicative feature selection without the repeated sequence-shift operations that reduced throughput to 390.1M tokens.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.91, "num_params_M": 50.1, "num_steps": 940.0, "peak_vram_mb": 44804.8, "total_tokens_M": 492.8, "training_seconds": 300.0, "val_bpb": 0.995144}

RECENT RESULT
hypothesis: Moving from 3×/5× to 2.5×/5.5× lower/upper MLPs will preserve aggregate parameters and roughly 500M-token throughput while reducing val_bpb below 0.992431.
change: Allocate 2.5× MLP width to the lower four layers and 5.5× width to the upper four, retaining the 4× average width and 50% linear warmdown.
mechanism: Stronger depth-tapered ReLU² capacity allocation
evidence_used: Strengthening the taper from 3.5×/4.5× to 3×/5× improved val_bpb from 0.992809 to 0.992431 at nearly unchanged throughput, motivating one further tensor-core-aligned step along the same axis.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.36, "num_params_M": 50.3, "num_steps": 971.0, "peak_vram_mb": 45066.0, "total_tokens_M": 509.1, "training_seconds": 300.1, "val_bpb": 0.991816}

RECENT RESULT
hypothesis: Moving from 2.5×/5.5× to 2×/6× lower/upper MLPs will preserve aggregate parameters and at least 500M-token throughput while reducing val_bpb below 0.991816.
change: Allocate 2× MLP width to the lower four layers and 6× width to the upper four, retaining the 4× average width and proven 50% linear warmdown.
mechanism: Aggressive depth-tapered ReLU² capacity allocation
evidence_used: Successively strengthening the taper from 3.5×/4.5× through 3×/5× to 2.5×/5.5× improved val_bpb from 0.992809 to 0.992431 to 0.991816 without reducing aggregate capacity or throughput, motivating one further tensor-core-aligned step.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.1, "num_params_M": 50.3, "num_steps": 964.0, "peak_vram_mb": 45060.2, "total_tokens_M": 505.4, "training_seconds": 300.0, "val_bpb": 0.9919}

RECENT RESULT
hypothesis: A 2.25×/5.75× lower/upper MLP taper will preserve the 4× average width and roughly 500M-token throughput while reducing val_bpb below the best observed 0.991816.
change: Give the lower four layers 2.25× MLP width and the upper four layers 5.75× width, retaining all other architecture, optimizer, batching, and schedule settings.
mechanism: Midpoint depth-tapered ReLU² capacity allocation
evidence_used: The taper improved monotonically through 2.5×/5.5× at 0.991816, while 2×/6× slightly regressed to 0.991900 at similar throughput; their tensor-core-aligned midpoint directly refines the bracketed optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.6, "num_params_M": 50.3, "num_steps": 952.0, "peak_vram_mb": 45066.6, "total_tokens_M": 499.1, "training_seconds": 300.3, "val_bpb": 0.992854}



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
