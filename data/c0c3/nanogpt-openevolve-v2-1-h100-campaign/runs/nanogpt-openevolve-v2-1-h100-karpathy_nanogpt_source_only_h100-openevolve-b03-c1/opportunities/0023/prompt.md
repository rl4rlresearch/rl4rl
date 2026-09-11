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
verified_results: {"depth": 8.0, "mfu_percent": 36.91, "num_params_M": 50.3, "num_steps": 2180.0, "peak_vram_mb": 39356.7, "total_tokens_M": 500.0, "training_seconds": 300.1, "val_bpb": 0.984506}
prior_hypothesis: Extending linear warmdown from 70% to 80% of training will reduce val_bpb below 0.984948 by further reducing high-learning-rate updates during the fixed training window.

## Recent verification evidence

RECENT RESULT
hypothesis: Halving the optimizer batch to 262,144 tokens will lower val_bpb below 0.993365 by roughly doubling parameter updates per trained token while retaining the proven 128-sequence device batch.
change: Reduce total batch size from 524,288 to 262,144 tokens, changing gradient accumulation from two microbatches to one.
mechanism: Higher-frequency single-microbatch optimization
evidence_used: The best design reached 0.993365 with only 977 optimizer updates, while the attempted 256-sequence single-pass batch could not be verified; this obtains single-pass updates without increasing the known-fitting device batch.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.74, "num_params_M": 50.3, "num_steps": 1900.0, "peak_vram_mb": 44908.2, "total_tokens_M": 498.1, "training_seconds": 300.1, "val_bpb": 0.986162}

RECENT RESULT
hypothesis: Halving the optimizer batch again to 131,072 tokens will reduce val_bpb below 0.986162 by providing roughly twice as many parameter updates, despite a possible throughput reduction from the smaller device batch.
change: Reduce both total and device batch sizes by half, preserving one forward/backward pass per optimizer step and all other model and optimizer settings.
mechanism: Higher-frequency single-microbatch optimization
evidence_used: Halving the optimizer batch from 524,288 to 262,144 tokens improved val_bpb from 0.993365 to 0.986162 even though trained tokens fell from 512.2M to 498.1M, strongly implicating update frequency rather than throughput as the gain.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.28, "num_params_M": 50.3, "num_steps": 3742.0, "peak_vram_mb": 22701.2, "total_tokens_M": 490.5, "training_seconds": 300.0, "val_bpb": 0.990069}

RECENT RESULT
hypothesis: A 196,608-token batch will lower val_bpb below 0.986162 by balancing the better update frequency of smaller batches against the optimization and throughput regression observed at 131,072 tokens.
change: Set both total and device batch sizes to a 96-sequence, 196,608-token single-pass optimizer step.
mechanism: Intermediate-frequency single-microbatch optimization
evidence_used: Reducing the batch from 524,288 to 262,144 improved val_bpb from 0.993365 to 0.986162, but reducing it further to 131,072 regressed to 0.990069; this directly tests the midpoint between the two measured settings.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.87, "num_params_M": 50.3, "num_steps": 2538.0, "peak_vram_mb": 33807.2, "total_tokens_M": 499.0, "training_seconds": 300.0, "val_bpb": 0.985961}

RECENT RESULT
hypothesis: A 229,376-token single-pass batch will lower val_bpb below 0.985961 by moving toward the approximately 111-sequence optimum implied by the measured 64-, 96-, and 128-sequence results.
change: Increase both total and device batch sizes from 96 to 112 sequences while preserving one optimizer update per forward/backward pass.
mechanism: Empirical batch-size interpolation
evidence_used: The 131,072-, 196,608-, and 262,144-token batches achieved val_bpb 0.990069, 0.985961, and 0.986162 respectively; quadratic interpolation of these evenly spaced measurements places the estimated minimum near 111 sequences, directly motivating a 112-sequence test.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.07, "num_params_M": 50.3, "num_steps": 2189.0, "peak_vram_mb": 39356.7, "total_tokens_M": 502.1, "training_seconds": 300.1, "val_bpb": 0.985757}

RECENT RESULT
hypothesis: A 108-sequence, 221,184-token single-pass batch will reduce val_bpb below 0.985757 by moving closer to the approximately 109-sequence minimum implied by the measured 96-, 112-, and 128-sequence results.
change: Reduce both total and device batch sizes from 112 to 108 sequences while preserving one optimizer update per forward/backward pass.
mechanism: Local batch-size response-surface refinement
evidence_used: Batch sizes of 96, 112, and 128 sequences produced val_bpb values of 0.985961, 0.985757, and 0.986162; quadratic interpolation of these equally spaced measurements estimates the local optimum near 109 sequences.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.5, "num_params_M": 50.3, "num_steps": 2235.0, "peak_vram_mb": 37969.4, "total_tokens_M": 494.3, "training_seconds": 300.0, "val_bpb": 0.986047}

RECENT RESULT
hypothesis: A 116-sequence, 237,568-token single-pass batch will reduce val_bpb below 0.985757 by moving toward the roughly 117-sequence minimum implied by the measured 108-, 112-, and 128-sequence results.
change: Increase both total and device batch sizes from 112 to 116 sequences while preserving one optimizer update per forward/backward pass.
mechanism: Local batch-size response-surface refinement
evidence_used: The 108-, 112-, and 128-sequence batches achieved val_bpb 0.986047, 0.985757, and 0.986162; quadratic interpolation of these local measurements places the estimated minimum near 117 sequences.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.77, "num_params_M": 50.3, "num_steps": 2097.0, "peak_vram_mb": 40744.2, "total_tokens_M": 498.2, "training_seconds": 300.1, "val_bpb": 0.985866}

RECENT RESULT
hypothesis: A 113-sequence, 231,424-token single-pass batch will reduce val_bpb below 0.985757 by matching the approximately 113-sequence minimum implied by the nearest measured batch sizes.
change: Increase both total and device batch sizes from 112 to 113 sequences while preserving one optimizer update per forward/backward pass.
mechanism: Local batch-size response-surface refinement
evidence_used: The 108-, 112-, and 116-sequence batches achieved val_bpb values of 0.986047, 0.985757, and 0.985866; quadratic interpolation of these closest measurements estimates the minimum near 113 sequences.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.72, "num_params_M": 50.3, "num_steps": 2149.0, "peak_vram_mb": 39704.1, "total_tokens_M": 497.3, "training_seconds": 300.0, "val_bpb": 0.985863}

RECENT RESULT
hypothesis: Replacing the linear half-run warmdown with a cosine warmdown of equal duration, endpoints, and mean learning rate will reduce val_bpb below 0.985757 by preserving larger updates early in annealing while damping updates more strongly near completion.
change: Preserve the winning 112-sequence batch and all optimizer settings, changing only the warmdown curve from linear to cosine.
mechanism: Cosine terminal learning-rate annealing
evidence_used: Batch sizes 108, 112, 113, and 116 produced val_bpb values within 0.000290, with 112 remaining best; this motivates holding batch size fixed and isolating learning-rate schedule geometry.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.87, "num_params_M": 50.3, "num_steps": 2178.0, "peak_vram_mb": 39356.7, "total_tokens_M": 499.6, "training_seconds": 300.1, "val_bpb": 0.98871}

RECENT RESULT
hypothesis: Extending linear warmdown from 50% to 60% of training will reduce val_bpb below 0.985757 by avoiding the larger early-annealing updates implicated by the worse cosine result.
change: Preserve the winning 112-sequence batch and optimizer settings, but begin the linear learning-rate decay at 40% rather than 50% of the training window.
mechanism: Earlier linear learning-rate annealing
evidence_used: Cosine warmdown regressed val_bpb from 0.985757 to 0.988710 while retaining larger learning rates early in annealing, motivating an earlier reduction using the proven linear schedule.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.54, "num_params_M": 50.3, "num_steps": 2158.0, "peak_vram_mb": 39356.2, "total_tokens_M": 495.0, "training_seconds": 300.1, "val_bpb": 0.985492}

RECENT RESULT
hypothesis: Applying a neutral-initialized, query-dependent scalar gate to each attention head will reduce val_bpb below 0.985492 by letting tokens selectively amplify or suppress retrieved context without sacrificing the proven local/global attention topology or meaningful throughput.
change: Challenge the assumption that every attention head’s retrieved context should enter the residual stream at uniform strength: add a learned per-token head gate after FlashAttention, initialized so the model is functionally unchanged at startup.
mechanism: Query-dependent attention-head gating
evidence_used: Removing global mixing worsened val_bpb to 0.994122 despite processing more tokens, while batch and schedule refinements have produced only marginal gains; this motivates improving how retrieved context is incorporated rather than reducing attention or further tuning the training schedule.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.49, "num_params_M": 50.3, "num_steps": 2095.0, "peak_vram_mb": 41166.9, "total_tokens_M": 480.5, "training_seconds": 300.1, "val_bpb": 0.985574}

RECENT RESULT
hypothesis: Extending linear warmdown from 60% to 70% of training will reduce val_bpb below 0.985492 by further limiting high-learning-rate updates during the middle of the fixed training window.
change: Preserve the winning 112-sequence batch and all optimizer settings, but begin linear learning-rate decay at 30% rather than 40% of the training window.
mechanism: Earlier linear learning-rate annealing
evidence_used: Extending linear warmdown from 50% to 60% improved val_bpb from 0.985757 to 0.985492, while cosine annealing that retained larger early-decay learning rates regressed to 0.988710; testing 70% directly probes whether the measured benefit continues.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.78, "num_params_M": 50.3, "num_steps": 2172.0, "peak_vram_mb": 39356.2, "total_tokens_M": 498.2, "training_seconds": 300.1, "val_bpb": 0.984948}

RECENT RESULT
hypothesis: Extending linear warmdown from 70% to 80% of training will reduce val_bpb below 0.984948 by further reducing high-learning-rate updates during the fixed training window.
change: Preserve the winning 112-sequence batch and all optimizer settings, but begin linear learning-rate decay at 20% rather than 30% of the training window.
mechanism: Earlier linear learning-rate annealing
evidence_used: Extending linear warmdown from 50% to 60% improved val_bpb from 0.985757 to 0.985492, and extending it to 70% improved further to 0.984948, providing direct monotonic evidence for testing 80%.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.91, "num_params_M": 50.3, "num_steps": 2180.0, "peak_vram_mb": 39356.7, "total_tokens_M": 500.0, "training_seconds": 300.1, "val_bpb": 0.984506}



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
