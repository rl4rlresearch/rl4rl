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
verified_results: {"depth": 8.0, "mfu_percent": 37.07, "num_params_M": 50.3, "num_steps": 2189.0, "peak_vram_mb": 39356.7, "total_tokens_M": 502.1, "training_seconds": 300.1, "val_bpb": 0.985757}
prior_hypothesis: A 229,376-token single-pass batch will lower val_bpb below 0.985961 by moving toward the approximately 111-sequence optimum implied by the measured 64-, 96-, and 128-sequence results.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a third full-context layer at index 5 will lower val_bpb below 0.993365; the added global mixing should outweigh its small throughput cost.
change: Change the repeating attention pattern from SSSL to SSL, producing full-context layers at indices 2, 5, and 7 while retaining 512-token local windows elsewhere.
mechanism: Denser periodic global attention
evidence_used: Removing one global layer increased tokens only from 512.2M to 516.9M but worsened val_bpb from 0.993365 to 0.994122, indicating global mixing contributes more than its modest compute cost.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.7, "num_params_M": 50.3, "num_steps": 965.0, "peak_vram_mb": 45060.2, "total_tokens_M": 505.9, "training_seconds": 300.3, "val_bpb": 0.994554}

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from index 3 to index 4 while retaining six 512-token local layers and the final global layer will reduce val_bpb below 0.993365 without changing throughput materially.
change: Use an explicit eight-layer SSSSLSSS pattern, placing full-context attention at layers 4 and 7.
mechanism: Later intermediate global mixing at constant compute
evidence_used: Two global layers achieved the best result (0.993365); one global layer regressed to 0.994122 and three regressed to 0.994554, motivating a controlled placement ablation with the winning attention counts and window sizes unchanged.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.58, "num_params_M": 50.3, "num_steps": 823.0, "peak_vram_mb": 45060.2, "total_tokens_M": 431.5, "training_seconds": 300.4, "val_bpb": 1.005964}

RECENT RESULT
hypothesis: Replacing the intermediate 2048-token global layer with a 1024-token window will retain enough cross-window mixing to beat 0.993365 val_bpb while recovering some of the throughput gained by removing that global layer entirely.
change: Add a medium-window attention type and use SSSM repetition, yielding 512-token local layers, a 1024-token layer at index 3, and mandatory full-context attention at the final layer.
mechanism: Hierarchical local, mid-range, and global attention
evidence_used: The two-global-layer design achieved 0.993365, while replacing its intermediate global layer with 512-token attention increased tokens from 512.2M to 516.9M but regressed to 0.994122; a 1024-token intermediate layer directly tests the quality-throughput midpoint.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 27.4, "num_params_M": 50.3, "num_steps": 736.0, "peak_vram_mb": 45060.2, "total_tokens_M": 385.9, "training_seconds": 300.3, "val_bpb": 1.012997}

RECENT RESULT
hypothesis: Representing the two full-context layers with FlashAttention’s native unrestricted window will preserve the winning SSSL receptive field while improving throughput enough to reduce val_bpb below 0.993365.
change: Keep 512-token local windows, but encode full-context attention as `(-1, -1)` instead of the equivalent `(2048, 0)` sliding window.
mechanism: Native full-attention kernel path
evidence_used: The best design uses full attention at layers 3 and 7; removing one gained only 4.7M tokens and worsened val_bpb to 0.994122, motivating a numerical optimization of full attention rather than reducing global mixing.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 27.44, "num_params_M": 50.3, "num_steps": 716.0, "peak_vram_mb": 45060.2, "total_tokens_M": 375.4, "training_seconds": 300.2, "val_bpb": 1.014553}

RECENT RESULT
hypothesis: Doubling the device batch to 256 will fit within H100 memory and reduce per-step launch and gradient-accumulation overhead, increasing trained tokens enough to lower val_bpb below 0.993365.
change: Process the full 524,288-token optimizer batch in one forward/backward pass instead of two accumulated microbatches.
mechanism: Single-microbatch optimizer steps
evidence_used: The best design peaks at 45,060 MB VRAM while requiring two microbatches per optimizer step, leaving device-batch scaling as an untested throughput lever without changing the successful model or attention topology.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reusing one value-embedding lookup across all value-residual layers will preserve the winning attention geometry while reducing redundant parameters, activations, and optimizer work enough to achieve val_bpb below 0.993365.
change: Replace four independent per-layer value-embedding tables with one shared table, looked up once per sequence and modulated by the existing layer-specific, input-dependent gates.
mechanism: Recurrently shared lexical value memory
evidence_used: Grouped-query attention reduced parameters but caused a throughput cliff to 421.0M tokens and val_bpb 1.012022 because it changed FlashAttention head geometry. This tests parameter sharing without changing the successful four-query-head SSSL attention kernels.
result: the implementation could not be verified

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
