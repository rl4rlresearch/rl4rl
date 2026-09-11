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
verified_results: {"depth": 8.0, "mfu_percent": 36.5, "num_params_M": 50.3, "num_steps": 2513.0, "peak_vram_mb": 33810.6, "total_tokens_M": 494.1, "training_seconds": 300.1, "val_bpb": 0.984034}
prior_hypothesis: A 196,608-token batch with all peak learning rates scaled by 6/7 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving the best run’s cumulative optimizer displacement while benefiting from more frequent, less correlated updates.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 36.64, "num_params_M": 50.3, "num_steps": 2164.0, "peak_vram_mb": 39360.4, "total_tokens_M": 496.4, "training_seconds": 300.1, "val_bpb": 0.984713}
prior_hypothesis: A 229,376-token batch with all peak learning rates increased by 2.86% will process at least 490M tokens and reduce val_bpb below 0.984024.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 36.84, "num_params_M": 50.3, "num_steps": 2342.0, "peak_vram_mb": 36584.2, "total_tokens_M": 498.8, "training_seconds": 300.1, "val_bpb": 0.983699}
prior_hypothesis: A 212,992-token batch with learning rates scaled by 13/14 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving cumulative optimizer displacement while interpolating between the two equally competitive normalized-batch endpoints.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 36.48, "num_params_M": 50.3, "num_steps": 2512.0, "peak_vram_mb": 33808.7, "total_tokens_M": 493.9, "training_seconds": 300.0, "val_bpb": 0.985044}
prior_hypothesis: A 196,608-token optimizer batch will retain at least 485M training tokens while reducing val_bpb below 0.985237.

## Recent verification evidence

RECENT RESULT
hypothesis: A 2.375×/2.625× lower-half taper with uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991682.
change: Restore the best architecture’s aggregate MLP allocation while testing the tensor-core-aligned midpoint between its 2.25×/2.75× lower taper and the competitive uniform 2.5× lower allocation.
mechanism: Refined weak-side lower-half feed-forward taper
evidence_used: The 2.25×/2.75× taper achieved the best observed 0.991682, outperforming uniform 2.5× at 0.991816; the previously tested stronger-side midpoint regressed, so this tests the unmeasured weaker-side midpoint.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.98, "num_params_M": 50.3, "num_steps": 961.0, "peak_vram_mb": 45067.3, "total_tokens_M": 503.8, "training_seconds": 300.1, "val_bpb": 0.992481}

RECENT RESULT
hypothesis: A 2×/2.5×/2.5×/3× lower-layer MLP ramp with uniform 5.5× upper MLPs will process at least 500M tokens and reduce val_bpb below 0.991682.
change: Preserve the best design’s lower-quarter averages and total MLP capacity while redistributing capacity within each lower-layer pair toward greater depth.
mechanism: Recursive per-layer lower-half feed-forward taper
evidence_used: The 2.25×/2.75× lower-half taper achieved the best observed val_bpb of 0.991682, outperforming both uniform 2.5× and stronger pairwise tapers; recursively applying the same ±0.25× shift within those pairs tests smoother depth localization without changing aggregate parameters or MLP compute.
result: the implementation could not be verified

RECENT RESULT
hypothesis: On the best 2.25×/2.75×/5.5× MLP taper, removing the redundant layer-3 full-context attention pass will increase throughput beyond 510.7M tokens while the forced full-context final layer preserves global mixing, reducing val_bpb below 0.991682.
change: Restore the best verified MLP allocation and change the attention pattern from SSSL to SSSS, leaving only the automatically forced final layer at full context.
mechanism: Single terminal global-attention layer
evidence_used: The 2.25×/2.75×/5.5× design achieved the best verified val_bpb of 0.991682 at 510.7M tokens; subsequent MLP refinements and schedule changes did not improve it, while every verified design retained two full-context layers, making removal of the earlier full-context pass an untested throughput–context tradeoff.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.08, "num_params_M": 50.3, "num_steps": 980.0, "peak_vram_mb": 45064.8, "total_tokens_M": 513.8, "training_seconds": 300.2, "val_bpb": 0.99232}

RECENT RESULT
hypothesis: Doubling short-layer context from 512 to 1024 tokens will retain at least 500M training tokens and reduce val_bpb below 0.991682.
change: Keep the best 2.25×/2.75×/5.5× architecture and SSSL pattern, but make each short-attention layer cover half rather than one quarter of the 2048-token sequence.
mechanism: Broader sliding-window attention
evidence_used: Removing the layer-3 full-context pass increased throughput only from 510.7M to 513.8M tokens but worsened val_bpb from 0.991682 to 0.992320, indicating that additional contextual mixing is worth modest attention cost.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.4, "num_params_M": 50.3, "num_steps": 944.0, "peak_vram_mb": 45064.8, "total_tokens_M": 494.9, "training_seconds": 300.3, "val_bpb": 0.99372}

RECENT RESULT
hypothesis: Halving the optimizer batch to 262K tokens on the best 2.25×/2.75×/5.5× architecture will provide roughly twice as many parameter updates, retain at least 450M training tokens, and reduce val_bpb below 0.991682.
change: Restore the best verified MLP taper and reduce gradient accumulation from two microbatches to one while preserving all learning rates and schedules.
mechanism: Single-microbatch higher-frequency optimization
evidence_used: The 2.25×/2.75×/5.5× design achieved the best val_bpb, 0.991682; subsequent width, schedule, auxiliary-loss, and attention changes regressed, while every verified design retained the same 524K-token optimizer batch, making update frequency an untested orthogonal mechanism.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.53, "num_params_M": 50.3, "num_steps": 1889.0, "peak_vram_mb": 44912.7, "total_tokens_M": 495.2, "training_seconds": 300.1, "val_bpb": 0.985237}

RECENT RESULT
hypothesis: Halving the optimizer batch again on the proven 2.25×/2.75×/5.5× architecture will retain at least 440M training tokens while increasing parameter-update frequency enough to reduce val_bpb below 0.985237.
change: Restore the best verified MLP taper, reduce the optimizer batch from 262K to 131K tokens, and use a 64-sequence device batch so each optimizer step remains one microbatch.
mechanism: 131K-token single-microbatch higher-frequency optimization
evidence_used: Reducing the optimizer batch from 524K to 262K tokens nearly doubled updates, retained 495.2M tokens, and improved val_bpb substantially from 0.991682 to 0.985237; another halving directly tests whether that strong update-frequency trend continues.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.73, "num_params_M": 50.3, "num_steps": 3685.0, "peak_vram_mb": 22705.3, "total_tokens_M": 483.0, "training_seconds": 300.0, "val_bpb": 0.989256}

RECENT RESULT
hypothesis: On the proven 2.25×/2.75×/5.5× taper and 262K-token optimizer batch, replacing four 128-dimensional attention heads with eight 64-dimensional heads will retain at least 480M training tokens and reduce val_bpb below 0.985237 by enabling more specialized contextual routing at essentially unchanged projection and attention FLOPs.
change: Restore the best verified MLP taper and optimizer batch, then challenge the shared assumption that four wide heads are the best context representation by doubling head count while preserving model width, KV width, parameter scale, sequence length, and attention-window pattern.
mechanism: Finer-grained eight-head context routing
evidence_used: The 262K-token design achieved the best observed val_bpb of 0.985237 at 495.2M tokens. All available designs fixed HEAD_DIM=128, while broader windows increased contextual compute but regressed to 0.993720; finer head factorization tests richer learned context selection without paying for longer attention.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.91, "num_params_M": 50.3, "num_steps": 1857.0, "peak_vram_mb": 45013.6, "total_tokens_M": 486.8, "training_seconds": 300.0, "val_bpb": 0.989288}

RECENT RESULT
hypothesis: A 196,608-token optimizer batch will retain at least 485M training tokens while reducing val_bpb below 0.985237.
change: Use a 96-sequence device batch and optimizer batch three-quarters the current size, preserving one microbatch per update and all learning rates and schedules.
mechanism: Intermediate single-microbatch update frequency
evidence_used: The 262K-token batch achieved 0.985237, while neighboring 524K and 131K batches regressed to 0.991682 and 0.989256; testing 197K refines the clearly bracketed update-frequency optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.48, "num_params_M": 50.3, "num_steps": 2512.0, "peak_vram_mb": 33808.7, "total_tokens_M": 493.9, "training_seconds": 300.0, "val_bpb": 0.985044}

RECENT RESULT
hypothesis: A 229,376-token optimizer batch will retain at least 490M training tokens and reduce val_bpb below 0.985044.
change: Increase the current batch to 112 sequences per single-microbatch optimizer step while preserving the best architecture, learning rates, and schedules.
mechanism: Bracketed optimizer-batch refinement
evidence_used: The 196,608-token batch achieved the best val_bpb, 0.985044, while neighboring 131,072- and 262,144-token batches scored 0.989256 and 0.985237; their results bracket an estimated optimum near 225K tokens.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.68, "num_params_M": 50.3, "num_steps": 2166.0, "peak_vram_mb": 39360.4, "total_tokens_M": 496.8, "training_seconds": 300.0, "val_bpb": 0.984024}

RECENT RESULT
hypothesis: A 196,608-token batch with all peak learning rates scaled by 6/7 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving the best run’s cumulative optimizer displacement while benefiting from more frequent, less correlated updates.
change: Restore four 128-dimensional heads, use a 96-sequence single microbatch, and scale every optimizer-group learning rate by the 196,608/229,376 batch ratio.
mechanism: Token-normalized smaller-batch learning rates
evidence_used: The 229,376-token batch achieved 0.984024, while 196,608 tokens achieved a close 0.985044 and 493.9M tokens using unchanged learning rates but 16% more updates; scaling rates by 6/7 isolates whether excessive cumulative update magnitude caused that regression.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.5, "num_params_M": 50.3, "num_steps": 2513.0, "peak_vram_mb": 33810.6, "total_tokens_M": 494.1, "training_seconds": 300.1, "val_bpb": 0.984034}

RECENT RESULT
hypothesis: A 229,376-token batch with all peak learning rates increased by 2.86% will process at least 490M tokens and reduce val_bpb below 0.984024.
change: Restore the best 112-sequence single-microbatch design and scale every optimizer-group peak learning rate by 36/35.
mechanism: Bracketed peak-learning-rate refinement
evidence_used: The 229,376-token run achieved 0.984024; moving to a higher effective per-token rate at 196,608 tokens regressed to 0.985044, while scaling that run back by 6/7 recovered 0.984034. The opposite 262,144-token direction regressed further to 0.985237, so the asymmetric bracket motivates a small upward refinement near the proven optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.64, "num_params_M": 50.3, "num_steps": 2164.0, "peak_vram_mb": 39360.4, "total_tokens_M": 496.4, "training_seconds": 300.1, "val_bpb": 0.984713}

RECENT RESULT
hypothesis: A 212,992-token batch with learning rates scaled by 13/14 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving cumulative optimizer displacement while interpolating between the two equally competitive normalized-batch endpoints.
change: Use a 104-sequence single microbatch and scale every optimizer-group peak learning rate by the batch ratio 212,992/229,376.
mechanism: Token-normalized optimizer-batch midpoint
evidence_used: The 229,376-token baseline reached 0.984024, while 196,608 tokens with proportional 6/7 learning rates reached 0.984034; their near-identical results bracket this untested midpoint and support keeping learning rate proportional to batch size.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.84, "num_params_M": 50.3, "num_steps": 2342.0, "peak_vram_mb": 36584.2, "total_tokens_M": 498.8, "training_seconds": 300.1, "val_bpb": 0.983699}



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
