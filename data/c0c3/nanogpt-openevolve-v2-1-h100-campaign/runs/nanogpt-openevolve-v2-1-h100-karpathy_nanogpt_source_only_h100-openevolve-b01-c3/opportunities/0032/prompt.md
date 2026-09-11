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
verified_results: {"depth": 8.0, "mfu_percent": 38.87, "num_params_M": 51.9, "num_steps": 1782.0, "peak_vram_mb": 46460.3, "total_tokens_M": 467.1, "training_seconds": 300.1, "val_bpb": 0.986294}
prior_hypothesis: At the proven 262,144-token batch, scaling all optimizer learning rates to 81.25% will retain at least 460M-token throughput and reduce val_bpb below 0.985487.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 38.52, "num_params_M": 53.0, "num_steps": 1744.0, "peak_vram_mb": 50625.8, "total_tokens_M": 457.2, "training_seconds": 300.1, "val_bpb": 0.985446}
prior_hypothesis: A zero-initialized rank-64 bigram expert at the proven 93.75% learning rates will retain at least 450M-token throughput and reduce val_bpb below 0.984614 by learning frequent token transitions outside the deep contextual path.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 38.72, "num_params_M": 51.9, "num_steps": 1775.0, "peak_vram_mb": 46708.3, "total_tokens_M": 465.3, "training_seconds": 300.1, "val_bpb": 0.985063}
prior_hypothesis: A zero-initialized output bias at the proven 93.75% learning rates will retain at least 460M-token throughput and reduce val_bpb below 0.984614 by learning global next-token frequencies without the rank-64 bigram expert’s compute and memory overhead.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 38.84, "num_params_M": 51.9, "num_steps": 1780.0, "peak_vram_mb": 46460.3, "total_tokens_M": 466.6, "training_seconds": 300.1, "val_bpb": 0.984614}
prior_hypothesis: At the proven 262,144-token batch, scaling all optimizer learning rates to 93.75% will retain at least 460M-token throughput and reduce val_bpb below 0.985487.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing the assumption that attention must reconstruct all local token interactions with a low-FLOP learned bigram representation, while restoring the proven 4.375× MLP, will process at least 440M tokens and reduce val_bpb below 0.994296.
change: Restore 2240-channel MLPs and augment each token embedding with a scaled, trainable bigram code formed by concatenating two independently hashed half-width embeddings.
mechanism: Dual-hash bigram memory in the residual stream
evidence_used: The 4.375× MLP achieved 0.994296 at 472.9M tokens, whereas 3.75× achieved 0.996902 despite 498.6M tokens, showing that representational capacity outweighed marginal throughput; hashed lookup capacity adds an explicit local-context mechanism without another dense matrix.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.62, "num_params_M": 54.0, "num_steps": 890.0, "peak_vram_mb": 46642.0, "total_tokens_M": 466.6, "training_seconds": 300.1, "val_bpb": 0.994892}

RECENT RESULT
hypothesis: Halving the global batch will preserve at least 465M-token throughput while roughly doubling optimizer updates, reducing val_bpb below 0.994296.
change: Reduce the global batch from 524,288 to 262,144 tokens while retaining the proven 4.375× MLP, device batch, learning rates, and time-based schedules.
mechanism: Finer-grained stochastic optimization
evidence_used: Increasing the batch to 786,432 reduced updates from 948 to 632 and worsened val_bpb from 0.995558 to 1.012938 at comparable token exposure, indicating that update frequency is valuable in this fixed-time regime.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.67, "num_params_M": 51.9, "num_steps": 1772.0, "peak_vram_mb": 46460.3, "total_tokens_M": 464.5, "training_seconds": 300.0, "val_bpb": 0.985506}

RECENT RESULT
hypothesis: A 131,072-token global batch will preserve at least 420M-token throughput while approximately doubling the proven 1,772 optimizer updates, reducing val_bpb below 0.985506.
change: Restore the proven uniform 4.375× MLP and halve both global and device batch sizes so each optimizer step processes 131,072 tokens without gradient accumulation.
mechanism: Quarter-batch update densification
evidence_used: Halving the global batch from 524,288 to 262,144 tokens increased updates from roughly 900 to 1,772 and improved val_bpb from 0.994296 to 0.985506 despite slightly lower token exposure, while the larger 786,432-token batch performed substantially worse.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.93, "num_params_M": 51.9, "num_steps": 3468.0, "peak_vram_mb": 23484.2, "total_tokens_M": 454.6, "training_seconds": 300.1, "val_bpb": 0.989237}

RECENT RESULT
hypothesis: A 196,608-token batch will retain at least 450M-token throughput while providing roughly 2,350 updates, reducing val_bpb below the 262,144-token batch’s 0.985506.
change: Remove the unsuccessful hashed-bigram path, restore the proven uniform 4.375× MLP model, and use a single-microbatch 196,608-token global batch.
mechanism: Intermediate-batch update densification
evidence_used: Reducing batch size from 524,288 to 262,144 improved val_bpb from 0.994296 to 0.985506, but reducing again to 131,072 worsened it to 0.989237; the untested midpoint directly probes the apparent optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.37, "num_params_M": 51.9, "num_steps": 2342.0, "peak_vram_mb": 34974.9, "total_tokens_M": 460.5, "training_seconds": 300.1, "val_bpb": 0.986435}

RECENT RESULT
hypothesis: A 196,608-token batch with every optimizer learning rate scaled by 0.75 will preserve at least 450M-token throughput while matching the successful 262,144-token run’s cumulative update magnitude, reducing val_bpb below 0.985506.
change: Restore uniform 4.375× MLPs, use the proven 196,608-token single-microbatch configuration, and scale all AdamW and Muon learning rates in proportion to batch size.
mechanism: Batch-linear learning-rate normalization
evidence_used: The 196,608-token design reached 0.986435 versus 0.985506 at 262,144 tokens, but used the same learning rates across 32% more updates; scaling rates by 0.75 preserves its finer update frequency without increasing cumulative learning-rate exposure.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.66, "num_params_M": 51.9, "num_steps": 2359.0, "peak_vram_mb": 34974.8, "total_tokens_M": 463.8, "training_seconds": 300.0, "val_bpb": 0.985713}

RECENT RESULT
hypothesis: A 229,376-token batch with optimizer learning rates scaled to 87.5% will retain at least 460M-token throughput and reduce val_bpb below 0.985506.
change: Use a single 112-sequence microbatch per optimizer step and scale all AdamW and Muon learning rates linearly with the batch reduction.
mechanism: Intermediate-batch learning-rate interpolation
evidence_used: Scaling the 196,608-token design’s learning rates to 75% improved val_bpb from 0.986435 to 0.985713, nearly matching the 262,144-token design’s 0.985506; testing their midpoint probes the remaining batch–update tradeoff while preserving cumulative learning-rate exposure.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.64, "num_params_M": 51.9, "num_steps": 2023.0, "peak_vram_mb": 40718.4, "total_tokens_M": 464.0, "training_seconds": 300.1, "val_bpb": 0.985963}

RECENT RESULT
hypothesis: At the proven 262,144-token batch, scaling every optimizer learning rate to 87.5% will retain at least 460M-token throughput and reduce val_bpb below 0.985506.
change: Restore the best-performing 262,144-token single-microbatch configuration and reduce all AdamW and Muon learning rates by 12.5%.
mechanism: Best-batch learning-rate refinement
evidence_used: Scaling learning rates down at 196,608 tokens improved val_bpb from 0.986435 to 0.985713; testing the same moderate reduction while holding the best 262,144-token batch fixed isolates learning rate from the batch-size confound in the unsuccessful 229,376-token run.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.1, "num_params_M": 51.9, "num_steps": 1792.0, "peak_vram_mb": 46458.8, "total_tokens_M": 469.8, "training_seconds": 300.1, "val_bpb": 0.985487}

RECENT RESULT
hypothesis: At the proven 262,144-token batch, scaling all optimizer learning rates to 81.25% will retain at least 460M-token throughput and reduce val_bpb below 0.985487.
change: Restore the best-performing 262,144-token single-microbatch configuration and reduce AdamW and Muon learning rates another 6.25% from the current best.
mechanism: Best-batch learning-rate bracketing
evidence_used: At 196,608 tokens, a 25% learning-rate reduction improved val_bpb from 0.986435 to 0.985713; at 262,144 tokens, a 12.5% reduction improved 0.985506 to 0.985487, motivating a measured step further in the same direction.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.87, "num_params_M": 51.9, "num_steps": 1782.0, "peak_vram_mb": 46460.3, "total_tokens_M": 467.1, "training_seconds": 300.1, "val_bpb": 0.986294}

RECENT RESULT
hypothesis: At the proven 262,144-token batch, scaling all optimizer learning rates to 93.75% will retain at least 460M-token throughput and reduce val_bpb below 0.985487.
change: Restore the 262,144-token single-microbatch configuration and set every AdamW and Muon learning rate midway between the tested 87.5% and 100% settings.
mechanism: Quadratic learning-rate interpolation at the optimal batch size
evidence_used: At 262,144 tokens, 87.5% learning rates achieved 0.985487, 100% achieved 0.985506, and 81.25% worsened to 0.986294; these measurements bracket the apparent optimum near 93.75%.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.84, "num_params_M": 51.9, "num_steps": 1780.0, "peak_vram_mb": 46460.3, "total_tokens_M": 466.6, "training_seconds": 300.1, "val_bpb": 0.984614}

RECENT RESULT
hypothesis: Replacing the linear warmdown with an equal-area cosine warmdown at the proven 93.75% peak learning rates will retain at least 460M-token throughput and reduce val_bpb below 0.984614.
change: Keep the architecture, batch size, peak learning rates, warmdown duration, endpoints, and cumulative learning-rate exposure fixed while changing only the warmdown curvature.
mechanism: Equal-area cosine terminal annealing
evidence_used: The 262,144-token runs bracket the best peak scale: 93.75% achieved 0.984614, while 87.5%, 100%, and 81.25% were worse. Holding that peak fixed and testing an equal-area schedule shape is therefore a controlled next optimization dimension.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.61, "num_params_M": 51.9, "num_steps": 1770.0, "peak_vram_mb": 46458.8, "total_tokens_M": 464.0, "training_seconds": 300.1, "val_bpb": 0.988064}

RECENT RESULT
hypothesis: A zero-initialized rank-64 bigram expert at the proven 93.75% learning rates will retain at least 450M-token throughput and reduce val_bpb below 0.984614 by learning frequent token transitions outside the deep contextual path.
change: Challenge the assumption that every prediction must be decoded solely from the final transformer state; add an exact, collision-free factorized bigram distribution directly to the logits while retaining the transformer for longer-context corrections.
mechanism: Residual low-rank bigram logit expert
evidence_used: The 93.75% learning-rate design is best at 0.984614. The hashed-bigram design reached only 0.994892 despite 466.6M tokens, indicating that indirect, collision-prone residual injection was ineffective; a zero-initialized direct logit expert tests the local-statistics idea without requiring eight layers to preserve and decode the added representation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.52, "num_params_M": 53.0, "num_steps": 1744.0, "peak_vram_mb": 50625.8, "total_tokens_M": 457.2, "training_seconds": 300.1, "val_bpb": 0.985446}

RECENT RESULT
hypothesis: A zero-initialized output bias at the proven 93.75% learning rates will retain at least 460M-token throughput and reduce val_bpb below 0.984614 by learning global next-token frequencies without the rank-64 bigram expert’s compute and memory overhead.
change: Restore the best 93.75% optimizer rates and add a zero-initialized bias to the language-model head.
mechanism: Near-free unigram logit prior
evidence_used: The rank-64 direct-logit bigram expert lost 9.4M tokens and worsened val_bpb from 0.984614 to 0.985446; a fused output bias tests whether a minimal direct statistical expert provides the useful prior without that throughput penalty.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.72, "num_params_M": 51.9, "num_steps": 1775.0, "peak_vram_mb": 46708.3, "total_tokens_M": 465.3, "training_seconds": 300.1, "val_bpb": 0.985063}



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
