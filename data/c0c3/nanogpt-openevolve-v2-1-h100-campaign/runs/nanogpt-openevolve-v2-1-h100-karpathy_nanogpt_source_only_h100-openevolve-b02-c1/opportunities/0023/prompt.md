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
verified_results: {"depth": 8.0, "mfu_percent": 38.3, "num_params_M": 50.3, "num_steps": 2429.0, "peak_vram_mb": 33808.0, "total_tokens_M": 477.6, "training_seconds": 300.0, "val_bpb": 0.986971}
prior_hypothesis: Reducing the update batch from 262,144 to 196,608 tokens will provide 33% more whole-model updates per token while retaining adequate H100 occupancy, lowering val_bpb below 0.987386.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the successful direct lexical expert from four head-level coefficients to 512 channel-level coefficients will improve its ability to select predictive token features and reduce val_bpb below 0.995511 without materially affecting throughput.
change: Replace the final value embedding’s per-head readout scaling with independent scaling for every model channel, retaining the same initialization, architecture, and optimizer.
mechanism: Channel-wise lexical-residual readout
evidence_used: The per-head lexical-residual readout improved val_bpb from 0.995558 to 0.995511 at similar throughput, while the faster grouped-query design regressed to 0.999976; this motivates increasing value-path expressivity with negligible added computation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.19, "num_params_M": 50.3, "num_steps": 939.0, "peak_vram_mb": 45061.2, "total_tokens_M": 492.3, "training_seconds": 300.3, "val_bpb": 0.994364}

RECENT RESULT
hypothesis: Modulating the successful channel-wise lexical residual with a learned context-dependent gate will reduce val_bpb below 0.994364 without materially reducing throughput.
change: Add a zero-initialized 32-to-512 readout gate conditioned on the final contextual state, preserving the current lexical path exactly at initialization while allowing token-feature scaling to vary by context.
mechanism: Context-conditioned channel-wise lexical gating
evidence_used: Expanding lexical scaling from four head-level coefficients to 512 channel-level coefficients improved val_bpb from 0.995511 to 0.994364 at comparable throughput, motivating a minimally expensive increase in the same path’s expressivity.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.37, "num_params_M": 50.3, "num_steps": 919.0, "peak_vram_mb": 46086.5, "total_tokens_M": 481.8, "training_seconds": 300.3, "val_bpb": 0.995708}

RECENT RESULT
hypothesis: Decoupling the successful channel-wise lexical expert from the final attention value embedding will eliminate competing attention/readout gradients and reduce val_bpb below 0.994364 without materially reducing throughput.
change: Add a dedicated token embedding for the lexical readout, initialize it as an exact copy of the final value embedding, and optimize it with the existing embedding schedule.
mechanism: Dedicated lexical-expert embedding
evidence_used: Channel-wise static lexical scaling improved val_bpb from 0.995511 to 0.994364, while context-dependent gating regressed to 0.995708; this motivates greater static lexical specialization while preserving the best model’s initial function.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.74, "num_params_M": 54.5, "num_steps": 928.0, "peak_vram_mb": 45349.2, "total_tokens_M": 486.5, "training_seconds": 300.2, "val_bpb": 0.995859}

RECENT RESULT
hypothesis: A neutral-initialized scalar gate keyed by the current token will reduce val_bpb below 0.994364 by learning when to trust the successful lexical residual, with negligible throughput cost.
change: Add one learned gate per vocabulary token, multiply it into the channel-wise lexical residual, and optimize it with the low-rate scalar parameter group.
mechanism: Token-conditioned lexical confidence gate
evidence_used: Channel-wise lexical scaling achieved the best val_bpb of 0.994364, while a dense context-conditioned gate regressed to 0.995708. A scalar token lookup preserves the best model’s initial function while cheaply restoring token-specific lexical confidence removed by normalizing the value embedding.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.05, "num_params_M": 50.3, "num_steps": 935.0, "peak_vram_mb": 45062.3, "total_tokens_M": 490.2, "training_seconds": 300.1, "val_bpb": 0.994764}

RECENT RESULT
hypothesis: A channel-wise mixture of all four existing value embeddings will reduce val_bpb below 0.994364 without materially reducing throughput.
change: Preserve the current final-value lexical path at initialization, while adding zero-initialized channel coefficients that let earlier-layer value embeddings contribute directly to the readout.
mechanism: Multi-depth static lexical ensemble
evidence_used: Channel-wise static lexical scaling achieved the best val_bpb of 0.994364, whereas context gating and a dedicated lexical embedding regressed; reusing already-computed value embeddings extends the successful static mechanism without adding another embedding table.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.73, "num_params_M": 50.3, "num_steps": 904.0, "peak_vram_mb": 45332.3, "total_tokens_M": 474.0, "training_seconds": 300.2, "val_bpb": 0.997915}

RECENT RESULT
hypothesis: Increasing only the channel-wise lexical expert’s Adam learning rate from 0.005 to 0.02 will reduce val_bpb below 0.994364 without affecting throughput.
change: Separate `bigram_lambdas` from the residual-scalar optimizer group and train them at four times their current learning rate while preserving all other parameters and schedules.
mechanism: Faster channel-wise lexical adaptation
evidence_used: Channel-wise lexical scaling improved val_bpb from 0.995511 to 0.994364, while added gating, embeddings, and multi-depth inputs regressed; this isolates optimization of the successful mechanism without adding computation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.74, "num_params_M": 50.3, "num_steps": 928.0, "peak_vram_mb": 45061.2, "total_tokens_M": 486.5, "training_seconds": 300.2, "val_bpb": 0.995174}

RECENT RESULT
hypothesis: Using batch size 192 with one 393,216-token microbatch per optimizer step will provide 33% more whole-model updates per token while maintaining high GPU occupancy, reducing val_bpb below 0.994364.
change: Increase the device batch from 128 to 192 and reduce total batch size to exactly one device microbatch, preserving the best architecture and optimizer settings.
mechanism: Single-microbatch 384K-token updates
evidence_used: Raising only the lexical expert learning rate regressed val_bpb to 0.995174, while the unchanged channel-wise expert remains best at 0.994364; this motivates increasing update frequency for the whole proven model instead of further accelerating or expanding the lexical path.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.14, "num_params_M": 50.3, "num_steps": 1246.0, "peak_vram_mb": 67119.7, "total_tokens_M": 489.9, "training_seconds": 300.1, "val_bpb": 0.990146}

RECENT RESULT
hypothesis: Reducing the single-microbatch update from 393,216 to 327,680 tokens will provide 20% more whole-model updates per token while retaining sufficient H100 occupancy, lowering val_bpb below 0.990146.
change: Set both total and device batch sizes to 160 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.
mechanism: Single-microbatch 320K-token updates
evidence_used: The 393K single-microbatch design improved val_bpb from 0.994364 to 0.990146 at essentially unchanged training volume, while increasing steps from 939 to 1246; this strongly motivates testing a further update-frequency increase.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.72, "num_params_M": 50.3, "num_steps": 1478.0, "peak_vram_mb": 56016.5, "total_tokens_M": 484.3, "training_seconds": 300.2, "val_bpb": 0.988487}

RECENT RESULT
hypothesis: Reducing the update batch from 327,680 to 262,144 tokens will provide 25% more whole-model updates per token while preserving the previously demonstrated throughput of 128-sequence microbatches, lowering val_bpb below 0.988487.
change: Set both total and device batch sizes to 128 sequences, retaining one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.
mechanism: Single-microbatch 256K-token updates
evidence_used: Reducing single-microbatch updates from 393,216 to 327,680 tokens improved val_bpb from 0.990146 to 0.988487, while prior 128-sequence runs demonstrated roughly 492M-token throughput; this motivates continuing the update-frequency trend at a known-efficient device batch size.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.48, "num_params_M": 50.3, "num_steps": 1833.0, "peak_vram_mb": 44909.2, "total_tokens_M": 480.5, "training_seconds": 300.1, "val_bpb": 0.987386}

RECENT RESULT
hypothesis: Adding a rank-512 multiplicative feature of the previous and current token embeddings will reduce val_bpb below 0.987386 while retaining nearly all throughput.
change: Replace the assumption that non-current lexical context must pass through attention with an asymmetric Hadamard-product expert that maps each ordered previous/current token pair directly into the pre-logit state; initialize its channel coefficients to zero and retain the proven 256K-token update regime.
mechanism: Ordered low-rank trigram readout
evidence_used: The static channel-wise current-token expert improved val_bpb from 0.995511 to 0.994364, whereas the additive multi-depth ensemble regressed to 0.997915. This motivates extending the successful direct readout with a genuinely new multiplicative token-pair interaction rather than another additive lexical source.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.27, "num_params_M": 50.3, "num_steps": 1823.0, "peak_vram_mb": 45166.2, "total_tokens_M": 477.9, "training_seconds": 300.0, "val_bpb": 0.987766}

RECENT RESULT
hypothesis: Reducing the update batch from 262,144 to 196,608 tokens will provide 33% more whole-model updates per token while retaining adequate H100 occupancy, lowering val_bpb below 0.987386.
change: Set both total and device batch sizes to 96 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.
mechanism: Single-microbatch 192K-token updates
evidence_used: Successive single-microbatch reductions from 393,216 to 327,680 to 262,144 tokens improved val_bpb from 0.990146 to 0.988487 to 0.987386 despite modest throughput losses, strongly motivating another update-frequency increase.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.3, "num_params_M": 50.3, "num_steps": 2429.0, "peak_vram_mb": 33808.0, "total_tokens_M": 477.6, "training_seconds": 300.0, "val_bpb": 0.986971}

RECENT RESULT
hypothesis: Reducing the update batch from 196,608 to 163,840 tokens will provide 20% more whole-model updates per token while retaining adequate H100 occupancy, lowering val_bpb below 0.986971.
change: Set both total and device batch sizes to 80 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.
mechanism: Single-microbatch 160K-token updates
evidence_used: Successive single-microbatch reductions from 393,216 to 327,680 to 262,144 to 196,608 tokens consistently improved val_bpb from 0.990146 to 0.988487 to 0.987386 to 0.986971, while throughput declined only modestly; a smaller incremental reduction tests whether the update-frequency benefit continues near the apparent optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.13, "num_params_M": 50.3, "num_steps": 2900.0, "peak_vram_mb": 28255.6, "total_tokens_M": 475.1, "training_seconds": 300.0, "val_bpb": 0.988167}



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
