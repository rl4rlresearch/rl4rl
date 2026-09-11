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
verified_results: {"depth": 8.0, "mfu_percent": 36.52, "num_params_M": 50.3, "num_steps": 1972.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.9, "training_seconds": 300.0, "val_bpb": 0.984467}
prior_hypothesis: Adding an intermediate full-context attention layer will lower val_bpb below 0.985229 by removing the single-final-layer global-information bottleneck, despite modestly reduced token throughput.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a learned direct path from the current-token embedding to the output logits will factorize common token transitions away from the contextual stack, allowing the preserved full-context model and 262K-token update cadence to lower val_bpb below 0.986167 without meaningful throughput loss.
change: Add a zero-initialized per-channel output gate that mixes the normalized input embedding into the final representation before the shared language-model head, with a dedicated conservative AdamW learning rate and complete parameter/FLOP accounting.
mechanism: Gated low-rank bigram prediction bypass
evidence_used: The 262K-token batch improved val_bpb to 0.986167, while removing full-context integration regressed to 1.015479. The old design assumes one final representation must encode both immediate token-transition statistics and global context; this patch preserves global attention but gives prediction a separate learned low-rank bigram route.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.81, "num_params_M": 50.3, "num_steps": 1984.0, "peak_vram_mb": 44908.2, "total_tokens_M": 520.1, "training_seconds": 300.1, "val_bpb": 0.986618}

RECENT RESULT
hypothesis: Starting warmdown at 40% rather than 50% of the training window will reduce late-stage optimization noise at the successful 262K-token batch size and lower val_bpb below 0.986167.
change: Increase WARMDOWN_RATIO from 0.5 to 0.6 while preserving the model, batch size, peak learning rates, and zero final learning rate.
mechanism: Earlier linear learning-rate warmdown
evidence_used: Halving the batch from 524K to 262K improved val_bpb from 0.992286 to 0.986167, but halving it again to 131K regressed to 0.991527, indicating excessive high-frequency optimization is harmful; a slightly earlier warmdown tests reduced late update magnitude without sacrificing throughput or update cadence.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 34.93, "num_params_M": 50.3, "num_steps": 1991.0, "peak_vram_mb": 44908.2, "total_tokens_M": 521.9, "training_seconds": 300.1, "val_bpb": 0.985493}

RECENT RESULT
hypothesis: Starting warmdown at 30% rather than 40% of the training window will further reduce late-stage optimization noise at the successful 262K-token batch size and lower val_bpb below 0.985493.
change: Increase WARMDOWN_RATIO from 0.6 to 0.7 while preserving architecture, batch size, peak learning rates, throughput, and zero final learning rate.
mechanism: Extended linear learning-rate warmdown
evidence_used: Increasing WARMDOWN_RATIO from 0.5 to 0.6 improved val_bpb from 0.986167 to 0.985493 at effectively unchanged tokens and steps, directly motivating a continuation of the warmdown-duration sweep.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.47, "num_params_M": 50.3, "num_steps": 1965.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.1, "training_seconds": 300.1, "val_bpb": 0.985706}

RECENT RESULT
hypothesis: A 196K-token batch will balance the improved update frequency seen at 262K tokens against the throughput and optimization regression at 131K tokens, lowering val_bpb below 0.985493.
change: Set both total and device batch size to 96 sequences (196,608 tokens), preserving one microbatch per optimizer step and the successful 0.6 warmdown schedule.
mechanism: Intermediate-batch optimization cadence
evidence_used: Reducing batch size from 524K to 262K improved val_bpb from 0.992286 to 0.986167, while reducing it further to 131K regressed to 0.991527; testing the midpoint directly probes the apparent interior optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.42, "num_params_M": 50.3, "num_steps": 2612.0, "peak_vram_mb": 33807.2, "total_tokens_M": 513.5, "training_seconds": 300.1, "val_bpb": 0.986221}

RECENT RESULT
hypothesis: Setting WARMDOWN_RATIO to 0.625 will better balance sustained peak learning with late-stage noise reduction and lower val_bpb below 0.985493.
change: Start linear learning-rate warmdown at 37.5% of the training window while preserving architecture, batch size, peak learning rates, and final zero learning rate.
mechanism: Quadratic-guided warmdown refinement
evidence_used: WARMDOWN_RATIO 0.6 improved val_bpb from 0.986167 to 0.985493, while 0.7 slightly regressed to 0.985706; the asymmetric three-point sweep suggests the local optimum lies just above 0.6.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 34.99, "num_params_M": 50.3, "num_steps": 1994.0, "peak_vram_mb": 44908.2, "total_tokens_M": 522.7, "training_seconds": 300.1, "val_bpb": 0.985229}

RECENT RESULT
hypothesis: Setting WARMDOWN_RATIO to 0.645 will lower val_bpb below 0.985229 by placing warmdown near the local optimum implied by the 0.6, 0.625, and 0.7 results.
change: Increase WARMDOWN_RATIO from 0.625 to 0.645 while preserving architecture, batching, peak learning rates, and final zero learning rate.
mechanism: Quadratic-interpolated warmdown timing
evidence_used: WARMDOWN_RATIO 0.625 achieved the best val_bpb of 0.985229, improving on 0.6 at 0.985493, while 0.7 regressed to 0.985706; quadratic interpolation of these three controlled results places the estimated minimum near 0.644.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.78, "num_params_M": 50.3, "num_steps": 1982.0, "peak_vram_mb": 44908.2, "total_tokens_M": 519.6, "training_seconds": 300.1, "val_bpb": 0.985678}

RECENT RESULT
hypothesis: Replacing linear warmdown with an equal-area cosine curve will shift learning-rate mass from noisier late updates to earlier warmdown updates and lower val_bpb below 0.985229.
change: Keep the successful 0.625 warmdown duration and all other settings unchanged, but use cosine interpolation between peak and final learning rates.
mechanism: Equal-area cosine warmdown
evidence_used: Linear WARMDOWN_RATIO 0.625 achieved 0.985229, while extending it to 0.7 regressed to 0.985706; reshaping the best-duration schedule preserves its average learning rate while reducing late-step magnitude.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.09, "num_params_M": 50.3, "num_steps": 2000.0, "peak_vram_mb": 44908.2, "total_tokens_M": 524.3, "training_seconds": 300.1, "val_bpb": 0.987814}

RECENT RESULT
hypothesis: Shifting learning-rate mass from early to late warmdown while preserving duration, endpoints, and average rate will lower val_bpb below 0.985229.
change: Replace linear warmdown with a monotonic cubic schedule that keeps WARMDOWN_RATIO at 0.625 but redistributes rate toward later updates.
mechanism: Equal-area late-biased cubic warmdown
evidence_used: The equal-area cosine schedule shifted rate toward early warmdown and away from late updates, regressing val_bpb from 0.985229 to 0.987814; testing the opposite redistribution directly probes whether later warmdown updates are more valuable.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.85, "num_params_M": 50.3, "num_steps": 1986.0, "peak_vram_mb": 44908.2, "total_tokens_M": 520.6, "training_seconds": 300.1, "val_bpb": 0.986641}

RECENT RESULT
hypothesis: Adding an intermediate full-context attention layer will lower val_bpb below 0.985229 by removing the single-final-layer global-information bottleneck, despite modestly reduced token throughput.
change: Change the attention pattern from seven local layers plus one forced full-context layer to six local layers and full-context attention at layers 4 and 8.
mechanism: Periodic full-context consolidation
evidence_used: Making the final layer local regressed sharply to 1.015479, showing that global integration is disproportionately valuable, while throughput-oriented 128-token windows and grouped-query attention also regressed; this motivates testing additional global capacity rather than pursuing throughput alone.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.52, "num_params_M": 50.3, "num_steps": 1972.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.9, "training_seconds": 300.0, "val_bpb": 0.984467}

RECENT RESULT
hypothesis: Adding full-context attention at layer 6 while retaining it at layers 4 and 8 will reduce val_bpb below 0.984467, despite lower token throughput, by letting later layers reintegrate global information before final prediction.
change: Change the attention schedule from full-context layers 4 and 8 to layers 4, 6, and 8; all other architecture and optimization settings remain unchanged.
mechanism: Three-stage full-context consolidation
evidence_used: Moving from one full-context layer to layers 4 and 8 improved val_bpb from 0.985229 to 0.984467 even as total tokens fell, while making attention entirely local regressed sharply to 1.015479; this motivates a controlled increase in global-attention depth.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.48, "num_params_M": 50.3, "num_steps": 1924.0, "peak_vram_mb": 44908.2, "total_tokens_M": 504.4, "training_seconds": 300.1, "val_bpb": 0.985972}

RECENT RESULT
hypothesis: Giving one attention head full context in every layer will lower val_bpb below 0.984467 by enabling continuous global information exchange while preserving the best design’s exact global/local attention-head compute budget.
change: Replace periodic full-width global layers with three local heads and one global head per layer; parameter count and theoretical attention FLOPs remain unchanged, but context scale becomes a learned channel-wise representation rather than a layer-wise choice.
mechanism: Head-wise continuous global context
evidence_used: Two full-context layers improved val_bpb to 0.984467, while three regressed to 0.985972. This suggests the load-bearing factor is a limited amount of global computation, challenging the old assumption that it must occur as occasional all-channel consolidation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.34, "num_params_M": 50.3, "num_steps": 1963.0, "peak_vram_mb": 46959.2, "total_tokens_M": 514.6, "training_seconds": 300.1, "val_bpb": 0.985719}

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from layer 4 to layer 5 will lower val_bpb below 0.984467 by allowing an additional local layer to construct features before global mixing while retaining two local refinement layers before the final global layer.
change: Use full-context attention at layers 5 and 8 instead of layers 4 and 8, preserving parameter count, attention-compute budget, batching, and optimization.
mechanism: Later two-stage full-context consolidation
evidence_used: Full-context layers 4 and 8 improved val_bpb from 0.985229 to 0.984467, but adding layer 6 regressed to 0.985972 and distributing global heads across every layer regressed to 0.985719; this motivates holding the successful two-layer global-attention budget fixed while testing its placement.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.11, "num_params_M": 50.3, "num_steps": 1950.0, "peak_vram_mb": 44908.2, "total_tokens_M": 511.2, "training_seconds": 300.0, "val_bpb": 0.985851}



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
