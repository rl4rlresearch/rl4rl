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
verified_results: {"depth": 8.0, "mfu_percent": 38.58, "num_params_M": 50.3, "num_steps": 2156.0, "peak_vram_mb": 39356.7, "total_tokens_M": 494.5, "training_seconds": 300.1, "val_bpb": 0.984083}
prior_hypothesis: Giving only layer 6 a 1024-token attention window will reduce val_bpb below 0.984227 by retaining useful late-context refinement without the throughput cost of another full-context layer.

## Recent verification evidence

RECENT RESULT
hypothesis: Halving short-attention windows to 256 tokens while retaining full-context layers at indices 3, 5, and 7 will reduce val_bpb below 0.984227 by increasing processed tokens without removing proven global mixing.
change: Change the five short-context layers from quarter-context (512 tokens) to eighth-context (256 tokens), preserving all other architecture and training settings.
mechanism: Compute reallocation from local context to token throughput
evidence_used: Widening short windows to 768 tokens regressed val_bpb to 0.985663, while adding global mixing improved val_bpb despite lower throughput; this suggests long-range capacity is best concentrated in the three global layers and local-attention compute can be reduced.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.03, "num_params_M": 50.3, "num_steps": 2172.0, "peak_vram_mb": 39356.7, "total_tokens_M": 498.2, "training_seconds": 300.1, "val_bpb": 0.984865}

RECENT RESULT
hypothesis: Splitting the unchanged 512-dimensional attention space into eight 64-dimensional heads will reduce val_bpb below 0.984227 without materially reducing throughput or changing parameter count.
change: Change HEAD_DIM from 128 to 64, increasing query and key/value heads from four to eight while preserving model width, depth, context windows, optimizer, and schedule.
mechanism: Finer-grained attention-head factorization
evidence_used: Adding a third global-attention layer improved val_bpb from 0.984312 to 0.984227 despite processing fewer tokens, indicating attention representation capacity is limiting; finer head partitioning tests additional relational subspaces without adding attention FLOPs or parameters.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.04, "num_params_M": 50.3, "num_steps": 2098.0, "peak_vram_mb": 39448.0, "total_tokens_M": 481.2, "training_seconds": 300.1, "val_bpb": 0.990109}

RECENT RESULT
hypothesis: Replacing the learned per-head value-embedding gate with its neutral fixed mixture will reduce val_bpb below 0.984227 by eliminating an unhelpful projection and increasing token throughput.
change: Remove the 32-channel value-gate projection and inject each alternating value embedding directly into the attention value stream.
mechanism: Ungated value-residual injection
evidence_used: The prior attention-head gate experiment processed only 480.5M tokens and did not improve quality; fixing the gate at its initialization-equivalent value of one preserves the original value-residual signal while removing its measured overhead.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.11, "num_params_M": 50.3, "num_steps": 2158.0, "peak_vram_mb": 38454.9, "total_tokens_M": 495.0, "training_seconds": 300.0, "val_bpb": 0.986944}

RECENT RESULT
hypothesis: Expanding each value-embedding gate from 32 input channels to the full 512-dimensional hidden state will reduce val_bpb below 0.984227 without materially affecting throughput.
change: Let the existing per-head value gate condition on every normalized hidden-state channel instead of an arbitrary 32-channel prefix.
mechanism: Full-state value-residual routing
evidence_used: Replacing the learned value gate with a fixed neutral mixture regressed val_bpb from 0.984227 to 0.986944 while preserving essentially the same throughput (495.5M versus 495.0M tokens), showing that learned value-residual routing is useful and its small projection is not a meaningful compute bottleneck.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.04, "num_params_M": 50.3, "num_steps": 2154.0, "peak_vram_mb": 39357.3, "total_tokens_M": 494.1, "training_seconds": 300.1, "val_bpb": 0.984493}

RECENT RESULT
hypothesis: Giving only layer 6 a 1024-token attention window will reduce val_bpb below 0.984227 by retaining useful late-context refinement without the throughput cost of another full-context layer.
change: Add a medium-window pattern symbol and change layer 6 from a 512-token local window to a 1024-token window, preserving the proven full-context layers at indices 3, 5, and 7.
mechanism: Concentrated medium-context late refinement
evidence_used: Promoting layer 6 to full context nearly matched the winner at 0.984308 despite processing 2.8M fewer tokens, suggesting useful late-context capacity with excessive compute cost; widening all local layers regressed to 0.985663, motivating a concentrated intermediate window.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.58, "num_params_M": 50.3, "num_steps": 2156.0, "peak_vram_mb": 39356.7, "total_tokens_M": 494.5, "training_seconds": 300.1, "val_bpb": 0.984083}

RECENT RESULT
hypothesis: Increasing layer 6’s attention window from 1024 to 1280 tokens will reduce val_bpb below 0.984083 by moving toward the approximately 1200-token optimum interpolated from the measured 512-, 1024-, and 2048-token results.
change: Change the single medium-context layer’s window from one-half to five-eighths of the 2048-token sequence while preserving all other settings.
mechanism: Late-context window response-surface refinement
evidence_used: Layer 6 windows of 512, 1024, and 2048 tokens produced val_bpb values of 0.984227, 0.984083, and 0.984308; quadratic interpolation places the local minimum near 1200 tokens.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.83, "num_params_M": 50.3, "num_steps": 2154.0, "peak_vram_mb": 39356.7, "total_tokens_M": 494.1, "training_seconds": 300.0, "val_bpb": 0.984367}

RECENT RESULT
hypothesis: Replacing every tokenwise squared-ReLU MLP with a parameter-matched MLP that jointly transforms the current and immediately preceding hidden states will reduce val_bpb below 0.984083 by giving each block a direct learned local-context pathway without adding attention or the separate projections that reduced SwiGLU throughput.
change: Concatenate each normalized hidden state with a one-token causal shift before the MLP, and reduce the expansion width from 4d to approximately 8d/3 so the two-linear-layer MLP retains essentially the same parameter count and matrix FLOPs.
mechanism: Parameter-matched causal bigram MLP
evidence_used: Attention-window changes around the winning design produced only small gains or regressions, suggesting that attention-only context formation is a load-bearing assumption worth challenging. The parameter-matched SwiGLU attempt fell to 474.6M tokens and 0.989071 with multiple expansion projections; this alternative keeps two matrix multiplications while testing a genuinely different mechanism in which the feature bank directly learns adjacent-token interactions.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.65, "num_params_M": 50.3, "num_steps": 2163.0, "peak_vram_mb": 36670.0, "total_tokens_M": 496.1, "training_seconds": 300.1, "val_bpb": 0.992404}

RECENT RESULT
hypothesis: Reducing layer 6’s attention window from 1024 to 768 tokens will lower val_bpb below 0.984083 by moving toward the sub-1024 optimum implied by the measured response.
change: Change only the single medium-context layer’s window from one-half to three-eighths of the 2048-token sequence.
mechanism: Asymmetric late-context window refinement
evidence_used: Layer 6 at 1024 tokens achieved 0.984083, while increasing it to 1280 regressed sharply to 0.984367 and reducing it to 512 caused only a smaller regression to 0.984227; this asymmetric response motivates testing between 512 and 1024.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.36, "num_params_M": 50.3, "num_steps": 2157.0, "peak_vram_mb": 39356.7, "total_tokens_M": 494.8, "training_seconds": 300.0, "val_bpb": 0.984119}

RECENT RESULT
hypothesis: Reducing layer 6’s attention window from 1024 to 896 tokens will lower val_bpb below 0.984083 by approaching the approximately 925-token optimum interpolated from the measured 768-, 1024-, and 1280-token results.
change: Change the single medium-context window from one-half to seven-sixteenths of the 2048-token sequence while preserving all other architecture and training settings.
mechanism: Late-context window response-surface refinement
evidence_used: Layer 6 windows of 768, 1024, and 1280 tokens yielded val_bpb values of 0.984119, 0.984083, and 0.984367; quadratic interpolation places the local minimum near 925 tokens, motivating an aligned 896-token test.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.19, "num_params_M": 50.3, "num_steps": 2141.0, "peak_vram_mb": 39356.7, "total_tokens_M": 491.1, "training_seconds": 300.1, "val_bpb": 0.984526}

RECENT RESULT
hypothesis: Giving layer 4 a 1024-token window will reduce val_bpb below 0.984083 by adding targeted post-global refinement without broadly widening all local layers.
change: Change layer 4 from a 512-token short window to a 1024-token medium window, preserving full-context layers 3, 5, and 7 and the proven medium window at layer 6.
mechanism: Alternating late-stage medium/global context refinement
evidence_used: A single 1024-token window at layer 6 improved val_bpb from 0.984227 to 0.984083, whereas widening every short layer to 768 regressed to 0.985663; this motivates another concentrated medium window rather than distributed widening.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.0, "num_params_M": 50.3, "num_steps": 2150.0, "peak_vram_mb": 39356.7, "total_tokens_M": 493.2, "training_seconds": 300.1, "val_bpb": 0.984595}

RECENT RESULT
hypothesis: Reducing only layer 4’s window from 512 to 256 tokens will lower val_bpb below 0.984083 by increasing throughput in a layer that did not benefit from wider context, while preserving the proven early 512-token, global, and layer-6 medium windows.
change: Add an eighth-context window symbol and apply it only to layer 4, changing the pattern from SSSL​SLML to SSSL​NLML.
mechanism: Targeted post-global context compression
evidence_used: Widening layer 4 to 1024 tokens regressed val_bpb to 0.984595, while reducing every short layer to 256 regressed to 0.984865; a targeted reduction tests whether layer 4’s attention compute can be reclaimed without removing useful early-layer context.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.35, "num_params_M": 50.3, "num_steps": 2157.0, "peak_vram_mb": 39356.7, "total_tokens_M": 494.8, "training_seconds": 300.1, "val_bpb": 0.984253}

RECENT RESULT
hypothesis: Expanding each value-embedding gate from 32 to 64 hidden-state channels will reduce val_bpb below 0.984083 by providing richer routing signals without the noisy overconditioning observed with all 512 channels.
change: Double the value-gate input width while preserving the winning attention pattern, initialization, optimizer, and training schedule.
mechanism: Moderately expanded value-residual routing
evidence_used: Removing learned gates regressed val_bpb to 0.986944, while full-state gates reached only 0.984493; this suggests learned routing is valuable but benefits from a low-dimensional bottleneck, motivating a modest expansion beyond 32 channels.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.1, "num_params_M": 50.3, "num_steps": 2129.0, "peak_vram_mb": 39357.2, "total_tokens_M": 488.3, "training_seconds": 300.1, "val_bpb": 0.984472}



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
