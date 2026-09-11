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
verified_results: {"depth": 8.0, "mfu_percent": 38.14, "num_params_M": 50.3, "num_steps": 2160.0, "peak_vram_mb": 39356.7, "total_tokens_M": 495.5, "training_seconds": 300.1, "val_bpb": 0.984227}
prior_hypothesis: Adding one well-spaced full-context attention layer will reduce val_bpb below 0.984312 because the measured penalty from removing global mixing outweighed its throughput gain.

## Recent verification evidence

RECENT RESULT
hypothesis: A 76% linear warmdown will reduce val_bpb below 0.984312 by approaching the approximately 76.3% minimum implied by the measured 70%, 78%, and 80% schedules.
change: Preserve the winning architecture, batch size, optimizer, and linear schedule while beginning learning-rate decay at 24% rather than 22% of the training window.
mechanism: Local warmdown-duration response-surface refinement
evidence_used: Warmdowns of 70%, 78%, and 80% achieved val_bpb values of 0.984948, 0.984312, and 0.984506; quadratic interpolation of these nearest measurements places the estimated minimum near 76.3%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.89, "num_params_M": 50.3, "num_steps": 2179.0, "peak_vram_mb": 39356.7, "total_tokens_M": 499.8, "training_seconds": 300.1, "val_bpb": 0.984631}

RECENT RESULT
hypothesis: A 78.25% linear warmdown will reduce val_bpb below 0.984312 by matching the approximately 78.24% minimum implied by the measured 76%, 78%, and 80% schedules.
change: Preserve the winning architecture, batch size, optimizer, and linear schedule while beginning learning-rate decay at 21.75% rather than 22% of the training window.
mechanism: Local warmdown-duration response-surface refinement
evidence_used: Warmdowns of 76%, 78%, and 80% produced val_bpb values of 0.984631, 0.984312, and 0.984506; quadratic interpolation of these equally spaced measurements places the local minimum near 78.24%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.72, "num_params_M": 50.3, "num_steps": 2168.0, "peak_vram_mb": 39356.7, "total_tokens_M": 497.3, "training_seconds": 300.0, "val_bpb": 0.984642}

RECENT RESULT
hypothesis: Redistributing learning rate from the first half of the 78% warmdown to its second half, while preserving its duration, endpoints, and mean, will reduce val_bpb below 0.984312.
change: Replace the linear warmdown with a monotonic sinusoidally tilted linear curve that is 0.05 lower early in annealing and 0.05 higher late in annealing.
mechanism: Mean-preserving front-loaded annealing
evidence_used: Equal-mean cosine annealing regressed val_bpb from 0.985757 to 0.988710 when it retained larger early-annealing updates, while progressively earlier linear decay improved val_bpb through the 78% schedule; this directly motivates testing the opposite redistribution without changing total learning-rate exposure.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.85, "num_params_M": 50.3, "num_steps": 2176.0, "peak_vram_mb": 39356.7, "total_tokens_M": 499.1, "training_seconds": 300.0, "val_bpb": 0.985007}

RECENT RESULT
hypothesis: Adding one well-spaced full-context attention layer will reduce val_bpb below 0.984312 because the measured penalty from removing global mixing outweighed its throughput gain.
change: Preserve the architecture, optimizer, batch size, and 78% linear warmdown while changing the eight-layer attention topology from two to three full-context layers at indices 3, 5, and 7.
mechanism: Denser late global-context mixing
evidence_used: Removing global mixing worsened val_bpb to 0.994122 despite processing more tokens, indicating that additional long-range information flow may be worth a modest throughput cost.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.14, "num_params_M": 50.3, "num_steps": 2160.0, "peak_vram_mb": 39356.7, "total_tokens_M": 495.5, "training_seconds": 300.1, "val_bpb": 0.984227}

RECENT RESULT
hypothesis: Replacing the homogeneous squared-ReLU feature bank with an approximately parameter-matched multiplicative SwiGLU bank will reduce val_bpb below 0.984227 by learning context-dependent feature interactions without materially reducing token throughput.
change: Challenge the assumption that each MLP should independently activate polynomial features: split its expansion into learned gate and value streams, combine them multiplicatively, and preserve activation scale with a factor of two.
mechanism: Variance-matched SwiGLU feature synthesis
evidence_used: The added attention-head gate processed only 480.5M tokens and did not improve quality, while the third global layer improved val_bpb by just 0.000085; this motivates changing the token representation mechanism using a compute-matched replacement rather than adding another attention-side branch.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.27, "num_params_M": 50.1, "num_steps": 2069.0, "peak_vram_mb": 39133.1, "total_tokens_M": 474.6, "training_seconds": 300.1, "val_bpb": 0.989071}

RECENT RESULT
hypothesis: Moving the three full-context layers from indices 3, 5, and 7 to evenly spaced indices 1, 4, and 7 will reduce val_bpb below 0.984227 by exposing earlier representations to global context while preserving throughput and parameter count.
change: Change only the attention window pattern, retaining three full-context layers, the 78% linear warmdown, batch size, optimizer, and all other architecture settings.
mechanism: Evenly spaced global-context mixing
evidence_used: Adding a third full-context layer improved val_bpb from 0.984312 to 0.984227 despite reducing processed tokens from 501.6M to 495.5M, showing global mixing is valuable; redistributing the same three layers isolates whether earlier, uniform spacing uses that capacity more effectively.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.08, "num_params_M": 50.3, "num_steps": 2156.0, "peak_vram_mb": 39356.7, "total_tokens_M": 494.5, "training_seconds": 300.0, "val_bpb": 0.986575}

RECENT RESULT
hypothesis: Widening each short-context layer from 512 to 768 tokens while retaining the proven global layers at indices 3, 5, and 7 will reduce val_bpb below 0.984227 by distributing additional context across five layers.
change: Change short attention windows from one-quarter to three-eighths of the 2048-token sequence, preserving all other architecture and training settings.
mechanism: Distributed medium-context attention
evidence_used: Adding a third global layer improved val_bpb from 0.984312 to 0.984227 despite reducing processed tokens from 501.6M to 495.5M, showing that extra attention context can outweigh its throughput cost; widening local windows tests a more distributed allocation of comparable attention compute.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.77, "num_params_M": 50.3, "num_steps": 2123.0, "peak_vram_mb": 39356.7, "total_tokens_M": 487.0, "training_seconds": 300.1, "val_bpb": 0.985663}

RECENT RESULT
hypothesis: Promoting layer 6 to full-context attention will reduce val_bpb below 0.984227 by letting every layer after the proven layer-5 global mixer refine globally integrated representations.
change: Preserve the winning three global layers and all training settings, while changing the final local attention layer at index 6 to full context.
mechanism: Dense late global-context refinement
evidence_used: Adding a third global layer improved val_bpb from 0.984312 to 0.984227 despite lower throughput, whereas widening every local window regressed to 0.985663; this favors concentrated full-context capacity over distributed medium-context compute.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.49, "num_params_M": 50.3, "num_steps": 2148.0, "peak_vram_mb": 39356.7, "total_tokens_M": 492.7, "training_seconds": 300.0, "val_bpb": 0.984308}

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
