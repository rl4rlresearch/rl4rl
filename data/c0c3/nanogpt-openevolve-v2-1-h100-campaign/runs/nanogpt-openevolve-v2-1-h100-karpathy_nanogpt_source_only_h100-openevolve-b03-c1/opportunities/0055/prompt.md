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
verified_results: {"depth": 8.0, "mfu_percent": 37.53, "num_params_M": 50.3, "num_steps": 2097.0, "peak_vram_mb": 41163.2, "total_tokens_M": 481.0, "training_seconds": 300.1, "val_bpb": 0.982693}
prior_hypothesis: Reducing each context gate from eight to four input channels will lower val_bpb below 0.983164 by suppressing noisy routing features while retaining token-conditional per-head write control.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Sampling 32 channels across the full hidden state for each value gate will reduce val_bpb below 0.984083 by preserving the effective low-dimensional routing bottleneck while providing broader state coverage.
change: Replace the gate’s contiguous 32-channel prefix with 32 evenly spaced hidden-state channels, without changing parameters or matrix-operation size.
mechanism: Distributed bottleneck value-residual routing
evidence_used: Learned 32-channel gates achieved 0.984083, while removing them regressed to 0.986944 and expanding them to 64 or 512 channels regressed to 0.984472 and 0.984493; this motivates retaining the 32-channel bottleneck while testing whether its arbitrary prefix selection limits routing information.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.44, "num_params_M": 50.3, "num_steps": 2147.0, "peak_vram_mb": 39358.4, "total_tokens_M": 492.5, "training_seconds": 300.0, "val_bpb": 0.984757}

RECENT RESULT
hypothesis: Moving the existing 32-channel value gates from Muon to AdamW will reduce val_bpb below 0.984083 by allowing the tiny 4×32 routing matrices to learn correlated per-head updates without Muon’s orthogonalization constraint.
change: Exclude value-gate weights from the Muon matrix groups and optimize them with AdamW at one-quarter of the matrix learning rate, preserving the architecture and all other settings.
mechanism: Adam-optimized low-dimensional value routing
evidence_used: Learned 32-channel gates achieved 0.984083 versus 0.986944 with fixed gates, while widening their inputs to 64 or 512 channels regressed to 0.984472 and 0.984493. This indicates the compact gates are useful and motivates improving their optimization rather than increasing their capacity.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.57, "num_params_M": 50.3, "num_steps": 2155.0, "peak_vram_mb": 39356.7, "total_tokens_M": 494.3, "training_seconds": 300.1, "val_bpb": 0.984226}

RECENT RESULT
hypothesis: Reducing the learned value-gate input from 32 to 16 channels will lower val_bpb below 0.984083 by suppressing noisy routing features while retaining the learned modulation whose removal caused a large regression.
change: Halve the contiguous hidden-state prefix used by each per-head value-embedding gate, without changing its optimizer, initialization, or the winning attention-window pattern.
mechanism: Tighter value-routing bottleneck
evidence_used: Learned 32-channel gates achieved 0.984083, while widening them to 64 or 512 channels regressed to 0.984472 and 0.984493; fixed gates regressed much further to 0.986944, motivating a narrower learned bottleneck rather than removing routing.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.54, "num_params_M": 50.3, "num_steps": 2153.0, "peak_vram_mb": 39357.2, "total_tokens_M": 493.8, "training_seconds": 300.0, "val_bpb": 0.983916}

RECENT RESULT
hypothesis: Reducing each learned value gate from 16 to 8 input channels will lower val_bpb below 0.983916 by further suppressing noisy routing features while retaining input-dependent value modulation.
change: Halve the contiguous hidden-state prefix used by each per-head value-embedding gate from 16 channels to 8, preserving all other architecture and optimization settings.
mechanism: Eight-channel value-routing bottleneck
evidence_used: Narrowing the gate from 32 to 16 channels improved val_bpb from 0.984083 to 0.983916, while widening it to 64 or 512 channels regressed to 0.984472 and 0.984493; fixed gates regressed substantially to 0.986944, motivating another measured bottleneck reduction rather than removing learned routing.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.59, "num_params_M": 50.3, "num_steps": 2156.0, "peak_vram_mb": 39357.2, "total_tokens_M": 494.5, "training_seconds": 300.1, "val_bpb": 0.9839}

RECENT RESULT
hypothesis: Allowing each token to suppress or amplify each attention head’s retrieved context will reduce val_bpb below 0.983900 without materially reducing throughput.
change: Replace the assumption that every softmax-attention head must write a value at every token with a zero-initialized, eight-channel per-head gate applied after attention and before output projection.
mechanism: Token-conditional attention-head write gates
evidence_used: Attention-window refinements yielded only marginal gains, while learned per-head value routing was useful and improved further with an eight-channel bottleneck; this motivates applying the proven compact routing mechanism to the entire retrieved context, where standard softmax attention otherwise lacks a token-dependent no-write option.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.58, "num_params_M": 50.3, "num_steps": 2100.0, "peak_vram_mb": 41162.2, "total_tokens_M": 481.7, "training_seconds": 300.1, "val_bpb": 0.983164}

RECENT RESULT
hypothesis: Reducing each context gate from eight to four input channels will lower val_bpb below 0.983164 by suppressing noisy routing features while retaining token-conditional per-head write control.
change: Halve only the contiguous hidden-state prefix used by every attention context gate, preserving the winning value gates, architecture, optimizer, and schedule.
mechanism: Four-channel attention-write routing bottleneck
evidence_used: Eight-channel context gates improved val_bpb from 0.983900 to 0.983164, while value-routing gates improved as their bottleneck narrowed from 32 to 16 to 8 channels; this motivates testing whether context routing benefits from the same stronger bottleneck.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.53, "num_params_M": 50.3, "num_steps": 2097.0, "peak_vram_mb": 41163.2, "total_tokens_M": 481.0, "training_seconds": 300.1, "val_bpb": 0.982693}

RECENT RESULT
hypothesis: Reducing each context gate from four to two input channels will lower val_bpb below 0.982693 by further suppressing noisy routing features while retaining token-conditional per-head write control.
change: Halve only the contiguous hidden-state prefix used by every attention context gate, preserving the eight-channel value gates and all other architecture, optimizer, and schedule settings.
mechanism: Two-channel attention-write routing bottleneck
evidence_used: Narrowing context gates from eight to four channels improved val_bpb from 0.983164 to 0.982693, while narrower value-routing bottlenecks also consistently improved prior designs; this motivates one further controlled halving.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.39, "num_params_M": 50.3, "num_steps": 2089.0, "peak_vram_mb": 41163.0, "total_tokens_M": 479.2, "training_seconds": 300.1, "val_bpb": 0.98431}

RECENT RESULT
hypothesis: Expanding each context gate from four to six input channels will lower val_bpb below 0.982693 by approaching the approximately six-channel optimum interpolated from the measured two-, four-, and eight-channel results.
change: Increase only the contiguous hidden-state prefix used by every attention context gate from four channels to six, preserving the eight-channel value gates and all other settings.
mechanism: Intermediate-width attention-write routing bottleneck
evidence_used: Four-channel context gates achieved 0.982693, outperforming eight channels at 0.983164, while narrowing further to two channels regressed sharply to 0.984310; this non-monotonic response motivates testing the intermediate six-channel width.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing context-gate width from four to six channels will lower val_bpb below 0.982693 by approaching the estimated optimum between the regressive two- and eight-channel designs.
change: Increase only each attention context gate’s contiguous hidden-state input from four channels to six, preserving all other architecture and training settings.
mechanism: Intermediate-width attention-write routing bottleneck
evidence_used: Four-channel gates achieved 0.982693 versus 0.983164 at eight channels and 0.984310 at two channels; the unverified six-channel attempt produced no contradictory metric, so the evidence-supported intermediate remains untested.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.54, "num_params_M": 50.3, "num_steps": 2097.0, "peak_vram_mb": 41163.0, "total_tokens_M": 481.0, "training_seconds": 300.0, "val_bpb": 0.98293}



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
