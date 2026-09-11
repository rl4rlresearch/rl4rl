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
verified_results: {"depth": 8.0, "mfu_percent": 35.91, "num_params_M": 50.3, "num_steps": 1981.0, "peak_vram_mb": 44908.2, "total_tokens_M": 519.3, "training_seconds": 300.1, "val_bpb": 0.983152}
prior_hypothesis: A monotone 144/144/136/136/128/128 local-window taper will lower val_bpb below 0.983275 by approximating the moderate early-context bias while keeping every window 8-token aligned.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 35.6, "num_params_M": 50.3, "num_steps": 1964.0, "peak_vram_mb": 44908.2, "total_tokens_M": 514.9, "training_seconds": 300.1, "val_bpb": 0.982662}
prior_hypothesis: Moving the first full-context layer from layer 4 to layer 3 while retaining the best 144/144/136/136/128/128 local-window taper will lower val_bpb below 0.983152 by giving five downstream layers access to globally mixed representations without changing aggregate attention compute.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 35.62, "num_params_M": 50.3, "num_steps": 1965.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.1, "training_seconds": 300.1, "val_bpb": 0.983933}
prior_hypothesis: Using 152-token windows in the first three local layers and 120-token windows in the final three will lower val_bpb below 0.983275 by extending the observed advantage of allocating more local context to earlier layers while preserving the six-layer average of 136 tokens.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983275}
prior_hypothesis: Using 144-token windows in the first three local layers and 128-token windows in the final three will lower val_bpb below 0.983317 by favoring early context formation while preserving the aggregate attention compute of uniform 136-token windows.

## Recent verification evidence

RECENT RESULT
hypothesis: Using 128-token windows in the first three local layers and 144-token windows in the final three local layers will lower val_bpb below 0.983317 by allocating more context to higher-level representations while matching the total local-attention compute of the best homogeneous 136-token design.
change: Replace uniform 128-token local attention with 128-token early windows and 144-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.
mechanism: Depth-progressive local-context allocation
evidence_used: Uniform 136-token windows achieved the best val_bpb of 0.983317, outperforming both 128 tokens at 0.983766 and 144 tokens at 0.983758; a 3×128 plus 3×144 schedule preserves exactly the same aggregate local-window budget as 6×136 while testing whether context is more valuable at greater depth.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.64, "num_params_M": 50.3, "num_steps": 1967.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.6, "training_seconds": 300.1, "val_bpb": 0.983641}

RECENT RESULT
hypothesis: Using 144-token windows in the first three local layers and 128-token windows in the final three will lower val_bpb below 0.983317 by favoring early context formation while preserving the aggregate attention compute of uniform 136-token windows.
change: Replace uniform 512-token local attention with 144-token early and 128-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.
mechanism: Reverse depth-progressive local-context allocation
evidence_used: Uniform 136-token windows achieved 0.983317, while the compute-matched 128-early/144-late allocation regressed to 0.983641; reversing that allocation directly tests whether local context is more valuable in earlier layers.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983275}

RECENT RESULT
hypothesis: Using 152-token windows in the first three local layers and 120-token windows in the final three will lower val_bpb below 0.983275 by extending the observed advantage of allocating more local context to earlier layers while preserving the six-layer average of 136 tokens.
change: Replace uniform 192-token local attention with 152-token early and 120-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.
mechanism: Compute-matched early-context reallocation
evidence_used: The compute-matched 144-early/128-late design achieved the best result, 0.983275, outperforming uniform 136 at 0.983317 and the reversed 128-early/144-late allocation at 0.983641; increasing the same early-layer bias tests whether that directional improvement continues.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.62, "num_params_M": 50.3, "num_steps": 1965.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.1, "training_seconds": 300.1, "val_bpb": 0.983933}

RECENT RESULT
hypothesis: Compute-matched 140-token early and 132-token late windows will lower val_bpb below 0.983275 by placing the depth bias near the quadratic minimum implied by the uniform, moderate-bias, and strong-bias results.
change: Reduce the early/late local-window contrast from 144/128 to 140/132 while preserving the 136-token average, full-context layers 4 and 8, and all optimizer settings.
mechanism: Quadratic refinement of depth-biased local context
evidence_used: Uniform 136/136 achieved 0.983317, moderate 144/128 improved to 0.983275, and stronger 152/120 regressed to 0.983933; quadratic interpolation places the estimated optimum near 140.5/131.5.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.53, "num_params_M": 50.3, "num_steps": 1960.0, "peak_vram_mb": 44908.2, "total_tokens_M": 513.8, "training_seconds": 300.1, "val_bpb": 0.984113}

RECENT RESULT
hypothesis: A monotone 144/144/136/136/128/128 local-window taper will lower val_bpb below 0.983275 by approximating the moderate early-context bias while keeping every window 8-token aligned.
change: Replace uniform 136-token local attention with a compute-matched depth taper, preserving full-context layers 4 and 8 and all optimizer settings.
mechanism: Hardware-aligned depth-tapered local context
evidence_used: The aligned 144-early/128-late design achieved the best 0.983275, while the stronger 152/120 bias and non-8-aligned 140/132 refinement regressed; a three-level aligned taper tests a gentler transition without changing aggregate attention compute.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.91, "num_params_M": 50.3, "num_steps": 1981.0, "peak_vram_mb": 44908.2, "total_tokens_M": 519.3, "training_seconds": 300.1, "val_bpb": 0.983152}

RECENT RESULT
hypothesis: An aligned 144/136/136/136/136/128 local-window schedule will lower val_bpb below 0.983152 by retaining the beneficial early-context bias while avoiding repeated extreme windows and preserving the 136-token average.
change: Replace the current two-level 128/144 allocation with a compute-matched six-layer monotone taper using 144-token endpoints and 136-token middle windows.
mechanism: Endpoint-weighted monotone local-context taper
evidence_used: The aligned 144/144/136/136/128/128 taper achieved the best val_bpb, 0.983152, outperforming the 144/128 split at 0.983275, while the stronger 152/120 bias regressed to 0.983933; this motivates a gentler aligned taper.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.53, "num_params_M": 50.3, "num_steps": 1961.0, "peak_vram_mb": 44908.2, "total_tokens_M": 514.1, "training_seconds": 300.1, "val_bpb": 0.984107}

RECENT RESULT
hypothesis: A 152/144/136/136/128/120 local-window taper will lower val_bpb below 0.983152 by concentrating context gradually toward early layers without exposing three layers each to the unsuccessful 152/120 extremes.
change: Replace the current two-level 152/120 allocation with an 8-token-aligned six-local-layer taper that preserves the 136-token average and full-context layers 4 and 8.
mechanism: Compute-matched expanded local-context taper
evidence_used: The 144/144/136/136/128/128 taper achieved the best val_bpb of 0.983152, while the abrupt 152/152/152/120/120/120 split regressed to 0.983933; a gradual expanded taper tests the same endpoint range without the abrupt depth partition.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.25, "num_params_M": 50.3, "num_steps": 1945.0, "peak_vram_mb": 44908.2, "total_tokens_M": 509.9, "training_seconds": 300.1, "val_bpb": 0.984234}

RECENT RESULT
hypothesis: Moving the second-stage 136-token window from the local layer immediately after the first full-context layer to the local layer immediately before final full-context integration will lower val_bpb below 0.983152 without changing attention compute or throughput.
change: Change the six local windows from 144/144/136/136/128/128 to 144/144/136/128/128/136, preserving the proven first-stage taper, 136-token average, and 8-token alignment.
mechanism: Post-global context deferral
evidence_used: The compute-matched 144/144/136/136/128/128 taper is the best result at 0.983152; this permutation isolates window placement within the second three-local-layer stage while keeping its multiset and every other setting unchanged.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.61, "num_params_M": 50.3, "num_steps": 1965.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.1, "training_seconds": 300.1, "val_bpb": 0.98366}

RECENT RESULT
hypothesis: A 144/144/136/136/128/120 local-window schedule will lower val_bpb below 0.983152 by retaining the best taper’s useful early context while shortening the local layer immediately before final full-context integration to gain throughput.
change: Replace the current early/late split with the best verified explicit taper, except reduce its final local window from 128 to 120 tokens.
mechanism: Terminal-local context pruning
evidence_used: The 144/144/136/136/128/128 taper achieved the best val_bpb of 0.983152, while moving a larger 136-token window to the final local layer regressed to 0.983660; this suggests additional context is least valuable immediately before the final full-context layer.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.81, "num_params_M": 50.3, "num_steps": 1976.0, "peak_vram_mb": 44908.2, "total_tokens_M": 518.0, "training_seconds": 300.1, "val_bpb": 0.983745}

RECENT RESULT
hypothesis: Moving the first full-context layer from layer 4 to layer 3 while retaining the best 144/144/136/136/128/128 local-window taper will lower val_bpb below 0.983152 by giving five downstream layers access to globally mixed representations without changing aggregate attention compute.
change: Restore the best verified six-window taper and place full-context attention at layers 3 and 8.
mechanism: Early global-context integration
evidence_used: The 144/144/136/136/128/128 taper achieved the best val_bpb of 0.983152, and its larger early windows indicate that context is especially valuable near the bottom of the network; advancing the first full-context layer tests that implication on a new architectural axis while preserving the exact local-window multiset.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.6, "num_params_M": 50.3, "num_steps": 1964.0, "peak_vram_mb": 44908.2, "total_tokens_M": 514.9, "training_seconds": 300.1, "val_bpb": 0.982662}

RECENT RESULT
hypothesis: Moving the first full-context layer from layer 3 to layer 2 while retaining the verified 144/144/136/136/128/128 local-window multiset will lower val_bpb below 0.982662 by exposing six downstream layers to globally mixed representations without changing aggregate attention compute.
change: Restore the best explicit local-window taper and place full-context attention at layers 2 and 8.
mechanism: Earlier global-context integration
evidence_used: Moving the first full-context layer from layer 4 to layer 3 improved val_bpb from 0.983152 to 0.982662 with the same local-window multiset, motivating one further step along that architectural axis.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.83, "num_params_M": 50.3, "num_steps": 1977.0, "peak_vram_mb": 44908.2, "total_tokens_M": 518.3, "training_seconds": 300.1, "val_bpb": 0.986773}

RECENT RESULT
hypothesis: Moving the second full-context layer from layer 8 to layer 7 while retaining full context at layer 3 and the verified local-window multiset will lower val_bpb below 0.982662 by allowing the final block to transform globally mixed representations without changing aggregate attention compute.
change: Place full-context attention at layers 3 and 7, with 144/144/136/136/128/128 windows across the remaining layers, and remove the unconditional final-layer override.
mechanism: Penultimate global-context integration
evidence_used: Advancing the first full-context layer from layer 4 to layer 3 improved val_bpb from 0.983152 to 0.982662 at unchanged compute; shifting the second full-context layer one step earlier tests the same benefit while preserving the best first-global placement, whereas the large regression at layer 2 cautions against a larger shift.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.84, "num_params_M": 50.3, "num_steps": 1977.0, "peak_vram_mb": 44908.2, "total_tokens_M": 518.3, "training_seconds": 300.1, "val_bpb": 0.988445}



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
