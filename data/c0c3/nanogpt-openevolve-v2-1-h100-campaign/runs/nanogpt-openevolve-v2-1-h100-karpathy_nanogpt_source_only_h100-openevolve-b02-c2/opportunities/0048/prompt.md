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
verified_results: {"depth": 8.0, "mfu_percent": 35.8, "num_params_M": 50.3, "num_steps": 1975.0, "peak_vram_mb": 44908.2, "total_tokens_M": 517.7, "training_seconds": 300.1, "val_bpb": 0.983275}
prior_hypothesis: Using 144-token windows in the first three local layers and 128-token windows in the final three will lower val_bpb below 0.983317 by favoring early context formation while preserving the aggregate attention compute of uniform 136-token windows.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 35.86, "num_params_M": 50.3, "num_steps": 1978.0, "peak_vram_mb": 44908.2, "total_tokens_M": 518.5, "training_seconds": 300.0, "val_bpb": 0.982455}
prior_hypothesis: Moving eight local-window tokens from layer 4 to layer 5 will lower val_bpb below 0.982662 by reducing context in the layer where isolated expansion regressed while strengthening the following transformation, without changing attention compute.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 35.83, "num_params_M": 50.3, "num_steps": 1977.0, "peak_vram_mb": 44908.2, "total_tokens_M": 518.3, "training_seconds": 300.1, "val_bpb": 0.982539}
prior_hypothesis: Doubling the verified layer-4-to-layer-5 context transfer will lower val_bpb below 0.982455 by further concentrating local attention in the second transformation after global mixing while preserving total attention compute.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 35.65, "num_params_M": 50.3, "num_steps": 1967.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.6, "training_seconds": 300.1, "val_bpb": 0.98273}
prior_hypothesis: A 12-token layer-4-to-layer-5 context transfer will lower val_bpb below 0.982455 by targeting the apparent optimum between the improving 8-token transfer and the slightly regressing 16-token transfer while preserving aggregate attention compute.

## Recent verification evidence

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

RECENT RESULT
hypothesis: With full-context layers 3 and 8, shifting eight local-window tokens from layer 7 to layer 4 will lower val_bpb below 0.982662 by strengthening the first transformation after global mixing without changing aggregate attention compute.
change: Place full-context attention at layers 3 and 8 and use 144/144 local windows before the first global layer followed by a 144/136/128/120 taper.
mechanism: Post-global local-context restart
evidence_used: Moving the first full-context layer from 4 to 3 improved val_bpb from 0.983152 to 0.982662. Separately, moving a larger window toward the terminal local layer regressed to 0.983660, motivating compute reallocation toward the first post-global layer.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.69, "num_params_M": 50.3, "num_steps": 1969.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.2, "training_seconds": 300.0, "val_bpb": 0.982945}

RECENT RESULT
hypothesis: Relocating the layer-4 value embedding to layer 3 while using the best verified layer-3/layer-8 full-context layout will lower val_bpb below 0.982662 by injecting token-specific values during the first global mixing operation without changing parameter count.
change: Restore the verified 144/144/136/136/128/128 local taper with full context at layers 3 and 8, and move one value embedding and its gate from layer 4 to layer 3.
mechanism: Full-context value-embedding alignment
evidence_used: Moving first full context from layer 4 to layer 3 improved val_bpb from 0.983152 to 0.982662, despite leaving the alternating value embedding at layer 4; this motivates aligning that fixed-cost resource with the newly superior global layer.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.69, "num_params_M": 50.3, "num_steps": 1969.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.2, "training_seconds": 300.1, "val_bpb": 0.983327}

RECENT RESULT
hypothesis: Increasing only the first post-global local window from 136 to 144 tokens will lower val_bpb below 0.982662 by strengthening the value-embedding layer immediately after global mixing without the confounding terminal-window pruning used previously.
change: Restore full-context layers 3 and 8 with the best verified taper, then expand layer 4 alone to produce 144/144/144/136/128/128 local windows while retaining all optimizer settings.
mechanism: Isolated post-global context expansion
evidence_used: The 144/144/136/136/128/128 layer-3/layer-8 design achieved the best val_bpb of 0.982662. Its compute-matched post-global restart scored 0.982945 but simultaneously expanded layer 4 and pruned layer 7; independently, terminal pruning regressed, motivating isolation of the layer-4 expansion.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.71, "num_params_M": 50.3, "num_steps": 1970.0, "peak_vram_mb": 44908.2, "total_tokens_M": 516.4, "training_seconds": 300.1, "val_bpb": 0.982881}

RECENT RESULT
hypothesis: Moving eight local-window tokens from layer 4 to layer 5 will lower val_bpb below 0.982662 by reducing context in the layer where isolated expansion regressed while strengthening the following transformation, without changing attention compute.
change: Change the local-window schedule from 144/144/136/136/128/128 to 144/144/128/144/128/128, retaining full-context layers 3 and 8 and all optimizer settings.
mechanism: Delayed post-global context allocation
evidence_used: Expanding layer 4 alone from 136 to 144 worsened val_bpb from 0.982662 to 0.982881, while the compute-matched layer-4 expansion plus layer-7 pruning scored 0.982945; this motivates reallocating context away from layer 4 rather than adding more there.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.86, "num_params_M": 50.3, "num_steps": 1978.0, "peak_vram_mb": 44908.2, "total_tokens_M": 518.5, "training_seconds": 300.0, "val_bpb": 0.982455}

RECENT RESULT
hypothesis: Doubling the verified layer-4-to-layer-5 context transfer will lower val_bpb below 0.982455 by further concentrating local attention in the second transformation after global mixing while preserving total attention compute.
change: Use 144/144/120/152/128/128 local windows with full-context layers 3 and 8, transferring another eight tokens from layer 4 to layer 5.
mechanism: Amplified delayed post-global context allocation
evidence_used: The compute-matched shift from 144/144/136/136/128/128 to 144/144/128/144/128/128 improved val_bpb from 0.982662 to 0.982455, while expanding layer 4 alone regressed to 0.982881; this motivates a dose-response test of the same transfer direction.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.83, "num_params_M": 50.3, "num_steps": 1977.0, "peak_vram_mb": 44908.2, "total_tokens_M": 518.3, "training_seconds": 300.1, "val_bpb": 0.982539}

RECENT RESULT
hypothesis: A 12-token layer-4-to-layer-5 context transfer will lower val_bpb below 0.982455 by targeting the apparent optimum between the improving 8-token transfer and the slightly regressing 16-token transfer while preserving aggregate attention compute.
change: Restore 128-token windows in layers 6 and 7, then use 124/148-token windows in layers 4/5 with full-context layers 3 and 8.
mechanism: Intermediate delayed post-global context allocation
evidence_used: Transfers of 0, 8, and 16 tokens produced val_bpb values of 0.982662, 0.982455, and 0.982539 respectively, motivating a midpoint test between the two transferred schedules.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.65, "num_params_M": 50.3, "num_steps": 1967.0, "peak_vram_mb": 44908.2, "total_tokens_M": 515.6, "training_seconds": 300.1, "val_bpb": 0.98273}



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
