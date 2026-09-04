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
verified_results: {"depth": 8.0, "mfu_percent": 35.34, "num_params_M": 50.3, "num_steps": 3178.0, "peak_vram_mb": 28254.4, "total_tokens_M": 520.7, "training_seconds": 300.0, "val_bpb": 0.981559}
prior_hypothesis: Halving only the final layer’s attention window to 1,024 tokens will increase throughput enough to achieve `val_bpb < 0.982068`, while the earlier full-context layer preserves long-range information.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the six local-attention layers from 512-token to 256-token windows while retaining two full-context layers will increase token throughput enough to achieve `val_bpb < 0.983023`.
change: Use one-eighth-context windows for `S` layers without changing model capacity, optimizer, batching, schedule, softcap, or full-context layers.
mechanism: Further short-window context-compute reallocation
evidence_used: Reducing local windows from 1,024 to 512 tokens increased throughput from 493.2M to 509.1M tokens and improved `val_bpb` from `0.984431` to `0.983023`; this continuation tests whether further compute reallocation remains beneficial.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.56, "num_params_M": 50.3, "num_steps": 3153.0, "peak_vram_mb": 28254.4, "total_tokens_M": 516.6, "training_seconds": 300.0, "val_bpb": 0.982793}

RECENT RESULT
hypothesis: Reducing the six local-attention layers from 256-token to 128-token windows while retaining two full-context layers will increase training throughput enough to achieve `val_bpb < 0.982793`.
change: Change `S` layers from one-eighth-context to one-sixteenth-context attention without altering model capacity, optimizer, batch size, schedule, softcap, or full-context layers.
mechanism: Further local-attention compute reallocation
evidence_used: Shortening local windows from 1,024 to 512 tokens improved `val_bpb` from `0.984431` to `0.983023`, and shortening them again to 256 tokens improved it to `0.982793` while increasing throughput from 509.1M to 516.6M tokens; this directly tests whether that compute-quality trend continues.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.04, "num_params_M": 50.3, "num_steps": 3179.0, "peak_vram_mb": 28254.4, "total_tokens_M": 520.8, "training_seconds": 300.0, "val_bpb": 0.982806}

RECENT RESULT
hypothesis: Using 192-token local-attention windows will achieve `val_bpb < 0.982793` by balancing the quality advantage of 256-token windows with the higher throughput observed at 128 tokens.
change: Set the six `S` layers to a 3/32-context (192-token) attention window while retaining two full-context layers and all other settings.
mechanism: Interpolated local-attention compute allocation
evidence_used: Moving from 512 to 256 tokens improved `val_bpb` from `0.983023` to `0.982793`, while 128 tokens increased throughput but slightly regressed to `0.982806`; quadratic interpolation over the 512/256/128 results places the estimated optimum near 192 tokens.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.96, "num_params_M": 50.3, "num_steps": 3050.0, "peak_vram_mb": 30816.8, "total_tokens_M": 499.7, "training_seconds": 300.1, "val_bpb": 0.984658}

RECENT RESULT
hypothesis: Using 128-token windows for four local layers while retaining 256-token windows immediately before each full-context layer will achieve `val_bpb < 0.982793` by preserving high-value local aggregation while recovering most of the 128-token throughput gain.
change: Shorten four local-attention layers to 128 tokens and keep the two local layers directly preceding full-context attention at 256 tokens.
mechanism: Hierarchical local-to-global attention allocation
evidence_used: Uniform 128-token windows increased throughput from 516.6M to 520.8M tokens with only a `0.000013` regression versus 256 tokens; concentrating 256-token windows at the local-to-global boundaries tests whether that small quality advantage can be retained efficiently.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.24, "num_params_M": 50.3, "num_steps": 3173.0, "peak_vram_mb": 28254.4, "total_tokens_M": 519.9, "training_seconds": 300.1, "val_bpb": 0.982545}

RECENT RESULT
hypothesis: Using 512-token windows only for the two local layers immediately before full-context layers will achieve `val_bpb < 0.982545` by strengthening high-value aggregation while retaining 128-token windows in the other four local layers.
change: Increase the pre-global attention window from 256 to 512 tokens, leaving all other architecture, optimizer, batching, and schedule settings unchanged.
mechanism: Expanded local-to-global boundary context
evidence_used: The mixed 128/256 design improved `val_bpb` from `0.982806` for uniform 128-token windows to `0.982545`, despite slightly lower throughput, indicating that extra context is especially valuable immediately before global attention; the 512-token uniform design also confirms this window size is efficient and trainable.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.42, "num_params_M": 50.3, "num_steps": 3142.0, "peak_vram_mb": 28254.4, "total_tokens_M": 514.8, "training_seconds": 300.1, "val_bpb": 0.982674}

RECENT RESULT
hypothesis: Retaining a 256-token window only before the first full-context layer will achieve `val_bpb < 0.982545` by preserving the longer-lived early aggregation benefit while recovering some throughput from the second boundary.
change: Use 256-token attention at the first pre-global layer and 128-token attention at all other local layers; retain both full-context layers.
mechanism: Front-loaded local-to-global context allocation
evidence_used: Using 256-token windows at both pre-global boundaries improved `val_bpb` from `0.982806` to `0.982545`, while expanding both to 512 tokens regressed to `0.982674`; isolating the early boundary tests whether the gain is concentrated where contextualized representations propagate through more downstream layers.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.35, "num_params_M": 50.3, "num_steps": 3194.0, "peak_vram_mb": 28254.4, "total_tokens_M": 523.3, "training_seconds": 300.0, "val_bpb": 0.982234}

RECENT RESULT
hypothesis: Retaining 128-token windows in the first half while shortening the three late local layers to 64 tokens will achieve `val_bpb < 0.982234` by preserving early contextualization and reallocating low-value late attention compute into additional training tokens.
change: Keep the first pre-global layer at 256 tokens and the other early local layers at 128 tokens, while reducing all second-half local layers to 64 tokens.
mechanism: Front-loaded hierarchical attention allocation
evidence_used: A 256-token window only before the first full-context layer improved `val_bpb` from `0.982545` to `0.982234` while increasing throughput to 523.3M tokens; removing extra context from the later boundary helped, motivating a broader reduction of late local attention.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 32.79, "num_params_M": 50.3, "num_steps": 2899.0, "peak_vram_mb": 28254.4, "total_tokens_M": 475.0, "training_seconds": 300.1, "val_bpb": 0.986404}

RECENT RESULT
hypothesis: Expanding only the first pre-global attention window from 256 to 512 tokens will achieve `val_bpb < 0.982234` by improving early contextualization without paying for the harmful late expansion.
change: Use a 512-token window immediately before the first full-context layer while retaining 128-token windows for every other local layer.
mechanism: Isolated expanded early local-to-global context
evidence_used: A 256-token window only at the first boundary achieved the best result (`0.982234`), outperforming both-boundary 256 (`0.982545`); therefore the prior both-boundary 512 result (`0.982674`) does not isolate whether additional early context is beneficial.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.49, "num_params_M": 50.3, "num_steps": 3183.0, "peak_vram_mb": 28254.4, "total_tokens_M": 521.5, "training_seconds": 300.0, "val_bpb": 0.982068}

RECENT RESULT
hypothesis: Expanding only the first pre-global attention window from 512 to 1,024 tokens will achieve `val_bpb < 0.982068` by strengthening early contextualization while retaining 128-token windows in the other five local layers.
change: Increase the first pre-global window from one-quarter to one-half of the 2,048-token context; preserve architecture, optimizer, batching, schedules, and all other attention windows.
mechanism: Continued isolated early-boundary context expansion
evidence_used: Increasing only the first pre-global window from 256 to 512 tokens improved `val_bpb` from `0.982234` to `0.982068` despite slightly lower throughput, directly motivating a power-of-two continuation that isolates whether the early-boundary benefit persists.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.64, "num_params_M": 50.3, "num_steps": 3148.0, "peak_vram_mb": 28254.4, "total_tokens_M": 515.8, "training_seconds": 300.0, "val_bpb": 0.983064}

RECENT RESULT
hypothesis: Using a 256-token window in the layer preceding the existing 512-token first pre-global layer will achieve `val_bpb < 0.982068` by improving early contextual aggregation at modest compute cost.
change: Retain the 512-token first pre-global window and 128-token default local windows, but expand the immediately preceding local layer to 256 tokens.
mechanism: Staged early local-to-global context funnel
evidence_used: Expanding only the first pre-global window from 256 to 512 tokens improved `val_bpb` from `0.982234` to `0.982068`, whereas expanding it to 1,024 regressed to `0.983064`; this motivates distributing additional early context into the preceding layer instead of further enlarging the boundary window.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.37, "num_params_M": 50.3, "num_steps": 3161.0, "peak_vram_mb": 28254.4, "total_tokens_M": 517.9, "training_seconds": 300.1, "val_bpb": 0.982585}

RECENT RESULT
hypothesis: Halving only the final layer’s attention window to 1,024 tokens will increase throughput enough to achieve `val_bpb < 0.982068`, while the earlier full-context layer preserves long-range information.
change: Retain the 512-token first pre-global window and all 128-token local windows, but reduce the final attention layer from 2,048-token full context to a power-of-two 1,024-token window.
mechanism: Front-loaded global-context compute reallocation
evidence_used: The best result (`0.982068`) concentrates extra local context before the first global layer, while adding context at the later pre-global boundary worsened `val_bpb` from `0.982234` to `0.982545`; this motivates testing whether late global-attention compute is also less valuable.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.34, "num_params_M": 50.3, "num_steps": 3178.0, "peak_vram_mb": 28254.4, "total_tokens_M": 520.7, "training_seconds": 300.0, "val_bpb": 0.981559}

RECENT RESULT
hypothesis: Halving the final layer’s attention window from 1,024 to 512 tokens will achieve `val_bpb < 0.981559` by reallocating late long-range attention compute while the layer-4 full-context attention preserves global information.
change: Reduce only the final attention layer’s window to one-quarter of the 2,048-token sequence; preserve all other architecture, optimizer, batching, and schedule settings.
mechanism: Progressive late-global context compression
evidence_used: Reducing only the final layer from 2,048-token full attention to 1,024 tokens improved `val_bpb` from `0.982068` to `0.981559`, the strongest result observed, directly motivating one further power-of-two reduction.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 33.64, "num_params_M": 50.3, "num_steps": 3074.0, "peak_vram_mb": 28254.4, "total_tokens_M": 503.6, "training_seconds": 300.1, "val_bpb": 0.985963}



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
