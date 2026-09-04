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
verified_results: {"depth": 8.0, "mfu_percent": 36.49, "num_params_M": 50.3, "num_steps": 3183.0, "peak_vram_mb": 28254.4, "total_tokens_M": 521.5, "training_seconds": 300.0, "val_bpb": 0.982068}
prior_hypothesis: Expanding only the first pre-global attention window from 256 to 512 tokens will achieve `val_bpb < 0.982234` by improving early contextualization without paying for the harmful late expansion.

## Recent verification evidence

RECENT RESULT
hypothesis: At the unchanged 80-sequence batch, scaling all optimizer learning rates to 2/3 of the 96-sequence baseline will achieve `val_bpb < 0.984431` by continuing the improvement observed as the scale decreased from 1.0 to 5/6 to 3/4.
change: Reduce every optimizer learning rate from 3/4 to 2/3 of its 96-sequence baseline while preserving architecture, batching, schedules, and compilation.
mechanism: Fixed-batch learning-rate continuation
evidence_used: At batch size 80, learning-rate scales of 1.0, 5/6, and 3/4 produced `val_bpb` values of `0.986929`, `0.985642`, and `0.984431`; the consistent improvement motivates one further comparable reduction to test and bracket the optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.5, "num_params_M": 50.3, "num_steps": 3004.0, "peak_vram_mb": 28256.4, "total_tokens_M": 492.2, "training_seconds": 300.0, "val_bpb": 0.98478}

RECENT RESULT
hypothesis: At the unchanged 80-sequence batch, scaling all optimizer learning rates to 72.5% of the 96-sequence baseline will achieve `val_bpb < 0.984431`.
change: Reduce embedding, unembedding, matrix, and scalar learning rates from 75% to 72.5% of their 96-sequence baselines while preserving all other settings.
mechanism: Quadratically interpolated fixed-batch learning-rate tuning
evidence_used: At batch size 80, the 75% scale achieved `0.984431`, while 83.3% and 66.7% achieved `0.985642` and `0.984780`; quadratic interpolation of these bracketed results estimates the optimum near 72.7%.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.28, "num_params_M": 50.3, "num_steps": 2987.0, "peak_vram_mb": 28256.4, "total_tokens_M": 489.4, "training_seconds": 300.0, "val_bpb": 0.984916}

RECENT RESULT
hypothesis: At the unchanged 80-sequence batch, scaling all optimizer learning rates to 77.5% of their 96-sequence baselines will achieve `val_bpb < 0.984431`.
change: Increase embedding, unembedding, matrix, and scalar learning rates from 75% to 77.5% of their established baselines while preserving every other setting.
mechanism: Upper-side fixed-batch learning-rate interpolation
evidence_used: The 75% scale is the observed optimum at `0.984431`; fitting the nearby 72.5%, 75%, and 83.3% results places a local minimum near 77%, making 77.5% the most informative untested upper-side interpolation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.06, "num_params_M": 50.3, "num_steps": 2970.0, "peak_vram_mb": 28256.4, "total_tokens_M": 486.6, "training_seconds": 300.0, "val_bpb": 0.985177}

RECENT RESULT
hypothesis: Reducing short-attention windows from 1,024 to 512 tokens while retaining two full-context layers will increase token throughput enough to achieve `val_bpb < 0.984431`.
change: Use quarter-context windows for `S` layers without changing model capacity, batching, optimizer, schedule, or full-context layers.
mechanism: Short-window context-compute reallocation
evidence_used: The best design processed 493.2M tokens, while recent slower designs processing 474.8M–489.4M tokens produced worse `val_bpb`; shortening only the six local-attention layers targets additional training tokens while preserving full-context attention in layers 4 and 8.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.64, "num_params_M": 50.3, "num_steps": 3107.0, "peak_vram_mb": 28254.4, "total_tokens_M": 509.1, "training_seconds": 300.0, "val_bpb": 0.983023}

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
