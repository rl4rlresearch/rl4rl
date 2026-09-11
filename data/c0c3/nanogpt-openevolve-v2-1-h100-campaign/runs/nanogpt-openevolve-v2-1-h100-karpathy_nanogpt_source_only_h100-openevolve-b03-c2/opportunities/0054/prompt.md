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
verified_results: {"depth": 8.0, "mfu_percent": 36.49, "num_params_M": 50.3, "num_steps": 1939.0, "peak_vram_mb": 44908.2, "total_tokens_M": 508.3, "training_seconds": 300.1, "val_bpb": 0.983556}
prior_hypothesis: Using 448-token windows before the intermediate full-context layer and 256-token windows afterward will reduce val_bpb below 0.983618 by interpolating between the near-best 512/256 allocation and the higher-throughput 384/256 allocation.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 36.88, "num_params_M": 50.3, "num_steps": 1949.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.9, "training_seconds": 300.1, "val_bpb": 0.983618}
prior_hypothesis: Using 384-token short-layer attention will reduce `val_bpb` below 0.984156 by preserving more local context than the regressing 256-token design while exposing more training tokens than the best 512-token design.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 36.56, "num_params_M": 50.3, "num_steps": 1953.0, "peak_vram_mb": 44908.2, "total_tokens_M": 512.0, "training_seconds": 300.1, "val_bpb": 0.983738}
prior_hypothesis: Using 384-token local attention before the intermediate full-context layer and 256-token attention afterward will reduce `val_bpb` below 0.983618 by retaining the stronger early-context allocation while increasing token exposure.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 36.71, "num_params_M": 50.3, "num_steps": 1948.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.7, "training_seconds": 300.1, "val_bpb": 0.983549}
prior_hypothesis: Using 464-token windows in the three local layers before layer-4 global attention and 256-token windows afterward will reduce `val_bpb` below 0.983556.

## Recent verification evidence

RECENT RESULT
hypothesis: Using 384-token short-layer attention will reduce `val_bpb` below 0.984156 by preserving more local context than the regressing 256-token design while exposing more training tokens than the best 512-token design.
change: Set the six short attention windows to 3/16 of the 2048-token sequence length, retaining two full-context layers and all verified optimizer schedules.
mechanism: Intermediate local-attention compute allocation
evidence_used: Reducing short attention from 1024 to 512 improved `val_bpb` from 0.985708 to 0.984156, while a further reduction to 256 slightly regressed to 0.984221 despite higher throughput; this brackets a likely context-throughput optimum between 256 and 512 tokens.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.88, "num_params_M": 50.3, "num_steps": 1949.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.9, "training_seconds": 300.1, "val_bpb": 0.983618}

RECENT RESULT
hypothesis: Using 384-token local attention in the first seven layers and retaining full context only in the final layer will reduce `val_bpb` below 0.983618 by increasing token exposure while preserving end-to-end access to the entire sequence.
change: Restore the best verified 35% AdamW cooldown and 384-token local window, then remove the intermediate full-context layer by changing the window pattern to all-short; the existing final-layer override remains full-context.
mechanism: Single global-attention consolidation layer
evidence_used: The 384-token design achieved the best result, `val_bpb` 0.983618 at 510.9M tokens, outperforming both 512-token and 256-token windows. Because seven stacked 384-token causal layers have a receptive field exceeding 2048 tokens and the final layer remains globally attentive, the intermediate global layer may be redundant compute.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 35.16, "num_params_M": 50.3, "num_steps": 1950.0, "peak_vram_mb": 44908.2, "total_tokens_M": 511.2, "training_seconds": 300.1, "val_bpb": 0.985116}

RECENT RESULT
hypothesis: Compute-neutral 256-token early local attention and 512-token late local attention, with full context at layers 4 and 8, will reduce val_bpb below 0.983618 by using early layers for cheap feature extraction and allocating wider context to later refinement.
change: Restore the best shared 35% AdamW schedule and replace uniform short attention with a 256/512-token stagewise split whose average local window remains 384 tokens.
mechanism: Stagewise context reallocation
evidence_used: Uniform 384-token attention achieved the best val_bpb of 0.983618, while uniform 256 and 512 produced similar regressions of 0.984221 and 0.984156; combining those endpoints preserves the winning design’s attention compute while testing whether context is more valuable after the intermediate global layer. Removing that global layer regressed to 0.985116, so both verified full-context layers are retained.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.75, "num_params_M": 50.3, "num_steps": 1942.0, "peak_vram_mb": 44908.2, "total_tokens_M": 509.1, "training_seconds": 300.1, "val_bpb": 0.984213}

RECENT RESULT
hypothesis: Using 512-token local attention before the intermediate full-context layer and 256-token local attention afterward will reduce `val_bpb` below 0.983618 by preserving broader token interactions during early feature construction while using cheaper late refinement.
change: Replace uniform 384-token short attention with a compute-neutral 512/256-token stagewise split, retaining full context at layers 4 and 8 and all optimizer settings.
mechanism: Front-loaded local context allocation
evidence_used: Uniform 384-token attention achieved 0.983618, while the opposite compute-neutral split—256 tokens early and 512 late—regressed to 0.984213; reversing that allocation directly tests whether local context is more valuable before the verified intermediate global-attention layer.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.73, "num_params_M": 50.3, "num_steps": 1941.0, "peak_vram_mb": 44908.2, "total_tokens_M": 508.8, "training_seconds": 300.1, "val_bpb": 0.98362}

RECENT RESULT
hypothesis: Using 512-token local attention before the intermediate full-context layer and 384-token attention afterward will reduce `val_bpb` below 0.983618.
change: Keep full context at layers 4 and 8, widen the first three local layers to 512 tokens, and retain the best uniform design’s 384-token window for the final three local layers.
mechanism: Front-loaded local-context hybrid
evidence_used: The 512/256 front-loaded split reached 0.983620, essentially matching uniform 384, while the reverse 256/512 split regressed to 0.984213. This suggests early context is more valuable, while uniform 512’s 0.984156 argues against widening the late layers too.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.97, "num_params_M": 50.3, "num_steps": 1932.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.5, "training_seconds": 300.1, "val_bpb": 0.98412}

RECENT RESULT
hypothesis: Using 448-token early windows and 320-token late windows will reduce `val_bpb` below 0.983618 by favoring early context without starving late refinement or changing total local-attention compute.
change: Restore an average 384-token local window while reallocating 64 tokens from each late local layer to its early counterpart; retain full context at layers 4 and 8.
mechanism: Compute-neutral moderate front-loading of local context
evidence_used: Uniform 384-token attention achieved 0.983618, and the front-loaded 512/256 split nearly matched it at 0.983620 while the reverse 256/512 split regressed to 0.984213; the midpoint tests the indicated direction with less extreme late-context reduction.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.57, "num_params_M": 50.3, "num_steps": 1932.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.5, "training_seconds": 300.0, "val_bpb": 0.984286}

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from layer 4 to layer 3 will reduce `val_bpb` below 0.983618 by giving four subsequent local layers access to globally integrated representations.
change: Restore two full-context layers while placing them at layers 3 and 8; retain 384-token local attention and all verified optimizer settings.
mechanism: Earlier global-context integration
evidence_used: Removing intermediate global attention regressed from 0.983618 to 0.985116, while early-heavy context allocation nearly matched the best design and late-heavy allocation regressed to 0.984213, motivating earlier global mixing without changing compute.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.32, "num_params_M": 50.3, "num_steps": 1919.0, "peak_vram_mb": 44908.2, "total_tokens_M": 503.1, "training_seconds": 300.1, "val_bpb": 0.98463}

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from layer 4 to layer 5 will reduce `val_bpb` below 0.983618 by allowing one additional 384-token local layer to build features before global mixing while retaining two post-integration refinement layers.
change: Change the attention pattern from full context at layers 4 and 8 to full context at layers 5 and 8, preserving all window sizes and optimizer settings.
mechanism: Delayed intermediate global-context integration
evidence_used: Moving intermediate global attention earlier to layer 3 regressed from 0.983618 to 0.984630, while removing it regressed to 0.985116; this motivates testing the opposite one-layer displacement while retaining the demonstrably necessary intermediate global layer.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.65, "num_params_M": 50.3, "num_steps": 1831.0, "peak_vram_mb": 44908.2, "total_tokens_M": 480.0, "training_seconds": 300.1, "val_bpb": 0.987948}

RECENT RESULT
hypothesis: Using 384-token local attention before the intermediate full-context layer and 256-token attention afterward will reduce `val_bpb` below 0.983618 by retaining the stronger early-context allocation while increasing token exposure.
change: Restore the best design’s 384-token early windows, narrow the three post-integration local layers to 256 tokens, and retain full context at layers 4 and 8.
mechanism: Late-layer context pruning
evidence_used: Uniform 384-token attention achieved 0.983618, while the 512/256 front-loaded split essentially tied it at 0.983620 and outperformed the reverse 256/512 split at 0.984213; this indicates early local context is more valuable and motivates pruning only late-layer context.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.56, "num_params_M": 50.3, "num_steps": 1953.0, "peak_vram_mb": 44908.2, "total_tokens_M": 512.0, "training_seconds": 300.1, "val_bpb": 0.983738}

RECENT RESULT
hypothesis: Using 448-token windows before the intermediate full-context layer and 256-token windows afterward will reduce val_bpb below 0.983618 by interpolating between the near-best 512/256 allocation and the higher-throughput 384/256 allocation.
change: Set the three early local-attention layers to 448 tokens and the three late local-attention layers to 256 tokens, retaining full context at layers 4 and 8 and all optimizer settings.
mechanism: Moderately front-loaded local context with late-layer pruning
evidence_used: The 512/256 design nearly tied the best result at 0.983620, while 384/256 reached 0.983738 with greater token exposure; 448/256 directly tests the midpoint of their early-context tradeoff while preserving the consistently favorable 256-token late windows.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.49, "num_params_M": 50.3, "num_steps": 1939.0, "peak_vram_mb": 44908.2, "total_tokens_M": 508.3, "training_seconds": 300.1, "val_bpb": 0.983556}

RECENT RESULT
hypothesis: Using 464-token windows in the three local layers before layer-4 global attention and 256-token windows afterward will reduce `val_bpb` below 0.983556.
change: Restore full attention at layers 4 and 8, set early local windows to 464 tokens, retain 256-token late windows, and preserve all optimizer settings.
mechanism: Fine-grained early-context interpolation with late-layer pruning
evidence_used: The 448/256 design achieved 0.983556, outperforming both 384/256 at 0.983738 and 512/256 at 0.983620; 464 tokens refines the bracket near its empirically favored interior.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.71, "num_params_M": 50.3, "num_steps": 1948.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.7, "training_seconds": 300.1, "val_bpb": 0.983549}

RECENT RESULT
hypothesis: Increasing the three pre-global local windows from 464 to 480 tokens will reduce `val_bpb` below 0.983549 by moving slightly toward the broader-context regime without reaching the regressing 512-token endpoint.
change: Set early local-attention windows to 480 tokens while retaining 256-token late windows, full attention at layers 4 and 8, and all optimizer settings.
mechanism: Fine-grained upper-side early-context interpolation
evidence_used: The 464/256 design achieved 0.983549, narrowly outperforming 448/256 at 0.983556 and 512/256 at 0.983620; testing 480/256 probes the unexplored upper half of this bracket.
result: the implementation could not be verified



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
