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
verified_results: {"depth": 8.0, "mfu_percent": 36.76, "num_params_M": 50.3, "num_steps": 1946.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.1, "training_seconds": 300.1, "val_bpb": 0.983497}
prior_hypothesis: Using 424-token windows before the depth-4 global layer and 320-token windows afterward will reduce val_bpb below 0.983635 by preserving important early context while increasing training throughput.

## Recent verification evidence

RECENT RESULT
hypothesis: A 432-token short window will reduce val_bpb below 0.983765 by preserving more local context than 416 tokens without incurring the full throughput loss observed at 448 tokens.
change: Increase the six short-attention layers from 416-token to 432-token windows while preserving both full-context layers and all other settings.
mechanism: Upper-bracket short-attention window refinement
evidence_used: The 416-token window achieved the best val_bpb of 0.983765, while 400 tokens regressed to 0.983933 and 448 tokens regressed to 0.984134; 432 tokens bisects the remaining unexplored upper bracket.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.85, "num_params_M": 50.3, "num_steps": 1931.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.2, "training_seconds": 300.1, "val_bpb": 0.983917}

RECENT RESULT
hypothesis: Using 320-token windows in the first three local layers and 416-token windows in the later three will reduce val_bpb below 0.983765 by recovering some throughput while preserving the best-performing context size near the output.
change: Make early short-attention layers use 320 tokens while retaining 416 tokens for later short-attention layers and full context for both existing global layers.
mechanism: Depth-dependent local-attention windows
evidence_used: Uniform 320-token attention processed 509.1M tokens but reached 0.983911, while uniform 416-token attention achieved the best 0.983765 with 506.5M tokens; this hybrid tests whether the 416-token quality benefit is concentrated in later layers.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.44, "num_params_M": 50.3, "num_steps": 1931.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.2, "training_seconds": 300.1, "val_bpb": 0.984264}

RECENT RESULT
hypothesis: A 408-token short window will reduce val_bpb below 0.983765 by retaining slightly more throughput than 416 tokens while preserving more local context than the worse 400-token design.
change: Change the six short-attention layers from 416-token to 408-token windows, preserving both full-context layers and all other settings.
mechanism: Lower-bracket short-attention window refinement
evidence_used: Uniform 416-token windows achieved the best val_bpb of 0.983765, while 400 and 432 tokens regressed to 0.983933 and 0.983917; 408 tokens bisects the unexplored lower bracket closest to the current optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.57, "num_params_M": 50.3, "num_steps": 1924.0, "peak_vram_mb": 44908.2, "total_tokens_M": 504.4, "training_seconds": 300.0, "val_bpb": 0.98424}

RECENT RESULT
hypothesis: A 424-token short window will reduce val_bpb below 0.983765 by preserving slightly more context than 416 tokens without incurring the regression observed at 432 tokens.
change: Increase all six short-attention windows from 416 to 424 tokens while preserving both full-context layers and every other setting.
mechanism: Upper-bracket local-attention window refinement
evidence_used: Uniform 416-token windows achieved the best val_bpb of 0.983765, while 432-token windows regressed to 0.983917; 424 tokens bisects the remaining unexplored upper bracket.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.79, "num_params_M": 50.3, "num_steps": 1931.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.2, "training_seconds": 300.1, "val_bpb": 0.983635}

RECENT RESULT
hypothesis: A 420-token short window will reduce val_bpb below 0.983635 by retaining slightly more throughput than 424 tokens while preserving more context than the worse 416-token design.
change: Reduce all six short-attention windows from 424 to 420 tokens while preserving both full-context layers and every other setting.
mechanism: Fine-grained local-attention window refinement
evidence_used: The best result moved from 0.983765 at 416 tokens to 0.983635 at 424 tokens, while 432 tokens regressed to 0.983917; 420 tokens bisects the remaining lower-side bracket around the current optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.5, "num_params_M": 50.3, "num_steps": 1917.0, "peak_vram_mb": 44908.2, "total_tokens_M": 502.5, "training_seconds": 300.1, "val_bpb": 0.984478}

RECENT RESULT
hypothesis: A 428-token short window will reduce val_bpb below 0.983635 by preserving slightly more context than the best 424-token design without reaching the regression observed at 432 tokens.
change: Increase all six short-attention windows from 424 to 428 tokens while preserving both full-context layers and every other setting.
mechanism: Fine-grained upper-bracket local-attention refinement
evidence_used: The 424-token window achieved the best val_bpb of 0.983635, while 432 tokens regressed to 0.983917; 428 tokens bisects the remaining unexplored upper-side bracket.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.21, "num_params_M": 50.3, "num_steps": 1899.0, "peak_vram_mb": 44908.2, "total_tokens_M": 497.8, "training_seconds": 300.1, "val_bpb": 0.984781}

RECENT RESULT
hypothesis: Three full-context layers at depths 3, 5, and 7 will reduce val_bpb below 0.983635 because the large regression from removing the intermediate global layer indicates that global mixing is worth more than its throughput cost.
change: Preserve the 424-token local window while changing layer 5 from local to full-context attention, producing five local and three global layers.
mechanism: Add a late intermediate global-attention layer
evidence_used: Replacing the intermediate full-context layer with local attention increased throughput from 506.2M to 513.3M tokens but worsened val_bpb from 0.983635 to 0.984677, showing that additional global mixing can dominate the value of extra training tokens.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.01, "num_params_M": 50.3, "num_steps": 1907.0, "peak_vram_mb": 44908.2, "total_tokens_M": 499.9, "training_seconds": 300.0, "val_bpb": 0.984553}

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from depth 4 to depth 5 will reduce val_bpb below 0.983635 by applying global mixing to more mature local features while retaining the empirically favored total of two global layers.
change: Change the eight-layer attention pattern from SSS-L-SSS-L to SSSS-L-SS-L, preserving the 424-token local window and all other settings.
mechanism: Shift intermediate global attention deeper
evidence_used: Removing the intermediate global layer regressed val_bpb to 0.984677, while adding a third global layer regressed it to 0.984553; this indicates that two global layers are preferable and motivates testing their placement instead of their count.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.62, "num_params_M": 50.3, "num_steps": 1922.0, "peak_vram_mb": 44908.2, "total_tokens_M": 503.8, "training_seconds": 300.1, "val_bpb": 0.985418}

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from depth 4 to depth 3 will reduce val_bpb below 0.983635 by enabling global information exchange earlier, since moving it later to depth 5 sharply regressed validation quality.
change: Change the eight-layer attention pattern from SSS-L-SSS-L to SS-L-SSSS-L while preserving the 424-token local window and all other settings.
mechanism: Earlier global feature mixing
evidence_used: The best design uses two global layers at depths 4 and 8 with val_bpb 0.983635, whereas shifting the intermediate global layer to depth 5 worsened val_bpb to 0.985418; testing the opposite direction is the most informative remaining placement change.
result: the implementation could not be verified

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from depth 4 to depth 3 will reduce val_bpb below 0.983635 by enabling global information exchange earlier.
change: Change the attention pattern from SSS-L-SSS-L to SS-L-SSSS-L while retaining 424-token local windows and all other settings.
mechanism: Earlier global feature mixing
evidence_used: Shifting the intermediate global layer later to depth 5 worsened val_bpb from 0.983635 to 0.985418; the earlier-placement attempt produced no verifiable implementation, so its quality remains untested.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.8, "num_params_M": 50.3, "num_steps": 1931.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.2, "training_seconds": 300.0, "val_bpb": 0.98409}

RECENT RESULT
hypothesis: Using 424-token windows before the depth-4 global layer and 320-token windows afterward will reduce val_bpb below 0.983635 by preserving important early context while increasing training throughput.
change: Keep the first three local layers at 424 tokens, shorten the three local layers after the intermediate full-context layer to 320 tokens, and retain full context at layers 4 and 8.
mechanism: Post-global attention-window compression
evidence_used: Uniform 320-token windows reached 0.983911, but shortening only the early layers while keeping later layers at 416 worsened val_bpb to 0.984264; this suggests local context is more valuable before global mixing and motivates testing the complementary allocation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.76, "num_params_M": 50.3, "num_steps": 1946.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.1, "training_seconds": 300.1, "val_bpb": 0.983497}



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
