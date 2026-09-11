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
verified_results: {"depth": 8.0, "mfu_percent": 36.79, "num_params_M": 50.3, "num_steps": 1931.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.2, "training_seconds": 300.1, "val_bpb": 0.983635}
prior_hypothesis: A 424-token short window will reduce val_bpb below 0.983765 by preserving slightly more context than 416 tokens without incurring the regression observed at 432 tokens.

## Recent verification evidence

RECENT RESULT
hypothesis: A 448-token short window will reduce val_bpb below 0.983803 by preserving more local context than 384 tokens while retaining most of its throughput advantage over the worse 512-token design.
change: Increase the six short-attention layers from 384-token to 448-token windows while preserving the two full-context layers and all optimizer and schedule settings.
mechanism: Bracketed short-attention window refinement
evidence_used: Increasing the short window from 320 to 384 tokens improved val_bpb from 0.983911 to 0.983803, while 512 tokens was worse at 0.984293; 448 tokens tests the midpoint of the remaining bracket.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.45, "num_params_M": 50.3, "num_steps": 1905.0, "peak_vram_mb": 44908.2, "total_tokens_M": 499.4, "training_seconds": 300.0, "val_bpb": 0.984134}

RECENT RESULT
hypothesis: Using seven 384-token local-attention layers and only the forced final full-context layer will lower val_bpb below 0.983803 by increasing training throughput while stacked local layers retain a sequence-wide receptive field.
change: Change the attention pattern from six short and two full-context layers to seven short layers followed by the existing forced full-context final layer.
mechanism: Replace redundant intermediate global attention with local attention
evidence_used: Increasing the short window from 384 to 448 tokens reduced training from 508.8M to 499.4M tokens and regressed val_bpb from 0.983803 to 0.984134, while windows below 384 also regressed. This motivates preserving the best 384-token window and seeking throughput by removing one expensive full-context layer instead.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.3, "num_params_M": 50.3, "num_steps": 1958.0, "peak_vram_mb": 44908.2, "total_tokens_M": 513.3, "training_seconds": 300.1, "val_bpb": 0.984677}

RECENT RESULT
hypothesis: A 416-token short window will reduce val_bpb below 0.983803 by preserving slightly more local context than 384 tokens without incurring the larger throughput loss observed at 448 tokens.
change: Increase the six short-attention layers from 384-token to 416-token windows while preserving both full-context layers and all other settings.
mechanism: Bracketed short-attention window refinement
evidence_used: The 384-token window improved val_bpb to 0.983803, while 320 tokens was slightly worse at 0.983911 and 448 tokens regressed to 0.984134; 416 tokens bisects the remaining upper-side bracket.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.76, "num_params_M": 50.3, "num_steps": 1932.0, "peak_vram_mb": 44908.2, "total_tokens_M": 506.5, "training_seconds": 300.1, "val_bpb": 0.983765}

RECENT RESULT
hypothesis: A 400-token short window will reduce val_bpb below 0.983765 by retaining more throughput than 416 tokens while preserving more local context than the slightly worse 384-token design.
change: Change the six short-attention layers from 416-token to 400-token windows, preserving both full-context layers and all other settings.
mechanism: Local-attention window interpolation
evidence_used: The 416-token window achieved 0.983765, improving on 384 tokens at 0.983803, while 448 tokens regressed to 0.984134; 400 tokens refines the lower half of the remaining bracket around the current optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.7, "num_params_M": 50.3, "num_steps": 1934.0, "peak_vram_mb": 44908.2, "total_tokens_M": 507.0, "training_seconds": 300.1, "val_bpb": 0.983933}

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
