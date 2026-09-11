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
verified_results: {"depth": 8.0, "mfu_percent": 37.61, "num_params_M": 50.3, "num_steps": 977.0, "peak_vram_mb": 45060.2, "total_tokens_M": 512.2, "training_seconds": 300.0, "val_bpb": 0.993365}
prior_hypothesis: Reducing short attention windows from 1024 to 512 tokens will increase training throughput enough to lower val_bpb below 0.995558, while the full-context layers at indices 3 and 7 preserve long-range information.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing short attention windows from 1024 to 512 tokens will increase training throughput enough to lower val_bpb below 0.995558, while the full-context layers at indices 3 and 7 preserve long-range information.
change: Use quarter-context windows for “S” layers while retaining the existing SSSL pattern and full-context final layer.
mechanism: Denser periodic global attention with cheaper local layers
evidence_used: The baseline reaches val_bpb 0.995558 after 497.0M tokens at 39.58% MFU; attention remains a substantial compute cost, and six of eight layers currently use relatively expensive half-context windows.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.61, "num_params_M": 50.3, "num_steps": 977.0, "peak_vram_mb": 45060.2, "total_tokens_M": 512.2, "training_seconds": 300.0, "val_bpb": 0.993365}

RECENT RESULT
hypothesis: Reducing short-layer windows from 512 to 256 tokens will increase token throughput enough to lower val_bpb below 0.993365, while full-context layers 3 and 7 preserve long-range information.
change: Use 256-token windows for “S” layers while retaining the existing SSSL pattern and full-context final layer.
mechanism: Eighth-context local attention with periodic global mixing
evidence_used: Cutting short windows from 1024 to 512 improved val_bpb from 0.995558 to 0.993365 and increased training tokens from 497.0M to 512.2M, motivating the next controlled window-size ablation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 27.15, "num_params_M": 50.3, "num_steps": 740.0, "peak_vram_mb": 45060.2, "total_tokens_M": 388.0, "training_seconds": 300.2, "val_bpb": 1.013136}

RECENT RESULT
hypothesis: Keeping the kernel-efficient 512-token local windows while removing the intermediate full-context layer will raise token throughput enough to reduce val_bpb below 0.993365; seven stacked local layers can propagate information across the 2048-token sequence before the final global layer.
change: Change the attention pattern from six local/two global layers to seven local/one final global layer.
mechanism: Stacked local receptive field with a single global readout
evidence_used: The 512-token design improved val_bpb to 0.993365 with 512.2M tokens, whereas shrinking windows to 256 caused a throughput cliff and val_bpb 1.013136; this motivates retaining 512-token windows and testing a less costly reduction in global-attention frequency.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.3, "num_params_M": 50.3, "num_steps": 986.0, "peak_vram_mb": 45060.2, "total_tokens_M": 516.9, "training_seconds": 300.3, "val_bpb": 0.994122}

RECENT RESULT
hypothesis: Halving the KV-head count while retaining four query heads and the successful SSSL 512-token pattern will reduce projection, value-embedding, and attention bandwidth enough to beat val_bpb 0.993365 within five minutes without materially reducing attention quality.
change: Use two KV heads for the four query heads, leaving model width, depth, batching, and attention windows unchanged.
mechanism: Two-to-one grouped-query attention
evidence_used: The one-global-layer design processed more tokens but regressed to 0.994122, showing that removing global mixing sacrifices useful capacity; grouped-query attention instead preserves the winning two-global-layer topology while targeting redundant KV computation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 29.05, "num_params_M": 39.8, "num_steps": 803.0, "peak_vram_mb": 41364.2, "total_tokens_M": 421.0, "training_seconds": 300.3, "val_bpb": 1.012022}

RECENT RESULT
hypothesis: Adding a third full-context layer at index 5 will lower val_bpb below 0.993365; the added global mixing should outweigh its small throughput cost.
change: Change the repeating attention pattern from SSSL to SSL, producing full-context layers at indices 2, 5, and 7 while retaining 512-token local windows elsewhere.
mechanism: Denser periodic global attention
evidence_used: Removing one global layer increased tokens only from 512.2M to 516.9M but worsened val_bpb from 0.993365 to 0.994122, indicating global mixing contributes more than its modest compute cost.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.7, "num_params_M": 50.3, "num_steps": 965.0, "peak_vram_mb": 45060.2, "total_tokens_M": 505.9, "training_seconds": 300.3, "val_bpb": 0.994554}

RECENT RESULT
hypothesis: Moving the intermediate full-context layer from index 3 to index 4 while retaining six 512-token local layers and the final global layer will reduce val_bpb below 0.993365 without changing throughput materially.
change: Use an explicit eight-layer SSSSLSSS pattern, placing full-context attention at layers 4 and 7.
mechanism: Later intermediate global mixing at constant compute
evidence_used: Two global layers achieved the best result (0.993365); one global layer regressed to 0.994122 and three regressed to 0.994554, motivating a controlled placement ablation with the winning attention counts and window sizes unchanged.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 31.58, "num_params_M": 50.3, "num_steps": 823.0, "peak_vram_mb": 45060.2, "total_tokens_M": 431.5, "training_seconds": 300.4, "val_bpb": 1.005964}

RECENT RESULT
hypothesis: Replacing the intermediate 2048-token global layer with a 1024-token window will retain enough cross-window mixing to beat 0.993365 val_bpb while recovering some of the throughput gained by removing that global layer entirely.
change: Add a medium-window attention type and use SSSM repetition, yielding 512-token local layers, a 1024-token layer at index 3, and mandatory full-context attention at the final layer.
mechanism: Hierarchical local, mid-range, and global attention
evidence_used: The two-global-layer design achieved 0.993365, while replacing its intermediate global layer with 512-token attention increased tokens from 512.2M to 516.9M but regressed to 0.994122; a 1024-token intermediate layer directly tests the quality-throughput midpoint.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 27.4, "num_params_M": 50.3, "num_steps": 736.0, "peak_vram_mb": 45060.2, "total_tokens_M": 385.9, "training_seconds": 300.3, "val_bpb": 1.012997}

RECENT RESULT
hypothesis: Representing the two full-context layers with FlashAttention’s native unrestricted window will preserve the winning SSSL receptive field while improving throughput enough to reduce val_bpb below 0.993365.
change: Keep 512-token local windows, but encode full-context attention as `(-1, -1)` instead of the equivalent `(2048, 0)` sliding window.
mechanism: Native full-attention kernel path
evidence_used: The best design uses full attention at layers 3 and 7; removing one gained only 4.7M tokens and worsened val_bpb to 0.994122, motivating a numerical optimization of full attention rather than reducing global mixing.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 27.44, "num_params_M": 50.3, "num_steps": 716.0, "peak_vram_mb": 45060.2, "total_tokens_M": 375.4, "training_seconds": 300.2, "val_bpb": 1.014553}



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
