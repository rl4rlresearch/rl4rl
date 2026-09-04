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
verified_results: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.98278}
prior_hypothesis: Restoring the verified-best 5× MLP while compensating its Muon learning rate for aspect-ratio scaling will retain at least 445M training tokens and reduce val_bpb below 0.982905.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 38.72, "num_params_M": 54.5, "num_steps": 1710.0, "peak_vram_mb": 49047.2, "total_tokens_M": 448.3, "training_seconds": 300.1, "val_bpb": 0.983844}
prior_hypothesis: Restoring the verified-best 5× MLP while removing the unsuccessful bigram expert and making the intermediate full-attention layer local will retain final-layer global context, raise throughput above 460M tokens, and reduce val_bpb below 0.982905.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 39.88, "num_params_M": 55.6, "num_steps": 1680.0, "peak_vram_mb": 50080.0, "total_tokens_M": 440.4, "training_seconds": 300.1, "val_bpb": 0.983115}
prior_hypothesis: A 5.25× MLP will retain at least 440M-token throughput and reduce val_bpb below 0.982905.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 39.98, "num_params_M": 54.5, "num_steps": 1724.0, "peak_vram_mb": 49047.2, "total_tokens_M": 451.9, "training_seconds": 300.1, "val_bpb": 0.982905}
prior_hypothesis: A 5× MLP at the proven 93.75% learning rates will retain at least 445M-token throughput and reduce val_bpb below 0.984614.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing the linear warmdown with an equal-area cosine warmdown at the proven 93.75% peak learning rates will retain at least 460M-token throughput and reduce val_bpb below 0.984614.
change: Keep the architecture, batch size, peak learning rates, warmdown duration, endpoints, and cumulative learning-rate exposure fixed while changing only the warmdown curvature.
mechanism: Equal-area cosine terminal annealing
evidence_used: The 262,144-token runs bracket the best peak scale: 93.75% achieved 0.984614, while 87.5%, 100%, and 81.25% were worse. Holding that peak fixed and testing an equal-area schedule shape is therefore a controlled next optimization dimension.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 38.61, "num_params_M": 51.9, "num_steps": 1770.0, "peak_vram_mb": 46458.8, "total_tokens_M": 464.0, "training_seconds": 300.1, "val_bpb": 0.988064}

RECENT RESULT
hypothesis: A zero-initialized rank-64 bigram expert at the proven 93.75% learning rates will retain at least 450M-token throughput and reduce val_bpb below 0.984614 by learning frequent token transitions outside the deep contextual path.
change: Challenge the assumption that every prediction must be decoded solely from the final transformer state; add an exact, collision-free factorized bigram distribution directly to the logits while retaining the transformer for longer-context corrections.
mechanism: Residual low-rank bigram logit expert
evidence_used: The 93.75% learning-rate design is best at 0.984614. The hashed-bigram design reached only 0.994892 despite 466.6M tokens, indicating that indirect, collision-prone residual injection was ineffective; a zero-initialized direct logit expert tests the local-statistics idea without requiring eight layers to preserve and decode the added representation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.52, "num_params_M": 53.0, "num_steps": 1744.0, "peak_vram_mb": 50625.8, "total_tokens_M": 457.2, "training_seconds": 300.1, "val_bpb": 0.985446}

RECENT RESULT
hypothesis: A zero-initialized output bias at the proven 93.75% learning rates will retain at least 460M-token throughput and reduce val_bpb below 0.984614 by learning global next-token frequencies without the rank-64 bigram expert’s compute and memory overhead.
change: Restore the best 93.75% optimizer rates and add a zero-initialized bias to the language-model head.
mechanism: Near-free unigram logit prior
evidence_used: The rank-64 direct-logit bigram expert lost 9.4M tokens and worsened val_bpb from 0.984614 to 0.985446; a fused output bias tests whether a minimal direct statistical expert provides the useful prior without that throughput penalty.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.72, "num_params_M": 51.9, "num_steps": 1775.0, "peak_vram_mb": 46708.3, "total_tokens_M": 465.3, "training_seconds": 300.1, "val_bpb": 0.985063}

RECENT RESULT
hypothesis: A 5× MLP at the proven 93.75% learning rates will retain at least 445M-token throughput and reduce val_bpb below 0.984614.
change: Restore the best verified optimizer rates and increase every MLP hidden width from 4.375× to 5×, yielding a 2560-channel hidden dimension aligned to 128-channel tensor-core tiles.
mechanism: Tensor-core-aligned MLP capacity expansion
evidence_used: The 4.375× MLP achieved 0.994296 versus 0.996902 for 3.75× despite processing 25.7M fewer tokens, showing that additional MLP capacity outweighed throughput; 93.75% learning rates independently produced the best verified baseline at 0.984614.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.98, "num_params_M": 54.5, "num_steps": 1724.0, "peak_vram_mb": 49047.2, "total_tokens_M": 451.9, "training_seconds": 300.1, "val_bpb": 0.982905}

RECENT RESULT
hypothesis: A 5.625× MLP will retain at least 435M-token throughput and reduce val_bpb below 0.982905.
change: Increase every MLP hidden width from 5× to 5.625× while retaining the proven batch size, architecture, and 93.75% learning rates.
mechanism: Continued MLP capacity scaling
evidence_used: Increasing MLP width from 4.375× to 5× improved val_bpb from 0.984614 to 0.982905 despite reducing token exposure from 466.6M to 451.9M, so another equal 0.625× capacity increment directly tests whether the favorable capacity-throughput tradeoff continues.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.75, "num_params_M": 57.1, "num_steps": 1618.0, "peak_vram_mb": 51631.9, "total_tokens_M": 424.1, "training_seconds": 300.0, "val_bpb": 0.983497}

RECENT RESULT
hypothesis: A 5.25× MLP will retain at least 440M-token throughput and reduce val_bpb below 0.982905.
change: Increase every MLP hidden width from 4.375× to 5.25× while retaining the proven 262,144-token batch, eight-layer architecture, linear warmdown, and 93.75% learning rates.
mechanism: Tensor-core-aligned MLP width interpolation
evidence_used: Increasing MLP width from 4.375× to 5× improved val_bpb from 0.984614 to 0.982905 despite lower token exposure, while 5.625× regressed to 0.983497 after throughput fell to 424.1M tokens; the 5.25× width is a conservative interpolation and remains aligned to 128-channel tensor-core tiles.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.88, "num_params_M": 55.6, "num_steps": 1680.0, "peak_vram_mb": 50080.0, "total_tokens_M": 440.4, "training_seconds": 300.1, "val_bpb": 0.983115}

RECENT RESULT
hypothesis: A 5.125× MLP at the proven 93.75% learning rates will retain at least 445M-token throughput and reduce val_bpb below 0.982905.
change: Remove the unsuccessful output bias and widen every MLP from 4.375× to 5.125× while retaining the proven batch, schedule, and optimizer rates.
mechanism: Fine-grained MLP capacity interpolation
evidence_used: The unbiased 5× MLP achieved 0.982905, while 5.25× reached 0.983115 and 5.625× reached 0.983497; testing the midpoint between 5× and 5.25× refines the bracket around the observed capacity optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.71, "num_params_M": 55.1, "num_steps": 1692.0, "peak_vram_mb": 49564.2, "total_tokens_M": 443.5, "training_seconds": 300.0, "val_bpb": 0.983318}

RECENT RESULT
hypothesis: Restoring the verified-best 5× MLP while removing the unsuccessful bigram expert and making the intermediate full-attention layer local will retain final-layer global context, raise throughput above 460M tokens, and reduce val_bpb below 0.982905.
change: Remove the rank-64 bigram path, widen all MLPs to 5×, and change the attention pattern from SSSL to SSSS; the existing window logic still forces the final layer to use full context.
mechanism: Final-layer-only global attention reallocation
evidence_used: The unbiased 5× MLP achieved the best observed val_bpb of 0.982905 despite processing 14.7M fewer tokens than the 4.375× baseline, while the bigram expert regressed to 0.985446 and cost 9.4M tokens; reclaiming one intermediate full-attention layer tests whether extra token exposure can improve the proven 5× design without eliminating global attention.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 38.72, "num_params_M": 54.5, "num_steps": 1710.0, "peak_vram_mb": 49047.2, "total_tokens_M": 448.3, "training_seconds": 300.1, "val_bpb": 0.983844}

RECENT RESULT
hypothesis: Adding one full-attention layer between the proven global layers will retain at least 440M training tokens and reduce val_bpb below 0.982905.
change: Keep the verified-best 5× MLP, optimizer, batch, and schedule unchanged while changing the eight-layer attention pattern from global layers 3 and 7 to global layers 3, 5, and 7.
mechanism: Additional late-intermediate global-context refresh
evidence_used: Removing the intermediate full-attention layer regressed val_bpb from 0.982905 to 0.983844 without improving measured throughput, showing that intermediate global-context mixing is valuable and motivating a controlled addition of one later global refresh.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 40.28, "num_params_M": 54.5, "num_steps": 1696.0, "peak_vram_mb": 49047.2, "total_tokens_M": 444.6, "training_seconds": 300.0, "val_bpb": 0.985077}

RECENT RESULT
hypothesis: A 4.875× MLP will process at least 455M tokens and reduce val_bpb below 0.982905 by preserving most of the 5× model’s capacity while increasing token exposure.
change: Reduce every MLP hidden width from 5.25× to 4.875× (2496 channels), retaining the proven architecture, batch, schedule, and 93.75% learning rates.
mechanism: Below-optimum MLP width interpolation
evidence_used: The 5× MLP achieved the best val_bpb of 0.982905; 5.125×, 5.25×, and 5.625× all regressed, while 4.375× was also worse. Testing 4.875× completes the nearest symmetric bracket around the observed 5× optimum.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.49, "num_params_M": 54.0, "num_steps": 1723.0, "peak_vram_mb": 48529.5, "total_tokens_M": 451.7, "training_seconds": 300.1, "val_bpb": 0.98374}

RECENT RESULT
hypothesis: Restoring the verified-best 5× MLP while compensating its Muon learning rate for aspect-ratio scaling will retain at least 445M training tokens and reduce val_bpb below 0.982905.
change: Restore the 2560-channel MLP and scale only its Muon parameter groups by sqrt(4.375/5), preserving the effective matrix update magnitude previously tuned at 4.375× while leaving attention and gate rates unchanged.
mechanism: Aspect-compensated MLP Muon rate
evidence_used: The 5× MLP achieved the best val_bpb, 0.982905, while 4.875×, 5.125×, 5.25×, and 5.625× were worse. The optimizer additionally multiplies Muon rates by the square root of matrix aspect ratio, so widening the MLP silently raised its effective rate despite the 93.75% rate having been selected on the 4.375× design.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 39.83, "num_params_M": 54.5, "num_steps": 1717.0, "peak_vram_mb": 49047.2, "total_tokens_M": 450.1, "training_seconds": 300.0, "val_bpb": 0.98278}

RECENT RESULT
hypothesis: On the restored 0.982780 SSSL/aspect-compensated baseline, making one of four heads in each full-context layer positionless will retain at least 445M training tokens and reduce val_bpb below 0.982780 by enabling content-addressed long-range retrieval without distance-dependent rotary phase, while the other three heads preserve ordered context.
change: Restore the verified-best SSSL attention pattern and compensated 5×-MLP Muon rate, then omit RoPE from the final head only in the two full-context layers.
mechanism: Positionless long-range retrieval head
evidence_used: The compensated SSSL design achieved the best result, 0.982780. Removing its intermediate global layer regressed to 0.983844, while adding another global layer regressed to 0.985077, suggesting the informative alternative is improving how the proven two global layers retrieve context rather than changing their count.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 39.24, "num_params_M": 54.5, "num_steps": 1692.0, "peak_vram_mb": 49046.7, "total_tokens_M": 443.5, "training_seconds": 300.1, "val_bpb": 0.984526}



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
