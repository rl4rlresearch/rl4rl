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
verified_results: {"depth": 8.0, "mfu_percent": 36.66, "num_params_M": 50.3, "num_steps": 2331.0, "peak_vram_mb": 36584.2, "total_tokens_M": 496.5, "training_seconds": 300.1, "val_bpb": 0.983937}
prior_hypothesis: A 212,992-token batch with all peak learning rates reduced from 13/14 to 9/10 of their base values will process at least 490M tokens and reduce val_bpb below 0.983699.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 36.93, "num_params_M": 50.3, "num_steps": 2348.0, "peak_vram_mb": 36584.6, "total_tokens_M": 500.1, "training_seconds": 300.1, "val_bpb": 0.984113}
prior_hypothesis: A 212,992-token batch with peak learning rates scaled by 33/35 will process at least 490M tokens and reduce val_bpb below 0.983699.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 36.84, "num_params_M": 50.3, "num_steps": 2342.0, "peak_vram_mb": 36584.2, "total_tokens_M": 498.8, "training_seconds": 300.1, "val_bpb": 0.983699}
prior_hypothesis: A 212,992-token batch with learning rates scaled by 13/14 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving cumulative optimizer displacement while interpolating between the two equally competitive normalized-batch endpoints.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 36.83, "num_params_M": 50.3, "num_steps": 2341.0, "peak_vram_mb": 36584.6, "total_tokens_M": 498.6, "training_seconds": 300.0, "val_bpb": 0.983505}
prior_hypothesis: A 212,992-token batch with peak learning rates scaled by 23/25 will process at least 490M tokens and reduce val_bpb below 0.983699.

## Recent verification evidence

RECENT RESULT
hypothesis: A 196,608-token batch with all peak learning rates scaled by 6/7 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving the best run’s cumulative optimizer displacement while benefiting from more frequent, less correlated updates.
change: Restore four 128-dimensional heads, use a 96-sequence single microbatch, and scale every optimizer-group learning rate by the 196,608/229,376 batch ratio.
mechanism: Token-normalized smaller-batch learning rates
evidence_used: The 229,376-token batch achieved 0.984024, while 196,608 tokens achieved a close 0.985044 and 493.9M tokens using unchanged learning rates but 16% more updates; scaling rates by 6/7 isolates whether excessive cumulative update magnitude caused that regression.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.5, "num_params_M": 50.3, "num_steps": 2513.0, "peak_vram_mb": 33810.6, "total_tokens_M": 494.1, "training_seconds": 300.1, "val_bpb": 0.984034}

RECENT RESULT
hypothesis: A 229,376-token batch with all peak learning rates increased by 2.86% will process at least 490M tokens and reduce val_bpb below 0.984024.
change: Restore the best 112-sequence single-microbatch design and scale every optimizer-group peak learning rate by 36/35.
mechanism: Bracketed peak-learning-rate refinement
evidence_used: The 229,376-token run achieved 0.984024; moving to a higher effective per-token rate at 196,608 tokens regressed to 0.985044, while scaling that run back by 6/7 recovered 0.984034. The opposite 262,144-token direction regressed further to 0.985237, so the asymmetric bracket motivates a small upward refinement near the proven optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.64, "num_params_M": 50.3, "num_steps": 2164.0, "peak_vram_mb": 39360.4, "total_tokens_M": 496.4, "training_seconds": 300.1, "val_bpb": 0.984713}

RECENT RESULT
hypothesis: A 212,992-token batch with learning rates scaled by 13/14 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving cumulative optimizer displacement while interpolating between the two equally competitive normalized-batch endpoints.
change: Use a 104-sequence single microbatch and scale every optimizer-group peak learning rate by the batch ratio 212,992/229,376.
mechanism: Token-normalized optimizer-batch midpoint
evidence_used: The 229,376-token baseline reached 0.984024, while 196,608 tokens with proportional 6/7 learning rates reached 0.984034; their near-identical results bracket this untested midpoint and support keeping learning rate proportional to batch size.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.84, "num_params_M": 50.3, "num_steps": 2342.0, "peak_vram_mb": 36584.2, "total_tokens_M": 498.8, "training_seconds": 300.1, "val_bpb": 0.983699}

RECENT RESULT
hypothesis: A 212,992-token batch with all peak learning rates reduced from 13/14 to 9/10 of their base values will process at least 490M tokens and reduce val_bpb below 0.983699.
change: Restore the best 104-sequence single-microbatch configuration and lower every optimizer-group peak learning rate by 3.08%.
mechanism: Downward peak-learning-rate refinement at the best optimizer batch
evidence_used: The 212,992-token design achieved the best val_bpb, 0.983699. At fixed 196,608- and 229,376-token batches, increasing peak learning rates worsened val_bpb by 0.001010 and 0.000689 respectively, motivating a conservative downward refinement while holding the best batch fixed.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.66, "num_params_M": 50.3, "num_steps": 2331.0, "peak_vram_mb": 36584.2, "total_tokens_M": 496.5, "training_seconds": 300.1, "val_bpb": 0.983937}

RECENT RESULT
hypothesis: A 212,992-token batch with peak learning rates scaled by 33/35 will process at least 490M tokens and reduce val_bpb below 0.983699.
change: Restore the best 104-sequence single-microbatch configuration and raise every optimizer-group peak learning rate from 13/14 to 33/35 of its base value.
mechanism: Upward peak-learning-rate refinement at the best optimizer batch
evidence_used: The 13/14 configuration achieved the best val_bpb of 0.983699, while lowering its learning-rate scale to 9/10 worsened val_bpb to 0.983937; a smaller upward step tests the remaining side of the local optimum without changing architecture, batch, or schedule.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.93, "num_params_M": 50.3, "num_steps": 2348.0, "peak_vram_mb": 36584.6, "total_tokens_M": 500.1, "training_seconds": 300.1, "val_bpb": 0.984113}

RECENT RESULT
hypothesis: A 212,992-token batch with peak learning rates scaled by 23/25 will process at least 490M tokens and reduce val_bpb below 0.983699.
change: Use a 104-sequence single microbatch and scale every optimizer-group peak learning rate to 23/25 of its base value.
mechanism: Quadratically interpolated peak-learning-rate refinement
evidence_used: At the fixed 212,992-token batch, scales of 9/10, 13/14, and 33/35 produced val_bpb of 0.983937, 0.983699, and 0.984113; quadratic interpolation places the estimated optimum near 0.919, motivating the nearby exact scale 23/25.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.83, "num_params_M": 50.3, "num_steps": 2341.0, "peak_vram_mb": 36584.6, "total_tokens_M": 498.6, "training_seconds": 300.0, "val_bpb": 0.983505}

RECENT RESULT
hypothesis: Replacing linear warmdown with an equal-area cosine decay will retain at least 490M training tokens and reduce val_bpb below 0.983505 by sustaining useful learning rates longer before decaying more sharply near the end.
change: Preserve the best 212,992-token batch, 23/25 peak learning rates, warmdown duration, and zero terminal rate while changing only the warmdown curve from linear to cosine.
mechanism: Equal-area cosine warmdown
evidence_used: At the fixed best batch, peak-rate scales of 9/10, 13/14, and 33/35 all underperformed the current 23/25 result of 0.983505, indicating that peak magnitude is tightly bracketed; an equal-area schedule change tests learning-rate allocation without changing peak or cumulative optimizer displacement.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.68, "num_params_M": 50.3, "num_steps": 2332.0, "peak_vram_mb": 36584.6, "total_tokens_M": 496.7, "training_seconds": 300.1, "val_bpb": 0.986677}

RECENT RESULT
hypothesis: On the verified 212,992-token, 23/25-rate baseline, replacing unary squared-ReLU features with parameter-matched multiplicative SwiGLU features will retain at least 480M training tokens and reduce val_bpb below 0.983505.
change: Restore the best verified peak learning rates and replace each dense squared-ReLU MLP with a nearly parameter-matched, 64-aligned SwiGLU MLP whose learned gate conditionally selects prediction features.
mechanism: Parameter-matched SwiGLU content gating
evidence_used: The 23/25-rate design achieved the best val_bpb of 0.983505, while nearby rate refinements, eight-head attention, and cosine warmdown failed to improve it; all retained the load-bearing assumption of an ungated squared-ReLU feature bank, making content-dependent MLP gating an untested computational mechanism.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.89, "num_params_M": 50.3, "num_steps": 2223.0, "peak_vram_mb": 36531.2, "total_tokens_M": 473.5, "training_seconds": 300.1, "val_bpb": 0.988103}

RECENT RESULT
hypothesis: Using two KV heads for four query heads will process at least 500M tokens while retaining sufficient contextual capacity to reduce val_bpb below 0.983505.
change: Restore the best 23/25 learning-rate scale and replace full multi-head K/V projections with two-group GQA, halving K/V and value-embedding width while preserving query heads, model width, batch, and schedule.
mechanism: Two-way grouped-query attention
evidence_used: The 212,992-token 23/25-rate design achieved the best val_bpb, 0.983505, while doubling full-width heads regressed to 0.989288; this motivates testing whether K/V head redundancy can be removed for fixed-time efficiency rather than adding query-head specialization.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 212,992-token batch with all peak learning rates scaled by 11/12 will process at least 490M tokens and reduce val_bpb below 0.983505.
change: Preserve the best architecture, batch, and linear schedule while moving the coupled peak-rate scale from 33/35 to 11/12.
mechanism: Locally interpolated peak learning rate
evidence_used: At fixed batch size, scales 0.90, 0.92, and 0.9286 yielded val_bpb 0.983937, 0.983505, and 0.983699; local quadratic interpolation estimates the minimum near 0.917.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.64, "num_params_M": 50.3, "num_steps": 2330.0, "peak_vram_mb": 36584.6, "total_tokens_M": 496.3, "training_seconds": 300.1, "val_bpb": 0.984321}

RECENT RESULT
hypothesis: Halving short-layer windows to 256 tokens while retaining full attention in layers 4 and 8 will process at least 510M tokens and reduce val_bpb below 0.983505.
change: Change the six short-attention layers from quarter-context to eighth-context windows without altering architecture, batch size, learning rates, or the two full-context layers.
mechanism: Denser global anchors with narrower local attention
evidence_used: The 212,992-token, 23/25-rate baseline reached 0.983505 after 498.6M tokens, while added attention specialization via eight heads regressed to 0.989288; narrower local windows test attention efficiency while preserving periodic global mixing.
result: the implementation could not be verified

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
