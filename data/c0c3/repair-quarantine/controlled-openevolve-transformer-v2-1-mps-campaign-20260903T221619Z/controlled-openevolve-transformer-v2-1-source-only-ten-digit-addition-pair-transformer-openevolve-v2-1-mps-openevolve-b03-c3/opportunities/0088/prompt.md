# Optimize a transformer for 10-digit addition

You are an autonomous ML engineer improving the source code for an
autoregressive transformer that adds two 10-digit numbers.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% accuracy under the fixed verification process. A
smaller implementation is useful only when it meets that accuracy requirement.
Every submitted implementation is trained from a fresh initialization.

## Learned-model requirement

Produce a smaller trained autoregressive transformer, not a hand-coded addition
program. The submitted implementation must:

- have nonzero trainable parameters;
- contain and use at least one learned causal self-attention module;
- map token inputs to token logits through the learned model;
- train from a fresh initialization during verification;
- write both `checkpoints/best.pt` and a positive-step `checkpoints/last.pt`;
- keep source code unchanged while training; and
- use the protected generic decoding interface exactly as supplied.

Do not implement or embed decimal arithmetic, carry propagation, place-value
rules, digit lookup tables, finite-state addition transitions, fixed answer
rules, or input-dependent Python logic that directly computes the sum. Do not
hide such a solver in model generation, token processing, training, or saved
weights. Do not add dummy or zero-length parameters to disguise a fixed
algorithm as a learned model.

Do not modify protected files. Do not perform post-training state-dictionary
surgery, substitute a different saved model, truncate weights after training,
or report a parameter count that differs from the submitted model.

## Work boundaries

Minimize parameters. Required result: accuracy >= 0.99.
Editable source files: src/model.py, src/train.py.
Results reported after each verification: accuracy, parameters, training_steps.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or verification yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.995, "parameters": 1386, "training_steps": 4999}
prior_hypothesis: A 1,386-parameter model will retain at least 99% accuracy because the verified 1,388-parameter relative-lag design reached 99.22%, while removing one additive constant from each head’s lag logits is an exact softmax symmetry with ambient AdamW dynamics preserved.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9945, "parameters": 1382, "training_steps": 4999}
prior_hypothesis: A 1,382-parameter relative-lag model will retain at least 99% accuracy because the verified 1,383-parameter design achieved 99.74%, while removing the common output component of a tenth `fc2` column uses the identical pre-final-LayerNorm symmetry already verified for nine columns.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9971, "parameters": 1380, "training_steps": 4999}
prior_hypothesis: A 1,380-parameter relative-lag model will retain at least 99% accuracy because the verified 1,381-parameter design achieved 99.75%, while gauge-fixing the twelfth and final `fc2` column applies the same exact pre-final-LayerNorm symmetry already verified for eleven columns.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9975, "parameters": 1381, "training_steps": 4999}
prior_hypothesis: A 1,381-parameter relative-lag model will retain at least 99% accuracy because the verified 1,382-parameter ten-column design achieved 99.45%, while gauge-fixing an eleventh `fc2` column applies the same exact pre-final-LayerNorm output-shift symmetry.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1,521-parameter model will retain at least 99% accuracy because the verified 1,522-parameter design achieved 99.96%, while a fifth positional row has the same exact pre-LayerNorm scalar-shift symmetry as the four successfully removed rows.
change: Reproduce the qualified two-scale ln1 absorption, complete ln2 absorption, and four positional gauges, then gauge-fix a fifth positional row while preserving every omitted coordinate through initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Fifth positional-row scalar-shift gauge with complete LayerNorm scale absorption
evidence_used: The 1,522-parameter four-position-gauge design reached 99.96%; extending this repeatedly successful exact symmetry is better supported than a third ln1 absorption, which reached only 98.85%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1521, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,520-parameter model will retain at least 99% accuracy because the verified 1,521-parameter design achieved 99.88%, while a sixth positional row has the same exact pre-LayerNorm scalar-shift symmetry as the five successfully removed rows.
change: Gauge-fix the sixth positional embedding row and preserve its omitted ambient coordinate through initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Sixth positional-row scalar-shift gauge with ambient AdamW
evidence_used: The current design removed five positional-row scalars and reached 99.88% at 1,521 parameters; all four preceding one-row extensions also exceeded 99%, making the same controlled one-parameter reduction the strongest supported next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1520, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,519-parameter model will retain at least 99% accuracy because the verified 1,520-parameter six-gauge design achieved 99.96%, and a seventh positional row has the same exact pre-LayerNorm scalar-shift symmetry.
change: Extend the current four positional gauges through the verified fifth and sixth gauges, then gauge-fix the seventh row while preserving every omitted coordinate during initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Seventh positional-row scalar-shift gauge with ambient AdamW
evidence_used: Reference Design 2 achieved 99.96% accuracy with six positional-row gauges at 1,520 parameters; every prior one-row positional extension remained above 99%, making the seventh identical one-parameter reduction the strongest supported next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1519, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,518-parameter model will retain at least 99% accuracy because the verified seven-gauge design achieved 99.98%, while the eighth positional row has the same exact pre-LayerNorm scalar-shift symmetry.
change: Reproduce the qualified seven positional gauges, then gauge-fix the eighth row while preserving every omitted coordinate through initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Eighth positional-row scalar-shift gauge with ambient AdamW
evidence_used: Reference Design 1 achieved 99.98% accuracy with seven positional-row gauges at 1,519 parameters; every tested positional-gauge extension has remained above 99%, making the identical eighth-row reduction the strongest supported next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9809, "parameters": 1518, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing full-width absolute position embeddings with per-head learned relative-lag logits will retain at least 99% accuracy while removing `6 * max_seq_len - 2` parameters, because addition dependencies repeat at relative offsets and content-dependent query/key attention remains intact.
change: Remove absolute positions from the residual stream and let each causal attention head learn its own lag preference directly; simplify gauge optimization accordingly. This challenges the prior assumption that every position needs a full learned vector.
mechanism: Learned relative-lag attention routing
evidence_used: Absolute-position compression stalled at 97.06% for a rank-4 subspace and 94.19% for fixed Chebyshev features, whereas the shared-key/value attention design reached 99.98%. This motivates changing positional computation itself instead of further compressing the same absolute-table representation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9922, "parameters": 1388, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,386-parameter model will retain at least 99% accuracy because the verified 1,388-parameter relative-lag design reached 99.22%, while removing one additive constant from each head’s lag logits is an exact softmax symmetry with ambient AdamW dynamics preserved.
change: Store each head’s learned relative-lag vector with its final coordinate fixed to zero, reconstruct the full vectors during attention, and optimize both omitted coordinates through the existing gauge-aware clipping and AdamW path.
mechanism: Per-head relative-logit softmax gauge fixing
evidence_used: The current learned relative-lag design achieved 99.22% at 1,388 parameters, and prior positional/output gauge reductions repeatedly preserved accuracy when omitted coordinates retained full ambient optimizer dynamics.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.995, "parameters": 1386, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,385-parameter model will retain at least 99% accuracy because the verified 1,386-parameter relative-lag design achieved 99.50%, while removing the common output component of a seventh `fc2` column is the same exact pre-LayerNorm scalar-shift symmetry already used successfully on six columns.
change: Replace absolute positional embeddings with gauge-fixed learned per-head relative-lag logits, then extend the terminal projection’s ambient-AdamW output gauge from six to seven columns.
mechanism: Relative-lag attention with a seventh residual-output gauge
evidence_used: The gauge-fixed relative-lag design reached 99.50% with 1,386 parameters, substantially outperforming other positional compression approaches; its terminal projection already removes six equivalent column-wise output shifts, motivating one controlled additional reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1385, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,384-parameter model will retain at least 99% accuracy because the verified 1,385-parameter relative-lag design achieved 99.64%, while removing the common output component of an eighth `fc2` column is the same exact pre-LayerNorm scalar-shift symmetry already applied to seven columns.
change: Replace absolute positional embeddings with per-head gauge-fixed learned relative-lag logits, reproduce the verified seven-column terminal gauge, and extend it to an eighth column while preserving omitted coordinates through gradient clipping and ambient AdamW updates.
mechanism: Gauge-fixed relative-lag attention with an eighth terminal-output gauge
evidence_used: The 1,385-parameter relative-lag design reached 99.64% with seven terminal-column gauges; this patch makes the smallest controlled reduction from that qualified design using the identical output-shift symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1384, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,383-parameter relative-lag model will retain at least 99% accuracy because the verified 1,384-parameter design achieved 99.73%, while removing the common output component of a ninth `fc2` column uses the same exact pre-final-LayerNorm symmetry already verified for eight columns.
change: Replace absolute position embeddings with gauge-fixed per-head relative-lag logits and extend the qualified terminal projection gauge from eight to nine columns, preserving omitted coordinates through clipping and ambient AdamW updates.
mechanism: Ninth terminal-output scalar-shift gauge
evidence_used: The 1,384-parameter relative-lag design reached 99.73% after extending the terminal gauge to eight columns; a ninth column is the smallest controlled reduction using the identical symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9974, "parameters": 1383, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,382-parameter relative-lag model will retain at least 99% accuracy because the verified 1,383-parameter design achieved 99.74%, while removing the common output component of a tenth `fc2` column uses the identical pre-final-LayerNorm symmetry already verified for nine columns.
change: Extend `GaugeFixedTerminalLinear` from nine to ten gauge-fixed weight columns, retaining the existing full ambient-gradient clipping and AdamW update path for the omitted coordinate.
mechanism: Tenth terminal-output scalar-shift gauge
evidence_used: The current nine-column-gauge design reached 99.74% at 1,383 parameters, and each prior extension from seven through nine columns remained above 99%, making the identical one-parameter reduction the most controlled next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9945, "parameters": 1382, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,381-parameter relative-lag model will retain at least 99% accuracy because the verified 1,382-parameter ten-column design achieved 99.45%, while gauge-fixing an eleventh `fc2` column applies the same exact pre-final-LayerNorm output-shift symmetry.
change: Reproduce the qualified ten-column terminal gauge and extend it to eleven columns, preserving the omitted coordinate through the existing ambient-gradient clipping and AdamW update path.
mechanism: Eleventh terminal-output scalar-shift gauge
evidence_used: Reference Design 1 reached 99.45% accuracy at 1,382 parameters with ten terminal-column gauges; every tested extension through the tenth column remained above 99%, motivating the smallest controlled one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1381, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,380-parameter relative-lag model will retain at least 99% accuracy because the verified 1,381-parameter design achieved 99.75%, while gauge-fixing the twelfth and final `fc2` column applies the same exact pre-final-LayerNorm symmetry already verified for eleven columns.
change: Extend `GaugeFixedTerminalLinear` from seven to all twelve gauge-fixed columns, omit the now-empty residual weight parameter, and retain the existing full ambient-gradient clipping and AdamW updates.
mechanism: Complete terminal-output scalar-shift gauge fixing
evidence_used: The qualified eleven-column design reached 99.75% accuracy at 1,381 parameters; every terminal-column extension through eleven remained above 99%, directly motivating the final identical one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9971, "parameters": 1380, "training_steps": 4999}



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
blocks have been applied. All blocks must apply. They may edit either or both
editable files, but together they must describe one implementation ready for
verification. The mechanism name is descriptive, not chosen from a fixed list.
Do not paste whole files, lengthy logs, or routine progress reports outside the
patch.
