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
verified_results: {"accuracy": 0.9945, "parameters": 1382, "training_steps": 4999}
prior_hypothesis: A 1,382-parameter relative-lag model will retain at least 99% accuracy because the verified 1,383-parameter design achieved 99.74%, while removing the common output component of a tenth `fc2` column uses the identical pre-final-LayerNorm symmetry already verified for nine columns.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9971, "parameters": 1380, "training_steps": 4999}
prior_hypothesis: A 1,380-parameter relative-lag model will retain at least 99% accuracy because the verified 1,381-parameter design achieved 99.75%, while gauge-fixing the twelfth and final `fc2` column applies the same exact pre-final-LayerNorm symmetry already verified for eleven columns.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9975, "parameters": 1381, "training_steps": 4999}
prior_hypothesis: A 1,381-parameter relative-lag model will retain at least 99% accuracy because the verified 1,382-parameter ten-column design achieved 99.45%, while gauge-fixing an eleventh `fc2` column applies the same exact pre-final-LayerNorm output-shift symmetry.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9939, "parameters": 1378, "training_steps": 4999}
prior_hypothesis: A 1,378-parameter model will retain at least 99% accuracy because the verified 1,379-parameter design achieved 99.67%, and absorbing a third `ln1` scale extends the exact downstream q/k/v factorization already used successfully for two scales.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 1,379-parameter relative-lag model will retain at least 99% accuracy because the verified 1,380-parameter complete-terminal-gauge design achieved 99.71%, while gauge-fixing a fourth attention-output column applies the same exact pre-final-LayerNorm scalar-shift symmetry already used on three columns.
change: Reproduce the qualified all-twelve-column terminal projection gauge, then extend the attention output projection from three to four gauge-fixed columns while preserving omitted coordinates through the existing ambient-gradient clipping and AdamW path.
mechanism: Fourth attention-output scalar-shift gauge atop complete terminal gauge
evidence_used: Reference Design 2 reached 99.71% accuracy at 1,380 parameters with every terminal-output column gauge-fixed; the attention projection already removes three equivalent output shifts, making a fourth column the smallest controlled reduction from that qualified design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9967, "parameters": 1379, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,378-parameter model will retain at least 99% accuracy because the verified 1,379-parameter design achieved 99.67%, while gauge-fixing a fifth attention-output column applies the same exact pre-final-LayerNorm symmetry already verified for four columns.
change: Extend `GaugeFixedAttentionProjection` from four to five gauge-fixed weight columns while preserving omitted-coordinate initialization and ambient AdamW optimization.
mechanism: Fifth attention-output scalar-shift gauge
evidence_used: The current four-column attention-output gauge reached 99.67% at 1,379 parameters; extending the identical symmetry by one column is the smallest controlled reduction from that qualified design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0017000000000000001, "parameters": 1378, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,270-parameter transformer will retain at least 99% accuracy because learned per-head lag distributions can route the fixed-offset operands without content-dependent query/key scores, while learned values, projections, and the MLP still perform token-dependent computation.
change: Replace query/key attention with two learned causal relative-lag routing heads over a shared learned value stream, and absorb all first-LayerNorm scales into that sole value projection.
mechanism: Content-independent learned relative-lag attention
evidence_used: The 1,388-parameter relative-lag design reached 99.22% despite already sharing keys and values; this suggests its load-bearing head distinction is learned lag routing. The patch directly tests the alternative to the old assumption that a separate 104-parameter content-addressing path is also necessary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.02, "parameters": 1270, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,378-parameter model will retain at least 99% accuracy because the qualified 1,379-parameter design reached 99.67%, while one common component of an `fc1` weight row is functionally invisible after parameter-free LayerNorm and its omitted ambient coordinate is preserved during optimization.
change: Reproduce the qualified complete terminal gauge and four-column attention-output gauge, then remove one parameter from the first `fc1` row while maintaining its omitted offset through gradient clipping, absorbed-scale AdamW updates, and initialization.
mechanism: First pre-MLP zero-mean row gauge with ambient factorized AdamW
evidence_used: Reference Design 3 achieved 99.67% accuracy at 1,379 parameters. The failed fifth attention-output gauge motivates testing a distinct exact LayerNorm symmetry rather than extending that attention gauge again.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1066, "parameters": 1378, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,378-parameter model will retain at least 99% accuracy because the qualified 1,379-parameter design achieved 99.67%, while gauge-fixing the final attention-projection input column uses the same exact downstream-LayerNorm symmetry without repeating the failed first-column-of-head-two gauge.
change: Reproduce the qualified complete terminal gauge and four first-head attention gauges, then gauge-fix attention column seven while preserving full ambient-gradient clipping and AdamW updates.
mechanism: Noncontiguous fifth attention-output scalar-shift gauge
evidence_used: The all-terminal, four-column attention-gauge design reached 99.67% at 1,379 parameters, whereas extending the contiguous prefix into column four collapsed to 0.17%; selecting column seven tests whether that failure was coordinate/trajectory-specific rather than a limit of the exact projection symmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1438, "parameters": 1378, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,378-parameter model will retain at least 99% accuracy because the verified 1,379-parameter design achieved 99.67%, and absorbing a third `ln1` scale extends the exact downstream q/k/v factorization already used successfully for two scales.
change: Remove one learned `ln1` scale, absorb it into the corresponding q/k/v input columns, and preserve its full ambient AdamW and gradient-clipping dynamics through the existing virtual-scale optimizer path.
mechanism: Third pre-attention LayerNorm scale absorption
evidence_used: The current design verifies at 99.67% while already absorbing two `ln1` scales. The failed 1,378-parameter trials altered attention-output or `fc1` gauges, so extending the previously qualified scale-absorption mechanism is the most informative distinct one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9939, "parameters": 1378, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, and absorbing a fourth `ln1` scale extends the same exact q/k/v factorization that succeeded for three scales.
change: Reproduce the qualified four-column attention-output gauge, then absorb four rather than three `ln1` scales while updating the ambient-gradient clipping and AdamW factorization slices accordingly.
mechanism: Fourth pre-attention LayerNorm scale absorption
evidence_used: Reference Design 3 achieved 99.39% accuracy at 1,378 parameters after three-scale absorption; unlike the failed fifth attention gauge and `fc1` row gauge, this applies the previously successful scale-absorption mechanism by one additional coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.49670000000000003, "parameters": 1377, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while gauge-fixing the untested attention column five with a zero-mean representative minimizes the numerically irrelevant common output shift that may have destabilized prior fifth-gauge trials.
change: Reproduce the qualified complete terminal gauge, four-column attention gauge, and three-scale `ln1` absorption, then remove one additional parameter by reconstructing attention column five in a centered scalar-shift gauge while retaining full ambient-gradient clipping and AdamW updates.
mechanism: Balanced fifth attention-output scalar-shift gauge
evidence_used: Reference Design 3 achieved 99.39% at 1,378 parameters. Prior fifth-gauge attempts using last-coordinate-zero representatives collapsed on columns four and seven, motivating a centered representative on the untested interior column five to distinguish numerical gauge conditioning from a true capacity limit.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9329999999999999, "parameters": 1377, "training_steps": 4999}



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
