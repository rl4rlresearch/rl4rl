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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1489, "training_steps": 4999}
prior_hypothesis: Shearing the untouched second-head bias-bearing query row against the already-successful dual-sheared bias-bearing row will reduce Reference Design 3 from 1490 to 1489 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9979, "parameters": 1486, "training_steps": 4999}
prior_hypothesis: Replacing the failed coordinate-aligned completion with a max-pivot-stable orthogonality chart will reduce the verified model from 1487 to 1486 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995, "parameters": 1491, "training_steps": 4999}
prior_hypothesis: Shearing the second head’s last bias-bearing query row against its normalized zero-bias target row will reduce the model from 1492 to 1491 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1485, "training_steps": 4999}
prior_hypothesis: Normalizing the remaining orthogonal first-head zero-bias query row with a max-pivot chart will reduce Reference Design 2 from 1486 to 1485 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Applying the untested complementary mutual shear between the first head’s freely biased query rows atop the verified 1487-parameter design will produce 1486 parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 1’s first-head terminal scale/shear chart, then omit one coordinate from the second biased query row using the leading biased row and compensate in the leading key row.
mechanism: Complementary first-head biased-row query/key shear
evidence_used: Reference Design 1 achieved 99.97% accuracy at 1487 parameters. The opposite first-head biased-row shear reached 91.85%, but the first-head zero-bias experiments showed strong directional asymmetry—the initially tested direction reached 97.09% while its complement reached 99.97%—making the untested complementary biased-row direction the closest distinct reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5919, "parameters": 1486, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the remaining first-head zero-bias shear after the verified terminal-row shear will reduce Reference Design 2 from 1487 to 1486 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified 1487-parameter construction, then omit the terminal pivot coordinate from the neighboring first-head query row and absorb its conditioned shear into the terminal key row.
mechanism: Conditioned completion of the first-head zero-bias query/key gauge
evidence_used: Reference Design 2 achieved 99.97% accuracy, and the second head successfully uses both sequential directions of the analogous zero-bias shear gauge; the failed 97.09% first-head experiment applied the opposite direction first, so testing it after the successful terminal-row shear isolates the observed ordering asymmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9776, "parameters": 1486, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the failed coordinate-aligned completion with a max-pivot-stable orthogonality chart will reduce the verified model from 1487 to 1486 parameters while retaining at least 99% accuracy.
change: Remove one coordinate from the first head’s remaining zero-bias query row, reconstruct it by enforcing orthogonality to the normalized terminal row, and apply the inverse initialization shear to the terminal key row.
mechanism: Orthogonal first-head zero-bias query/key gauge
evidence_used: The current complementary terminal-row shear achieved 99.97% at 1487 parameters, while the coordinate-aligned completion reached 97.76%; this tests the same remaining gauge with a balanced orthogonal constraint instead of another pivot-zero constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1486, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing the remaining orthogonal first-head zero-bias query row with a max-pivot chart will reduce Reference Design 2 from 1486 to 1485 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified first-head orthogonality construction, then fix the neighboring row’s remaining scale and transfer its initialization scale into the matching key row.
mechanism: Max-pivot normalized orthogonal first-head query frame
evidence_used: Reference Design 2 achieved 99.79% at 1486 parameters using a max-pivot-stable orthogonality chart; unlike the failed coordinate-aligned scale completion, this reduction normalizes the already-orthogonalized row and therefore completes the balanced two-row query/key gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1485, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1485-parameter normalized orthogonal-frame design with a balanced orthogonality chart for the first head’s freely biased query pair will reduce the model to 1484 parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 3, then omit one coordinate from the second biased first-head query row, reconstruct it by projection orthogonal to the first biased row, and apply the inverse initialization shear to the matching key row.
mechanism: Max-pivot projected orthogonality gauge for the first-head biased query pair
evidence_used: Reference Design 3 reached 99.92% at 1485 parameters. Direct coordinate shears between the first-head biased rows failed, whereas replacing a coordinate-aligned first-head zero-bias completion with a max-pivot orthogonality chart raised accuracy from 97.76% to 99.79%, and normalizing that orthogonal frame reached 99.92%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8047, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1485-parameter normalized orthogonal-query design with an orthogonality gauge on the first head’s neighboring projection row will produce 1484 parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 3, then remove one projection-row coordinate, reconstruct it orthogonal to the normalized first-head target row, and apply the inverse value transformation.
mechanism: Max-pivot orthogonal value/output frame completion
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters. A max-pivot orthogonality chart previously rescued an analogous query reduction from 97.76% to 99.79%, motivating a balanced alternative to the failed coordinate-aligned third value/output shear.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Constraining the first head’s neighboring projection row to be orthogonal to its normalized target row will reduce the model to 1484 parameters while retaining at least 99% accuracy.
change: Remove one learned coordinate from that projection row, reconstruct it from the orthogonality constraint, absorb the initialization shear into the matching value row, and update optimizer-side projection reconstruction.
mechanism: Max-pivot orthogonal value/output neighbor gauge
evidence_used: The current normalized orthogonal-query design reached 99.92% at 1485 parameters, and max-pivot orthogonality previously improved an analogous query reduction from 97.76% to 99.79%; the prior value/output attempt was not tested because its search matched multiple locations.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9723999999999999, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding one direction-preserving second-head value/output scale gauge to the verified 1485-parameter normalized orthogonal-query design will yield 1484 parameters while retaining at least 99% accuracy.
change: Complete the qualified first-head query-frame normalization, then normalize the second head’s penultimate projection row and transfer its scale into the matching value row; update projection reconstruction used by training.
mechanism: Max-pivot diagonal value/output gauge on the second-head neighboring row
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters, and identical diagonal value/output gauges already succeed on both target rows. The 1484-parameter neighboring-row orthogonality attempt reached 97.24%; preserving the row’s direction while removing only its scale is a less restrictive test of the remaining gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5554, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing only the first head’s neighboring projection-row scale will reduce Reference Design 3 from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified normalized orthogonal-query design, then replace the first head’s neighboring projection row with a max-pivot normalized chart and transfer its initialization scale into the corresponding value row; update optimizer-side projection reconstruction.
mechanism: First-head neighboring value/output scale gauge
evidence_used: Reference Design 3 achieved 99.92% accuracy at 1485 parameters. The first-head neighboring-row orthogonality constraint reached 97.24%, so removing only its independent scale while preserving its direction is the closest less-restrictive untested reduction; the poor second-head scale result also makes testing the distinct first-head gauge informative.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5366, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1485-parameter normalized orthogonal-query design and additionally shearing its remaining second-head biased anchor row against the normalized zero-bias target row will yield 1484 parameters while retaining at least 99% accuracy.
change: Add Reference Design 3’s first-head normalized orthogonal frame and full second-head construction, then omit a second anchor-row coordinate using a target-first shear that preserves the existing anchor pivot and transfers the inverse transformation into the target key row.
mechanism: Pivot-preserving second-head biased-to-zero-bias query/key shear
evidence_used: Reference Design 3 achieved 99.92% accuracy at 1485 parameters, while analogous coordinate-aligned biased-to-zero-bias shears already succeeded in the second head; unlike the failed biased-pair and value/output reductions, this uses a zero-bias target with a normalized, well-conditioned pivot.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9115000000000001, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Shearing the remaining second-head biased anchor row against the normalized penultimate zero-bias row will reduce the verified model from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Omit the anchor row’s penultimate-pivot coordinate, reconstruct both omitted coordinates, and transfer the inverse shear into the corresponding key row.
mechanism: Penultimate-first second-head anchor shear
evidence_used: The current normalized orthogonal-query design achieved 99.92% at 1485 parameters. The analogous target-first anchor shear reached only 91.15%, but prior first-head experiments showed strong directional asymmetry, making the complementary penultimate-first shear the closest distinct untested reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.63, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing the first head’s unconstrained biased-coordinate key row atop the verified normalized orthogonal-query frame will reduce the model from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Complete Reference Design 3’s normalization of the first-head zero-bias query frame, then remove the scale of the first biased-coordinate key row and transfer its initialization scale into the matching query row and bias.
mechanism: Max-pivot diagonal gauge on the first head’s biased-coordinate key row
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters. Prior 1484 failures constrained query shears or value/output rows; this tests an untried independent diagonal query/key gauge while leaving the biased query direction unconstrained.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3367, "parameters": 1484, "training_steps": 4999}



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
