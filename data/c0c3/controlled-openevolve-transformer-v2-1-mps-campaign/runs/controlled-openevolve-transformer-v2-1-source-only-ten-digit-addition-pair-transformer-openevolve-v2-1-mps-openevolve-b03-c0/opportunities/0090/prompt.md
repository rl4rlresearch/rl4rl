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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9998, "parameters": 1573, "training_steps": 20999}
prior_hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 7` will reduce the model from 1,574 to 1,573 parameters while retaining at least 99% accuracy after 21,000 steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-anchoring the initialized midpoint positional row will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split positional row `max_seq_len // 2` into seven learned relative coordinates plus a zero anchor, reconstruct it in place during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Nonadjacent midpoint positional-row shift gauge
evidence_used: Rows 0 and 1 retained 99.98% and 99.94% accuracy, while row 2 and the fifth-to-last row failed near 74%; testing the distant midpoint is the clearest next test of location-specific optimization sensitivity without disturbing the initialization draw stream.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1580, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 1` will reduce the model from 1,580 to 1,579 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the row immediately after the successful midpoint row into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: The isolated midpoint gauge achieved 99.84% at 1,580 parameters, and adjacent extensions of the same gauge succeeded across four terminal rows; testing the midpoint’s immediate neighbor is the clearest next test of whether that successful region can support another reduction despite failures at row 2 and the fifth-to-last row.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1579, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 2` will reduce the model from 1,579 to 1,578 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the second row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: The midpoint and immediately following positional rows retained 99.84% and 99.92% accuracy with the identical gauge; extending this successful contiguous region by one row is the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1578, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 3` will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the third row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: The midpoint through `max_seq_len // 2 + 2` formed a contiguous successful region, with the latest extension reaching 99.88% at 1,578 parameters; extending the identical gauge to its next adjacent row is the most informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1577, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 4` will reduce the model from 1,577 to 1,576 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the fourth row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: The contiguous midpoint region through `max_seq_len // 2 + 3` has four successful gauges, most recently reaching 99.92% at 1,577 parameters; extending the identical symmetry to the next adjacent row is the most informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1576, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 5` will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the fifth row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: The contiguous midpoint region from `max_seq_len // 2` through `max_seq_len // 2 + 4` has five successful gauges, most recently reaching 99.83% at 1,576 parameters; extending the identical symmetry to the next adjacent row is the most informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1575, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 6` will reduce the model from 1,575 to 1,574 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the sixth row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: The contiguous midpoint region from `max_seq_len // 2` through `max_seq_len // 2 + 5` has six successful gauges, with the latest extension reaching 99.98% at 1,575 parameters; extending the identical symmetry to the next adjacent row is the most informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1574, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 7` will reduce the model from 1,574 to 1,573 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the seventh row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: All seven contiguous rows from `max_seq_len // 2` through `max_seq_len // 2 + 6` succeeded with this exact gauge; the latest achieved 99.97% accuracy at 1,574 parameters, making the next adjacent row the most informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1573, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the eighth row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Adjacent-midpoint positional-row shift gauge
evidence_used: Every contiguous row from `max_seq_len // 2` through `max_seq_len // 2 + 7` succeeded with this gauge; the latest achieved 99.98% accuracy at 1,573 parameters, directly motivating the next adjacent extension.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.647, "parameters": 1580, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 10` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the tenth row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Nonadjacent interior positional-row shift gauge
evidence_used: The adjacent `+8` trial failed at 64.7% with a malformed 1,580-parameter result, while earlier nonadjacent gauge placement recovered after adjacent failures; testing `+10` probes the interior between the failed `+8` and fifth-to-last regions without repeating the failed boundary extension.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.93, "parameters": 1580, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 - 1` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the row immediately before the successful midpoint region into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Opposite-edge positional-row shift gauge
evidence_used: All eight rows from the midpoint through `+7` succeeded, reaching 99.98% at 1,573 parameters; after unsuccessful or malformed attempts beyond the upper edge, extending the identical gauge from the untested opposite edge is the most informative next reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gauge-anchoring row `max_seq_len // 2 - 1` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy, and halving evaluation frequency will let the previously timed-out experiment complete without changing its training trajectory.
change: Add a seven-coordinate gauge representation for the positional row immediately before the successful midpoint region, reconstruct it with a zero anchor, optimize it with `GaugeAdamW`, and change the default evaluation interval from 300 to 600 steps.
mechanism: Opposite-edge positional-row shift gauge with validation-cadence thinning
evidence_used: All eight contiguous rows from the midpoint through `+7` succeeded, reaching 99.98% accuracy at 1,573 parameters; the only `-1` trial timed out rather than producing adverse accuracy evidence, so repeating that reduction with fewer expensive validation passes is the most informative next test.
result: training did not finish within the verification time limit



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
