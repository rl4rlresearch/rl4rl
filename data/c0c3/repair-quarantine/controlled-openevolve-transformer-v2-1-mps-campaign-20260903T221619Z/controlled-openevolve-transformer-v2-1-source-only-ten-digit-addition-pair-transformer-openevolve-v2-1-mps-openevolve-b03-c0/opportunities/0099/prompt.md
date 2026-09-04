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
hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 10` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the tenth row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Nonadjacent interior positional-row shift gauge
evidence_used: The adjacent `+8` trial failed at 64.7% with a malformed 1,580-parameter result, while earlier nonadjacent gauge placement recovered after adjacent failures; testing `+10` probes the interior between the failed `+8` and fifth-to-last regions without repeating the failed boundary extension.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.93, "parameters": 1580, "training_steps": 20999}

RECENT RESULT
hypothesis: Correctly gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Represent the `+8` positional row with seven learned relative coordinates and an implicit zero anchor, remove its original eight-parameter row, reconstruct it in place, and train it with `GaugeAdamW`.
mechanism: Corrected adjacent-midpoint positional-row shift gauge
evidence_used: All contiguous rows from the midpoint through `+7` succeeded, most recently at 99.98% accuracy; the prior `+8` attempt reported 1,580 parameters instead of the intended 1,572, so it was malformed and did not conclusively test this adjacent extension.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"accuracy": 0.9944, "parameters": 1580, "training_steps": 20999}

RECENT RESULT
hypothesis: Replacing positional row `max_seq_len // 2 + 8` with seven learned relative coordinates and an implicit zero anchor will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy.
change: Add a gauged `+8` positional row, explicitly remove its original eight-parameter row from the base embedding, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Corrected adjacent positional-row shift gauge
evidence_used: Gauging every contiguous midpoint row through `+7` succeeded, reaching 99.98% at 1,573 parameters; the malformed `+8` trial still reached 99.44% but reported 1,580 parameters, so the correctly deduplicated extension remains untested.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 parameters while retaining at least 99% accuracy.
change: Remove the `+8` row from the base positional embedding, represent it with seven learned relative coordinates and an implicit zero anchor, reconstruct it during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Correct adjacent positional-row shift gauge
evidence_used: All contiguous midpoint rows through `+7` succeeded, most recently reaching 99.98% at 1,573 parameters; a malformed `+8` implementation still reached 99.44% but reported 1,580 parameters, indicating the row can tolerate the gauge while leaving the intended deduplicated implementation untested.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Correctly removing positional row `max_seq_len // 2 + 8` from the base embedding and representing it with seven learned relative coordinates plus an implicit zero anchor will reduce the verified model from 1,573 to 1,572 parameters while retaining at least 99% accuracy.
change: Add a gauged `+8` positional row, exclude all nine midpoint-through-`+8` rows from the base positional parameter, reconstruct the new row during forward passes, and train it with `GaugeAdamW`.
mechanism: Extended adjacent positional-row shift gauge
evidence_used: Gauging every contiguous row from the midpoint through `+7` succeeded, reaching 99.98% at 1,573 parameters; the malformed `+8` trial still reached 99.44% but counted 1,580 parameters, so a correctly deduplicated `+8` implementation remains the most informative next test.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Correctly gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy.
change: Remove the `+8` row from the base positional embedding, represent it with seven learned relative coordinates and an implicit zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.
mechanism: Extended adjacent positional-row shift gauge
evidence_used: Gauging every contiguous midpoint row through `+7` reached 99.98% at 1,573 parameters, while the malformed `+8` model still reached 99.44%; the correctly deduplicated extension remains the strongest unverified reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the `+8` row from the base positional embedding, represent it with seven learned relative coordinates and an implicit zero anchor, reconstruct it during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row shift-gauge extension
evidence_used: Gauging all contiguous midpoint rows through `+7` achieved 99.98% at 1,573 parameters, while a malformed `+8` model still achieved 99.44% but counted 1,580 parameters; this directly motivates a correctly deduplicated `+8` extension.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` while removing its original eight-parameter row will reduce the model from 1,573 to 1,572 parameters and retain at least 99% accuracy after 21,000 steps.
change: Represent the `+8` positional row using seven learned relative coordinates and an implicit zero anchor, exclude its original row from the base embedding, reconstruct it in forward passes, and optimize it with `GaugeAdamW`.
mechanism: Correct adjacent positional-row shift-gauge extension
evidence_used: The verified contiguous gauges through `+7` reached 99.98% accuracy at 1,573 parameters; the malformed `+8` model still reached 99.44% but counted 1,580 parameters, supporting the row’s tolerance while leaving the correctly deduplicated extension untested.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` and removing its original eight-parameter row will reduce the verified model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Add a seven-coordinate gauged `+8` positional row, exclude its original row from the base embedding, reconstruct it during forward passes, and train it with `GaugeAdamW`.
mechanism: Correct adjacent positional-row shift-gauge extension
evidence_used: Gauges through `+7` achieved 99.98% at 1,573 parameters, and the malformed `+8` implementation achieved 99.44% despite counting 1,580 parameters; this supports the row’s tolerance while making correct deduplication the most informative test.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Correctly replacing positional row `max_seq_len // 2 + 8` with seven learned relative coordinates and an implicit zero anchor will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Exclude the `+8` row from the base positional embedding, reconstruct it from a seven-parameter gauged row during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row shift-gauge extension
evidence_used: Gauging every contiguous midpoint row through `+7` achieved 99.98% at 1,573 parameters, while the malformed `+8` implementation still achieved 99.44% but counted 1,580 parameters; this indicates the row tolerates the gauge and leaves correct deduplication as the most informative test.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Correctly gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the `+8` row from the base positional embedding, represent it with seven learned relative coordinates and an implicit zero anchor, reconstruct it during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row shift-gauge extension
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` implementation still achieved 99.44% but counted 1,580 parameters; this supports the row’s tolerance and makes correct deduplication the most informative next test.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Correctly replacing positional row `max_seq_len // 2 + 8` with seven learned relative coordinates and an implicit zero anchor will reduce the verified model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the `+8` row from the base positional embedding, reconstruct it from a seven-parameter gauged row during forward passes, and train that parameter with `GaugeAdamW`.
mechanism: Adjacent positional-row shift-gauge extension
evidence_used: Gauging every contiguous midpoint row through `+7` achieved 99.98% accuracy at 1,573 parameters, and the malformed `+8` implementation still achieved 99.44% while retaining eight extra parameters; this directly supports testing the correctly deduplicated extension.
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
blocks have been applied. All blocks must apply. They may edit either or both
editable files, but together they must describe one implementation ready for
verification. The mechanism name is descriptive, not chosen from a fixed list.
Do not paste whole files, lengthy logs, or routine progress reports outside the
patch.
