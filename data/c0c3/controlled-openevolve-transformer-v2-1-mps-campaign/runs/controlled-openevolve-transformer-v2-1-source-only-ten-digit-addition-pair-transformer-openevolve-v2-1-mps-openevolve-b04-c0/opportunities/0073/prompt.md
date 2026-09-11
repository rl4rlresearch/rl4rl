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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1592, "training_steps": 4999}
prior_hypothesis: Fixing `fc2.bias[1]` instead of coordinate 0 will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because coordinate-specific optimization effects may make coordinate 1 removable even though fixing coordinate 0 failed.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `fc1.bias[11]` in addition to coordinates 0 through 7, 9, and 10 will reduce the model from 1597 to 1596 parameters while retaining at least 99% accuracy, because the current design retained 99.97% accuracy with only coordinates 8 and 11 learned, while prior evidence specifically identifies coordinate 8—not coordinate 11—as initialization-sensitive.
change: Retain a learned `fc1` bias only for hidden unit 8 and reconstruct every other bias coordinate as a fixed zero while preserving ordinary linear-layer initialization RNG use.
mechanism: Third alternative hidden-unit bias anchoring
evidence_used: The 1597-parameter design achieved 99.97% accuracy after removing coordinate 10, whereas removing coordinate 8 previously reduced accuracy to 77.57%; this makes coordinate 11 the most informative remaining hidden-bias constraint to test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[0]` at zero will reduce the model from 1596 to 1595 parameters while retaining at least 99% accuracy, because one query-shift direction is invisible to attention softmax and the remaining seven query-bias coordinates preserve trainable content-independent attention control.
change: Replace the eight-parameter query bias with seven learned coordinates, reconstructing coordinate 0 as a fixed zero while preserving the existing zero initialization.
mechanism: Query-bias softmax-gauge anchoring
evidence_used: The 1596-parameter design achieved 99.94%, while removing the sole remaining `fc1.bias[8]` previously collapsed accuracy to 77.57%; this motivates leaving that initialization-sensitive parameter intact and testing an untouched attention-side redundancy instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9956, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[1]` at zero will reduce the model from 1595 to 1594 parameters while retaining at least 99% accuracy, because coordinate 0 was removed successfully and the zero-initialized query-bias coordinates are architecturally symmetric.
change: Replace the seven learned query-bias coordinates with six, reconstructing coordinates 0 and 1 as fixed zeros while preserving initialization behavior.
mechanism: Second query-bias gauge anchoring
evidence_used: Fixing `q_bias[0]` reduced the model to 1595 parameters with 99.56% accuracy; extending that successful attention-side constraint is the most direct next capacity test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[2]` at zero will reduce the model from 1594 to 1593 parameters while retaining at least 99% accuracy, because fixing coordinates 0 and 1 consecutively preserved 99.97% accuracy and all query-bias coordinates share zero initialization.
change: Replace the six learned query-bias coordinates with five, reconstructing coordinates 0 through 2 as fixed zeros.
mechanism: Third query-bias coordinate anchoring
evidence_used: The 1594-parameter design achieved 99.97% accuracy after fixing `q_bias[1]`; extending this successful query-bias constraint is the most informative next capacity test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[3]` at zero will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because coordinates 0 through 2 were removed consecutively without violating the accuracy requirement and all query-bias coordinates share zero initialization.
change: Replace the five learned query-bias coordinates with four, reconstructing coordinates 0 through 3 as fixed zeros.
mechanism: Fourth query-bias coordinate anchoring
evidence_used: The 1593-parameter design achieved 99.90% accuracy after fixing `q_bias[2]`; extending this three-step successful query-bias constraint is the most informative next capacity test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.14800000000000002, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[4]` instead of `q_bias[3]` will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because coordinate-specific initialization effects previously allowed `fc1.bias[9]` to be removed successfully after removing the symmetric coordinate 8 failed.
change: Retain learned query-bias coordinates 3, 5, 6, and 7 while reconstructing coordinates 0, 1, 2, and 4 as fixed zeros.
mechanism: Alternative query-bias coordinate anchoring
evidence_used: Fixing `q_bias[3]` collapsed accuracy to 14.80%, but the analogous hidden-bias sequence showed that failure of one symmetric coordinate need not imply failure of another: fixing `fc1.bias[8]` reached only 77.57%, whereas fixing coordinate 9 produced 100% accuracy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.451, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[5]` while retaining coordinates 3, 4, 6, and 7 will reduce the model from 1593 to 1592 parameters while maintaining at least 99% accuracy, because coordinate-specific training effects may make coordinate 5 removable even though removing coordinates 3 or 4 failed.
change: Store four learned query-bias coordinates and reconstruct coordinates 0, 1, 2, and 5 as fixed zeros.
mechanism: Alternative fifth-coordinate query-bias anchoring
evidence_used: Fixing query-bias coordinates 0 through 2 preserved 99.90% accuracy, while alternative removals of coordinates 3 and 4 failed; earlier hidden-bias results showed that one symmetric coordinate can fail badly while another succeeds, motivating the next untested coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0677, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[6]` while retaining coordinates 3, 4, 5, and 7 will reduce the model from 1593 to 1592 parameters while maintaining at least 99% accuracy, because coordinate-specific training effects may make coordinate 6 removable even though removing coordinates 3, 4, or 5 failed.
change: Store four learned query-bias coordinates and reconstruct coordinates 0, 1, 2, and 6 as fixed zeros.
mechanism: Alternative sixth-coordinate query-bias anchoring
evidence_used: The 1593-parameter design achieved 99.90% accuracy with coordinates 0 through 2 fixed; prior hidden-bias experiments showed that failure of one architecturally symmetric coordinate did not predict failure of another, motivating the next untested query-bias coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5329999999999999, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[7]` while retaining coordinates 3 through 6 will reduce the model from 1593 to 1592 parameters while maintaining at least 99% accuracy, because coordinate-specific training effects may make the sole untested query-bias coordinate removable even though removing coordinates 3 through 6 failed.
change: Store four learned query-bias coordinates and reconstruct coordinates 0, 1, 2, and 7 as fixed zeros.
mechanism: Alternative seventh-coordinate query-bias anchoring
evidence_used: Fixing query-bias coordinates 0 through 2 preserved 99.90% accuracy, while removing coordinates 3, 4, 5, or 6 individually failed; coordinate 7 is the only remaining untested alternative, and earlier hidden-bias results showed that failures of symmetric coordinates did not rule out success for another coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9323, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `fc2.bias[0]` at zero will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because any removed value can be subtracted uniformly from all eight bias coordinates without changing the following final LayerNorm output.
change: Replace the eight-parameter `fc2` bias with seven learned coordinates, reconstruct coordinate 0 as zero, and preserve ordinary `nn.Linear` initialization RNG consumption.
mechanism: Final-residual bias gauge anchoring
evidence_used: The 1593-parameter design achieved 99.90%, while every attempted fourth query-bias removal failed; this motivates testing an untouched exact gauge in the final MLP residual rather than further reducing query capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0279, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `fc2.bias[1]` instead of coordinate 0 will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because coordinate-specific optimization effects may make coordinate 1 removable even though fixing coordinate 0 failed.
change: Replace the eight-parameter `fc2` bias with seven learned coordinates, reconstruct coordinate 1 as zero, and preserve ordinary `nn.Linear` initialization RNG consumption.
mechanism: Alternative final-residual bias gauge anchoring
evidence_used: Fixing `fc2.bias[0]` reached only 2.79%, but the hidden-bias experiments showed that one symmetric coordinate can fail while another succeeds: fixing `fc1.bias[8]` reached 77.57%, whereas fixing coordinate 9 reached 100%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1.weight[3]` at one will reduce the model from 1592 to 1591 parameters while retaining at least 99% accuracy, because its scale can be absorbed into column 3 of the learned QKV projection.
change: Replace the eight-parameter first LayerNorm gain with seven learned coordinates, reconstructing gain coordinate 3 as one while retaining the existing fixed shift at coordinate 3.
mechanism: Attention-input gain gauge anchoring
evidence_used: The 1592-parameter design achieved 99.93% accuracy while already fixing six second-LayerNorm gains that are absorbed by the downstream MLP; this tests the analogous unused multiplicative gauge before removing additional empirically sensitive biases.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7401000000000001, "parameters": 1591, "training_steps": 4999}



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
