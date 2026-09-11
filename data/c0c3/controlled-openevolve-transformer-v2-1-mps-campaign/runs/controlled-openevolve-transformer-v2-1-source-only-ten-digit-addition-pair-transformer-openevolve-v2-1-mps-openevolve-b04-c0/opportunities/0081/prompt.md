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
verified_results: {"accuracy": 0.9963, "parameters": 1587, "training_steps": 4999}
prior_hypothesis: Fixing `ln1.weight[5]` alongside coordinates 0, 2, 4, and 6 will reduce the model from 1588 to 1587 parameters while retaining at least 99% accuracy, because its scale is absorbable by QKV column 5 and coordinate-specific outcomes leave untested coordinate 5 informative despite earlier failures at odd coordinates 1 and 3.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing `ln1.weight[0]` at one will reduce the model from 1592 to 1591 parameters while retaining at least 99% accuracy, because coordinate 0 may be optimization-compatible even though fixing coordinate 3—whose shift is already anchored—failed.
change: Replace the eight-parameter first LayerNorm gain with seven learned coordinates, reconstructing gain coordinate 0 as one while continuing to fix shift coordinate 3.
mechanism: Alternative attention-input gain anchoring
evidence_used: Fixing `ln1.weight[3]` reached only 74.01%, but the final-MLP bias experiments demonstrated strong coordinate sensitivity: fixing `fc2.bias[0]` reached 2.79%, whereas fixing `fc2.bias[1]` achieved 99.93% at 1592 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1.weight[1]` at one will reduce the model from 1591 to 1590 parameters while retaining at least 99% accuracy, because fixing coordinate 0 achieved 99.91% and each LayerNorm gain can be absorbed into the corresponding learned QKV input column.
change: Store six learned first-LayerNorm gains and reconstruct gain coordinates 0 and 1 as fixed ones, while continuing to fix shift coordinate 3.
mechanism: Second attention-input gain anchoring
evidence_used: The current 1591-parameter design achieved 99.91% after fixing `ln1.weight[0]`; the earlier failure at coordinate 3 demonstrates coordinate-sensitive optimization, making adjacent untested coordinate 1 the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9889, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1.weight[2]` alongside coordinate 0 will reduce the model from 1591 to 1590 parameters while retaining at least 99% accuracy, because LayerNorm-gain removability is coordinate-sensitive and the alternative coordinate 1 narrowly missed at 98.89%.
change: Store six learned first-LayerNorm gains and reconstruct gain coordinates 0 and 2 as fixed ones, while continuing to fix shift coordinate 3.
mechanism: Alternative second attention-input gain anchoring
evidence_used: Fixing gain coordinate 0 achieved 99.91% at 1591 parameters, while additionally fixing coordinate 1 reached 98.89%; prior `fc2.bias` results showed that changing the anchored coordinate can move accuracy from 2.79% to 99.93%, motivating the next untested gain coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1.weight[4]` alongside coordinates 0 and 2 will reduce the model from 1590 to 1589 parameters while retaining at least 99% accuracy, because LayerNorm-gain removability is strongly coordinate-sensitive and coordinate 2 preserved 99.87% whereas coordinates 1 and 3 failed.
change: Store five learned first-LayerNorm gains and reconstruct gain coordinates 0, 2, and 4 as fixed ones, while continuing to fix shift coordinate 3.
mechanism: Alternative third attention-input gain anchoring
evidence_used: The current design achieved 99.87% with gain coordinates 0 and 2 fixed; the 98.89% result for coordinate 1 and 74.01% result for coordinate 3 make the next untested coordinate 4 the most informative capacity test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1.weight[6]` alongside coordinates 0, 2, and 4 will reduce the model from 1589 to 1588 parameters while retaining at least 99% accuracy, because every tested even gain coordinate so far has been removable whereas coordinates 1 and 3 failed.
change: Store four learned first-LayerNorm gains and reconstruct gain coordinates 0, 2, 4, and 6 as fixed ones, while continuing to fix shift coordinate 3.
mechanism: Fourth even-coordinate attention-input gain anchoring
evidence_used: Fixing gain coordinates 0, 2, and 4 achieved 99.88% at 1589 parameters; the successful even-coordinate sequence, contrasted with 98.89% for coordinate 1 and 74.01% for coordinate 3, makes coordinate 6 the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1.weight[5]` alongside coordinates 0, 2, 4, and 6 will reduce the model from 1588 to 1587 parameters while retaining at least 99% accuracy, because its scale is absorbable by QKV column 5 and coordinate-specific outcomes leave untested coordinate 5 informative despite earlier failures at odd coordinates 1 and 3.
change: Store three learned first-LayerNorm gains and reconstruct gain coordinates 0, 2, 4, 5, and 6 as fixed ones, while continuing to fix shift coordinate 3.
mechanism: Fifth attention-input gain anchoring
evidence_used: The current design achieved 99.90% with gain coordinates 0, 2, 4, and 6 fixed; prior results also showed strong coordinate sensitivity, including `fc2.bias[0]` failing at 2.79% while coordinate 1 achieved 99.93%, so the next untested gain coordinate cannot be inferred solely from failures at coordinates 1 and 3.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9963, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1.weight[7]` alongside coordinates 0, 2, 4, 5, and 6 will reduce the model from 1587 to 1586 parameters while retaining at least 99% accuracy, because coordinate 7 is the only untested first-LayerNorm gain and its scale is absorbable by QKV input column 7.
change: Store only gains 1 and 3 as learned parameters and reconstruct all other first-LayerNorm gains as fixed ones, while continuing to fix shift coordinate 3.
mechanism: Final-coordinate attention-input gain anchoring
evidence_used: The current design achieved 99.63% after fixing gains 0, 2, 4, 5, and 6; although additional fixes at gains 1 and 3 failed, prior coordinate-sensitive results show those failures do not determine whether the sole remaining coordinate 7 is removable.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9628, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing `ln1.weight[7]` with learned gain coordinate 1 will reduce the model to 1586 parameters while retaining at least 99% accuracy, because QKV columns can absorb coordinate-specific scaling while the shared value remains trainable.
change: Store only first-LayerNorm gains 1 and 3, and reuse gain 1 for coordinate 7 instead of fixing coordinate 7 at one.
mechanism: Shared adaptive attention-input gain
evidence_used: Independently fixing coordinate 7 reduced accuracy to 96.28%, while the 1587-parameter design reached 99.63%; sharing preserves adaptive scaling for coordinate 7 while retaining the especially sensitive coordinate 3 independently.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9867, "parameters": 1586, "training_steps": 4999}



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
