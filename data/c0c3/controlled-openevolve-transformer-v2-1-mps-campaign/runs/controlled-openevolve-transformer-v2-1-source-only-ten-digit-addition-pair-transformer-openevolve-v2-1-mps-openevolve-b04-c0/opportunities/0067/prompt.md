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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1593, "training_steps": 4999}
prior_hypothesis: Fixing `q_bias[2]` at zero will reduce the model from 1594 to 1593 parameters while retaining at least 99% accuracy, because fixing coordinates 0 and 1 consecutively preserved 99.97% accuracy and all query-bias coordinates share zero initialization.

## Recent verification evidence

RECENT RESULT
hypothesis: Representing `fc2.bias` in an orthonormal seven-dimensional zero-sum basis will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because it removes the exact common-shift null direction without the asymmetric optimization geometry of fixing one output coordinate.
change: Replace the ordinary eight-parameter `fc2` bias with seven learned orthonormal quotient coordinates whose reconstructed bias sums to zero.
mechanism: Orthonormal final-LayerNorm common-shift quotient
evidence_used: Fixing `fc2.bias` coordinate 3 failed at 59% despite the common-shift invariance of the final LayerNorm; the current design reaches 99.91%, so a minimal-norm orthonormal parameterization directly tests whether that failure arose from the coordinate-anchored optimization geometry rather than loss of functional capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0298, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `fc1.weight[8,0]` at zero will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because LayerNorm’s normalized coordinates sum to zero and the still-learned hidden-unit-8 bias preserves the corresponding affine function class.
change: Replace the full `fc1` weight with 95 learned coordinates, reconstruct `weight[8,0]` as zero, and gauge-transform its ordinary initialization without changing the initial model function.
mechanism: Zero-mean LayerNorm incoming-weight gauge anchoring
evidence_used: Removing `fc1.bias[8]` reduced accuracy to 77.57%, showing that bias should remain learned; the 1599-parameter design reached 99.91%, motivating removal of an exact incoming-weight redundancy from that same unit instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6819, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 3 of the pre-attention LayerNorm gain at one will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because its scale can be learned equivalently by coordinate 3 of every QKV projection row.
change: Replace the eight learned pre-attention LayerNorm gains with seven learned coordinates, reconstructing gain coordinate 3 as a fixed one while preserving the existing initialization and anchored shift.
mechanism: Attention preprojection gain gauge fixing
evidence_used: The verified 1599-parameter design already absorbs six pre-MLP LayerNorm gains into its downstream projection, while removing `fc1.bias[8]` failed at 77.57%; applying the analogous exact gain redundancy on the previously untouched attention side is therefore the most informative alternative.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9882, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-attention LayerNorm gain coordinate 0 at one will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because unlike the 98.82%-accurate coordinate-3 attempt, it does not compound the existing fixed shift on the same coordinate.
change: Replace the eight learned pre-attention LayerNorm gains with seven learned gains, reconstructing gain coordinate 0 as one while leaving the existing coordinate-3 shift anchor unchanged.
mechanism: Distributed pre-attention gain gauge fixing
evidence_used: Fixing gain coordinate 3 reached 98.82%, close to the threshold, showing the gain redundancy is nearly viable; distributing the gain and shift constraints across different coordinates directly tests whether the remaining loss was caused by coordinate-3’s compounded affine anchoring.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7234999999999999, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `fc1.bias[9]` instead of `fc1.bias[8]` will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because the four remaining hidden units are architecturally symmetric but receive different fixed-seed weight initializations.
change: Retain learned biases for hidden units 8, 10, and 11 while reconstructing bias coordinates 0 through 7 and coordinate 9 as fixed zeros.
mechanism: Alternative hidden-unit bias anchoring
evidence_used: Fixing coordinates 0 through 7 retained 99.91% accuracy, whereas additionally fixing coordinate 8 produced 77.57%; testing a different remaining unit isolates whether that failure reflects the particular initialized row rather than insufficient three-bias capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `fc1.bias[10]` in addition to coordinates 0 through 7 and 9 will reduce the model from 1598 to 1597 parameters while retaining at least 99% accuracy, because fixing coordinate 9 achieved 100% accuracy and showed that performance depends on which initialized hidden-unit bias is removed rather than requiring all four previously remaining biases.
change: Retain learned biases only for hidden units 8 and 11, reconstructing bias coordinates 0 through 7, 9, and 10 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Second alternative hidden-unit bias anchoring
evidence_used: The 1598-parameter design reached 100% accuracy after fixing coordinate 9, whereas fixing coordinate 8 reached only 77.57%; testing coordinate 10 is the most direct next probe of whether another initialization-specific hidden bias is redundant.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1597, "training_steps": 4999}

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
