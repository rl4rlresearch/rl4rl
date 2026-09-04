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
verified_results: {"accuracy": 0.9991, "parameters": 1599, "training_steps": 4999}
prior_hypothesis: Fixing coordinate 7 of `fc1.bias` at zero will reduce the model from 1600 to 1599 parameters while retaining at least 99% accuracy, because coordinates 0 through 6 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing coordinate 3 of `fc2.bias` at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the sole block is followed immediately by final LayerNorm, which makes any common shift across all MLP output coordinates unobservable.
change: Replace the eight-parameter second MLP bias with seven learned coordinates and reconstruct coordinate 3 as a fixed zero during the forward pass.
mechanism: Final-LayerNorm common-shift gauge fixing
evidence_used: The 1607-parameter design reached 99.91% accuracy, while the next attention-side bias constraint collapsed to 36.34%; this motivates testing a distinct exact null direction, using coordinate 3 because its pre-MLP gain and bias constraints both previously retained at least 99.9% accuracy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.59, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 0 of `fc1.bias` at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because either remaining learned pre-MLP LayerNorm shift can absorb that scalar while the other hidden-unit biases compensate for its distributed effect.
change: Replace the first MLP projection with an initialization-equivalent linear layer whose bias has eleven learned coordinates and a fixed-zero coordinate 0.
mechanism: Reverse pre-MLP shift-to-bias gauge fixing
evidence_used: The 1607-parameter design achieved 99.91% accuracy while pre-MLP bias coordinates 2 and 5 remained learned; attempts to fix those shifts failed, so this patch preserves both sensitive coordinates and tests the reverse redundancy after six other pre-MLP shifts were successfully absorbed by `fc1.bias`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 1 of `fc1.bias` at zero will reduce the model from 1606 to 1605 parameters while retaining at least 99% accuracy, because the second remaining learned pre-MLP LayerNorm shift provides another independent degree of freedom for absorbing a hidden-unit bias constraint.
change: Replace the first MLP projection’s eleven learned bias coordinates with ten, reconstructing bias coordinates 0 and 1 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Second reverse pre-MLP shift-to-bias gauge fixing
evidence_used: Fixing `fc1.bias` coordinate 0 achieved 99.96% accuracy at 1606 parameters; hidden units are initialization-symmetric, and two pre-MLP LayerNorm shifts remain learned, motivating the corresponding second constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the first shared embedding anchor at zero will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because subtracting the same scalar from every token and position embedding only adds a common-coordinate shift before LayerNorm and a vocabulary-wide common logit offset.
change: Replace the eight learned embedding anchors with seven learned anchors and one fixed-zero anchor, applying the corresponding function-preserving common shift during initialization.
mechanism: Joint token-position common-offset gauge fixing
evidence_used: The current 1605-parameter design achieved 99.95% accuracy while already quotienting eight token-position embedding gauges; the remaining simultaneous common offset is a distinct exact redundancy that does not further constrain the sensitive LayerNorm or MLP-bias coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7189, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 2 of `fc1.bias` at zero will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because coordinates 0 and 1 were removed independently with 99.96% and 99.95% accuracy, and the identically initialized hidden units remain symmetric.
change: Replace the first MLP projection’s ten learned bias coordinates with nine, reconstructing bias coordinates 0 through 2 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Third hidden-unit bias constraint
evidence_used: Consecutively fixing `fc1.bias` coordinates 0 and 1 retained 99.96% and 99.95% accuracy; after the distinct shared-embedding reduction failed at 71.89%, extending the demonstrated MLP-bias trend is the most direct test of whether its practical redundancy continues beyond the two exact LayerNorm-shift degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 3 of `fc1.bias` at zero will reduce the model from 1604 to 1603 parameters while retaining at least 99% accuracy, because coordinates 0 through 2 were removed consecutively with at least 99.95% accuracy and all hidden units use symmetric initialization.
change: Replace the first MLP projection’s nine learned bias coordinates with eight, reconstructing bias coordinates 0 through 3 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Fourth hidden-unit bias constraint
evidence_used: The current design achieved 99.98% accuracy after fixing `fc1.bias` coordinate 2, extending two earlier successful removals; testing coordinate 3 is the most direct probe of whether this demonstrated hidden-bias redundancy continues.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 4 of `fc1.bias` at zero will reduce the model from 1603 to 1602 parameters while retaining at least 99% accuracy, because coordinates 0 through 3 were removed consecutively with at least 99.95% accuracy and the hidden units have symmetric initialization.
change: Replace the first MLP projection’s eight learned bias coordinates with seven, reconstructing bias coordinates 0 through 4 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Fifth hidden-unit bias constraint
evidence_used: The current 1603-parameter design achieved 99.95% accuracy after fixing `fc1.bias` coordinate 3, continuing three earlier successful consecutive removals; extending this established sequence is the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 5 of `fc1.bias` at zero will reduce the model from 1602 to 1601 parameters while retaining at least 99% accuracy, because coordinates 0 through 4 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.
change: Replace the first MLP projection’s seven learned bias coordinates with six, reconstructing bias coordinates 0 through 5 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Sixth hidden-unit bias constraint
evidence_used: The current design achieved 99.87% accuracy after fixing `fc1.bias` coordinate 4, following four consecutive successful removals; extending this established sequence is the most direct test of the remaining hidden-bias capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 6 of `fc1.bias` at zero will reduce the model from 1601 to 1600 parameters while retaining at least 99% accuracy, because coordinates 0 through 5 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.
change: Replace the first MLP projection’s six learned bias coordinates with five, reconstructing bias coordinates 0 through 6 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Seventh hidden-unit bias constraint
evidence_used: The current design achieved 99.92% accuracy after fixing `fc1.bias` coordinate 5, extending five earlier consecutive successful removals; continuing this established sequence is the most informative next capacity test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 7 of `fc1.bias` at zero will reduce the model from 1600 to 1599 parameters while retaining at least 99% accuracy, because coordinates 0 through 6 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.
change: Replace the first MLP projection’s five learned bias coordinates with four, reconstructing bias coordinates 0 through 7 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Eighth hidden-unit bias constraint
evidence_used: The current 1600-parameter design achieved 99.77% accuracy after fixing `fc1.bias` coordinate 6, extending six earlier consecutive successful removals; continuing this established sequence is the most informative next capacity test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 8 of `fc1.bias` at zero will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because coordinates 0 through 7 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.
change: Replace the first MLP projection’s four learned bias coordinates with three, reconstructing bias coordinates 0 through 8 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Ninth hidden-unit bias constraint
evidence_used: Fixing `fc1.bias` coordinate 7 achieved 99.91% accuracy at 1599 parameters, extending seven earlier consecutive successful removals; continuing this established sequence is the most informative next capacity test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7757, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing `fc2.bias` in an orthonormal seven-dimensional zero-sum basis will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because it removes the exact common-shift null direction without the asymmetric optimization geometry of fixing one output coordinate.
change: Replace the ordinary eight-parameter `fc2` bias with seven learned orthonormal quotient coordinates whose reconstructed bias sums to zero.
mechanism: Orthonormal final-LayerNorm common-shift quotient
evidence_used: Fixing `fc2.bias` coordinate 3 failed at 59% despite the common-shift invariance of the final LayerNorm; the current design reaches 99.91%, so a minimal-norm orthonormal parameterization directly tests whether that failure arose from the coordinate-anchored optimization geometry rather than loss of functional capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0298, "parameters": 1598, "training_steps": 4999}



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
