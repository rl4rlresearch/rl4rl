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
verified_results: {"accuracy": 0.9995, "parameters": 1605, "training_steps": 4999}
prior_hypothesis: Fixing a fourth `ln2` bias coordinate to zero will reduce the model from 1,606 to 1,605 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed additive contribution.

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining the first two `fc2` weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy.
change: Store and reconstruct the first two MLP output-projection weight columns using independent orthonormal contrast coordinates while preserving the anchored output bias.
mechanism: Second MLP-output common-mode gauge elimination
evidence_used: Constraining the first `fc2` column retained 99.93% accuracy at 1,615 parameters; extending the identical parameterization by one column is the closest controlled test of that successful mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4955, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying one additional value-projection bias coordinate will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy, because a value bias contributes a position-independent attention output that can be absorbed by the learned attention output bias modulo common-mode shifts removed by downstream LayerNorms.
change: Store six independent value-bias coordinates instead of seven, preserve the existing final-to-first tie, and reconstruct the penultimate coordinate by tying it to the second coordinate.
mechanism: Incremental value-bias/output-bias gauge elimination
evidence_used: Extending the `fc2` gauge to a second column collapsed to 49.55%, whereas the distinct attention output-bias gauge previously retained 99.89% accuracy; this motivates testing a one-parameter redundancy coupled directly to that successful output bias rather than further constraining `fc2`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the third-to-last value-projection bias coordinate to the third coordinate will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because the preceding symmetric value-bias tie retained 99.90% accuracy and a position-independent value bias remains absorbable by the attention output bias modulo downstream LayerNorm common-mode invariance.
change: Store five independent value-bias coordinates instead of six and reconstruct the eight-coordinate value bias symmetrically as `[a, b, c, d, e, c, b, a]`.
mechanism: Incremental symmetric value-bias gauge elimination
evidence_used: The immediately preceding value-bias reduction tied the penultimate coordinate to the second coordinate and achieved 99.90% accuracy at 1,614 parameters; extending the same symmetric tying pattern by one coordinate is the closest controlled next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the fifth value-projection bias coordinate to the fourth will reduce the model from 1,613 to 1,612 parameters while retaining at least 99% accuracy.
change: Store four independent value-bias coordinates and reconstruct the eight-coordinate bias symmetrically as `[a, b, c, d, d, c, b, a]`.
mechanism: Incremental symmetric value-bias gauge elimination
evidence_used: The two immediately preceding symmetric value-bias ties retained 99.90% and 99.93% accuracy at 1,614 and 1,613 parameters; extending the same parameterization by one coordinate is the closest controlled next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the fourth independent value-projection bias coordinate to the third will reduce the model from 1,612 to 1,611 parameters while retaining at least 99% accuracy.
change: Store three independent value-bias coordinates and reconstruct the eight-coordinate bias as `[a, b, c, c, c, c, b, a]`.
mechanism: Incremental symmetric value-bias gauge elimination
evidence_used: The three consecutive symmetric value-bias reductions retained 99.90%, 99.93%, and 99.98% accuracy at 1,614, 1,613, and 1,612 parameters; extending the same successful tying pattern by one coordinate is the closest controlled next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1732, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the mirrored central value-bias pair to zero will reduce the model from 1,612 to 1,611 parameters while retaining at least 99% accuracy, because its position-independent contribution can be absorbed by the attention output bias without forcing the third value-bias coordinate to serve four dimensions.
change: Store three independent value-bias coordinates and reconstruct the bias as `[a, b, c, 0, 0, c, b, a]`.
mechanism: Central value-bias anchoring
evidence_used: The four-coordinate symmetric value bias achieved 99.98% accuracy, while tying its central coordinate to the third collapsed to 17.32%; anchoring the central pair tests whether that failure came from destructive coordinate conflation rather than removing the redundant additive degree itself.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the next mirrored value-bias pair to zero will reduce the model from 1,611 to 1,610 parameters while retaining at least 99% accuracy, because value bias contributes a position-independent attention output absorbable by the learned output bias.
change: Store two independent value-bias coordinates and reconstruct the eight-coordinate bias as `[a, b, 0, 0, 0, 0, b, a]`.
mechanism: Expanded central value-bias anchoring
evidence_used: Anchoring the central mirrored pair retained 99.97% accuracy at 1,611 parameters, whereas conflating it with another coordinate collapsed accuracy; expanding the zero-anchored center tests another reduction without introducing destructive coordinate tying.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the remaining inner mirrored value-bias pair to zero will reduce the model from 1,610 to 1,609 parameters while retaining at least 99% accuracy.
change: Store one value-bias coordinate and reconstruct the eight-dimensional value bias as `[a, 0, 0, 0, 0, 0, 0, a]`.
mechanism: Expanded value-bias zero anchoring
evidence_used: Expanding the zero-anchored center from two to four coordinates retained 99.98% accuracy at 1,610 parameters, directly motivating the same incremental removal of the next mirrored pair without conflating learned coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final mirrored value-projection bias pair to zero will reduce the model from 1,609 to 1,608 parameters while retaining at least 99% accuracy.
change: Remove the final learned value-bias coordinate and reconstruct the entire eight-dimensional value bias as zeros.
mechanism: Complete value-bias zero anchoring
evidence_used: Successive zero anchoring of the central and inner mirrored value-bias pairs retained 99.97%, 99.98%, and 99.90% accuracy; the latest 1,609-parameter result directly motivates removing the sole remaining value-bias coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second `ln2` bias coordinate to zero will reduce the model from 1,608 to 1,607 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the unrestricted `fc1` bias.
change: Store six rather than seven `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final two coordinates as zeros.
mechanism: Incremental pre-MLP LayerNorm bias absorption
evidence_used: The 1,608-parameter design achieved 99.96% accuracy with one `ln2` bias coordinate already anchored; extending that constraint by one coordinate is a controlled reduction, and unlike the failed additional `fc2` column constraint, it precedes a linear layer with a learned bias that can absorb the removed additive degree.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9967, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln2` bias coordinate to zero will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed additive contribution.
change: Store five rather than six `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final three coordinates as zeros.
mechanism: Incremental pre-MLP LayerNorm bias absorption
evidence_used: Fixing the second `ln2` bias coordinate retained 99.67% accuracy at 1,607 parameters; extending the identical constraint by one coordinate is the closest controlled test of this successful redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fourth `ln2` bias coordinate to zero will reduce the model from 1,606 to 1,605 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed additive contribution.
change: Store four rather than five `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final four coordinates as zeros.
mechanism: Incremental pre-MLP LayerNorm bias absorption
evidence_used: Fixing the third `ln2` bias coordinate retained 99.97% accuracy at 1,606 parameters; extending the identical constraint by one coordinate is the closest controlled test of this successful redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1605, "training_steps": 4999}



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
