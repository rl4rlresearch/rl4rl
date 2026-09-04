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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1613, "training_steps": 4999}
prior_hypothesis: Tying the third-to-last value-projection bias coordinate to the third coordinate will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because the preceding symmetric value-bias tie retained 99.90% accuracy and a position-independent value bias remains absorbable by the attention output bias modulo downstream LayerNorm common-mode invariance.

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining the ninth-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,621 to 1,620 parameters while retaining at least 99% accuracy.
change: Compact six adjacent positional rows into independent orthonormal contrast coordinates while preserving the two learned positional ties and final-row zero anchor.
mechanism: Incremental independent orthonormal positional gauge extension
evidence_used: Extending independent orthonormal positional gauges from one through five rows retained 99.82%, 99.93%, 99.94%, 99.98%, and 99.83% accuracy respectively; this directly motivates one further incremental extension of the same successful parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the tenth-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,620 to 1,619 parameters while retaining at least 99% accuracy.
change: Compact seven adjacent positional rows into independent orthonormal contrast coordinates while preserving the two learned positional ties and final-row zero anchor.
mechanism: Incremental independent orthonormal positional gauge extension
evidence_used: Extending independent orthonormal positional gauges from one through six rows consistently retained at least 99.82% accuracy, with the sixth extension reaching 99.93% at 1,620 parameters; this directly motivates one further incremental extension.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8177, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining positional row 1 to its seven-dimensional zero-mean subspace will reduce the model from 1,620 to 1,619 parameters while retaining at least 99% accuracy, indicating that the failed tenth-to-last-row extension was position-specific rather than a general limit of orthonormal gauge removal.
change: Compact positional row 1 into independent orthonormal contrast coordinates while preserving the six successful late-row gauges, two learned ties, and final-row anchor.
mechanism: Nonadjacent orthonormal positional gauge elimination
evidence_used: Six consecutive orthonormal positional gauge reductions retained 99.82%–99.98% accuracy, including 99.93% at 1,620 parameters, while only the adjacent tenth-to-last row failed at 81.77%; testing a nonadjacent early row isolates positional sensitivity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7354, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the first attention output-projection weight column to its seven-dimensional zero-mean subspace will reduce the model from 1,620 to 1,619 parameters while retaining at least 99% accuracy, because the removed component produces only token-wise common-mode residual shifts erased by subsequent LayerNorms.
change: Store the first attention output-projection weight column as seven orthonormal contrast coordinates and reconstruct the full zero-mean column during forward passes, while retaining the successful zero-mean output bias.
mechanism: Orthonormal attention-output weight gauge elimination
evidence_used: The orthonormal attention-output bias gauge retained 99.89% accuracy at 1,625 parameters, demonstrating that this projection tolerates orthonormal common-mode quotienting; this applies the same successful parameterization to one weight column rather than extending the position-specific gauges that failed at 1,619 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the first two attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,619 to 1,618 parameters while retaining at least 99% accuracy, because the first-column constraint achieved 100% accuracy and each removed component produces only a token-wise common-mode residual shift erased by downstream LayerNorms.
change: Store and reconstruct the first two attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the existing zero-mean output bias.
mechanism: Second attention-output weight gauge elimination
evidence_used: The immediately prior first-column orthonormal gauge reduction reached 100% accuracy at 1,619 parameters; extending the identical parameterization to one additional column is the closest controlled test of that successful mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the first three attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy.
change: Store and reconstruct the first three attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the successful zero-mean output bias.
mechanism: Third attention-output weight gauge elimination
evidence_used: Extending this exact orthonormal gauge from one column at 100% accuracy and 1,619 parameters to two columns retained 99.83% accuracy at 1,618 parameters, directly motivating one further incremental extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the first four attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,617 to 1,616 parameters while retaining at least 99% accuracy.
change: Store and reconstruct the first four attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the successful zero-mean output bias.
mechanism: Fourth attention-output weight gauge elimination
evidence_used: The same incremental gauge reduction retained 100% accuracy with one constrained column, 99.83% with two, and 99.89% with three at 1,617 parameters; extending it by one column is the closest controlled next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the first five attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,616 to 1,615 parameters while retaining at least 99% accuracy.
change: Store and reconstruct the first five attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the successful zero-mean output bias.
mechanism: Fifth attention-output weight gauge elimination
evidence_used: The identical incremental gauge retained 100%, 99.83%, 99.89%, and 99.94% accuracy with one through four constrained columns respectively; the latest result met the requirement at 1,616 parameters, directly motivating one further column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7399, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the first MLP output-projection weight column to its seven-dimensional zero-mean subspace will reduce the model from 1,616 to 1,615 parameters while retaining at least 99% accuracy.
change: Store the first `fc2` weight column as orthonormal contrast coordinates and reconstruct it during forward passes, while preserving the existing anchored output bias.
mechanism: MLP-output common-mode gauge elimination
evidence_used: Four attention output-projection columns accepted the identical common-mode quotient with 99.94% accuracy at 1,616 parameters; applying it to the separate MLP output projection tests the same successful invariance without extending into the fifth attention column that collapsed to 73.99%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1615, "training_steps": 4999}

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
