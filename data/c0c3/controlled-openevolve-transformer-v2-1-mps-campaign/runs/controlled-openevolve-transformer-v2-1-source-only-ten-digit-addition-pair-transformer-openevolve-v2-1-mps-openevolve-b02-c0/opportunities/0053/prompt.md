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
verified_results: {"accuracy": 0.9983, "parameters": 1618, "training_steps": 4999}
prior_hypothesis: Constraining the first two attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,619 to 1,618 parameters while retaining at least 99% accuracy, because the first-column constraint achieved 100% accuracy and each removed component produces only a token-wise common-mode residual shift erased by downstream LayerNorms.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying one `ln2` scale coordinate to its first learned coordinate will reduce the model from 1,626 to 1,625 parameters while retaining at least 99% accuracy, because the corresponding single dynamic scale tie in `ln1` achieved 99.78% and preserves the all-ones initialization.
change: Reuse `AnchoredLayerNorm` for `ln2`, retaining its successful zero bias anchor while dynamically reconstructing the final scale coordinate from the first.
mechanism: Independent post-attention LayerNorm scale tying
evidence_used: A single learned `ln1` scale tie met the threshold at 99.78%, and the independent `ln2` bias anchor reached 99.95% after an additional within-`ln1` bias tie failed; this motivates applying one proven scale tie to the separate `ln2` normalization rather than further constraining `ln1`.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7129000000000001, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing the attention output-projection bias in a seven-dimensional zero-mean orthonormal basis will reduce the model from 1,626 to 1,625 parameters while retaining at least 99% accuracy, because common-mode attention bias is removed by downstream LayerNorms and the orthonormal parameterization preserves initialization and avoids coordinate-anchor optimization distortion.
change: Add an output projection whose eight-dimensional bias is reconstructed from seven orthonormal contrast coordinates, and use it for the causal self-attention projection.
mechanism: Orthonormal attention-output common-mode gauge elimination
evidence_used: Orthonormal elimination of a positional common-mode gauge achieved 99.82% at 1,627 parameters after shared and stop-gradient parameterizations of the same gauge failed badly; this motivates applying the successful independent orthonormal quotient to the previously fragile attention-projection common-mode gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating all seven remaining value-projection bias parameters will reduce the model from 1,625 to 1,618 parameters while retaining at least 99% accuracy, because attention weights sum to one at the configured zero dropout, making the value bias a position-independent output offset already represented by the attention output bias up to the common mode removed by `ln2`.
change: Retain the eight learned query-bias coordinates while fixing every key and value bias coordinate to zero.
mechanism: Redundant value-bias elimination
evidence_used: The seven-dimensional orthonormal attention-output bias achieved 99.89% at 1,625 parameters, showing that the complete output-bias contrast space trains successfully; the remaining value bias only supplies another parameterization of that same downstream offset.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6767, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fifth-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,625 to 1,624 parameters while retaining at least 99% accuracy.
change: Compact two adjacent positional rows into independent orthonormal contrast coordinates, preserving the existing learned ties and final-row anchor.
mechanism: Independent orthonormal positional gauge extension
evidence_used: The first orthonormal positional gauge achieved 99.82% at 1,627 parameters, whereas adding a third cross-position tie collapsed to 53.13%; an independent orthonormal gauge extends the successful parameterization without cross-position gradient coupling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the sixth-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,624 to 1,623 parameters while retaining at least 99% accuracy.
change: Compact three adjacent positional rows into independent orthonormal contrast coordinates while preserving the two learned positional ties and final-row zero anchor.
mechanism: Independent orthonormal positional gauge extension
evidence_used: Extending from one to two independently parameterized orthonormal positional gauges improved the verified result from 99.82% at 1,627 parameters to 99.93% at 1,624 parameters, whereas adding another cross-position tie failed; this motivates extending the successful independent parameterization by one row.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the seventh-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,623 to 1,622 parameters while retaining at least 99% accuracy.
change: Compact four adjacent positional rows into independent orthonormal contrast coordinates while preserving the two learned positional ties and final-row zero anchor.
mechanism: Incremental independent orthonormal positional gauge extension
evidence_used: Extending independent orthonormal positional gauges from one row to two and then three retained 99.82%, 99.93%, and 99.94% accuracy respectively, while shared and stop-gradient alternatives failed; this motivates one further independent extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the eighth-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,622 to 1,621 parameters while retaining at least 99% accuracy.
change: Compact five adjacent positional rows into independent orthonormal contrast coordinates while preserving the two learned positional ties and final-row zero anchor.
mechanism: Incremental independent orthonormal positional gauge extension
evidence_used: Extending independent orthonormal positional gauges from one through four rows retained 99.82%, 99.93%, 99.94%, and 99.98% accuracy respectively; this directly motivates one further incremental extension of the same parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1621, "training_steps": 4999}

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
