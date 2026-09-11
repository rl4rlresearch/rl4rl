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
verified_results: {"accuracy": 0.9998, "parameters": 1622, "training_steps": 4999}
prior_hypothesis: Constraining the seventh-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,623 to 1,622 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Reusing the first learned positional scalar as the penultimate row’s final coordinate will reduce the model from 1,630 to 1,629 parameters while retaining at least 99% accuracy, because that coordinate is a pure positional common-mode gauge but remains randomly initialized and dynamically learned.
change: Remove the penultimate positional row’s final scalar, reconstruct it from the first positional scalar, and preserve the successful final-row gauge initialization.
mechanism: Dynamic cross-position gauge tying
evidence_used: Fixing the second positional anchor locally reached 96.45%, while a learned LayerNorm tie reached 99.78% after its fixed counterpart reached only 37.17%; this motivates replacing the fixed second positional anchor with a learned tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the second learned positional scalar as the third-to-last row’s final coordinate will reduce the model from 1,629 to 1,628 parameters while retaining at least 99% accuracy, because it extends the successful randomly initialized learned positional tie to one adjacent row.
change: Remove the third-to-last positional row’s final scalar, reconstruct it from the second positional scalar, and retain the existing penultimate learned tie and final-row zero anchor.
mechanism: Incremental dynamic cross-position gauge tying
evidence_used: Replacing the penultimate row’s fixed-zero coordinate, which achieved only 96.45%, with a learned cross-position tie achieved 99.91% at 1,629 parameters; this directly motivates one incremental tie using a distinct learned scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the third learned positional scalar as the fourth-to-last row’s final coordinate will reduce the model from 1,628 to 1,627 parameters while retaining at least 99% accuracy.
change: Remove the fourth-to-last positional row’s final scalar, reconstruct it from the third positional scalar, and retain the two successful learned ties and final-row zero anchor.
mechanism: Incremental dynamic cross-position gauge tying
evidence_used: The first learned positional tie achieved 99.91% at 1,629 parameters, and extending it to an adjacent row with a distinct learned scalar again achieved 99.91% at 1,628; this directly motivates one further incremental tie using another distinct learned scalar.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5313, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the third learned positional scalar for the fourth-to-last row while blocking that row’s reverse gradient will reduce the model to 1,627 parameters and retain at least 99% accuracy by avoiding the cross-position gradient interference implicated by the prior 53.13% result.
change: Remove the fourth-to-last positional row’s final scalar, reconstruct it from a detached view of the third learned positional scalar, and retain the two successful learned ties and final-row zero anchor.
mechanism: Stop-gradient cross-position gauge tying
evidence_used: Two dynamic positional ties retained 99.91% accuracy at 1,629 and 1,628 parameters, but adding a third tie collapsed to 53.13%; isolating the new tie’s backward coupling tests whether optimization interference, rather than the gauge reduction itself, caused that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.27690000000000003, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing the fourth-to-last positional row in an orthonormal seven-dimensional zero-mean basis will reduce the model from 1,628 to 1,627 parameters while retaining at least 99% accuracy, because it removes only the exact common-mode gauge while preserving the randomly initialized function and avoiding cross-position gradient interference.
change: Compact the fourth-to-last positional row into seven orthonormal contrast coordinates, reconstruct its eight zero-mean coordinates during forward passes, and retain the two successful learned positional ties and final-row zero anchor.
mechanism: Orthonormal positional common-mode gauge elimination
evidence_used: Two learned cross-position gauge ties achieved 99.91%, but a third tie collapsed to 53.13% and blocking its reverse gradient fell to 27.69%; this motivates eliminating the same redundant scalar with an independent, orthonormal within-row gauge parameterization instead of another shared scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` bias coordinate at zero will reduce the model from 1,627 to 1,626 parameters while retaining at least 99% accuracy, because the following biased MLP projection can absorb that LayerNorm offset and zero initialization preserves the original initial function.
change: Add a bias-only anchored LayerNorm and use it for `ln2`, leaving its scale unconstrained and all existing successful reductions unchanged.
mechanism: Downstream-bias-absorbed LayerNorm anchoring
evidence_used: A single `ln1` bias anchor previously achieved 99.95% accuracy, while tying an additional coordinate within that same bias fell to 97.52%; applying one independent anchor to `ln2` tests the successful single-anchor mechanism without adding another within-vector tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1626, "training_steps": 4999}

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
