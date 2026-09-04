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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1498, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified final zero-bias query-coordinate chart will reduce the model from 1499 to 1498 learned parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987, "parameters": 1493, "training_steps": 4999}
prior_hypothesis: Fixing one off-diagonal shear between the second head’s two normalized zero-bias query coordinates will produce a 1493-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1495, "training_steps": 4999}
prior_hypothesis: Fixing a second query-bias coordinate in the second head will reduce the qualified 1496-parameter design to 1495 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1494, "training_steps": 4999}
prior_hypothesis: Normalizing the penultimate zero-bias query row in the second head will reduce the qualified 1495-parameter design to 1494 parameters while retaining at least 99% accuracy, because the same head’s final zero-bias query-row chart already trained successfully.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the qualified 1499-parameter design with a normalized chart for the zero-bias query coordinate will produce a 1498-parameter model with at least 99% accuracy.
change: Reproduce the verified balanced projection-row gauges, then normalize the final query-weight row and absorb its omitted scale into the matching key-weight row.
mechanism: Max-pivot query/key coordinate scale gauge
evidence_used: Independent normalized value/output gauges reached 99.35–99.64% at 1499 parameters, whereas additional zero-coordinate restrictions failed; the proposed reduction uses the analogous exact diagonal query/key symmetry on the query coordinate whose bias is already fixed to zero.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1498, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing one first-head query weight-and-bias coordinate and absorbing its scale into the matching key row will reduce the verified 1498-parameter model to 1497 parameters while maintaining at least 99% accuracy.
change: Add a max-pivot chart for the first head’s final affine query coordinate, remove its row and bias from the ordinary parameters, reconstruct them during inference, and transfer the omitted scale into the corresponding key row at initialization.
mechanism: First-head affine query/key scale gauge
evidence_used: The current query/key scale gauge achieved 99.93% accuracy at 1498 parameters; extending the same exact diagonal symmetry to an independent head is better supported than another ordinary value/output-row normalization, which fell to 67.77%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.42829999999999996, "parameters": 1497, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified final zero-bias query-coordinate chart will reduce the model from 1499 to 1498 learned parameters while retaining at least 99% accuracy.
change: Normalize the final query-weight row through a max-pivot chart, omit one learned coordinate, and absorb its initialization scale into the matching key row.
mechanism: Zero-bias query/key scale gauge
evidence_used: Reference Design 2 verified this exact 1498-parameter construction at 99.93% accuracy; the failed 1497-parameter affine extension indicates the proven zero-bias coordinate is the better-supported boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1498, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final query-bias coordinate in each head while applying the verified scale chart to the second head’s zero-bias query row will produce a 1497-parameter model with at least 99% accuracy.
change: Reproduce the qualified 1498-parameter query/key chart, additionally omit the first head’s final query-bias coordinate, and reconstruct both per-head omitted biases as zero.
mechanism: Independent per-head query-bias gauge plus qualified zero-bias query/key scale chart
evidence_used: The zero-bias query/key chart reached 99.90–99.93% at 1498 parameters, while the 1497 affine chart on a bias-bearing first-head coordinate reached only 42.83%; isolating the already-successful query-bias basis gauge in the first head tests whether the affine bias coupling caused that failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1497, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified zero-bias query/key scale gauge to the first head as well as the second will reduce the qualified 1497-parameter model to 1496 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified per-head query-bias and normalized value/output gauges, then normalize both heads’ final zero-bias query rows and absorb their omitted scales into the matching key rows.
mechanism: Dual zero-bias query/key scale charts
evidence_used: Reference Design 1 achieved 99.93% accuracy at 1497 parameters with both final query biases fixed and the second-head zero-bias query chart; the failed 1497 affine chart normalized a bias-bearing first-head coordinate, so removing that bias coupling makes the independent first-head scale gauge the most direct next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8273, "parameters": 1496, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second first-head query-bias coordinate will reduce the qualified model to 1496 parameters while retaining at least 99% accuracy, because the first head has unused query/key basis freedom and does not use the failed additional query-row scale chart.
change: Omit the first head’s penultimate query-bias coordinate and reconstruct the final two first-head coordinates and final second-head coordinate as zero.
mechanism: Additional first-head query-bias basis gauge
evidence_used: The current per-head query-bias gauge achieved 99.93% at 1497 parameters, while the 1496 dual-scale chart achieved only 82.73%; this tests an unused basis-direction gauge without imposing another normalized query-weight chart.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1496, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second query-bias coordinate in the second head will reduce the qualified 1496-parameter design to 1495 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified two-coordinate first-head bias gauge, then omit the penultimate second-head query bias so each head reconstructs its final two query-bias coordinates as zero.
mechanism: Balanced per-head query-bias basis gauge
evidence_used: Fixing a second first-head query-bias coordinate achieved 99.98% accuracy at 1496 parameters, whereas the alternative 1496 dual query-row scale chart reached only 82.73%; applying the successful basis gauge symmetrically to the second head is the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1495, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third first-head query-bias coordinate will reduce the qualified 1495-parameter design to 1494 parameters while retaining at least 99% accuracy, because the first head previously tolerated the additional bias-basis restriction at 99.98% accuracy and retains one free affine query-bias coordinate.
change: Reproduce the balanced two-coordinate-per-head bias gauge and additionally omit the first head’s second query-bias coordinate, reconstructing three trailing first-head coordinates and two trailing second-head coordinates as zero.
mechanism: Asymmetric third query-bias basis gauge
evidence_used: The balanced query-bias design reached 99.93% accuracy at 1495 parameters, and introducing the second fixed coordinate in the first head reached 99.98%; this makes extending the same first-head basis gauge the closest supported one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9851000000000001, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing the penultimate zero-bias query row in the second head will reduce the qualified 1495-parameter design to 1494 parameters while retaining at least 99% accuracy, because the same head’s final zero-bias query-row chart already trained successfully.
change: Reproduce the balanced per-head query-bias and projection gauges, then represent both trailing second-head query rows with independent max-pivot normalized charts and absorb their initialization scales into the matching key rows.
mechanism: Second-head dual zero-bias query/key scale charts
evidence_used: The 1495-parameter balanced design achieved 99.93%, including a successful second-head zero-bias query/key scale chart; the alternative third query-bias restriction reached only 98.51%, motivating another exact scale gauge within the already-successful head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing the first head’s penultimate zero-bias query row will reduce the verified 1494-parameter model to 1493 parameters while retaining at least 99% accuracy.
change: Add a max-pivot chart for the first head’s penultimate query row, omit one learned coordinate, and absorb its initialization scale into the matching key row.
mechanism: First-head penultimate zero-bias query/key scale chart
evidence_used: The corresponding second-head penultimate chart achieved 99.88% accuracy at 1494 parameters; testing the same row position in the other head is the closest supported reduction, while differing from the failed first-head final-row chart.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7448999999999999, "parameters": 1493, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the remaining trailing second-head query-bias coordinate while reproducing the qualified 1494-parameter dual query-row charts will yield 1493 parameters and retain at least 99% accuracy.
change: Reconstruct three trailing second-head query biases as zero and represent its final two query rows with independent max-pivot query/key scale charts.
mechanism: Asymmetric third query-bias gauge with qualified second-head dual scale charts
evidence_used: The second-head dual-chart design reached 99.88% at 1494 parameters, while the analogous first-head penultimate chart collapsed to 74.49%; testing the unused third bias gauge in the already-successful second head is the closest distinct reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4799, "parameters": 1493, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one off-diagonal shear between the second head’s two normalized zero-bias query coordinates will produce a 1493-parameter model with at least 99% accuracy.
change: Reproduce the qualified balanced query-bias and dual second-head query-row charts, then zero one penultimate-row coordinate by shearing it against the final query row and apply the inverse transformation to the matching key rows.
mechanism: Within-head zero-bias query/key shear-and-scale gauge
evidence_used: Reference Design 3 reached 99.88% accuracy at 1494 parameters with both trailing second-head query rows normalized. Unlike the failed 1493 first-head chart and third-bias restrictions, this removes an off-diagonal gauge wholly within the already-successful two-row, zero-bias subspace.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1493, "training_steps": 4999}



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
