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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1497, "training_steps": 4999}
prior_hypothesis: Fixing the final query-bias coordinate in each head while applying the verified scale chart to the second head’s zero-bias query row will produce a 1497-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1498, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1499-parameter design with a normalized chart for the zero-bias query coordinate will produce a 1498-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1498, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified final zero-bias query-coordinate chart will reduce the model from 1499 to 1498 learned parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1501, "training_steps": 4999}
prior_hypothesis: Fixing a second relative attention-output coordinate in the second head will reduce the qualified 1502-parameter design to 1501 parameters while maintaining at least 99% accuracy, because each head admits an independent invertible value/output basis change and a well-conditioned two-pivot initialization preserves the represented function.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one disjoint relative projection coordinate in another row of the first attention head will reduce the model from 1501 to 1500 parameters while retaining at least 99% accuracy, because a best-pivot scalar shear preserves the initialized function without the three-coordinate same-row conditioning that reached only 98.82%.
change: Add a six-coordinate auxiliary projection-row parameter, reconstruct its omitted third-from-last relative coordinate as zero, and initialize it through a function-preserving one-pivot value/output basis change.
mechanism: Secondary-row scalar value/output basis gauge
evidence_used: The balanced two-coordinate-per-head design achieved 100% at 1501 parameters, while extending one row to three simultaneously fixed coordinates scored 98.82%; this tests an unused, independently conditioned scalar gauge on a different row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6437, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the second attention head from one to two function-preserving projection-coordinate gauges will reduce the model to 1501 parameters while retaining at least 99% accuracy.
change: Store five relative coordinates for each head’s final projection row, reconstruct two omitted coordinates per row, and initialize the second head with the same conditioned two-pivot basis change already used for the first head.
mechanism: Balanced per-head value/output basis gauge fixing
evidence_used: Reference Design 3 verified this balanced construction at 100% accuracy with 1501 parameters; both tested 1500-parameter extensions fell below 99%, making the qualified 1501 design the strongest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1501, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the failed three-pivot shear with a norm-preserving SVD basis rotation will reduce the balanced 1501-parameter design to 1500 parameters while retaining at least 99% accuracy.
change: Fix three relative projection coordinates in the first attention head and two in the second; use an orthogonal first-head initialization transform and the verified conditioned two-pivot transform for the second head.
mechanism: Orthogonal nullspace value/output basis gauge
evidence_used: The balanced two-coordinate-per-head design reached 100% at 1501 parameters, while the three-coordinate pivot construction narrowly missed at 98.82%; an orthogonal transform tests whether conditioning, rather than the additional gauge itself, caused that loss.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8768, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: A scalar value/output gauge on the balanced 1501-parameter model will reduce it to 1500 parameters while retaining at least 99% accuracy, because normalizing one already-gauged projection row and absorbing its scale into the matching value row preserves the initialized function without imposing a third zero-coordinate constraint.
change: Reproduce the verified final-scale, query-bias, embedding, and balanced two-coordinate projection quotients, then represent the first head’s gauged projection row by four max-pivot chart coordinates with fixed norm and transfer its omitted scale into the corresponding value weight.
mechanism: Max-pivot normalized value/output scale gauge
evidence_used: The balanced two-coordinate-per-head design achieved 100% at 1501 parameters, while adding a third zero coordinate reached only 98.82% and its SVD variant reached 87.68%; this tests an unused diagonal value/output symmetry with a bounded max-pivot chart instead of another projection-direction restriction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified normalized scalar value/output gauge independently to the second attention head will reduce the model from 1500 to 1499 parameters while retaining at least 99% accuracy.
change: Store both gauged projection rows with four chart coordinates and fixed norm, absorb each omitted scale into its matching value-weight row at initialization, and propagate both pivot indices through optimizer-side reconstruction.
mechanism: Independent second-head max-pivot value/output scale gauge
evidence_used: The current first-head scalar gauge achieved 99.94% accuracy at 1500 parameters, while prior third-coordinate restrictions failed; the second head has the same independent diagonal value/output symmetry and therefore offers the closest evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1499, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing one ordinary first-head projection row and absorbing its scale into the matching value-weight row will reduce the qualified 1499-parameter model to 1498 parameters while maintaining at least 99% accuracy.
change: Apply the verified normalized scalar gauges to both target rows, then use the same function-preserving chart for one additional first-head projection row and propagate its pivot through reconstruction and optimization.
mechanism: Additional max-pivot value/output scale gauge
evidence_used: Independent scalar gauges achieved 99.94% at 1500 parameters and 99.64% at 1499, while additional zero-coordinate constraints failed; extending the successful scale symmetry is therefore the most supported next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6777, "parameters": 1498, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing both two-coordinate-gauged projection rows and absorbing their scales into the matching value rows will produce a 1499-parameter model with at least 99% accuracy.
change: Replace both five-coordinate projection-row parameters with four-coordinate normalized charts, preserve their pivot indices, and use chart-aware reconstruction during training.
mechanism: Independent per-head max-pivot value/output scale gauges
evidence_used: Reference Design 2 verified this exact construction at 99.64% accuracy with 1499 parameters; the preceding single-head scalar gauge also reached 99.94% at 1500 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9934999999999999, "parameters": 1499, "training_steps": 4999}

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
