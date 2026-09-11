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
verified_results: {"accuracy": 0.9998, "parameters": 1488, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1489-parameter mutual-shear design and normalizing the untouched final zero-bias query row of the first head will yield 1488 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1489, "training_steps": 4999}
prior_hypothesis: Shearing the untouched second-head bias-bearing query row against the already-successful dual-sheared bias-bearing row will reduce Reference Design 3 from 1490 to 1489 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9997, "parameters": 1487, "training_steps": 4999}
prior_hypothesis: Shearing the normalized first-head terminal zero-bias query row against its neighboring zero-bias row will reduce Reference Design 3 from 1488 to 1487 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1491, "training_steps": 4999}
prior_hypothesis: Shearing the second head’s last bias-bearing query row against its normalized zero-bias target row will reduce the model from 1492 to 1491 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified 1491-parameter construction with an exact shear from the remaining bias-bearing second-head query row into its normalized zero-bias target row will yield 1490 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified two-coordinate scale/shear chart and neighboring bias-preserving shear, then omit the target-pivot coordinate from the other bias-bearing query row and absorb its inverse shear into the target key row.
mechanism: Second bias-preserving query/key shear
evidence_used: Reference Design 2 reached 99.95% accuracy at 1491 parameters after the first bias-preserving shear; the remaining second-head bias-bearing row admits the same transformation without imposing the failed additional query-bias restriction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7182999999999999, "parameters": 1490, "training_steps": 4999}

RECENT RESULT
hypothesis: Shearing the already-successful second-head bias-bearing query row against both normalized zero-bias rows will reduce the verified 1491-parameter model to 1490 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified two-coordinate scale/shear chart, then omit both target-pivot coordinates from the same neighboring bias-bearing query row and absorb the inverse shears into the corresponding key rows.
mechanism: Dual-target bias-preserving query/key shear
evidence_used: The first bias-preserving shear on this row achieved 99.95% at 1491 parameters, whereas the failed 1490 design changed the other bias-bearing row and reached 71.83%; using the unused zero-bias target with the already-successful row isolates a distinct exact gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1490, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1490-parameter dual-target query construction with a third value/output shear on the second head will produce 1489 parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 3’s qualified query scale/shear gauges, then zero a third relative output coordinate in the second-head target value row and absorb the inverse transformation into the corresponding value weights.
mechanism: Third conditioned second-head value/output shear
evidence_used: Reference Design 3 achieved 99.91% accuracy at 1490 parameters, while conditioned two-coordinate value/output shears remained successful throughout the qualified designs; extending that untouched exact gauge is distinct from the failed restriction on the other bias-bearing query row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8123999999999999, "parameters": 1489, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying one exact shear from the first head’s neighboring bias-bearing query row into its freely learned zero-bias target row will reduce the model to 1489 parameters while retaining at least 99% accuracy.
change: Omit one pivot coordinate from the first head’s second query row, reconstruct it as zero, and absorb the inverse initialization shear into the matching key row.
mechanism: First-head bias-preserving query/key shear
evidence_used: The analogous second-head bias-preserving shears achieved 99.91% at 1490 parameters. This tests the same successful gauge in the untouched first head without imposing the failed first-head target-row normalization or the failed third value/output shear.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7667, "parameters": 1489, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing the combined relative weights and learned bias of the already-successful dual-sheared second-head query row will reduce the qualified 1490-parameter construction to 1489 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified dual-target bias-preserving shears, then omit one additional query-weight coordinate by fixing an affine-row scale chart and absorbing its initialization scale into the matching key row.
mechanism: Affine query/key scale gauge on the dual-sheared second-head row
evidence_used: The dual-target shear on this same second-head bias-bearing row achieved 99.91% at 1490 parameters, while changing the other bias-bearing row, the first head, or the value/output chart failed; an exact diagonal gauge on the successful row is the closest distinct reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8501000000000001, "parameters": 1489, "training_steps": 4999}

RECENT RESULT
hypothesis: Shearing the untouched second-head bias-bearing query row against the already-successful dual-sheared bias-bearing row will reduce Reference Design 3 from 1490 to 1489 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified second-head scale/shear construction, then omit one coordinate from the remaining bias-bearing query row and absorb the inverse shear into the matching key row.
mechanism: Mutual bias-bearing query/key shear
evidence_used: Reference Design 3 achieved 99.91% at 1490 parameters. Unlike failed reductions involving the other bias-bearing row and a zero-bias target, this uses a mutual shear between two freely biased rows, so both transformed biases remain learned and no additional bias or scale restriction is imposed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1489, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1489-parameter construction with the complementary shear between the same two freely biased second-head query rows will produce 1488 parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 1, then eliminate one additional coordinate from its previously sheared bias-bearing query row using the other bias-bearing row, with compensating zero-bias-row components and the inverse key transformation.
mechanism: Conditioned complementary mutual query/key shear
evidence_used: Reference Design 1 achieved 99.99% at 1489 parameters using the first mutual bias-bearing shear, whereas reductions involving another bias restriction, the first head, an affine scale, or a third value/output shear failed; the complementary gauge stays within the successful freely biased pair.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7104, "parameters": 1488, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing only the relative query weights of the newly qualified mutually sheared anchor row, while leaving its query bias independently learned, will reduce the model to 1488 parameters and retain at least 99% accuracy.
change: Add a max-pivot scale chart for the second head’s anchor query row, omit one learned weight coordinate, and transfer the initialization scale into the matching key row.
mechanism: Bias-free diagonal query/key scale chart on the mutually sheared anchor row
evidence_used: The current mutual bias-bearing shear reached 99.99% accuracy at 1489 parameters, and diagonal query/key scale charts already succeed on both zero-bias target rows. Unlike the failed affine chart, this construction does not normalize or couple the learned query bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8371, "parameters": 1488, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1489-parameter mutual-shear design and normalizing the untouched final zero-bias query row of the first head will yield 1488 parameters while retaining at least 99% accuracy.
change: Add Reference Design 1’s mutual bias-bearing second-head shear, then omit one coordinate and the scalar scale from the first head’s final query row while transferring its initialization scale to the matching key row.
mechanism: First-head terminal query/key scale gauge atop the qualified mutual-shear chart
evidence_used: Reference Design 1 achieved 99.99% accuracy at 1489 parameters. Although a first-head penultimate-row chart failed, the terminal zero-bias row remains untested, making this a row-specific test of the same exact diagonal query/key gauge already successful on second-head zero-bias rows.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1488, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1488-parameter design with a shear between the first head’s two zero-bias query rows will yield 1487 parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 3, then omit one coordinate from the first head’s penultimate query row and absorb the inverse shear into the normalized terminal row’s key weights.
mechanism: First-head zero-bias query/key shear
evidence_used: Reference Design 3 achieved 99.98% accuracy at 1488 parameters with the first-head terminal row normalized; analogous zero-bias query/key shears in the second head retained at least 99.87%, while failed first-head experiments altered a scale or a bias-bearing row rather than this remaining off-diagonal gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9709, "parameters": 1487, "training_steps": 4999}

RECENT RESULT
hypothesis: Shearing the normalized first-head terminal zero-bias query row against its neighboring zero-bias row will reduce Reference Design 3 from 1488 to 1487 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified 1488-parameter construction, then omit one terminal-row coordinate using the complementary first-head shear and absorb the inverse transformation into the corresponding key rows.
mechanism: Complementary first-head terminal-row query/key shear
evidence_used: Reference Design 3 achieved 99.98% at 1488 parameters; the opposite first-head shear reached 97.09%, while both directions of the analogous second-head zero-bias gauge succeeded, making the untested complementary direction the closest distinct reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1487, "training_steps": 4999}

RECENT RESULT
hypothesis: Shearing one freely biased first-head query row against the other will reduce the verified 1487-parameter model to 1486 parameters while retaining at least 99% accuracy.
change: Omit one pivot coordinate from the first head’s leading biased query row and apply the inverse initialization shear to the other biased row’s key weights.
mechanism: First-head mutual bias-bearing query/key shear
evidence_used: The analogous mutual shear between freely biased second-head rows achieved 99.99% at 1489 parameters; unlike the failed first-head bias-to-zero-bias shear, this preserves independently learned biases on both transformed rows.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9185, "parameters": 1486, "training_steps": 4999}



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
