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
verified_results: {"accuracy": 0.9976999999999999, "parameters": 1590, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,590-parameter design by fixing a sixth `ln2` bias coordinate will retain at least 99% accuracy because `fc1`’s independent biases absorb the removed LayerNorm offset.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1590, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,591-parameter design and fixing a sixth `ln2` bias coordinate will yield 1,590 parameters with at least 99% accuracy, because `fc1`’s independent biases can absorb that LayerNorm offset.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1593, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,594-parameter design and Helmert-parameterizing a second `fc2` weight column will yield 1,593 parameters with at least 99% accuracy, because each removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1588, "training_steps": 4999}
prior_hypothesis: Fixing the first and last `ln2` scale coordinates will yield 1,588 parameters with at least 99% accuracy because `fc1` can absorb both scales, while retaining trainable bias on the newly fixed leading coordinate may avoid the conditioning failure of fixing the last two adjacent scales.

## Recent verification evidence

RECENT RESULT
hypothesis: Reproducing the verified 1,593-parameter design and Helmert-parameterizing a third `fc2` weight column will yield 1,592 parameters with at least 99% accuracy, because each removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.
change: Apply the qualified seven-translation and three-position-row embedding compactions, then reconstruct the first three `fc2` weight columns from independent seven-dimensional zero-mean coordinates.
mechanism: Third activation-scaled residual-uniform MLP output gauge quotient
evidence_used: The verified two-column quotient achieved 99.96% accuracy at 1,593 parameters; extending the same exact output-space gauge to the adjacent third column is more directly supported than the failed third-value-row, fourth-position-row, or second-`fc1` reductions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.508, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,593-parameter design and Helmert-parameterizing the first attention output-projection column will yield 1,592 parameters with at least 99% accuracy, because its removed activation-scaled uniform residual component is eliminated by downstream LayerNorm.
change: Apply the qualified seven-translation, three-position-row, four-query-row, two-value-row, and two-`fc2`-column compactions, then reconstruct the first attention projection column from seven zero-mean coordinates.
mechanism: Attention-projection residual-uniform column quotient
evidence_used: The two-`fc2`-column residual-uniform quotient achieved 99.96% accuracy at 1,593 parameters. Since extending it to an adjacent third MLP column fell to 50.8%, applying the same exact output-space quotient in the attention projection tests an independent branch without further constraining the MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5284, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing one coordinate of the first head’s zero-bias query-row pair, with the matching orthogonal key-row rotation, will reduce the verified 1,593-parameter model to 1,592 parameters while retaining at least 99% accuracy because it preserves the initialized attention score matrix exactly.
change: Givens-rotate query rows 2–3 and matching key rows 2–3, omit the resulting zero query coordinate, and reconstruct that coordinate as zero during forward passes.
mechanism: Zero-bias attention Q/K rotation gauge
evidence_used: The current two-column `fc2` quotient achieved 99.96% at 1,593 parameters, while a third `fc2` column and an attention-projection column both failed; an independent internal Q/K factorization symmetry is therefore the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9470000000000001, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,593-parameter two-column `fc2` design and fixing a fourth `ln2` bias coordinate will yield 1,592 parameters with at least 99% accuracy, because the retained independent `fc1` biases absorb that LayerNorm offset without restricting the learned function.
change: Reconstruct the first two `fc2` columns from zero-mean coordinates and increase `ln2`’s fixed bias coordinates from three to four.
mechanism: Fourth downstream-affine LayerNorm-bias gauge
evidence_used: The two-column `fc2` quotient achieved 99.96% accuracy at 1,593 parameters. Unlike the failed third-`fc2` column, attention-projection column, and Q/K rotation, this tests an independent downstream-affine bias redundancy while preserving the qualified `fc2` layout.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,592-parameter design and fixing a fifth `ln2` bias coordinate will yield 1,591 parameters with at least 99% accuracy, because the independent `fc1` biases can absorb that LayerNorm offset without restricting the learned function.
change: Apply the qualified seven-translation, three-position-row, two-`fc2`-column compactions and increase `ln2`’s fixed bias coordinates from four to five.
mechanism: Fifth downstream-affine LayerNorm-bias gauge
evidence_used: The 1,592-parameter reference achieved 99.90% accuracy after fixing the fourth `ln2` bias coordinate, while reductions involving a third `fc2` column, an attention-projection column, and a Q/K rotation failed; extending the successful downstream-affine bias redundancy is the most directly supported next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,591-parameter design and fixing a sixth `ln2` bias coordinate will yield 1,590 parameters with at least 99% accuracy, because `fc1`’s independent biases can absorb that LayerNorm offset.
change: Apply the qualified seven-translation, three-position-row, four-query-row, two-value-row, and two-`fc2`-column compactions, then increase `ln2`’s fixed bias coordinates from five to six.
mechanism: Sixth downstream-affine LayerNorm-bias gauge
evidence_used: Fixing the fifth `ln2` bias coordinate achieved 99.89% accuracy at 1,591 parameters; extending that successful downstream-affine redundancy is better supported than the failed third-`fc2` column, second-`fc1` row, attention-projection column, or Q/K rotation reductions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a seventh `ln2` bias coordinate will reduce the verified 1,590-parameter model to 1,589 parameters while retaining at least 99% accuracy, because `fc1`’s independent biases can absorb the removed LayerNorm offset.
change: Increase `ln2`’s fixed bias coordinates from six to seven while preserving the qualified architecture and training configuration.
mechanism: Seventh downstream-affine LayerNorm-bias gauge
evidence_used: Successive fourth, fifth, and sixth `ln2` bias reductions achieved 99.90%, 99.89%, and 99.88% accuracy respectively; extending this repeatedly successful local redundancy is better supported than the failed third-`fc2` column, second-`fc1` row, attention-projection column, or Q/K rotation reductions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9717, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,590-parameter design by fixing a sixth `ln2` bias coordinate will retain at least 99% accuracy because `fc1`’s independent biases absorb the removed LayerNorm offset.
change: Increase `ln2`’s fixed bias coordinates from five to six, reducing the learned parameter count from 1,591 to 1,590.
mechanism: Sixth downstream-affine LayerNorm-bias gauge
evidence_used: The identical six-coordinate design previously achieved 99.88% accuracy with 1,590 parameters, while fixing a seventh coordinate fell to 97.17%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate in the verified six-bias-coordinate design will produce 1,589 parameters while retaining at least 99% accuracy, because the corresponding trainable `fc1` column absorbs the removed scale without reducing the learned function class.
change: Use the qualified six-coordinate `ln2` bias reduction, reconstruct one `ln2` scale coordinate as constant one, and make the compact first MLP row use the reconstructed full scale vector.
mechanism: Downstream-linear LayerNorm scale gauge quotient
evidence_used: The six-coordinate `ln2` design reached 99.88% and 99.77% accuracy at 1,590 parameters, whereas fixing a seventh bias coordinate fell to 97.17%; this tests an independent exact downstream-affine redundancy instead of repeating that failed bias reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,589-parameter design and fixing a second `ln2` scale coordinate will yield 1,588 parameters with at least 99% accuracy, because the corresponding trainable `fc1` column absorbs the removed scale without reducing the learned function class.
change: Apply the qualified six-coordinate `ln2` bias reduction, reconstruct two `ln2` scale coordinates as constant one, and make the compact first MLP row use the reconstructed full scale vector.
mechanism: Second downstream-linear LayerNorm scale gauge quotient
evidence_used: Fixing the first `ln2` scale coordinate achieved 99.96% accuracy at 1,589 parameters; extending the same downstream-linear scale redundancy is more directly supported than the failed seventh-bias, third-`fc2`-column, attention-projection, Q/K-rotation, or second-`fc1`-row reductions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9879000000000001, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the first and last `ln2` scale coordinates will yield 1,588 parameters with at least 99% accuracy because `fc1` can absorb both scales, while retaining trainable bias on the newly fixed leading coordinate may avoid the conditioning failure of fixing the last two adjacent scales.
change: Preserve the verified final-coordinate scale anchor, add a leading-coordinate anchor, and reconstruct the full scale consistently in `ln2` and the compact first MLP row.
mechanism: Nonadjacent downstream LayerNorm scale gauge quotient
evidence_used: Fixing one trailing `ln2` scale achieved 99.96% accuracy at 1,589 parameters, while fixing the two trailing scales reached 98.79%; selecting a nonadjacent second coordinate with trainable `ln2` bias tests whether that failure was specific to the adjacent fully bias-fixed coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinates 0, 1, and 7 will reduce the qualified 1,588-parameter design to 1,587 parameters while retaining at least 99% accuracy, because `fc1` absorbs the scales and coordinate 1 retains an independent trainable LayerNorm bias.
change: Reconstruct three `ln2` scale coordinates as constant one and use the reconstructed full scale in the compact first MLP row.
mechanism: Third downstream LayerNorm scale gauge with trainable-bias leading anchor
evidence_used: The nonadjacent coordinate-0-and-7 anchors achieved 99.99% accuracy at 1,588 parameters, whereas anchoring the two trailing, bias-fixed coordinates achieved only 98.79%; extending the leading side tests a third anchor whose bias remains trainable.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4777, "parameters": 1587, "training_steps": 4999}



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
