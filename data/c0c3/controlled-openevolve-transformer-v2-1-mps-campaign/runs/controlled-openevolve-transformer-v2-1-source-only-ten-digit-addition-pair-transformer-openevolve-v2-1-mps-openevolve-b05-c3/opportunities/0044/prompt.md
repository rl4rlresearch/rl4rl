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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9983, "parameters": 1602, "training_steps": 4999}
prior_hypothesis: Learning five query-bias coordinates while retaining the qualified gauge-aware positional parameterization will produce 1602 parameters and maintain at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1599, "training_steps": 4999}
prior_hypothesis: Removing one first-key-row weight along the exact LayerNorm input-shift symmetry will produce 1599 learned parameters and retain at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9989, "parameters": 1597, "training_steps": 4999}
prior_hypothesis: Reconstructing the third query-bias coordinate as the mean of two learned coordinates will produce 1,597 parameters and maintain at least 99% accuracy by preserving three active query offsets.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing the qualified trailing `fc1` bias and quotienting unit 10’s eight effective normalized-input coefficients to seven will produce 1599 learned parameters and retain at least 99% accuracy.
change: Restore the qualified four-coordinate query bias, retain eleven learned `fc1` thresholds, and represent unit 10 directly on the zero-mean LayerNorm coordinates with one omitted gauge coefficient and full-space-equivalent AdamW moments and clipping.
mechanism: LayerNorm-hyperplane quotient for one MLP input row
evidence_used: The 1600-parameter design with one fixed `fc1` bias achieved 99.91%, while removing a second threshold missed at 98.57%; the unverified unit-10 weight-quotient proposal therefore remains the most informative way to remove one scalar without sacrificing that unit’s learned threshold.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Removing one first-key-row weight along the exact LayerNorm input-shift symmetry will produce 1599 learned parameters and retain at least 99% accuracy.
change: Adopt the qualified positional quotient, four-coordinate query bias, and fixed trailing `fc1` bias, then gauge-fix one key-projection coefficient while preserving full-space initialization, AdamW moments, weight decay, and clipping.
mechanism: LayerNorm-hyperplane key-weight quotient with gauge-aware AdamW
evidence_used: The 1600-parameter reference achieved 99.91%, while direct 1599-parameter capacity ablations failed; the positional quotient succeeded with gauge-aware optimization, and key-vector constant offsets cancel exactly in softmax attention without the threshold coupling that complicated the unverified MLP-weight quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the final input coefficient of the second key-projection row will reduce the verified model from 1599 to 1598 learned parameters while retaining at least 99% accuracy, because each key row independently has the same exact input-shift symmetry already validated for the first row.
change: Extend the key projection representation, initialization, full-space AdamW update, and gradient clipping from one omitted key-row coefficient to two.
mechanism: Second LayerNorm-hyperplane key-weight quotient
evidence_used: The current first-key-row quotient achieved 99.88% at 1599 parameters, whereas multiple direct 1599-parameter capacity ablations failed; this directly motivates applying the same successful exact symmetry to one additional key row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2961, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified first-key-row gauge quotient will yield 1,599 learned parameters and at least 99% accuracy.
change: Gauge-fix one coefficient in the first key-projection row while preserving full-shape initialization, AdamW moments, weight decay, and gradient clipping.
mechanism: Single key-row LayerNorm-hyperplane quotient
evidence_used: Reference Design 3 achieved 99.88% accuracy with 1,599 parameters; the failed second-row extension shows the validated single quotient should be retained without further gauge removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9948999999999999, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding an exact common-shift gauge for the MLP output bias to the qualified single-key-row quotient will produce 1,598 learned parameters and retain at least 99% accuracy.
change: Adopt the verified four-query-bias, trailing-`fc1`-bias, and first-key-row reductions, then represent each `fc2` bias with seven learned relative coordinates and preserve full-space AdamW and gradient-clipping dynamics for all gauges.
mechanism: Final-residual channel-shift quotient
evidence_used: The single key-row quotient achieved 99.88% at 1,599 parameters; the failed second-key-row extension motivates removing an orthogonal redundancy instead. The `fc2` bias is immediately upstream of the final LayerNorm, which exactly removes any channel-common residual shift.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.611, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing unit 10’s final `fc1` input coefficient while retaining its learned bias will reduce the verified model to 1598 parameters and maintain at least 99% accuracy.
change: Represent one `fc1` row with seven learned coefficients, reconstruct its omitted LayerNorm-gauge gradient, and optimize its weight and bias with gauge-aware AdamW and clipping.
mechanism: LayerNorm-affine quotient for one MLP input row
evidence_used: The current single-key-row quotient achieved 99.88% at 1599 parameters, while removing unit 10’s bias nearly passed at 98.57%; the previously proposed unit-10 weight quotient was not verified, so preserving its threshold while removing an exact affine redundancy remains the most informative test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9662000000000001, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining three learned query-bias coordinates will reduce the verified model to 1,598 parameters while maintaining at least 99% accuracy.
change: Reduce the learned query bias from four coordinates to three, leaving the qualified positional, MLP-bias, and single-key-row quotients unchanged.
mechanism: Third query-bias coordinate ablation
evidence_used: Reducing query bias from five coordinates to four was compatible with 99.91% accuracy at 1,600 parameters, and the current single-key-row quotient remains qualified at 1,599; this isolates one previously untested scalar instead of repeating failed MLP or second-key-row reductions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining two learned query-bias coordinates in the verified 1,598-parameter design will produce 1,597 parameters while maintaining at least 99% accuracy.
change: Adopt the qualified positional quotient, fixed trailing `fc1` bias, and single-key-row quotient, then reduce the learned query bias from three coordinates to two.
mechanism: Fourth query-bias coordinate ablation atop qualified gauge reductions
evidence_used: Reference Design 3 achieved 99.97% accuracy at 1,598 parameters with three query-bias coordinates. Earlier reductions from five to four and then three coordinates also retained high accuracy, while alternative one-scalar MLP and second-key-row reductions failed, making the next query-bias coordinate the most informative isolated ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.21050000000000002, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the full eight-dimensional tied token matrix with a learned rank-seven factorization will reduce the model from 1,644 to 1,586 parameters while retaining at least 99% accuracy, because the attention and MLP streams remain width eight and final LayerNorm already restricts centered output states to seven degrees of freedom.
change: Encode every token through a learned seven-dimensional code and shared learned projection to width eight, then reuse the same factorization for output logits.
mechanism: Learned rank-seven tied token codebook
evidence_used: The width-six two-block design failed completely, showing that narrowing the entire computational stream is destructive, while the width-eight baseline reached 99.96%. This patch challenges the separate assumption that token identity and output classification require eight independent embedding dimensions without narrowing attention or MLP computation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0506, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing the third query-bias coordinate as the mean of two learned coordinates will produce 1,597 parameters and maintain at least 99% accuracy by preserving three active query offsets.
change: Replace three independent query-bias parameters with two learned parameters whose mean supplies the third effective coordinate.
mechanism: Shared effective query-bias coordinate
evidence_used: The three-coordinate design achieved 99.97% at 1,598 parameters, while fixing the third coordinate to zero collapsed accuracy to 21.05%; this tests whether the third coordinate must remain active rather than independently parameterized.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the qualified three-active-query-bias construction and removing the exact global common-shift redundancy from the tied token matrix will produce 1,596 learned parameters while retaining at least 99% accuracy.
change: Adopt the verified two-parameter query-bias mean reconstruction, represent the tied token matrix with one globally omitted coefficient, and preserve full-space initialization, AdamW moments, weight decay, and gradient clipping for the new gauge.
mechanism: Global tied-token common-shift quotient
evidence_used: Reference Design 3 achieved 99.89% accuracy at 1,597 parameters; unlike failed capacity ablations and additional key/MLP quotients, a common scalar added to every tied token-matrix entry only shifts residual channels and all output logits uniformly, making it an orthogonal exact symmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.731, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing the penultimate `fc1` bias as the mean of two learned biases atop the qualified 1,597-parameter design will produce 1,596 parameters and retain at least 99% accuracy.
change: Adopt the qualified positional and single-key-row quotients plus the three-active-query-bias construction, then replace eleven independent `fc1` biases with ten learned biases, one mean-reconstructed bias, and the qualified trailing zero bias.
mechanism: Shared adaptive MLP threshold
evidence_used: Fixing a second MLP threshold at zero nearly passed at 98.57%, while mean reconstruction preserved a necessary query coordinate and achieved 99.89% at 1,597 parameters; this tests whether the missing MLP threshold likewise needs activity rather than independence.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7447, "parameters": 1596, "training_steps": 4999}



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
