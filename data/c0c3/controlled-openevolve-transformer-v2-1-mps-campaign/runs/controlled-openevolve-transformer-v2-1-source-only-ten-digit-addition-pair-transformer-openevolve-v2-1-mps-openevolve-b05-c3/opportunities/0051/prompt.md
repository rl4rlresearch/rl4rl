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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1596, "training_steps": 4999}
prior_hypothesis: Reusing the functionally redundant common shift of the attention projection bias as the second independent query-bias coordinate will produce 1,596 parameters and retain at least 99% accuracy while preserving three active query offsets.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1593, "training_steps": 4999}
prior_hypothesis: Gauge-fixing the common shift of the final MLP output bias will produce 1,593 learned parameters while retaining at least 99% accuracy, because this shift adds only a channel-uniform residual offset removed by the final LayerNorm.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1476, "training_steps": 4999}
prior_hypothesis: Replacing independent absolute positional residuals with two learned head-specific relative-lag tables will produce 1,476 parameters and retain at least 99% accuracy, because addition decoding can learn stationary attention routes while preserving the full-rank token representations shown to be load-bearing.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate at one atop the qualified 1,597-parameter design will produce 1,596 parameters and retain at least 99% accuracy, because its multiplicative degree of freedom can be absorbed exactly into the corresponding `fc1` input column and LayerNorm bias.
change: Adopt the qualified positional and first-key-row gauges plus the three-active-query-bias construction, then replace `ln2` with a LayerNorm whose final scale is fixed at its baseline initialization value.
mechanism: Final-MLP LayerNorm scale/linear-column quotient
evidence_used: Reference Design 3 achieved 99.89% accuracy at 1,597 parameters; failed 1,596-parameter bias sharing and token-shift experiments motivate testing an orthogonal scale redundancy that preserves the full MLP function class and initial forward pass.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.682, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the functionally redundant common shift of the attention projection bias as the second independent query-bias coordinate will produce 1,596 parameters and retain at least 99% accuracy while preserving three active query offsets.
change: Store one query-bias scalar explicitly and the other in the final projection-bias coordinate, then use a custom AdamW update and gauge-aware clipping to reproduce independent query-bias and full projection-bias optimization.
mechanism: Optimizer-preserving reuse of the attention projection-bias shift gauge
evidence_used: The current three-active-coordinate construction achieved 99.89% at 1,597 parameters, while zeroing its third coordinate collapsed accuracy to 21.05%; the failed standalone output-bias quotient motivates reusing the exact shift redundancy while explicitly preserving the original virtual AdamW dynamics.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified projection-bias shift gauge to one attention projection weight column will produce 1,595 learned parameters and retain at least 99% accuracy because the omitted coefficient changes the residual stream only by a channel-common, LayerNorm-invisible shift.
change: Adopt the qualified three-active-query construction, then gauge-fix one attention projection weight coefficient while preserving full-shape initialization, AdamW moments, weight decay, and gradient clipping.
mechanism: Single attention-output weight-column shift quotient
evidence_used: Reference Design 3 achieved 99.93% accuracy at 1,596 parameters by exploiting the attention projection’s common-output-shift symmetry; applying the same already-qualified symmetry to one input-dependent projection column is a conservative one-parameter extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,595-parameter projection gauge to a second input column will produce 1,594 learned parameters and retain at least 99% accuracy, because either omitted coefficient contributes only a token-dependent channel-common residual shift removed by downstream LayerNorms.
change: Adopt the qualified positional, query-bias-sharing, fixed-MLP-bias, and key-row reductions, then gauge-fix the final-row coefficients of two attention-output projection columns while preserving full-shape initialization, virtual AdamW moments, weight decay, and gradient clipping.
mechanism: Second attention-output weight-column shift quotient
evidence_used: The single-column projection quotient achieved 99.93% accuracy with 1,595 parameters; extending that same exact symmetry is more directly supported than the unrelated 1,596-parameter token, MLP-bias, and LayerNorm reductions that failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified two-column projection gauge to a third input column will reduce the model to 1,593 parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only an input-dependent channel-common residual shift removed by downstream LayerNorms.
change: Adopt the qualified positional, query-bias-sharing, fixed-MLP-bias, and key-row reductions, then gauge-fix three attention-output projection columns while preserving full-space initialization, AdamW dynamics, weight decay, and gradient clipping.
mechanism: Third attention-output weight-column shift quotient
evidence_used: The two-column projection quotient achieved 99.95% accuracy at 1,594 parameters after the one-column quotient achieved 99.93% at 1,595; the third column applies the same exact symmetry, unlike the unrelated 1,596-parameter reductions that failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7164, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the common shift of the final MLP output bias will produce 1,593 learned parameters while retaining at least 99% accuracy, because this shift adds only a channel-uniform residual offset removed by the final LayerNorm.
change: Represent each `fc2` bias with seven learned coordinates and a fixed final zero, while using virtual full-space AdamW moments, weight decay, and gradient clipping for the omitted coordinate.
mechanism: Final-MLP common-output-bias quotient
evidence_used: The two-column attention-output quotient reached 99.95% at 1,594 parameters, demonstrating that common-output shifts can be removed successfully; the third attention-column quotient failed at 71.64%, motivating an orthogonal common-output gauge immediately before the final LayerNorm instead of another attention input-column reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing independent absolute positional residuals with two learned head-specific relative-lag tables will produce 1,476 parameters and retain at least 99% accuracy, because addition decoding can learn stationary attention routes while preserving the full-rank token representations shown to be load-bearing.
change: Remove the 161-parameter absolute position embedding, add a gauge-fixed 44-parameter causal lag-bias table, and retain the qualified two-column projection and final-MLP output-bias quotients.
mechanism: Learned relative-lag attention routing
evidence_used: The qualified 1,593-parameter design reached 99.96%, whereas rank-seven token factorization collapsed to 5.06%; this motivates preserving token identity capacity while challenging the shared assumption that every sequence position needs an independent learned residual vector.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1476, "training_steps": 4999}



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
