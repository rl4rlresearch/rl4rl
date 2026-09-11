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
verified_results: {"accuracy": 0.9994, "parameters": 1538, "training_steps": 4999}
prior_hypothesis: A 1,538-parameter model will retain at least 99% accuracy because the two omitted `ln2` scales are exactly absorbed into their `fc1` columns, while optimizer-side factorization reproduces the full model’s separate AdamW updates instead of the optimization geometry that caused the naive two-scale design to reach only 80.28%.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1536, "training_steps": 4999}
prior_hypothesis: Extending the verified three-scale optimizer-side factorization to a fourth `ln2` scale will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, because the additional scale is exactly absorbable into its `fc1` column and its separate AdamW dynamics remain preserved outside the deduplicated model.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1539, "training_steps": 4999}
prior_hypothesis: Fixing one `ln2` scale to one will reduce the model from 1,540 to 1,539 parameters while retaining at least 99% accuracy, because the following `fc1` weight column can absorb that coordinate’s learned scaling without reducing functional capacity.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-fixing the single global all-ones shift of the verified 1,542-parameter tied embedding will yield 1,541 parameters while retaining at least 99% accuracy, because LayerNorm removes the shift throughout the input path and the tied output path changes every vocabulary logit by the same scalar.
change: Reproduce the qualified four-column, bias-free-value, bias-free-`ln1` design, then store the tied embedding relative to one fixed scalar and train it with full ambient-coordinate AdamW moments and clipping.
mechanism: Global tied-embedding shift gauge
evidence_used: The qualified 1,542-parameter design achieved 99.91% accuracy. Its existing positional and terminal gauges show that exact shift invariances remain trainable when omitted coordinates retain ambient AdamW dynamics, motivating a distinct one-scalar gauge instead of extending the positional gauge that failed at 1,549 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9157, "parameters": 1541, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the all-ones component of the attention output bias will reduce the qualified 1,542-parameter model to 1,541 parameters while retaining at least 99% accuracy, because the resulting scalar residual shift is erased by both the second pre-norm and final LayerNorm.
change: Reproduce the verified shared key/value, bias-free value, bias-free `ln1` design and represent the attention projection bias with seven learned differences, preserving its full eight-coordinate AdamW dynamics.
mechanism: Attention output-bias shift gauge
evidence_used: The shared-key/value, bias-free-value, bias-free-`ln1` design achieved 99.91% accuracy at 1,542 parameters; four analogous terminal output-direction gauges remained qualified, while the failed tied-embedding gauge motivates testing this distinct local invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9912000000000001, "parameters": 1541, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the all-ones output component of the first attention projection weight column will reduce the model from 1,541 to 1,540 parameters while retaining at least 99% accuracy, because it adds only a token-dependent scalar residual shift that is erased by the second pre-norm and final LayerNorm.
change: Represent the first attention projection weight column with seven learned differences, reconstruct its eighth coordinate during forward passes, and preserve its full eight-coordinate AdamW dynamics alongside the existing attention-bias gauge.
mechanism: First-column attention output-direction gauge
evidence_used: The attention output-bias shift gauge achieved 99.12% at 1,541 parameters, while four analogous terminal weight-column gauges remained qualified; this tests the smallest adjacent output-direction gauge within the same projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1540, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified attention projection gauge from one to two weight columns will reduce the model from 1,540 to 1,539 parameters while retaining at least 99% accuracy, because either column’s all-ones output component produces only a token-dependent scalar residual shift erased by the second pre-norm and final LayerNorm.
change: Reproduce the qualified attention-bias and first-column gauges, gauge-fix the second attention projection column, and preserve full eight-coordinate AdamW dynamics for all three attention gauges.
mechanism: Second-column attention output-direction gauge
evidence_used: Reference Design 3 achieved 99.99% accuracy with 1,540 parameters after gauge-fixing the attention output bias and first weight column; applying the identical exact invariance to one adjacent column is the smallest controlled next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7332, "parameters": 1539, "training_steps": 4999}

RECENT RESULT
hypothesis: Decoupling query/key width from value width will reduce the verified model from 1,555 to 1,503 parameters while retaining at least 99% accuracy, because two learned routing coordinates can encode fixed positional correspondences while the full four-dimensional value stream preserves digit content.
change: Replace each four-dimensional query/key head with a two-dimensional learned routing head, while retaining two independent query heads, the shared four-dimensional learned value stream, and all qualified residual and MLP pathways.
mechanism: Two-dimensional learned routing with four-dimensional shared content
evidence_used: Shared-key/value attention achieved 100% accuracy at 1,555 parameters, showing that routing and content need not use independent per-head representations. This cleanly challenges the remaining assumption that their widths must match; unlike the failed additional gauges and gated MLP, it preserves the qualified optimizer and nonlinear computation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8188, "parameters": 1503, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale to one will reduce the model from 1,540 to 1,539 parameters while retaining at least 99% accuracy, because the following `fc1` weight column can absorb that coordinate’s learned scaling without reducing functional capacity.
change: Replace `ln2` with a LayerNorm that learns seven scales and fixes the eighth to one, while leaving all other qualified components and training behavior unchanged.
mechanism: Single-coordinate second-pre-norm scale absorption
evidence_used: Removing all eight `ln2` scales collapsed accuracy to 73.68%, whereas the current 1,540-parameter model achieved 99.99%; removing only one scale is the smallest controlled test of whether the failure resulted from simultaneous loss of all eight optimization degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1539, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified single fixed `ln2` scale to two fixed scales will reduce the qualified 1,539-parameter design to 1,538 parameters while retaining at least 99% accuracy, because both scales can be absorbed independently by the following `fc1` weight columns.
change: Reproduce the qualified attention projection bias/first-column gauges and replace `ln2` with a bias-free LayerNorm that learns six scales while fixing the final two to one.
mechanism: Two-coordinate second-pre-norm scale absorption with qualified attention gauges
evidence_used: Reference Design 3 achieved 99.93% accuracy at 1,539 parameters with one fixed `ln2` scale; fixing all eight failed, so fixing one adjacent additional scale is the smallest controlled test along this successful reduction axis.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8028, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified terminal projection gauge from four to five weight columns will produce a 1,538-parameter model with at least 99% accuracy, because the removed all-ones output component contributes only a token-dependent scalar residual shift erased by the final LayerNorm.
change: Reproduce the qualified shared key/value, bias-free value, bias-free `ln1`, attention projection gauge, and single fixed `ln2` scale design, then gauge-fix a fifth terminal projection column while preserving full ambient-coordinate AdamW dynamics.
mechanism: Fifth terminal output-direction gauge atop the qualified 1,539-parameter design
evidence_used: The 1,539-parameter design achieved 99.93% accuracy, and four terminal-column gauges previously retained 99.96%; unlike the failed second attention-column and second `ln2`-scale reductions, this tests the next coordinate along the terminal gauge axis with the strongest prior tolerance.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9499, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln1` scale to one will reduce the qualified model from 1,539 to 1,538 parameters while retaining at least 99% accuracy, because the corresponding columns of the learned query, key, and value projections can independently absorb that scale.
change: Reuse the qualified seven-scale bias-free LayerNorm for `ln1`, while preserving the current attention, MLP, gauges, optimizer, and decoding behavior.
mechanism: Single-coordinate first-pre-norm scale absorption
evidence_used: Fixing one `ln2` scale achieved 99.93% accuracy at 1,539 parameters, while removing all eight scales failed; applying the same minimal one-coordinate constraint to the distinct `ln1` scale vector is the most controlled untested reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7165, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,538-parameter model will retain at least 99% accuracy because the two omitted `ln2` scales are exactly absorbed into their `fc1` columns, while optimizer-side factorization reproduces the full model’s separate AdamW updates instead of the optimization geometry that caused the naive two-scale design to reach only 80.28%.
change: Reproduce the qualified attention projection bias/first-column gauges, replace `ln2` with a six-scale LayerNorm, and train the two absorbed scale/weight factorizations in ambient coordinates while continuously materializing only their products in the learned model.
mechanism: Ambient AdamW for two multiplicatively absorbed LayerNorm scales
evidence_used: The full-scale 1,540-parameter attention-gauged design achieved 99.99%, whereas directly fixing a second `ln2` scale fell to 80.28%; since each scale is functionally absorbable into the following `fc1` column, this isolates and restores the lost optimizer dynamics while removing both parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-scale optimizer-side factorization to a third `ln2` scale will produce a 1,537-parameter model with at least 99% accuracy, because the third scale is exactly absorbable into its corresponding `fc1` column and its separate AdamW dynamics are preserved outside the deduplicated learned model.
change: Reproduce the qualified 1,538-parameter shared-key/value, bias-free-value, bias-free-`ln1`, attention-gauged, four-terminal-gauge design, then absorb three rather than two `ln2` scales into `fc1` with ambient-coordinate moments and clipping.
mechanism: Third multiplicatively absorbed LayerNorm scale with ambient AdamW
evidence_used: The two-scale ambient-factorization design achieved 99.94% accuracy at 1,538 parameters, whereas directly fixing two scales achieved only 80.28%; this isolates optimizer geometry as the failure cause and motivates extending the verified factorization by one coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1537, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-scale optimizer-side factorization to a fourth `ln2` scale will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, because the additional scale is exactly absorbable into its `fc1` column and its separate AdamW dynamics remain preserved outside the deduplicated model.
change: Absorb four rather than three trailing `ln2` scales into the corresponding `fc1` columns, expanding the existing ambient-coordinate moments, clipping correction, and materialization update accordingly.
mechanism: Fourth multiplicatively absorbed LayerNorm scale with ambient AdamW
evidence_used: The three-scale ambient-factorization design achieved 99.97% accuracy at 1,537 parameters, improving on the already-qualified two-scale design; this makes one additional coordinate along the same successful exact-factorization axis the smallest controlled next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1536, "training_steps": 4999}



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
