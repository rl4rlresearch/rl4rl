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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1127, "training_steps": 4999}
prior_hypothesis: Extending the qualified four-way final-bias tie to five channels will reduce the model from 1,128 to 1,127 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

REFERENCE DESIGN 1
verified_results: {"accuracy": 1.0, "parameters": 1120, "training_steps": 4999}
prior_hypothesis: Eliminating one projection coefficient in the second attention head via an exactly compensated value-basis shear will reduce the model from 1,121 to 1,120 parameters while retaining accuracy >= 0.99.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1118, "training_steps": 4999}
prior_hypothesis: Fixing the qualified second-head shear pivot at a constant scale while compensating its value row will reduce the model from 1,119 to 1,118 parameters and retain accuracy >= 0.99.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1120, "training_steps": 4999}
prior_hypothesis: Applying one shear to the previously untouched second attention head, with its largest-magnitude initialization coefficient as the pivot and smallest as the eliminated coordinate, will reduce the qualified 1,121-parameter design to 1,120 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Eliminating one projection coefficient in the second attention head via an exactly compensated value-basis shear will reduce the model from 1,121 to 1,120 parameters while retaining accuracy >= 0.99.
change: Add the verified second-head projection shear, including initialization, reconstruction, gradient lifting, and optimizer regauging.
mechanism: Second-head value-basis shear gauge fixing
evidence_used: Reference Design 3 used this mechanism with 1,120 parameters and achieved 1.0 accuracy after 4,999 training steps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1120, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating a second projection coefficient in the second attention head through the same compensated shear symmetry will reduce the qualified 1,120-parameter design to 1,119 parameters while retaining accuracy >= 0.99.
change: Extend the verified two-head projection gauge with another zeroed second-head coordinate, including exact initialization compensation, gradient reconstruction, and optimizer regauging.
mechanism: Additional second-head value-basis shear gauge fixing
evidence_used: Reference Design 3 achieved 1.0 accuracy with 1,120 parameters after 4,999 steps using the same largest-magnitude second-head pivot for one compensated shear; the additional shear removes another coordinate without changing the represented attention function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1119, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,119-parameter design to eliminate the final nonpivot coefficient of the second attention-head projection will produce a 1,118-parameter model with accuracy >= 0.99, because the compensating value-basis shear preserves the represented attention function.
change: Adopt the qualified bias-free final normalization and five-coordinate projection gauge, then add a third compensated second-head shear in reconstruction, initialization, gradient lifting, and optimizer regauging.
mechanism: Complete second-head value-basis shear gauge fixing
evidence_used: Reference Design 2 achieved 1.0 accuracy with 1,119 parameters after 4,999 steps using two compensated second-head shears; the remaining nonpivot coordinate has the same exact shear symmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the qualified second-head shear pivot at a constant scale while compensating its value row will reduce the model from 1,119 to 1,118 parameters and retain accuracy >= 0.99.
change: Remove the learned second-head pivot magnitude, reconstruct it as a fixed anchor, and add exact initialization and optimizer compensation in the corresponding value coordinate.
mechanism: Second-head pivot scale gauge fixing
evidence_used: The 1,119-parameter design achieved 1.0 accuracy with two second-head shears, while eliminating its final nonpivot coefficient collapsed accuracy; this preserves that coefficient and instead removes the pivot’s redundant scale using the already-qualified projection/value scale symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1,118-parameter projection gauge, eliminating one row-1 projection coefficient through a compensated shear between the two second-head columns already zero in row 0 will produce a 1,117-parameter model with accuracy >= 0.99.
change: Adopt the qualified fixed second-head pivot and two-shear parameterization, then add a magnitude-pivoted secondary shear with exact initialization compensation, gradient reconstruction, and optimizer regauging.
mechanism: Residual second-head value-basis shear gauge fixing
evidence_used: Reference Design 2 achieved 0.9999 accuracy with 1,118 parameters after 4,999 steps. Unlike the failed removal of the remaining first-row coefficient, this patch retains that coefficient and uses the residual basis symmetry between two columns whose first-row entries are already fixed to zero.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1117, "training_steps": 4999}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
