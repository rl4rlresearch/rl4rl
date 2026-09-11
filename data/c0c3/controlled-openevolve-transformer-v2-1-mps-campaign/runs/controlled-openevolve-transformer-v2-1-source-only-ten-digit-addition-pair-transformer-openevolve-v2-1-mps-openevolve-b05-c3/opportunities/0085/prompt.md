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
verified_results: {"accuracy": 0.9997, "parameters": 1282, "training_steps": 4999}
prior_hypothesis: Extending the verified three-coordinate `ln2` scale quotient to four coordinates will reduce the model from 1,283 to 1,282 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1284, "training_steps": 4999}
prior_hypothesis: Extending the verified one-coordinate `ln2` scale quotient to two coordinates will reduce the qualified 1,285-parameter model to 1,284 parameters while retaining at least 99% accuracy, because `fc1` can absorb both fixed channel scales.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1285, "training_steps": 4999}
prior_hypothesis: Fixing one `ln2` scale coordinate at one will reduce the verified 1,286-parameter model to 1,285 parameters while retaining at least 99% accuracy, because `fc1` can absorb that channel scale and the analogous pre-attention quotient retained accuracy with seven fixed coordinates.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998, "parameters": 1283, "training_steps": 4999}
prior_hypothesis: Extending the verified two-coordinate `ln2` scale quotient to three coordinates will reduce the model from 1,284 to 1,283 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified four-coordinate `ln1` scale quotient to five coordinates will reduce the model from 1,316 to 1,315 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while three adaptive scales remain.
change: Fix the final five `ln1` scale coordinates at one instead of the final four, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Five-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The four-coordinate quotient achieved 99.96% accuracy at 1,316 parameters, after the one-, two-, and three-coordinate variants all exceeded 99%; this next one-coordinate extension most directly probes the boundary indicated by the failed all-eight-scale removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1315, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified five-coordinate `ln1` scale quotient to six coordinates will reduce the model from 1,315 to 1,314 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while two adaptive scales remain.
change: Fix the final six `ln1` scale coordinates at one, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Six-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The five-coordinate quotient achieved 99.97% accuracy with 1,315 parameters after every one-through-four-coordinate quotient also exceeded 99%; this incremental extension most directly locates the boundary suggested by the failed all-eight-scale removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1314, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified six-coordinate `ln1` scale quotient to seven coordinates will reduce the model from 1,314 to 1,313 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while one adaptive scale preserves optimization flexibility absent from the failed fully non-affine design.
change: Fix the final seven `ln1` scale coordinates at one, leaving one learned scale and preserving the learned attention, initialization, optimizer, checkpoints, and decoding interface.
mechanism: Seven-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The six-coordinate quotient achieved 99.97% accuracy with 1,314 parameters, while fixing all eight scales failed at 42.51%; testing seven fixed coordinates is the direct incremental experiment that locates the boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1313, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing one input coefficient in every value-projection row will reduce Reference Design 1 from 1,313 to 1,305 parameters while retaining at least 99% accuracy, because LayerNorm outputs lie in a seven-dimensional zero-sum subspace and the omitted coefficients are exact function-space redundancies.
change: Adopt the verified seven-coordinate `ln1` scale quotient, then remove eight redundant value-projection coefficients while preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware clipping.
mechanism: LayerNorm-nullspace value-projection quotient
evidence_used: Reference Design 1 achieved 99.96% accuracy with 1,313 parameters and one adaptive `ln1` scale; complete attention-output and MLP projection gauges also retained accuracy, supporting another exact, optimizer-aware linear-map quotient without repeating the failed removal of the final adaptive scale.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1305, "training_steps": 4999}

RECENT RESULT
hypothesis: Absorbing `ln2` bias into a restored full `fc1` bias and gauge-fixing one input coefficient in every `fc1` row will reduce the verified model from 1,305 to 1,286 parameters while retaining at least 99% accuracy.
change: Remove eight `ln2` bias parameters, restore the omitted `fc1` bias, remove twelve LayerNorm-nullspace-redundant `fc1` weights, and train those reduced weights with the existing full-space gauge optimizer and clipping.
mechanism: Pre-MLP affine absorption and LayerNorm-nullspace quotient
evidence_used: The current value-projection nullspace quotient removed eight coefficients and achieved 99.89% accuracy; the earlier pre-attention LayerNorm-bias absorption also retained 99.66%. Here bias absorption is direct because `fc1` already supplies pre-GELU biases.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Building on the verified 1,305-parameter value-projection quotient, absorbing `ln2.bias` into a restored full `fc1.bias` will produce 1,298 parameters and retain at least 99% accuracy.
change: Adopt the verified eight-parameter value-map quotient and its gauge-aware optimizer, then remove the eight `ln2` bias parameters while restoring the twelfth `fc1` bias parameter.
mechanism: LayerNorm-nullspace value quotient with staged pre-MLP bias absorption
evidence_used: Reference Design 2 achieved 99.89% accuracy with 1,305 parameters; because the combined 1,286-parameter experiment could not be verified, isolating the exact seven-parameter bias-absorption reduction avoids simultaneously introducing the untested `fc1` weight quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1298, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing one input coefficient in every `fc1` row will reduce the verified 1,298-parameter design to 1,286 parameters while retaining at least 99% accuracy, because bias-free LayerNorm outputs have an exact one-dimensional nullspace that already supported the verified value-map quotient.
change: Adopt the verified seven-scale `ln1`, value-map quotient, and `ln2`-bias absorption, then represent each `fc1` row with seven coefficients and train both input-map quotients using full-space AdamW moments, weight decay, and gauge-aware clipping.
mechanism: Pre-MLP LayerNorm-nullspace weight quotient
evidence_used: The 1,298-parameter reference achieved 99.92% accuracy, and its analogous eight-parameter value-map nullspace quotient achieved 99.89% in isolation; the earlier combined 1,286-parameter attempt could not be verified, so applying only the remaining `fc1` quotient atop the qualified staged base is the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1286, "training_steps": 4999}

RECENT RESULT
hypothesis: A single unrestricted discrete lag potential, used with opposite signs by the two heads, will retain at least 99% accuracy while reducing the current model from 1,315 to 1,285 parameters because the heads primarily need complementary operand routes rather than independent lag tables.
change: Replace the two independent 30-parameter lag tables with one gauge-fixed 30-parameter table whose positive and negative logits define complementary learned causal routes.
mechanism: Complementary signed-lag attention
evidence_used: Dense stationary lag routing reached 99.97%, while the Gaussian mixture reached only 49.85%; this suggests discrete per-lag freedom is load-bearing, but does not establish that the two heads require independent tables. The signed table preserves unrestricted discrete resolution and direct optimization while testing head-route independence.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0015, "parameters": 1293, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate at one will reduce the verified 1,286-parameter model to 1,285 parameters while retaining at least 99% accuracy, because `fc1` can absorb that channel scale and the analogous pre-attention quotient retained accuracy with seven fixed coordinates.
change: Represent `ln2` with seven learned scales and one fixed unit scale, leaving the learned attention, MLP width, initialization, optimizer structure, checkpoints, and decoding interface unchanged.
mechanism: One-coordinate pre-MLP LayerNorm-scale quotient
evidence_used: The current 1,286-parameter design achieved 99.93% accuracy, and the verified `ln1` sequence retained at least 99% accuracy while progressively fixing one through seven scale coordinates; this motivates the conservative first-coordinate test at `ln2`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1285, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified one-coordinate `ln2` scale quotient to two coordinates will reduce the qualified 1,285-parameter model to 1,284 parameters while retaining at least 99% accuracy, because `fc1` can absorb both fixed channel scales.
change: Adopt the verified `fc1` LayerNorm-nullspace quotient, represent `ln2` with six learned scales and two fixed unit scales, and train both reduced input maps with the existing full-space gauge optimizer and clipping.
mechanism: Two-coordinate pre-MLP LayerNorm-scale quotient
evidence_used: The 1,285-parameter design achieved 99.88% accuracy with one fixed `ln2` scale, while successive `ln1` quotients retained accuracy through seven fixed scales; a second incremental `ln2` coordinate is therefore the most direct boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1284, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified two-coordinate `ln2` scale quotient to three coordinates will reduce the model from 1,284 to 1,283 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale.
change: Adopt the qualified `fc1` LayerNorm-nullspace quotient and `ln2` bias absorption, then represent `ln2` with five learned scales and three fixed unit scales while preserving full-space gauge-aware optimization.
mechanism: Three-coordinate pre-MLP LayerNorm-scale quotient
evidence_used: The two-coordinate `ln2` quotient achieved 99.96% accuracy with 1,284 parameters, after the one-coordinate version achieved 99.88%; the analogous incremental `ln1` quotient remained above 99% through seven fixed scales, motivating the next single-coordinate test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1283, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-coordinate `ln2` scale quotient to four coordinates will reduce the model from 1,283 to 1,282 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale.
change: Adopt the qualified seven-scale `ln1`, value-map and `fc1` LayerNorm-nullspace quotients, and `ln2` bias absorption, then represent `ln2` with four learned scales and four fixed unit scales while preserving full-space gauge-aware optimization.
mechanism: Four-coordinate pre-MLP LayerNorm-scale quotient
evidence_used: Reference Design 3 achieved 99.98% accuracy with 1,283 parameters after the one-, two-, and three-coordinate `ln2` quotients all exceeded 99%; the next single-coordinate extension is the most direct boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1282, "training_steps": 4999}



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
