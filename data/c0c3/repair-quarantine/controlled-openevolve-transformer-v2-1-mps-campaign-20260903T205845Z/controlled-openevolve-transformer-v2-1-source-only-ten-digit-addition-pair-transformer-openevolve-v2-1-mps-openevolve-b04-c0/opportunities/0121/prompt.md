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
verified_results: {"accuracy": 0.9989, "parameters": 1567, "training_steps": 4999}
prior_hypothesis: Fixing `proj.weight[0,2]` alongside the successful component-zero anchors will produce a 1567-parameter model with at least 99% accuracy, because the removed projection coefficient contributes only a feature-uniform residual shift, and component 2 previously tolerated a QKV gauge reduction while component 1 did not.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `qkv.weight[2,4]` at zero through a shear against query row 0 will produce a 1572-parameter model with at least 99% accuracy, because row 2 already tolerates its scale anchor and the transformation uses the stable fixed `qkv.weight[0,4] = 0.02` pivot while exactly preserving attention scores.
change: Store 181 learned QKV weights, reconstruct `qkv.weight[2,4]` as zero, and initialize it by shearing query row 2 against query row 0 while applying the inverse shear to key row 8.
mechanism: Stable-pivot head-0 query/key shear gauge anchor
evidence_used: The first shear anchor achieved 99.69% at 1573 parameters, while repeated scale, norm, and second-shear constraints specifically targeting query row 3 failed. Applying the same stable row-0 shear to the previously successful scale-anchored row 2 tests whether the last failure was row-3-specific.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9959, "parameters": 1572, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[1,4]` at zero through a shear against query row 0 will produce a 1571-parameter model with at least 99% accuracy, because it exactly preserves attention scores and mirrors the successful row-2 shear while retaining row 1’s fixed scale coefficient.
change: Store 180 learned QKV weights, reconstruct `qkv.weight[1,4]` as zero, and initialize it by shearing query row 1 against query row 0 while applying the inverse shear to key row 8.
mechanism: Parallel stable-pivot head-0 query/key shear anchor
evidence_used: The identical stable-pivot shear on query row 2 achieved 99.59% at 1572 parameters, while the analogous row-3 constraint failed; row 1 resembles the successful row 2 because its scale anchor remains unchanged by the row-0 shear.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.41619999999999996, "parameters": 1571, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[12,3]` at zero will produce a 1571-parameter model with at least 99% accuracy, because a LayerNorm output has a zero-sum standardized component, so the removed key-row coefficient is gauge-redundant up to a position-independent key shift that causal softmax attention cancels.
change: Store 180 learned QKV weights, reconstruct `qkv.weight[12,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row, preserving the initialized attention function.
mechanism: LayerNorm-null key-row coefficient anchor
evidence_used: The current 1572-parameter model reached 99.59%, while the next query-shear reduction collapsed to 41.62%; this motivates using an untested additive key-only invariance that requires neither another query shear nor reciprocal rescaling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1571, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[13,3]` at zero will produce a 1570-parameter model with at least 99% accuracy, because the same LayerNorm-null, softmax-invariant key-row gauge used successfully for row 12 applies independently to row 13.
change: Store 179 learned QKV weights, reconstruct `qkv.weight[13,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row.
mechanism: Second LayerNorm-null key-row coefficient anchor
evidence_used: Fixing `qkv.weight[12,3]` with the identical key-only canonicalization achieved 99.97% accuracy at 1571 parameters, making an adjacent independent key row the closest supported next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4498, "parameters": 1570, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[8,3]` at zero will produce a 1570-parameter model with at least 99% accuracy, because it is the head-0 counterpart of the successful `qkv.weight[12,3]` anchor and removes the same softmax-invariant key-row direction.
change: Store 179 learned QKV weights, reconstruct `qkv.weight[8,3]` as zero, and canonicalize its initialized key row after the existing score-preserving scale and shear transformations.
mechanism: Cross-head component-zero LayerNorm-null key gauge
evidence_used: The component-zero key anchor `qkv.weight[12,3]` achieved 99.97% at 1571 parameters, while the component-one row-13 anchor failed; testing the corresponding component-zero row in the other head is the closest way to distinguish component-specific viability from a one-row anomaly.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9945999999999999, "parameters": 1570, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the tied value/projection bias so its combined post-projection contribution is mean-zero will produce a 1569-parameter model with at least 99% accuracy, because it removes the exact uniform residual-shift gauge while preserving the original parameter’s value-bias and projection-bias coupling.
change: Replace the eight-coordinate tied bias with seven learned coordinates, reconstruct its final coordinate from the projection matrix, and continue using the reconstructed bias both before and after attention projection.
mechanism: Tied value/output bias gauge canonicalization
evidence_used: The current tied-bias design achieved 99.46% at 1570 parameters, while directly replacing it with a mean-zero post-projection bias collapsed to 64.92%; retaining the original coupled computational path tests whether that failed reduction was caused by its altered optimization geometry rather than by loss of functional capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.19940000000000002, "parameters": 1569, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[0,0]` at zero will produce a 1569-parameter model with at least 99% accuracy, because subtracting that coefficient from every output row changes each token only by a feature-uniform residual shift, which downstream LayerNorms remove under the fixed zero-dropout configuration.
change: Replace the attention output projection with a 63-weight parameterization, reconstruct `proj.weight[0,0]` as zero, and canonicalize its initialization while leaving the tied value/projection bias unchanged.
mechanism: Attention-output residual-shift weight gauge
evidence_used: The current 1570-parameter design reached 99.46%, whereas constraining the tied bias reached only 19.94%; this tests the independent projection-weight residual gauge without perturbing the bias coupling implicated by that failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1569, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[0,1]` at zero alongside the verified `proj.weight[0,0]` anchor will produce a 1568-parameter model with at least 99% accuracy, because independently subtracting each coefficient from its full output column changes every token only by a feature-uniform shift removed by downstream LayerNorms.
change: Store 62 learned attention-output projection weights, reconstruct `proj.weight[0,0:2]` as zeros, and canonicalize both corresponding columns at initialization.
mechanism: Second attention-output residual-shift weight gauge
evidence_used: The first attention-output residual-shift anchor achieved 99.93% accuracy with 1569 parameters; applying the identical exact gauge to the adjacent input column is the closest supported one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1551, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[0,4]` alongside the verified `proj.weight[0,0]` anchor will produce a 1568-parameter model with at least 99% accuracy, because column 4 is the component-zero counterpart in the second attention head and its removal adds only a feature-uniform residual shift eliminated by downstream LayerNorms.
change: Store 62 learned attention-output projection weights, reconstruct `proj.weight[0,0]` and `proj.weight[0,4]` as zeros, and canonicalize both columns at initialization.
mechanism: Cross-head attention-output residual-shift gauge
evidence_used: The component-zero projection anchor at `proj.weight[0,0]` achieved 99.93% at 1569 parameters, while the adjacent component-one anchor at `[0,1]` collapsed to 15.51%; prior QKV evidence also showed component-zero anchors succeeding across both heads while component one failed, motivating the cross-head component-zero counterpart.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9963, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[0,2]` alongside the successful component-zero anchors will produce a 1567-parameter model with at least 99% accuracy, because the removed projection coefficient contributes only a feature-uniform residual shift, and component 2 previously tolerated a QKV gauge reduction while component 1 did not.
change: Store 61 learned attention-output projection weights, reconstruct `proj.weight[0,0]`, `[0,2]`, and `[0,4]` as zeros, and canonicalize all three columns at initialization.
mechanism: Third attention-output residual-shift gauge
evidence_used: Projection anchors `[0,0]` and `[0,4]` achieved 99.63% at 1568 parameters, whereas `[0,1]` collapsed to 15.51%; separately, the component-2 QKV shear achieved 99.59%, making projection component 2 the most supported untested column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1567, "training_steps": 4999}

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
