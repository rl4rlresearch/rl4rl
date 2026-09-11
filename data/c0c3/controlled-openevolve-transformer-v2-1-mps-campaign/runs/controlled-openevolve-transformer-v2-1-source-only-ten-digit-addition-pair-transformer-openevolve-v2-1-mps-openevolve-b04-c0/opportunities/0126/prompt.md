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
verified_results: {"accuracy": 0.9998, "parameters": 1562, "training_steps": 4999}
prior_hypothesis: Fixing `qkv.weight[10,3]` at zero will produce a 1562-parameter model with at least 99% accuracy because it is the head-0 counterpart of the successful `[14,3]` component-two anchor.

## Recent verification evidence

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
hypothesis: Fixing `proj.weight[0,2]` alongside `[0,0]` and `[0,4]` will produce a 1567-parameter model with at least 99% accuracy because subtracting its coefficient from the full output column adds only a feature-uniform shift removed by downstream LayerNorms.
change: Store 61 learned projection weights, reconstruct `proj.weight[0,0]`, `[0,2]`, and `[0,4]` as zeros, and canonicalize all three columns at initialization.
mechanism: Third attention-output residual-shift gauge anchor
evidence_used: The `[0,0]` and `[0,4]` projection anchors achieved 99.63% at 1568 parameters, while the component-1 `[0,1]` anchor failed; separately, a component-2 QKV constraint succeeded at 99.59%, motivating component 2 as the next projection coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1567, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[0,6]` alongside `[0,0]`, `[0,2]`, and `[0,4]` will produce a 1566-parameter model with at least 99% accuracy because column 6 is the second-head counterpart of the successful column-2 anchor, and its removal adds only a feature-uniform residual shift eliminated by downstream LayerNorms.
change: Store 60 learned projection weights, reconstruct `proj.weight[0,0]`, `[0,2]`, `[0,4]`, and `[0,6]` as zeros, and canonicalize all four columns at initialization.
mechanism: Cross-head component-two attention-output residual gauge
evidence_used: The component-two projection anchor at `[0,2]` achieved 99.99% while the cross-head component-zero anchor at `[0,4]` achieved 99.63%; `[0,6]` combines the successful component-two choice with the successful cross-head symmetry, unlike the failed component-one `[0,1]` anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1566, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[1,1]` will produce a 1565-parameter model with at least 99% accuracy because subtracting this coefficient from its full input column adds only a feature-uniform residual shift removed by downstream LayerNorms, while avoiding a fifth anchor in projection output row 0.
change: Store 59 learned projection weights, reconstruct the existing four row-0 anchors plus `proj.weight[1,1]` as zero, and canonicalize column 1 using output row 1.
mechanism: Distributed attention-output residual-shift gauge
evidence_used: Four even-column row-0 projection anchors reached 99.96% at 1566 parameters, whereas `proj.weight[0,1]` collapsed to 15.51%; moving the column-1 gauge anchor to row 1 directly tests whether that failure came from concentrating another constraint in output row 0 rather than from removing the exact column gauge itself.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1565, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[3,3]` will produce a 1564-parameter model with at least 99% accuracy because subtracting this coefficient from column 3 adds only a feature-uniform residual shift removed by downstream LayerNorms, while distributing the new constraint away from projection row 0.
change: Store 58 learned projection weights, reconstruct the existing five anchors plus `proj.weight[3,3]` as zero, and canonicalize column 3 using output row 3.
mechanism: Distributed diagonal attention-output residual gauge
evidence_used: Moving the column-1 anchor from row 0, which failed at 15.51%, to its diagonal location `[1,1]` achieved 100% accuracy at 1565 parameters; applying the same distributed diagonal strategy to the next unanchored odd column is the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6825, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[5,5]` will produce a 1564-parameter model with at least 99% accuracy because it is the second-head counterpart of the successful distributed `[1,1]` anchor, while avoiding the component-three direction implicated by the failed `[3,3]` experiment.
change: Store 58 learned projection weights, reconstruct the existing five anchors plus `proj.weight[5,5]` as zero, and canonicalize column 5 using output row 5.
mechanism: Cross-head diagonal attention-output residual gauge
evidence_used: Moving the column-1 anchor from `[0,1]`, which achieved only 15.51%, to `[1,1]` achieved 100% accuracy at 1565 parameters; testing the matching diagonal component-one anchor in the second head is better supported than repeating the component-three strategy that reached only 68.25%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[14,3]` at zero will produce a 1563-parameter model with at least 99% accuracy because component-two projection anchors succeeded in both attention heads, while the same key-row canonicalization succeeded for component zero in both heads.
change: Store 177 learned QKV weights, reconstruct `qkv.weight[14,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row.
mechanism: Second-head component-two LayerNorm-null key gauge
evidence_used: The current design reached 99.96% at 1564 parameters; `qkv.weight[8,3]` and `[12,3]` established cross-head viability for component-zero key gauges, while projection columns 2 and 6 established cross-head viability for component two, making row 14 better supported than the failed component-one or component-three directions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9919, "parameters": 1563, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[10,3]` at zero will produce a 1562-parameter model with at least 99% accuracy because it is the head-0 counterpart of the successful `[14,3]` component-two anchor.
change: Store one fewer QKV weight, reconstruct `qkv.weight[10,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row.
mechanism: Cross-head component-two LayerNorm-null key gauge
evidence_used: `qkv.weight[14,3]` retained 99.19% accuracy at 1563 parameters, while component-zero key anchors succeeded in both heads and component-two projection anchors also succeeded in both heads, making the matching head-0 component-two key anchor the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1562, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[3,1]` at `0.02` while inversely scaling key row 11 will produce a 1561-parameter model with at least 99% accuracy, because it uses the eighth exact query/key scale symmetry already exploited by seven successful anchors.
change: Store 176 learned QKV weights, reconstruct `qkv.weight[3,1]` as fixed, and canonicalize its initialization after the existing row-3 shear through reciprocal query/key scaling.
mechanism: Completion of the query–key reciprocal scale gauge
evidence_used: The current 1562-parameter model achieved 99.98% accuracy while using seven query-scale anchors; the remaining query row already has a preserved zero anchor and an unconstrained reciprocal scale gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7675, "parameters": 1561, "training_steps": 4999}



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
