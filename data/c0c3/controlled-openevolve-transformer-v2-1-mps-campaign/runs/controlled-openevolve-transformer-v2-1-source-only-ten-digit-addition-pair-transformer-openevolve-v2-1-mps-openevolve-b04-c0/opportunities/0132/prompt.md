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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1561, "training_steps": 4999}
prior_hypothesis: Fixing `qkv.weight[1,2]` at zero through a reciprocal query/key shear will produce a 1561-parameter model with at least 99% accuracy, because the 99.98%-accurate 1562-parameter design already contains two successful query–key shear anchors, and this component-two reduction leaves query row 3’s scale unconstrained unlike the failed eighth scale anchor.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing `qkv.weight[1,2]` at zero through a reciprocal query/key shear will produce a 1561-parameter model with at least 99% accuracy, because the 99.98%-accurate 1562-parameter design already contains two successful query–key shear anchors, and this component-two reduction leaves query row 3’s scale unconstrained unlike the failed eighth scale anchor.
change: Store 176 learned QKV weights, reconstruct `qkv.weight[1,2]` as zero, and canonicalize initialization by shearing query row 1 against row 3 while applying the inverse shear to key row 11.
mechanism: Additional head-0 query–key shear gauge
evidence_used: The current design achieved 99.98% with existing shear eliminations at `qkv.weight[2,4]` and `[3,0]`; component-two projection and key anchors also succeeded, while fixing the remaining query scale collapsed to 76.75%, motivating another shear rather than another scale constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1561, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[3,4]` at zero will produce a 1560-parameter model with at least 99% accuracy because a shear of query row 3 against row 0 preserves its existing zero at `[3,0]`, leaves row 3’s scale unconstrained, and can be inverted in key row 8.
change: Store 175 learned QKV weights, reconstruct `qkv.weight[3,4]` as zero, canonicalize it through a reciprocal query–key shear, and restore key row 8’s existing LayerNorm-null anchor.
mechanism: Head-0 query–key shear gauge with preserved query scale
evidence_used: The 1561-parameter design achieved 99.93% after a third query–key shear, while directly fixing query row 3’s remaining scale achieved only 76.75%; this adds another shear without constraining that scale.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9789, "parameters": 1560, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[4,2]` at zero will produce a 1560-parameter model with at least 99% accuracy because it mirrors the successful `[0,2]` anchor while preserving all head-1 query scales and attention scores.
change: Store 175 learned QKV weights, reconstruct `qkv.weight[4,2]` as zero, and canonicalize it by shearing query row 4 against the scale-neutral difference of rows 5 and 7 with inverse updates to key rows 13 and 15.
mechanism: Head-1 scale-preserving reciprocal query–key shear
evidence_used: The 1561-parameter design retained 99.93% with `qkv.weight[0,2]` and an additional component-two shear; cross-head component-two key and projection anchors also succeeded, while the failed `[3,4]` experiment motivates leaving query row 3 unconstrained.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7612000000000001, "parameters": 1560, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[1,3]` will produce a 1560-parameter model with at least 99% accuracy because column 3 has an exact feature-uniform residual gauge, while output row 1 previously supported the difficult odd-component `[1,1]` anchor at 100% accuracy.
change: Store 57 learned projection weights, reconstruct `proj.weight[1,3]` as zero alongside the six existing anchors, and canonicalize column 3 using output row 1.
mechanism: Reused robust output-row residual-shift gauge
evidence_used: The diagonal column-3 anchor `[3,3]` failed at 68.25%, but relocating the column-1 anchor from failing `[0,1]` to `[1,1]` achieved 100%; reusing the empirically robust row 1 distinguishes a column-3 limitation from an anchor-row optimization problem.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing `proj.weight[5,7]` will produce a 1560-parameter model with at least 99% accuracy because column 7 has the same exact feature-uniform residual gauge as the successful `[5,5]` anchor, while retaining the empirically robust output row 5 and avoiding the failed output row 3.
change: Store 57 learned projection weights, reconstruct `proj.weight[5,7]` as zero alongside the six existing anchors, and canonicalize column 7 using output row 5.
mechanism: Second-head robust-row residual-shift gauge
evidence_used: `proj.weight[5,5]` retained 99.96% accuracy at 1564 parameters, and the current 1561-parameter design reached 99.93%; using row 5 for the remaining second-head column tests the closest robust-row analogue while the failed `[3,3]` result argues against output row 3.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing `proj.weight[1,7]` will produce a 1560-parameter model with at least 99% accuracy because column 7 has an exact feature-uniform residual gauge and output row 1 already supported the successful difficult odd-column anchor.
change: Store 57 learned projection weights, reconstruct `proj.weight[1,7]` as zero alongside the six existing anchors, and canonicalize column 7 using output row 1.
mechanism: Cross-head robust-row residual-shift gauge
evidence_used: The `[1,1]` row-1 anchor achieved 100% accuracy, while `[3,3]` fell to 68.25%; the `[5,7]` trial timed out without accuracy evidence, so testing column 7 at the empirically strongest odd-anchor row is the most informative next reduction.
result: training did not finish within the verification time limit



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
