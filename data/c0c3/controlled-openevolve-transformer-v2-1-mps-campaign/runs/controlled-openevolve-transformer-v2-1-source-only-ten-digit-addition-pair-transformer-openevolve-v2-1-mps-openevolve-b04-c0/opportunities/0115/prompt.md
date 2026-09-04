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
verified_results: {"accuracy": 0.9945999999999999, "parameters": 1570, "training_steps": 4999}
prior_hypothesis: Fixing `qkv.weight[8,3]` at zero will produce a 1570-parameter model with at least 99% accuracy, because it is the head-0 counterpart of the successful `qkv.weight[12,3]` anchor and removes the same softmax-invariant key-row direction.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing attention LayerNorm gain coordinate 3 at one will produce a 1576-parameter model with at least 99% accuracy because its corresponding LayerNorm shift is already fixed at zero, making the gain exactly absorbable into QKV column 3 while leaving initialization unchanged.
change: Remove the second learned attention LayerNorm gain and reconstruct coordinate 3 as one; retain the learned gain shared by coordinates 1 and 7.
mechanism: Zero-shift LayerNorm gain absorption into the QKV input column
evidence_used: The 1577-parameter model reached 99.13%, while attempted 1576-parameter changes involving query biases, fixed Q/K coefficients, reciprocal value scaling, and folded biases failed. This targets an untested exact downstream scale redundancy without reciprocal initialization or perturbing the verified initial function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing MLP LayerNorm gain coordinate 1 at one will produce a 1575-parameter model with at least 99% accuracy because its shift is already fixed at zero, so the gain is exactly absorbable into `fc1.weight` column 1 without changing initialization.
change: Remove the learned MLP LayerNorm gain for coordinate 1 and reconstruct it as one, leaving coordinate 2 as the sole learned gain.
mechanism: Zero-shift MLP LayerNorm gain absorption
evidence_used: Fixing attention LayerNorm gain coordinate 3 by the same zero-shift downstream-column absorption mechanism achieved 99.92% with 1576 parameters; the MLP LayerNorm already successfully fixes six other gains, making its zero-shift coordinate 1 the closest supported next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1525, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the remaining attention LayerNorm gain with MLP LayerNorm gain coordinate 1 will produce a 1575-parameter model with at least 99% accuracy, because both gains are downstream-absorbable gauge scalars while the gain remains learned rather than frozen.
change: Rebind the attention LayerNorm gain parameter to the MLP LayerNorm’s two-coordinate gain parameter; its first coordinate controls both norms, while the second remains exclusive to the MLP norm.
mechanism: Cross-sublayer LayerNorm gain sharing
evidence_used: Fixing attention LayerNorm gain coordinate 3 achieved 99.92% at 1576 parameters, but fixing MLP LayerNorm gain coordinate 1 collapsed to 15.25%. Sharing removes the independent redundant scalar while preserving learned gain adaptation, directly testing whether the failure was caused by freezing that gain.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing MLP LayerNorm gain coordinate 2 at one will produce a 1574-parameter model with at least 99% accuracy, because coordinate 1 remains learned and shared with attention while coordinate 2 is zero-shift and absorbable into its downstream `fc1.weight` column.
change: Store one MLP LayerNorm gain instead of two, reconstruct coordinate 2 as one, and retain the verified cross-sublayer sharing through coordinate 1.
mechanism: Fix the untested second MLP LayerNorm gain
evidence_used: Fixing coordinate 1 while leaving coordinate 2 learned collapsed to 15.25%, whereas keeping coordinate 1 learned through cross-sublayer sharing achieved 99.96% at 1575 parameters; this directly motivates preserving coordinate 1 and testing the unanchored coordinate 2.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1574, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[3,0]` at zero through a shear against the already anchored query row 1 will produce a 1573-parameter model with at least 99% accuracy, because the reciprocal key-row shear exactly preserves attention scores without the ill-conditioned scaling used by prior failed row-3 anchors.
change: Store 182 learned QKV weights, reconstruct `qkv.weight[3,0]` as zero, and shear query row 3 against query row 1 while applying the inverse shear to key row 9 during initialization.
mechanism: Attention-preserving query/key shear gauge anchor
evidence_used: The current 1574-parameter design achieved 99.95%, while fixed-scale and norm constraints involving head-0 query row 3 failed despite representing exact Q/K gauges. The successful fixed coefficient in query row 1 provides a stable nonzero pivot for a different, initialization-preserving shear parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9969, "parameters": 1573, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[3,4]` at zero through a shear against query row 0 will produce a 1572-parameter model with at least 99% accuracy, because the transformation exactly preserves attention scores and avoids scaling query row 3.
change: Store 181 learned QKV weights, reconstruct `qkv.weight[3,4]` as zero, and initialize it by shearing query row 3 against anchored query row 0 while applying the inverse shear to key row 8.
mechanism: Second attention-preserving head-0 query/key shear gauge anchor
evidence_used: The first row-3 shear anchor achieved 99.69% at 1573 parameters, whereas directly scaling row 3 through the same column-4 coefficient achieved only 39.61%; this tests another exact shear using the stable fixed `qkv.weight[0,4] = 0.02` pivot.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4032, "parameters": 1572, "training_steps": 4999}

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
