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
verified_results: {"accuracy": 0.9959, "parameters": 1572, "training_steps": 4999}
prior_hypothesis: Fixing `qkv.weight[2,4]` at zero through a shear against query row 0 will produce a 1572-parameter model with at least 99% accuracy, because row 2 already tolerates its scale anchor and the transformation uses the stable fixed `qkv.weight[0,4] = 0.02` pivot while exactly preserving attention scores.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `q_bias[4]` at zero will reduce the model to 1576 parameters while retaining at least 99% accuracy, because head 0 already tolerates three fixed query-bias coordinates and head 1 retains three learned bias coordinates after this first analogous anchor.
change: Store four learned query-bias values and reconstruct coordinates 0, 1, 2, and 4 as zero without disturbing the zero-bias initialization.
mechanism: Independent-head query-bias gauge anchor
evidence_used: The verified 1577-parameter design achieves 99.13% accuracy with query-bias coordinates 0 through 2 already fixed at zero; applying the first bias anchor to the independent second head tests the same successful redundancy while avoiding the failed fourth head-0 scale anchor.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7001999999999999, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[3,4]` at 0.02 with reciprocal scaling of head-0 key row 3 will produce a 1576-parameter model with at least 99% accuracy, because it uses the exact fourth head-0 coordinate-scale gauge without repeating the unsuccessful column-0 anchor.
change: Store 182 learned QKV weights, reconstruct `qkv.weight[3,4]` as 0.02, and initialize query row 3 and key row 11 with reciprocal scaling that preserves initial attention scores.
mechanism: Alternative-coefficient query–key scale-gauge anchor
evidence_used: The fourth scale anchor in head 1 passed at 99.13%, showing four coordinate anchors can work, while the first head-0 anchor at column 4 previously passed at 99.79%; this motivates testing column 4 for head-0 row 3 after its column-0 anchor failed at 71.61%. The subsequent query-bias reduction failed at 70.02%, favoring another function-preserving scale gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3961, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[16,0]` at 0.02 while reciprocally scaling output-projection column 0 will reduce the model to 1576 parameters and retain at least 99% accuracy because it preserves the initialized network function and uses the independent value/output gauge instead of the optimization-sensitive fourth head-0 query gauge.
change: Store 182 learned QKV weights, reconstruct value weight 128 as 0.02, scale value row 16 during initialization, and inversely scale output-projection column 0 after its ordinary initialization.
mechanism: Function-preserving value/output coordinate-scale gauge
evidence_used: Seven function-preserving query/key scale anchors produced the verified 1577-parameter model at 99.13%, while both tested versions of a fourth head-0 query anchor failed; moving to the analogous independent value/output scale symmetry is the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5537, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining head-0 query row 3 and its matching key row to equal norms will produce a 1576-parameter model with at least 99% accuracy, because it removes the same exact scale redundancy as the failed coefficient anchors while preserving initialization and avoiding a potentially ill-conditioned fixed coefficient.
change: Store 182 learned QKV values, encode key row 11’s direction with seven stereographic coordinates, derive its norm from query row 3, and initialize the pair through reciprocal scaling that exactly preserves attention scores.
mechanism: Balanced query–key norm gauge with stereographic direction coordinates
evidence_used: The seven coefficient-based Q/K scale anchors reached 99.13% at 1577 parameters, proving this gauge family is viable, but both fixed-coefficient attempts for head-0 row 3 failed at 71.61% and 39.61%. A balanced norm constraint tests whether those failures were caused by the coordinate-pivot parameterization rather than loss of necessary capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5456, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Folding the tied value/projection bias into a seven-parameter mean-zero post-projection bias will produce a 1576-parameter model with at least 99% accuracy, because with zero dropout each attention row sums to one, while the final LayerNorm removes the remaining uniform-shift degree of freedom.
change: Replace the eight-parameter bias shared by the value and output projections with seven learned coordinates that reconstruct a zero-sum attention output bias.
mechanism: Mean-zero folded attention output bias
evidence_used: The current 1577-parameter design reached 99.13%, whereas the multiplicative value/output weight gauge reached only 55.37%; this additive reparameterization preserves the zero initialization and avoids reciprocal scaling.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6492, "parameters": 1576, "training_steps": 4999}

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
