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
verified_results: {"accuracy": 0.9979, "parameters": 1583, "training_steps": 4999}
prior_hypothesis: Fixing `qkv.weight[0,4]` at 0.02 while inversely rescaling head-0 key row 0 will reduce the model to 1583 parameters and retain at least 99% accuracy, because this consumes the remaining query-key scale gauge without eliminating another query input direction.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `ln1.weight[7]` alongside coordinates 0, 2, 4, 5, and 6 will reduce the model from 1587 to 1586 parameters while retaining at least 99% accuracy, because coordinate 7 is the only untested first-LayerNorm gain and its scale is absorbable by QKV input column 7.
change: Store only gains 1 and 3 as learned parameters and reconstruct all other first-LayerNorm gains as fixed ones, while continuing to fix shift coordinate 3.
mechanism: Final-coordinate attention-input gain anchoring
evidence_used: The current design achieved 99.63% after fixing gains 0, 2, 4, 5, and 6; although additional fixes at gains 1 and 3 failed, prior coordinate-sensitive results show those failures do not determine whether the sole remaining coordinate 7 is removable.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9628, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing `ln1.weight[7]` with learned gain coordinate 1 will reduce the model to 1586 parameters while retaining at least 99% accuracy, because QKV columns can absorb coordinate-specific scaling while the shared value remains trainable.
change: Store only first-LayerNorm gains 1 and 3, and reuse gain 1 for coordinate 7 instead of fixing coordinate 7 at one.
mechanism: Shared adaptive attention-input gain
evidence_used: Independently fixing coordinate 7 reduced accuracy to 96.28%, while the 1587-parameter design reached 99.63%; sharing preserves adaptive scaling for coordinate 7 while retaining the especially sensitive coordinate 3 independently.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9867, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing `ln1.weight[7]` with learned gain coordinate 3 will reduce the model from 1587 to 1586 parameters while maintaining at least 99% accuracy, because coordinate 3 is substantially more optimization-sensitive than coordinate 1 and may provide a better adaptive scale for coordinate 7.
change: Store only first-LayerNorm gains 1 and 3, and reuse gain 3 for coordinate 7 instead of learning coordinate 7 independently.
mechanism: Sensitive-coordinate adaptive gain sharing
evidence_used: Fixing gain 7 reached 96.28%, while sharing it with gain 1 improved accuracy to 98.67%, showing that adaptive sharing recovers substantial performance. Independently fixing gain 3 caused the largest tested degradation, to 74.01%, making its learned scale the most informative remaining sharing candidate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing gain 1 for gain 7 while detaching the coordinate-7 branch will achieve at least 99% accuracy with 1586 parameters; ordinary sharing reached 98.67% versus 96.28% when fixed, suggesting the learned value helps but its additional gradient may create harmful optimization interference.
change: Store only gains 1 and 3, reconstruct gain 7 from a detached view of gain 1, and keep gains 0, 2, 4, 5, and 6 fixed.
mechanism: Gradient-isolated adaptive gain sharing
evidence_used: Sharing gain 7 with gain 1 recovered accuracy from 96.28% to 98.67%, whereas sharing with gain 3 fell to 90%; the existing embedding anchors also use detachment specifically to preserve shared values without additional gradient coupling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing learned gain 3 for gains 1 and 7 through detached views will reduce the model to 1585 parameters while retaining at least 99% accuracy, because detaching gain 7 previously improved sharing from 98.67% to 99.90%, and gain 3 is the most optimization-sensitive remaining gain.
change: Store only first-LayerNorm gain 3, use it normally at coordinate 3, and reuse detached copies at coordinates 1 and 7 while keeping the other gains fixed.
mechanism: Gradient-isolated sensitive-gain reuse
evidence_used: Fixing gain 3 reached only 74.01%, so it should remain directly learned; meanwhile, gradient-isolating the gain-7 reuse eliminated the harmful coupling that caused ordinary sharing with gain 1 to miss the threshold.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9853000000000001, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[0,0]` at zero will reduce the model to 1585 parameters while retaining at least 99% accuracy, because an invertible query/key basis transformation can absorb one query-coordinate constraint without reducing the attention score functions representable by the head.
change: Replace the QKV linear layer with an otherwise identical learned projection whose first query weight is fixed at zero, while preserving initialization random-number consumption and all 191 remaining learned QKV weights.
mechanism: Query-key basis gauge anchoring
evidence_used: The 1586-parameter design achieved 99.90%, while eliminating another LayerNorm gain through detached sharing reached only 98.53%; this tests a separate attention redundancy while preserving both empirically sensitive gains.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9954999999999999, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[1,0]` alongside `qkv.weight[0,0]` will reduce the model to 1584 parameters while retaining at least 99% accuracy, because an invertible basis transformation can orient the four-dimensional head-0 query column into the remaining two coordinates without changing attention scores.
change: Store 190 learned QKV weights, reconstruct flattened QKV weights 0 and 8 as zeros, and preserve ordinary initialization random-number consumption for every learned weight.
mechanism: Second query-key basis gauge anchor
evidence_used: The first query-weight anchor achieved 99.55% at 1585 parameters, while the attempted LayerNorm-gain reduction reached only 98.53%; extending the successful query-key gauge mechanism is therefore the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9717, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[0,2]` alongside `qkv.weight[0,0]` will reduce the model to 1584 parameters while retaining at least 99% accuracy, because a query-basis covector can be oriented orthogonally to two input-column vectors without suppressing a second query coordinate.
change: Reconstruct flattened QKV weights 0 and 2 as zeros while learning the other 190 weights and preserving ordinary initialization random-number consumption.
mechanism: Two-column query-row gauge anchoring
evidence_used: The first query anchor achieved 99.55% at 1585 parameters, whereas anchoring a second query coordinate on the same input column reached only 97.17%; anchoring a second input column on the already-constrained query coordinate tests a distinct gauge layout, and input coordinate 2 has already tolerated a fixed first-LayerNorm gain.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[0,4]` alongside `qkv.weight[0,0]` and `qkv.weight[0,2]` will reduce the model to 1583 parameters while retaining at least 99% accuracy, because a four-dimensional query-basis covector can be oriented orthogonally to three input-column vectors.
change: Store 189 learned QKV weights, reconstruct flattened QKV weights 0, 2, and 4 as zeros, and preserve ordinary initialization random-number consumption.
mechanism: Third same-row query-basis gauge anchor
evidence_used: Anchoring weights 0 and 2 in the same query row achieved 100% accuracy at 1584 parameters, whereas anchoring a second query coordinate on input column 0 reached only 97.17%; extending the successful same-row layout to coordinate 4 is the most direct capacity test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4131, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[4,0]` in the previously unconstrained second attention head will reduce the model to 1583 parameters while retaining at least 99% accuracy, because each head has an independent query-key basis gauge and the unsuccessful anchors constrained additional structure within the first head.
change: Store 189 learned QKV weights and reconstruct flattened weights 0, 2, and 32 as zeros while preserving ordinary initialization random-number consumption.
mechanism: Independent-head query-basis gauge anchoring
evidence_used: Anchoring weights 0 and 2 in head 0 achieved 100% accuracy, whereas anchoring row 1 of that same head reached 97.17% and adding a third same-row constraint reached 41.31%; testing the untouched head isolates an independent gauge redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7390000000000001, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the model to 1583 parameters by fixing `qkv.weight[0,4]` will retain at least 99% accuracy when initialization orthogonally transforms both head-0 query and key bases, preserving their initial attention scores instead of simply deleting a random query coefficient.
change: Fix flattened QKV weights 0, 2, and 4, store the remaining 189 weights, and initialize the constrained head through a joint orthogonal query-key basis transformation.
mechanism: Function-preserving third query-basis gauge anchor
evidence_used: The two-anchor layout at weights 0 and 2 achieved 100% accuracy, while naively adding weight 4 fell to 41.31%; preserving the unconstrained initialization’s attention function directly tests whether that failure arose from the uncompensated initialization disturbance rather than the gauge constraint itself.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7356, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[0,4]` at 0.02 while inversely rescaling head-0 key row 0 will reduce the model to 1583 parameters and retain at least 99% accuracy, because this consumes the remaining query-key scale gauge without eliminating another query input direction.
change: Store 189 learned QKV weights, reconstruct weights 0 and 2 as zeros and weight 4 as 0.02, and initialize the corresponding query/key rows with reciprocal scaling that preserves the successful 1584-parameter model’s initial attention scores.
mechanism: Nonzero query-key scale-gauge anchoring
evidence_used: The two-zero layout reached 100% at 1584 parameters, whereas making weight 4 a third zero reached 41.31% and function-preserving orthogonal initialization only recovered 73.56%. A nonzero anchor preserves weight 4’s contribution while using scale rather than another directional constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1583, "training_steps": 4999}



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
