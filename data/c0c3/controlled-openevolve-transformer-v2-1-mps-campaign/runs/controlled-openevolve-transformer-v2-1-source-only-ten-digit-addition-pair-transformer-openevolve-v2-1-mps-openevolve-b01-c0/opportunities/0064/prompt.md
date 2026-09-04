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
verified_results: {"accuracy": 0.9989, "parameters": 1606, "training_steps": 4999}
prior_hypothesis: Reusing `attn.qv_bias[4]` for the fifth-coordinate `ln2` contrast will reduce the model to 1606 parameters while maintaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying the remaining third-coordinate `ln1` bias contrast to the third query-bias parameter will reduce the model to 1612 parameters while retaining at least 99% accuracy, because the contrast remains learned and its projection-space effect is gauge-coupled to the query/value biases.
change: Remove the standalone `ln1` bias parameter and reuse `attn.qv_bias[2]` as its learned third-coordinate contrast.
mechanism: Shared pre-attention/query-bias gauge
evidence_used: The standalone third-coordinate contrast achieved 99.90% at 1613 parameters, whereas fixing it to zero reached 97.43%; sharing it with an existing zero-initialized query bias tests whether retaining an adaptive contrast, rather than an independent degree of freedom, supplies the optimization benefit.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second value-bias coordinate to zero will reduce the model to 1611 parameters while retaining at least 99% accuracy, because attention’s position-independent value bias is representable by the learned output-projection bias.
change: Store six value-bias coordinates instead of seven and reconstruct the final two coordinates as fixed zeros, leaving all query biases and the shared `ln1` contrast unchanged.
mechanism: Second value/output-bias redundancy
evidence_used: The 1612-parameter design achieved 100% accuracy while already fixing the final value-bias coordinate and retaining a seven-degree centered projection bias; extending that successful redundancy by one adjacent coordinate is the smallest informative reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8159000000000001, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the seventh value-bias coordinate to the seventh projection-bias parameter will reduce the model to 1611 parameters while retaining at least 99% accuracy, because it preserves an adaptive position-independent value offset without adding an independent degree of freedom.
change: Store six independent value-bias coordinates and reconstruct the seventh from `proj_bias[-1]`, while retaining the existing fixed-zero eighth value bias.
mechanism: Shared value/output-projection bias gauge
evidence_used: Fixing the seventh value bias to zero reduced accuracy to 81.59%, while sharing the otherwise necessary `ln1` contrast with an existing query bias achieved 100% at 1612 parameters; this tests the same successful adaptive-sharing strategy on the most recent failed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the sixth value-bias coordinate to the sixth projection-bias parameter will reduce the model to 1610 parameters while retaining at least 99% accuracy, because the successful seventh-coordinate tie shows that adaptive sharing can preserve capacity lost when a value bias is fixed.
change: Store five independent value-bias coordinates and reconstruct the sixth and seventh coordinates from the final two projection-bias parameters, retaining the fixed-zero eighth coordinate.
mechanism: Adjacent value/output-projection bias sharing
evidence_used: Tying the seventh value bias to `proj_bias[-1]` achieved 99.98% at 1611 parameters, whereas fixing that value bias to zero achieved only 81.59%; extending the same sharing mechanism to the adjacent sixth coordinate is the smallest evidence-backed reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.28309999999999996, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the fifth active `ln2` bias coordinate instead of the fourth will reduce the model to 1610 parameters while maintaining at least 99% accuracy, indicating that the prior four-contrast failure was coordinate-specific.
change: Store four learned `ln2` bias contrasts and place them in the first three and fifth active coordinates before centering.
mechanism: Coordinate-swapped fourth post-attention LayerNorm bias reduction
evidence_used: A coordinate swap rescued the analogous two-contrast `ln1` reduction from 96.77% to 99.97%; this directly motivates testing an alternative contrast basis for the previously failed fourth `ln2` reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing `fc1.bias[4]` as the fifth-coordinate `ln2` contrast will reduce the model to 1609 parameters while retaining at least 99% accuracy, because it preserves the adaptive coordinate that rescued the 1610-parameter design.
change: Store three independent `ln2` contrasts and reconstruct the fifth active coordinate from the existing fifth MLP input-bias parameter.
mechanism: Shared post-attention LayerNorm/MLP bias contrast
evidence_used: Retaining the fifth `ln2` coordinate achieved 99.77% at 1610 parameters, while sharing the required `ln1` contrast with an existing query bias achieved 100% where fixing it to zero achieved only 97.43%; this motivates adaptive sharing rather than deleting the important `ln2` contrast.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6807, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the first, third, and fifth active `ln2` bias coordinates will reduce the model to 1609 parameters while maintaining at least 99% accuracy.
change: Remove the second-coordinate `ln2` bias degree of freedom while preserving learned contrasts in the first, third, and fifth active coordinates before centering.
mechanism: Coordinate-selected three-contrast post-attention LayerNorm bias
evidence_used: Swapping the fourth `ln2` coordinate for the fifth raised the 1610-parameter design to 99.77%, and the analogous `ln1` evidence showed that retaining the third rather than second coordinate rescued a reduced-contrast design from 96.77% to 99.97%; this motivates preserving the third and fifth coordinates while testing removal of the second.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining only the third and fifth active `ln2` bias coordinates will reduce the model to 1608 parameters while maintaining at least 99% accuracy.
change: Remove the first-coordinate `ln2` bias degree of freedom and reconstruct the seven active coordinates from two learned contrasts in positions three and five.
mechanism: Third-and-fifth-coordinate post-attention LayerNorm bias
evidence_used: The fifth coordinate rescued the four-contrast `ln2` design to 99.77%, while the third coordinate was the strongest reduced-bias coordinate for `ln1`; the current first/third/fifth design reaches 99.86%, making removal of the less-supported first coordinate the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the third-coordinate `ln2` contrast while retaining the fifth will reduce the model to 1607 parameters and maintain at least 99% accuracy.
change: Store one learned `ln2` bias contrast and place it in the fifth active coordinate before centering.
mechanism: Fifth-coordinate-only post-attention LayerNorm bias
evidence_used: Adding the fifth coordinate rescued the reduced `ln2` design to 99.77%, and the current third/fifth design reaches 99.90%; this makes the less directly supported third coordinate the most informative next removal.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9497, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the third-coordinate `ln2` contrast while removing the fifth will reduce the model to 1607 parameters and maintain at least 99% accuracy.
change: Store one learned `ln2` bias contrast and place it only in the third active coordinate before centering.
mechanism: Third-coordinate-only post-attention LayerNorm bias
evidence_used: The complementary fifth-only design scored 94.97%, while the third coordinate was the strongest reduced-bias coordinate for `ln1`; testing third-only isolates whether it supplies the essential contribution in the 99.90% third/fifth design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6685, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing `attn.qv_bias[2]` for the third-coordinate `ln2` contrast while retaining an independent fifth-coordinate contrast will reduce the model to 1607 parameters and maintain at least 99% accuracy.
change: Remove one standalone `ln2` parameter, source its third-coordinate contrast from the parameter already shared by `ln1`, and keep its fifth-coordinate contrast independently learned.
mechanism: Shared third-coordinate normalization contrast
evidence_used: Neither third-only nor fifth-only `ln2` met the threshold, indicating both contrasts contribute, while sharing the required third-coordinate `ln1` contrast with `qv_bias[2]` achieved 100% at one fewer parameter; this motivates adaptive sharing of the aligned `ln2` contrast rather than deleting it.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9936, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing `attn.qv_bias[4]` for the fifth-coordinate `ln2` contrast will reduce the model to 1606 parameters while maintaining at least 99% accuracy.
change: Remove the standalone `ln2` bias parameter and source its third- and fifth-coordinate contrasts from the corresponding learned query-bias coordinates.
mechanism: Coordinate-aligned normalization/query-bias sharing
evidence_used: Sharing the third-coordinate `ln2` contrast with `qv_bias[2]` retained both required contrasts and achieved 99.36% at 1607 parameters; applying the same successful coordinate-aligned sharing mechanism to the remaining fifth contrast is the most direct one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1606, "training_steps": 4999}



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
