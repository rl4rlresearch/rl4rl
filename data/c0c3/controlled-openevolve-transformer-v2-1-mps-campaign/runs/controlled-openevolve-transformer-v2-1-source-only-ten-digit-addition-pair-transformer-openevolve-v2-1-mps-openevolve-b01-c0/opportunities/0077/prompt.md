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
verified_results: {"accuracy": 1.0, "parameters": 1601, "training_steps": 4999}
prior_hypothesis: Anchoring the final input coordinate of one key-projection row at zero will reduce the model to 1601 parameters while maintaining at least 99% accuracy, because LayerNorm’s centered activation makes that coordinate redundant up to a position-independent key offset that cancels in attention softmax.

## Recent verification evidence

RECENT RESULT
hypothesis: Reusing the sixth query-bias coordinate as the sixth value-bias coordinate will reduce the model to 1605 parameters while maintaining at least 99% accuracy.
change: Store five independent value-bias coordinates and reconstruct the sixth from `qv_bias[5]`, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.
mechanism: Coordinate-aligned query/value bias sharing
evidence_used: Tying the sixth value bias to a projection-bias parameter failed at 28.31%, but coordinate-aligned sharing of the third and fifth normalization contrasts with query biases achieved 99.89% at 1606 parameters; this tests whether the successful aligned query-sharing mechanism preserves the required adaptive sixth value offset.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9490999999999999, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final `fc2` column will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because it removes the same downstream-LayerNorm-invariant component already removed successfully from the first two columns while avoiding the sensitive third column.
change: Store the middle `fc2` columns in full and represent the final column with seven learned coordinates whose eighth coordinate enforces a zero column sum.
mechanism: Final-column MLP output common-mode gauge removal
evidence_used: Centering the first two `fc2` columns supports 99.89% accuracy at 1606 parameters, whereas centering the third fell to 13.83%; prior LayerNorm results show that changing which coordinate receives an otherwise equivalent reduction can restore accuracy, motivating a distant final-column test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6812, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the first attention projection column will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because its removed all-ones component propagates only as a featurewise common-mode shift that is eliminated by downstream LayerNorms.
change: Represent the first attention output-projection column with seven learned coordinates and reconstruct its eighth coordinate to enforce zero column sum, while preserving the original initialization draw.
mechanism: First-column attention-output common-mode gauge removal
evidence_used: The verified 1606-parameter design achieves 99.89% accuracy with the same centered-column reconstruction applied successfully to the first two `fc2` columns; testing the first column of the attention projection transfers that proven gauge reduction to the other residual projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7056, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering one key-projection row will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because the removed LayerNorm-input gauge component produces only a position-independent key offset that cancels in the attention softmax.
change: Replace the combined QKV weight with equivalent storage that reconstructs the first key row from seven learned coordinates with zero sum, while preserving the original initialization draw and initial model function.
mechanism: Single-row key-projection gauge removal
evidence_used: The 1606-parameter design reached 99.89%, while sharing sensitive value-bias coordinates and constraining attention-output columns failed; this motivates reducing a key-only degree of freedom whose effect cancels directly inside attention rather than constraining the residual output path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9584, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the sixth value-bias coordinate to zero will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because a value bias contributes only a constant attention output that the learned projection bias can absorb, without coupling it to a query coordinate.
change: Store five independent value-bias coordinates instead of six and insert a fixed zero for the sixth coordinate while preserving the existing seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.
mechanism: Uncoupled sixth value-bias removal
evidence_used: The 1606-parameter design achieves 99.89% with the eighth value bias fixed to zero and the seventh shared with projection bias, while sharing the sixth value bias with its query coordinate scored 94.91%; fixing the sixth coordinate isolates whether that failure arose from harmful query/value coupling rather than loss of an independently learned value offset.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the fifth value-bias coordinate to zero will reduce the model to 1604 parameters while maintaining at least 99% accuracy, because value bias contributes only a constant attention output that the learned centered projection bias can absorb.
change: Store four independent value-bias coordinates and insert fixed zeros for the fifth and sixth coordinates, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.
mechanism: Uncoupled fifth value-bias removal
evidence_used: Fixing the sixth value-bias coordinate independently to zero achieved 99.97% at 1605 parameters, whereas coupling it to the aligned query bias achieved only 94.91%; extending the successful uncoupled removal to the adjacent fifth value coordinate is the most direct test of the constant-output redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the fourth value-bias coordinate to zero will reduce the model to 1603 parameters while maintaining at least 99% accuracy, because value bias contributes a constant attention output that the learned centered projection bias can absorb.
change: Store three independent value-bias coordinates and insert fixed zeros for the fourth, fifth, and sixth coordinates, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.
mechanism: Uncoupled fourth value-bias removal
evidence_used: Successive uncoupled removal of the sixth and then fifth value-bias coordinates achieved 99.97% at 1605 parameters and 99.99% at 1604 parameters; removing the adjacent fourth coordinate is the most direct test of whether this demonstrated redundancy extends one coordinate further.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965999999999999, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the third value-bias coordinate to zero will reduce the model to 1602 parameters while maintaining at least 99% accuracy, because value bias contributes a constant attention output that the learned centered projection bias can absorb.
change: Store two independent value-bias coordinates and insert fixed zeros for the third through sixth coordinates, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.
mechanism: Uncoupled third value-bias removal
evidence_used: Successive uncoupled removal of the sixth, fifth, and fourth value-bias coordinates achieved 99.97%, 99.99%, and 99.66% accuracy respectively; extending the same demonstrated redundancy to the adjacent third coordinate is the most informative one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9620000000000001, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final tied-token column will reduce the model to 1602 parameters while maintaining at least 99% accuracy, because subtracting its mean from every embedding feature changes inputs only by a LayerNorm-invariant all-feature shift and logits only by a class-common offset.
change: Store the final token-embedding column with one fewer coordinate, reconstruct it centered, and transfer its initial mean across all embedding features to preserve the initialized model function modulo exact input and output gauges.
mechanism: Joint embedding common-mode gauge removal
evidence_used: The verified 1603-parameter design achieved 99.66% while already using centered tied-token columns and centered positional columns; extending that successful embedding parameterization removes the remaining joint all-feature common shift without constraining the sensitive attention or MLP paths.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1783, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing only the second value-bias coordinate to zero will reduce the model to 1602 parameters while maintaining at least 99% accuracy, because the failed third-coordinate removal may reflect coordinate-specific importance rather than a need for all three remaining learned value offsets.
change: Store the first and third value-bias coordinates, reconstruct the second as zero, and preserve the existing zero fourth-through-sixth and eighth coordinates plus the seventh-coordinate projection-bias tie.
mechanism: Coordinate-selective value-bias removal
evidence_used: Removing value-bias coordinates four through six successively retained 99.66% or better accuracy, while removing the third reduced accuracy to 96.20%; retaining that sensitive third coordinate while testing the second is the most direct way to distinguish localized coordinate importance from a general capacity limit.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1465, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing only the first value-bias coordinate to zero will reduce the model to 1602 parameters while maintaining at least 99% accuracy, because the failures from removing the second or third coordinates indicate coordinate-specific importance while leaving the first coordinate as the remaining untested value-bias redundancy.
change: Store the second and third value-bias coordinates, reconstruct the first as zero, and preserve the existing zero fourth-through-sixth and eighth coordinates plus the seventh-coordinate projection-bias tie.
mechanism: Coordinate-selective first value-bias removal
evidence_used: The 1603-parameter design reached 99.66% after removing value-bias coordinates four through six, while selective removal of the second scored 14.65% and removal of the third scored 96.20%; testing the first coordinate completes the coordinate-selective comparison without constraining any bias already shown to be important.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final input coordinate of one key-projection row at zero will reduce the model to 1601 parameters while maintaining at least 99% accuracy, because LayerNorm’s centered activation makes that coordinate redundant up to a position-independent key offset that cancels in attention softmax.
change: Replace the combined QKV weight with a 1601-parameter representation that stores seven coordinates of the first key row, fixes its eighth coordinate to zero, and transforms the original initialization to preserve the initial attention function.
mechanism: Anchored key-projection LayerNorm gauge
evidence_used: Centering the first key row reached 95.84% at 1605 parameters, making this exact key-only gauge a near-threshold candidate; the sharply different outcomes of coordinate-selective value-bias removals further motivate testing an anchored, less gradient-coupled parameterization instead of the failed zero-sum reconstruction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1601, "training_steps": 4999}



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
