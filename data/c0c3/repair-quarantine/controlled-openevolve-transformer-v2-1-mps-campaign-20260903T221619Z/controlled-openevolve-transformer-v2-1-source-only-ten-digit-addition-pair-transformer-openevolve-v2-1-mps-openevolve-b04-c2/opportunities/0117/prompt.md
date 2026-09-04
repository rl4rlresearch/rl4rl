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
verified_results: {"accuracy": 0.9983, "parameters": 1565, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified vocabulary-centering design while reducing `ln2.bias` from five to four orthonormal coordinates will produce a 1,565-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1566, "training_steps": 4999}
prior_hypothesis: Removing one more `ln2.bias` direction from the qualified 1,567-parameter design will produce a 1,566-parameter model with at least 99% accuracy, because every constant shift entering `fc1` is representable by its unrestricted bias.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1567, "training_steps": 4999}
prior_hypothesis: Removing one additional `ln2` bias direction will reduce the verified 1,568-parameter model to 1,567 parameters while maintaining at least 99% accuracy, because any constant shift it induces before `fc1` is exactly absorbable by the unrestricted `fc1` bias.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9953, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,583-parameter design by quotienting positional row 6 will produce a 1,582-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first and final LayerNorms.

## Recent verification evidence

RECENT RESULT
hypothesis: Quotienting positional row 14 while retaining sensitive rows 7 and 13 unrestricted will reduce the verified 1,576-parameter model to 1,575 parameters while maintaining at least 99% accuracy.
change: Add row 14 to the compact positional rows, reconstruct the original positional order exactly, and extend the full-coordinate AdamW mapping accordingly.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The current 1,576-parameter model achieved 99.99%; positional compaction through row 12 achieved 99.78%, and bypassing sensitive row 7 before compacting row 8 achieved 99.80%. Row 13 was demonstrably sensitive, whereas prior row-14 attempts were unverifiable and therefore provide no contrary accuracy result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding the verified head-1 value rotation and quotienting positional row 14 with index-based reconstruction will produce a 1,575-parameter model with at least 99% accuracy.
change: Apply the qualified ninth value-basis rotation, compact positional row 14, and use explicit row-index maps for inference and full-coordinate AdamW updates.
mechanism: Indexed row-14 positional common-mode quotient atop the qualified value gauge
evidence_used: The head-1 rotation achieved 99.99% accuracy at 1,576 parameters. Row-12 positional compaction achieved 99.78%; row 13 was sensitive, while row-14 attempts produced no accuracy result because they could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding the qualified head-1 value rotation and compacting positional rows 12 and 14 while preserving rows 7 and 13 will produce a 1,575-parameter model with at least 99% accuracy.
change: Apply the verified ninth value-basis rotation, then compact positional rows 0–6, 8–12, 14, and the final nine rows with balanced reconstruction and full-coordinate AdamW updates.
mechanism: Head-local value gauge plus skip-sensitive-row positional common-mode quotient
evidence_used: The row-12 design with the added head-1 rotation achieved 99.99% accuracy at 1,576 parameters. Row 13 was optimization-sensitive, while row-14 attempts produced no accuracy measurement and therefore provide no contrary evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding the verified head-1 value rotation and quotienting positional row 14 while retaining rows 7 and 13 unrestricted will produce a 1,575-parameter model with at least 99% accuracy.
change: Add the qualified ninth value-basis rotation; compact positional rows 0–6, 8–12, 14, and the final nine rows; and reconstruct gradients, updates, and inference rows in their exact original order.
mechanism: Head-local value gauge plus explicit skip-sensitive positional quotient
evidence_used: The added head-1 rotation achieved 99.99% accuracy at 1,576 parameters, while row-12 positional compaction achieved 99.78%. Row 13 was accuracy-sensitive, but row-14 attempts yielded no verification result and therefore leave this parameter reduction untested.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Fixing the final head-0 value-gauge coordinate will produce a 1,575-parameter model with at least 99% accuracy.
change: Add a third-column rotation between the two head-0 value rows already zero on columns 0 and 1, then omit the resulting zero coefficient.
mechanism: Head-0 terminal value-basis triangularization
evidence_used: The current nine-rotation model achieved 99.99% at 1,576 parameters, while adding the tenth rotation in head 1 reached only 98.58%. Head 0 already tolerates deeper triangularization, motivating testing its remaining exact gauge direction instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6724, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding the qualified head-1 second-column rotation will reduce the model from 1,577 to 1,576 parameters while maintaining at least 99% accuracy.
change: Omit one additional value-weight coefficient through a head-1 rotation with the matching output-projection counter-rotation.
mechanism: Head-local value-basis gauge fixation
evidence_used: The identical ninth value-coordinate rotation achieved 99.99% accuracy with 1,576 parameters; further triangularization failed, so this applies only the verified reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the tied token embedding across vocabulary, absorbing its removed mean into positional embeddings, and retaining the verified row-12 and head-1 value reductions will produce a 1,568-parameter model with at least 99% accuracy.
change: Apply the qualified ninth value rotation and row-12 positional quotient, then remove eight redundant tied-embedding parameters using an orthonormal zero-sum vocabulary basis with full-coordinate AdamW updates.
mechanism: Vocabulary-common embedding gauge quotient
evidence_used: The combined row-12 and head-1 rotation design achieved 99.99% accuracy at 1,576 parameters. A vocabulary-common embedding shift changes output logits only by a softmax-invariant common scalar and can be absorbed from every input by the positional embeddings, motivating an exact eight-parameter quotient without the failed deeper value or query-key constraints.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1,568-parameter design and additionally removing positional row 14’s LayerNorm-invariant common mode will produce a 1,567-parameter model with at least 99% accuracy.
change: Add the verified head-1 value rotation and row-12 positional quotient, center the tied token embedding across vocabulary, and compact positional row 14 with exact inference reconstruction and full-coordinate AdamW moments.
mechanism: Vocabulary-common embedding gauge plus indexed row-14 positional quotient
evidence_used: The combined token-centering, row-12, and head-1 design achieved 100% accuracy at 1,568 parameters. Row 13 was accuracy-sensitive, whereas every row-14 attempt was unverifiable and supplied no contrary accuracy measurement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Removing one additional `ln2` bias direction will reduce the verified 1,568-parameter model to 1,567 parameters while maintaining at least 99% accuracy, because any constant shift it induces before `fc1` is exactly absorbable by the unrestricted `fc1` bias.
change: Restrict `ln2.bias` from seven to six orthonormal directions while leaving all other verified token, positional, and value-basis reductions unchanged.
mechanism: Downstream-bias LayerNorm gauge quotient
evidence_used: The current 1,568-parameter design achieved 100% accuracy; unlike the failed deeper value rotations and query-key constraint, this reduction targets a representationally redundant affine direction immediately followed by a biased linear layer.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1567, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one more `ln2.bias` direction from the qualified 1,567-parameter design will produce a 1,566-parameter model with at least 99% accuracy, because every constant shift entering `fc1` is representable by its unrestricted bias.
change: Add the verified vocabulary-common embedding quotient and reduce `ln2.bias` to five orthonormal coordinates, preserving full-coordinate AdamW updates.
mechanism: Progressive post-LayerNorm bias quotient
evidence_used: Vocabulary centering achieved 100% at 1,568 parameters, and the first additional `ln2.bias` removal achieved 99.90% at 1,567; this tests the next direction of the same downstream-bias redundancy while avoiding failed deeper value and query-key constraints.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1566, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified vocabulary-centering design while reducing `ln2.bias` from five to four orthonormal coordinates will produce a 1,565-parameter model with at least 99% accuracy.
change: Center the tied token embedding across vocabulary, transfer its mean into positional embeddings, use full-coordinate AdamW updates for the compact embedding, and remove one additional `ln2.bias` direction beyond the qualified 1,566-parameter design.
mechanism: Progressive downstream-affine LayerNorm bias quotient
evidence_used: Vocabulary centering achieved 100% accuracy at 1,568 parameters, while successive `ln2.bias` reductions achieved 99.90% at 1,567 and 99.88% at 1,566; the next reduction tests the same downstream-bias redundancy without adding a previously failed value, query-key, or positional constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1565, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1,565-parameter design while reducing `ln2.bias` from four to three orthonormal coordinates will produce a 1,564-parameter model with at least 99% accuracy.
change: Add the qualified head-1 value rotation, vocabulary-centering quotient, and row-12 positional compaction, then remove one additional `ln2.bias` direction while preserving full-coordinate AdamW updates.
mechanism: Progressive downstream-affine LayerNorm bias quotient
evidence_used: The progressive `ln2.bias` reductions achieved 99.90% at 1,567 parameters, 99.88% at 1,566, and 99.83% at 1,565; testing the next single-coordinate reduction is the most direct extension of this consistently qualified mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9862000000000001, "parameters": 1564, "training_steps": 4999}



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
