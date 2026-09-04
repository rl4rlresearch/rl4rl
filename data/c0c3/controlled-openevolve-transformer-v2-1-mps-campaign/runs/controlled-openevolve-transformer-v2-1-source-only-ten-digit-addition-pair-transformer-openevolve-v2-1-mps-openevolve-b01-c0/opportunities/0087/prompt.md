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
verified_results: {"accuracy": 0.9998, "parameters": 1597, "training_steps": 4999}
prior_hypothesis: Anchoring the final output coordinate of the third attention projection column will reduce the model to 1597 parameters while retaining at least 99% accuracy, because the identical parameterization succeeded on the first two adjacent columns.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Anchoring the final input coordinate of a second key-projection row will reduce the model to 1600 parameters while maintaining at least 99% accuracy, because each key row has the same LayerNorm-induced redundant direction whose position-independent key offset cancels in attention softmax.
change: Store seven coordinates for each of the first two key rows, fix both eighth coordinates to zero, and transform both rows from the original initialization while shrinking the remaining full-row storage.
mechanism: Second-row anchored key-projection LayerNorm gauge
evidence_used: The identical anchored parameterization on the first key row achieved 100% accuracy at 1601 parameters, whereas the gradient-coupled zero-sum key-row parameterization reached only 95.84%; extending the successful anchored form to one adjacent key row is the most direct one-parameter test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final input coordinate of a third key-projection row will reduce the model to 1599 parameters while maintaining at least 99% accuracy, because the identical anchored parameterization succeeded on the first two key rows.
change: Store seven coordinates for each of the first three key rows, fix their eighth coordinates to zero, and transform all three rows from the original initialization.
mechanism: Third-row anchored key-projection LayerNorm gauge
evidence_used: Anchoring one key row achieved 100% accuracy at 1601 parameters, and extending it to a second row achieved 99.93% at 1600; applying the same successful reduction to the adjacent third row is the most direct one-parameter test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7415, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the first input coordinate of the third key-projection row will reach at least 99% accuracy with 1599 parameters, because it removes the same LayerNorm key gauge as the failed final-coordinate anchor but uses a less disruptive optimization chart.
change: Add a third compressed key row, fixing its first coordinate to zero while retaining final-coordinate anchors for the first two key rows.
mechanism: Coordinate-selective third key-row anchoring
evidence_used: Final-coordinate anchoring succeeded for two key rows at 1600 parameters but failed when extended to the third; the sharply different outcomes from coordinate-selective value-bias removals show that equivalent-size reductions can be highly coordinate-sensitive.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7426, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final input coordinate of the fourth key-projection row will achieve at least 99% accuracy with 1599 parameters, because the first two rows tolerate this reduction while the third row’s failure under two anchoring charts indicates row-specific importance.
change: Keep the third key row fully learned, compress the fourth key row to seven learned coordinates with its eighth fixed at zero, and preserve the initialization-equivalent key transformation.
mechanism: Coordinate-selective fourth key-row anchoring
evidence_used: Final-coordinate anchors on the first two key rows achieved 99.93% at 1600 parameters, whereas anchoring either the first or final coordinate of the third row scored about 74%; this motivates testing the adjacent fourth row without constraining the sensitive third row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7440000000000001, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final input coordinate of the fifth key-projection row will retain at least 99% accuracy with 1599 parameters, because it applies the successful anchored gauge to the first key coordinate of the previously unconstrained second attention head.
change: Compress key rows one, two, and five to seven learned coordinates each, reconstructing their eighth coordinates as zero while preserving initialization-equivalent row transformations.
mechanism: Second-head key-row anchoring
evidence_used: Anchoring the first two key rows achieved 99.93% at 1600 parameters, while anchoring the third or fourth rows produced about 74% accuracy; because those four rows comprise the first attention head, testing the fifth row isolates whether the failure is specific to later coordinates of that head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5467, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the third `fc2` output column will reduce the model to 1599 parameters while retaining at least 99% accuracy, because its eliminated all-feature component is canceled by downstream LayerNorm just like the first two already-centered columns.
change: Store seven coordinates for each of the first three `fc2` columns, reconstruct every column with zero feature mean, and preserve initialization modulo the exact downstream LayerNorm invariance.
mechanism: Third MLP output-column common-mode gauge removal
evidence_used: The verified 1600-parameter model reaches 99.93% while centering the first two `fc2` columns; extending that established parameterization is more informative than further key-row anchoring, which failed on rows three through five.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7454000000000001, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final output coordinate of the first attention projection column will reduce the model to 1599 parameters while retaining at least 99% accuracy, because the removed component produces only a feature-common residual shift that downstream LayerNorm removes.
change: Store seven coordinates for the first attention output-projection column, reconstruct its eighth coordinate as zero, and transform the original initialization by subtracting that coordinate from the column.
mechanism: Anchored attention-output common-mode gauge
evidence_used: The 1600-parameter design reached 99.93%, and analogous final-coordinate anchoring succeeded for two key rows while the more gradient-coupled third `fc2` centering failed; this tests an untouched downstream common-mode gauge using the successful anchored chart.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final output coordinate of the second attention projection column will reduce the model to 1598 parameters while retaining at least 99% accuracy, because it removes the same downstream LayerNorm-invariant common-mode direction as the successfully anchored first column.
change: Store seven coordinates for each of the first two attention output-projection columns, reconstruct both eighth coordinates as zero, and preserve initialization modulo feature-common residual shifts.
mechanism: Second attention-output common-mode gauge anchor
evidence_used: Anchoring the first attention projection column achieved 99.83% accuracy at 1599 parameters; extending that exact successful parameterization to the adjacent column is the most direct one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final output coordinate of the third attention projection column will reduce the model to 1597 parameters while retaining at least 99% accuracy, because the identical parameterization succeeded on the first two adjacent columns.
change: Store seven coordinates for each of the first three attention output-projection columns, reconstruct their eighth coordinates as zero, and preserve initialization modulo downstream LayerNorm-invariant feature-common residual shifts.
mechanism: Third attention-output common-mode gauge anchor
evidence_used: Anchoring the first projection column achieved 99.83% at 1599 parameters, and extending the same construction to the second achieved 99.89% at 1598; testing the adjacent third column is the most direct one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final output coordinate of the fourth attention projection column will reduce the model to 1596 parameters while retaining at least 99% accuracy, because the identical parameterization succeeded sequentially on the first three adjacent columns.
change: Store seven coordinates for each of the first four attention output-projection columns, reconstruct their eighth coordinates as zero, and preserve initialization modulo downstream LayerNorm-invariant feature-common residual shifts.
mechanism: Fourth attention-output common-mode gauge anchor
evidence_used: Anchoring the first three attention projection columns successively achieved 99.83%, 99.89%, and 99.98% accuracy at 1599, 1598, and 1597 parameters; extending the same construction to the adjacent fourth column is the most direct one-parameter test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.845, "parameters": 1596, "training_steps": 4999}



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
