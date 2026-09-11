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
verified_results: {"accuracy": 0.9981, "parameters": 1586, "training_steps": 4999}
prior_hypothesis: Compressing the eighth `fc2` output column with a second-coordinate anchor and virtual full-coordinate AdamW dynamics will reduce the model to 1586 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Representing the twelfth `fc2` column in the symmetric zero-mean gauge while emulating full eight-coordinate AdamW updates will reach at least 99% accuracy with 1589 parameters.
change: Replace the two-column `fc2_rest` tensor with a full eleventh column and a seven-parameter centered twelfth column, then give the twelfth column virtual full-coordinate moments, exact gradient-norm accounting, and quotient-projected AdamW updates.
mechanism: Centered optimizer-preserving twelfth MLP output-column quotient
evidence_used: Anchored virtual compression of column twelve reached 97.15%, the closest failed 1589-parameter result; testing the same redundant column with a symmetric centered gauge is therefore more informative than another key row after rows three and four achieved only 54.56% and 74.53%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2353, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the twelfth `fc2` column at its first output coordinate will reduce the model to 1589 parameters while reaching at least 99% accuracy, showing that the prior twelfth-column failure was sensitive to the chosen gauge representative.
change: Replace the two full remaining `fc2` columns with a full eleventh column and a seven-parameter twelfth column whose first coordinate is fixed to zero, then apply the existing virtual full-coordinate AdamW dynamics.
mechanism: First-coordinate optimizer-preserving twelfth-column gauge
evidence_used: Last-coordinate compression of column twelve reached 97.15%, the closest 1589-parameter failure, while its symmetric centered gauge reached only 23.53%; testing another asymmetric anchor is the most informative remaining variation of that near-success.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the eleventh `fc2` column at its first output coordinate, as already succeeded for the twelfth column, will reduce the model to 1588 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for both the eleventh and twelfth `fc2` columns, reconstruct each with a zero first coordinate, and include both in the established virtual full-coordinate optimizer.
mechanism: First-coordinate anchored eleventh-column quotient
evidence_used: The eleventh column failed with a final-coordinate anchor at 77.85%, while changing the twelfth column from a final-coordinate anchor at 97.15% to a first-coordinate anchor achieved 99.64%; this directly motivates testing the same gauge representative on the eleventh column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6546, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the eleventh `fc2` column at its second output coordinate will reduce the model to 1588 parameters while retaining at least 99% accuracy, demonstrating that its prior failures were sensitive to the chosen gauge representative.
change: Store seven relative coordinates for the eleventh `fc2` column, reconstruct it with a zero second coordinate, and apply the established virtual full-coordinate AdamW dynamics.
mechanism: Second-coordinate anchored eleventh-column gauge quotient
evidence_used: Changing the twelfth column from a final-coordinate anchor at 97.15% to a first-coordinate anchor achieved 99.64%; since eleventh-column final- and first-coordinate anchors failed, the next distinct asymmetric anchor is the most informative remaining one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9940000000000001, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the seventh `fc2` output column at its second coordinate will reduce the model to 1587 parameters while retaining at least 99% accuracy, because the same anchor rescued eleventh-column compression after its first- and final-coordinate anchors failed.
change: Replace the full seventh/eighth-column tensor with a seven-parameter seventh column anchored at its second coordinate and a full eighth column, then include the seventh column in virtual full-coordinate AdamW optimization.
mechanism: Second-coordinate anchored seventh-column gauge quotient
evidence_used: Seventh-column compression with a final-coordinate anchor reached only 74.55%, but the eleventh column improved from 77.85% and 65.46% with final and first anchors to 99.40% with a second-coordinate anchor, directly motivating the same gauge representative for the seventh column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the eighth `fc2` output column at its second coordinate will reduce the model to 1586 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for the eighth `fc2` column, reconstruct it with a zero second coordinate, initialize it in that gauge, and include it in virtual full-coordinate AdamW optimization.
mechanism: Second-coordinate anchored eighth-column gauge quotient
evidence_used: Seventh-column accuracy improved from 74.55% with a final-coordinate anchor to 99.98% with a second-coordinate anchor; since the eighth column’s final-coordinate anchor achieved only 59.66%, applying the same rescued gauge is the most informative next reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compressing the eighth `fc2` output column with a second-coordinate anchor and virtual full-coordinate AdamW dynamics will reduce the model to 1586 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for the eighth `fc2` column, reconstruct it with a zero second coordinate, initialize it in that gauge, and include it in virtual full-coordinate optimization.
mechanism: Second-coordinate anchored eighth-column gauge quotient
evidence_used: A second-coordinate anchor raised seventh-column compression from 74.55% to 99.98%; the analogous eighth-column change remains unresolved because its previous implementation could not be verified.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the fifth `fc2` output column with a second-coordinate anchor and virtual full-coordinate AdamW dynamics will reduce the model to 1585 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for the fifth `fc2` column, reconstruct it with a zero second coordinate, initialize it in that gauge, and include it in virtual full-coordinate optimization.
mechanism: Second-coordinate anchored fifth-column gauge quotient
evidence_used: The fifth, seventh, and eighth columns previously failed under compression, but second-coordinate anchors subsequently raised the seventh and eighth columns to 99.98% and 99.81%; applying the same successful gauge to the remaining failed fifth column is the most informative next reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compressing the fifth `fc2` output column with a second-coordinate anchor and virtual full-coordinate AdamW dynamics will reduce the model to 1585 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for the fifth `fc2` column, reconstruct it with a zero second coordinate, initialize it in that gauge, and include it in virtual full-coordinate optimization.
mechanism: Second-coordinate anchored fifth-column gauge quotient
evidence_used: Second-coordinate anchors enabled seventh- and eighth-column compression at 99.98% and 99.81%; the analogous fifth-column experiment was not verified, so a corrected implementation is the most informative next reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compressing the fifth `fc2` output column with a second-coordinate anchor and virtual full-coordinate AdamW dynamics will reduce the model to 1585 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for the fifth `fc2` column, reconstruct it with a zero second coordinate, initialize it in that gauge, and add it to virtual full-coordinate optimization.
mechanism: Second-coordinate anchored fifth-column gauge quotient
evidence_used: Second-coordinate anchors enabled seventh- and eighth-column compression at 99.98% and 99.81%; the analogous fifth-column experiment was never verified, making a corrected implementation the most informative next reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compressing the fifth `fc2` output column with the successful second-coordinate anchor and virtual full-coordinate AdamW dynamics will reduce the model from 1586 to 1585 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for the fifth `fc2` column, reconstruct its second coordinate as zero, initialize it in that gauge, and include it in virtual full-coordinate optimization.
mechanism: Second-coordinate anchored fifth-column gauge quotient
evidence_used: Second-coordinate anchoring preserved 99.98% and 99.81% accuracy for the seventh and eighth columns; the analogous fifth-column experiment remains unresolved because its prior implementations could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compressing the fifth `fc2` output column with the successful second-coordinate anchor and virtual full-coordinate AdamW dynamics will reduce the model from 1586 to 1585 parameters while retaining at least 99% accuracy.
change: Store seven relative coordinates for the fifth `fc2` column, reconstruct its second coordinate as zero, initialize it in that gauge, and include it in virtual full-coordinate optimization.
mechanism: Second-coordinate anchored fifth-column gauge quotient
evidence_used: Second-coordinate anchoring preserved 99.98% and 99.81% accuracy for the seventh and eighth columns; the analogous fifth-column experiment remains unresolved only because its prior implementations could not be verified.
result: the implementation could not be verified



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
