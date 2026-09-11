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
verified_results: {"accuracy": 0.9940000000000001, "parameters": 1588, "training_steps": 4999}
prior_hypothesis: Anchoring the eleventh `fc2` column at its second output coordinate will reduce the model to 1588 parameters while retaining at least 99% accuracy, demonstrating that its prior failures were sensitive to the chosen gauge representative.

## Recent verification evidence

RECENT RESULT
hypothesis: Compressing the seventh `fc2` output column with virtual full-coordinate AdamW dynamics will reduce the model to 1591 parameters while retaining at least 99% accuracy, because the sixth column succeeded despite the fifth column’s failure, indicating neuron-specific sensitivity.
change: Store the seventh `fc2` column with seven anchored coordinates, shift `fc2_rest` to columns eight onward, reconstruct the final coordinate as zero, and include the column in virtual full-coordinate optimization.
mechanism: Optimizer-preserving seventh MLP output-column gauge quotient
evidence_used: The current design compressed the sixth `fc2` column and achieved 99.96% accuracy at 1592 parameters, whereas the fifth-column attempt achieved only 54.49%; testing the adjacent seventh column is the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7454999999999999, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the eighth `fc2` output column with virtual full-coordinate AdamW dynamics will reduce the model to 1591 parameters while retaining at least 99% accuracy, because the tested even-numbered `fc2` columns four and six succeeded while columns three, five, and seven failed.
change: Keep the seventh `fc2` column full, store the eighth with seven anchored coordinates, shift `fc2_rest` to columns nine onward, reconstruct the eighth coordinate as zero, and optimize it with virtual full-coordinate moments.
mechanism: Optimizer-preserving eighth MLP output-column gauge quotient
evidence_used: The sixth-column compression achieved 99.96% accuracy at 1592 parameters, whereas the adjacent fifth and seventh columns achieved only 54.49% and 74.55%; testing column eight is the clearest test of the observed neuron-specific, even-column success pattern.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5966, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the ninth `fc2` output column while retaining failed columns five, seven, and eight in full will reduce the model to 1591 parameters and maintain at least 99% accuracy, confirming that compressibility remains neuron-specific.
change: Store `fc2` columns seven and eight in full, store column nine with seven anchored coordinates, shift `fc2_rest` to columns ten onward, and apply virtual full-coordinate AdamW updates to column nine.
mechanism: Optimizer-preserving ninth MLP output-column gauge quotient
evidence_used: Column six compression achieved 99.96% accuracy at 1592 parameters despite failures on columns five, seven, and eight; testing the next independent column is the clearest remaining one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the tenth `fc2` output column with virtual full-coordinate AdamW dynamics will reduce the model to 1590 parameters while retaining at least 99% accuracy, because the ninth column succeeded despite failures on columns seven and eight, confirming column-specific compressibility.
change: Store the tenth `fc2` column with seven anchored coordinates, shift `fc2_rest` to columns eleven onward, reconstruct its final coordinate as zero, and include it in virtual full-coordinate optimization.
mechanism: Optimizer-preserving tenth MLP output-column gauge quotient
evidence_used: Ninth-column compression achieved 99.90% accuracy at 1591 parameters after adjacent seventh- and eighth-column failures, making the next untested `fc2` column the clearest one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the eleventh `fc2` output column with virtual full-coordinate AdamW dynamics will reduce the model to 1589 parameters while retaining at least 99% accuracy.
change: Store the eleventh `fc2` column with seven anchored coordinates, leave only column twelve in `fc2_rest`, reconstruct the final coordinate as zero, and include the column in virtual full-coordinate optimization.
mechanism: Optimizer-preserving eleventh MLP output-column gauge quotient
evidence_used: Ninth- and tenth-column compression achieved 99.90% and 99.92% accuracy respectively, despite failures on columns seven and eight, showing column-specific compressibility and making the adjacent untested eleventh column the most informative reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7785, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the twelfth `fc2` output column while retaining the failed eleventh column in full will reduce the model to 1589 parameters and maintain at least 99% accuracy, demonstrating that the eleventh-column failure is column-specific.
change: Store `fc2` column eleven in full and column twelve with seven anchored coordinates, reconstruct the twelfth coordinate as zero, and emulate full eight-coordinate AdamW dynamics for column twelve.
mechanism: Optimizer-preserving twelfth MLP output-column gauge quotient
evidence_used: Ninth and tenth column compression retained 99.90% and 99.92% accuracy, while column eleven fell to 77.85%; this established column-specific sensitivity makes the remaining untested twelfth column the most informative one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9715, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the fourth key-projection row to seven coordinates while emulating full eight-coordinate AdamW dynamics will reduce the model to 1589 parameters and retain at least 99% accuracy.
change: Keep the third key row full, anchor the fourth row’s final coordinate, reconstruct its redundant gradient from the current `ln1` gains, and optimize it with virtual full-coordinate moments.
mechanism: Gamma-aware optimizer-preserving fourth key-row gauge quotient
evidence_used: The 1590-parameter design retains 99.92% accuracy with two anchored key rows, while the third-row attempt failed at 54.56%; testing the fourth row independently is the clearest test of row-specific key-gauge compressibility.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7453, "parameters": 1589, "training_steps": 4999}

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
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
