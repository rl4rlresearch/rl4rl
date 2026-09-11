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
verified_results: {"accuracy": 0.9992, "parameters": 1590, "training_steps": 4999}
prior_hypothesis: Compressing the tenth `fc2` output column with virtual full-coordinate AdamW dynamics will reduce the model to 1590 parameters while retaining at least 99% accuracy, because the ninth column succeeded despite failures on columns seven and eight, confirming column-specific compressibility.

## Recent verification evidence

RECENT RESULT
hypothesis: Compressing the seventh attention output-projection column while emulating its full eight-coordinate AdamW dynamics will retain at least 99% accuracy with 1593 parameters, because virtual optimizer geometry enabled the same sequential reduction for sensitive columns four through six.
change: Store seven learned coordinates for each of the first seven projection columns, reconstruct their eighth coordinates as zero, and extend virtual full-coordinate AdamW updates to the seventh compressed column.
mechanism: Optimizer-preserving seventh-column gauge quotient
evidence_used: Extending virtual full-coordinate optimization from the fifth to the sixth compressed column reduced the model from 1595 to 1594 parameters while retaining 99.48% accuracy; applying the established mechanism to the adjacent seventh column is the most direct next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5287, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the eighth attention output-projection column while leaving the failed seventh column unconstrained will achieve at least 99% accuracy with 1593 parameters, indicating that the seventh-column failure is column-specific.
change: Store seven coordinates for attention projection column eight, reconstruct its final coordinate as zero, and emulate its full eight-coordinate AdamW dynamics alongside compressed columns four through six.
mechanism: Optimizer-preserving eighth-column gauge quotient
evidence_used: Virtual full-coordinate optimization preserved accuracy while compressing columns four through six, reaching 99.48% at 1594 parameters, whereas extending it specifically to column seven fell to 52.87%; testing column eight independently is the most informative adjacent reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4351, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the third `fc2` output column while emulating its full eight-coordinate AdamW dynamics will achieve at least 99% accuracy with 1593 parameters, because virtual optimizer geometry rescued analogous attention-column gauges that failed under naïve parameterizations.
change: Store seven anchored coordinates for the third `fc2` column, reconstruct its eighth coordinate as zero, and include it in the existing virtual full-coordinate AdamW update.
mechanism: Optimizer-preserving third MLP output-column gauge quotient
evidence_used: Naïve zero-mean and anchored third-`fc2` reductions scored only 74.54% and 61.21%, while virtual full-coordinate optimization rescued similarly failing attention columns four through six and retained 99.48% at 1594 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2382, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the third key-projection row to seven coordinates while emulating full eight-coordinate AdamW updates along the LayerNorm-scaled key-shift gauge will retain at least 99% accuracy with 1593 parameters.
change: Anchor the third key row’s final coordinate, reconstruct its redundant gradient using the current `ln1` gains, and optimize it with virtual full-coordinate moments alongside the existing compressed attention-output columns.
mechanism: Gamma-aware optimizer-preserving key-row gauge quotient
evidence_used: The verified 1594-parameter model already tolerates two anchored key rows, while virtual full-coordinate optimization rescued sensitive attention-output reductions through the sixth column and retained 99.48% accuracy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5456, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the fourth `fc2` output column while emulating its full eight-coordinate AdamW dynamics will retain at least 99% accuracy with 1593 parameters; the third-column failure is neuron-specific rather than evidence that all remaining MLP column gauges are untrainable.
change: Store the fourth `fc2` column with seven anchored coordinates, retain the third column in full, and include the new compressed column in the existing virtual full-coordinate optimizer.
mechanism: Optimizer-preserving fourth MLP output-column gauge quotient
evidence_used: Virtual optimizer geometry successfully compressed attention columns four through six, while later attention columns and the third MLP column failed, demonstrating coordinate-specific sensitivity and motivating an independent test of the next MLP output column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9956, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the fifth `fc2` output column with virtual full-coordinate AdamW dynamics will reduce the model to 1592 parameters while retaining at least 99% accuracy, because the same method successfully compressed the adjacent fourth column to produce the current 99.56%-accurate design.
change: Store seven anchored coordinates for the fifth `fc2` column, reconstruct its eighth coordinate as zero, shift `fc2_rest` to columns six onward, and include the new column in virtual full-coordinate optimization.
mechanism: Optimizer-preserving fifth MLP output-column gauge quotient
evidence_used: The fourth `fc2` column succeeded with optimizer-preserving compression at 1593 parameters despite the third column failing, demonstrating neuron-specific sensitivity and making the adjacent fifth column the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5449, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the sixth `fc2` output column while retaining the failed fifth column in full will reduce the model to 1592 parameters and retain at least 99% accuracy, showing that MLP-column sensitivity is neuron-specific.
change: Store the fifth `fc2` column in full, store the sixth with seven anchored coordinates, shift `fc2_rest` to columns seven onward, and apply virtual full-coordinate AdamW updates to the sixth column.
mechanism: Optimizer-preserving sixth MLP output-column gauge quotient
evidence_used: Virtual optimization successfully compressed the fourth `fc2` column at 99.56% accuracy, while the adjacent fifth-column attempt fell to 54.49%; testing the sixth independently is the clearest way to determine whether that failure is isolated.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1592, "training_steps": 4999}

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
