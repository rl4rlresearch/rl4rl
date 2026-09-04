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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1591, "training_steps": 20999}
prior_hypothesis: Gauge-anchoring a fifth `fc2` column after generating the successful initialization draw stream will reduce the model from 1,592 to 1,591 parameters while retaining at least 99% accuracy after 21,000 steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Anchoring one positional-embedding row’s uniform-shift coordinate while preserving its full initialization and implicit eight-coordinate AdamW updates will reduce the model from 1,599 to 1,598 parameters and retain at least 99% accuracy after 18,000 steps.
change: Store the final positional row as seven relative coordinates plus a zero anchor, reconstruct it during forward passes, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.
mechanism: Initialization-preserving positional-row shift gauge
evidence_used: A naive positional-embedding anchor reached only 56.28%, but initialization-preserving gauge optimization rescued the attention-projection column mechanism from 40.05% to 99.98%; the same treatment directly tests whether the earlier positional failure was caused by initialization and optimizer geometry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1598, "training_steps": 17999}

RECENT RESULT
hypothesis: Anchoring a second positional-embedding row’s uniform-shift coordinate will reduce the model from 1,598 to 1,597 parameters while retaining at least 99% accuracy after 19,000 steps, because each positional row has the same exact shift symmetry and the first anchored row achieved 99.94%.
change: Store the penultimate positional row as seven relative coordinates plus a zero anchor, preserve its full initialization and implicit eight-coordinate AdamW updates, and add 1,000 low-rate refinement steps.
mechanism: Incremental initialization-preserving positional-row shift gauge
evidence_used: Initialization-preserving anchoring of the final positional row achieved 99.94% at 1,598 parameters after 18,000 steps, directly validating this symmetry and optimizer treatment for one positional row.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1597, "training_steps": 18999}

RECENT RESULT
hypothesis: Anchoring a third positional-embedding row’s uniform-shift coordinate will reduce the model from 1,597 to 1,596 parameters while retaining at least 99% accuracy after 20,000 steps, because the first two identically treated positional-row anchors achieved 99.94% and 99.97%.
change: Store the antepenultimate positional row as seven learned relative coordinates plus a zero anchor, preserve its equivalent full initialization and implicit eight-coordinate AdamW updates, and add 1,000 low-rate refinement steps.
mechanism: Incremental initialization-preserving positional-row shift gauge
evidence_used: Initialization-preserving positional-row anchoring succeeded consecutively at 1,598 and 1,597 parameters, with the second anchor reaching 99.97% after 19,000 steps; this directly motivates testing the same exact symmetry on the adjacent row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7494, "parameters": 1596, "training_steps": 19999}

RECENT RESULT
hypothesis: Anchoring the uniform-shift coordinate of the final MLP bias will reduce the model from 1,597 to 1,596 parameters while retaining at least 99% accuracy after 20,000 steps, because this shift is removed by the final LayerNorm.
change: Store seven relative `fc2` bias coordinates, reconstruct an eighth zero anchor, train them with implicit eight-coordinate GaugeAdamW, and add 1,000 low-rate refinement steps.
mechanism: Gauge-anchored MLP output bias
evidence_used: Gauge-aware anchoring of the analogous attention-projection bias retained 99.89% accuracy, while the current 1,597-parameter design reached 99.97%; this tests the same validated uniform-output symmetry in the MLP instead of repeating the failed third positional-row anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1596, "training_steps": 19999}

RECENT RESULT
hypothesis: Anchoring one `fc2` input column’s uniform-output coordinate will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy after 21,000 steps, because adding a uniform vector to that column only creates positionwise residual shifts removed by subsequent LayerNorm.
change: Store the final `fc2` column as seven relative coordinates plus a zero anchor, initialize it from an equivalent full eight-coordinate column, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.
mechanism: Initialization-preserving MLP output-column gauge
evidence_used: Initialization-preserving attention-output column gauges achieved 99.98% and 99.96%, while the current gauge-anchored MLP output bias achieved 99.96%; together these validate the same uniform-output symmetry and optimizer treatment in the MLP without repeating the failed third attention-column or positional-row anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1595, "training_steps": 20999}

RECENT RESULT
hypothesis: Anchoring a second `fc2` input column with equivalent full eight-coordinate initialization and implicit AdamW updates will reduce the model from 1,595 to 1,594 parameters while retaining at least 99% accuracy after 22,000 steps.
change: Represent the final two `fc2` columns with seven learned relative coordinates and zero anchors, initialize and optimize both through `GaugeAdamW`, and add 1,000 low-rate refinement steps.
mechanism: Second initialization-preserving MLP output-column gauge
evidence_used: The first identically treated `fc2` column anchor achieved 100% accuracy at 1,595 parameters, and the analogous second attention-projection column anchor achieved 99.96%, directly supporting one incremental application of the same exact uniform-output symmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7406999999999999, "parameters": 1594, "training_steps": 21999}

RECENT RESULT
hypothesis: A second `fc2` column anchor will reach at least 99% accuracy with 26,000 training steps, because the prior 22,000-step attempt reached 74.07% rather than collapsing and the first identical anchor reached 100%.
change: Gauge-anchor the final two `fc2` columns, initialize and optimize both as implicit eight-coordinate columns, and provide 4,000 more minimum-rate refinement steps than the failed attempt.
mechanism: Extended refinement for a second gauge-anchored MLP output column
evidence_used: The prior second-column attempt reduced the model to 1,594 parameters and reached 74.07% after 22,000 steps, while the first column anchor reached 100%; this supports testing whether the exact symmetry needs a longer optimization trajectory.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gauge-anchoring a second `fc2` column while exactly preserving the successful 1,595-parameter design’s initialization draw stream will retain at least 99% accuracy at 1,594 parameters after 21,000 steps.
change: Initialize `fc2` in the current storage geometry, split its final two columns into gauge coordinates only afterward, and train both with `GaugeAdamW`.
mechanism: Fixed-seed trajectory-preserving second MLP output-column gauge
evidence_used: The first `fc2` anchor achieved 100% accuracy at 1,595 parameters, whereas the earlier second anchor reached 74.07% after perturbing the fixed-seed parameter layout; this patch isolates the same exact gauge without changing the successful initialization trajectory.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1594, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring a third `fc2` column after the successful initialization draw stream has been generated will reduce the model from 1,594 to 1,593 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the third-to-last initialized `fc2` column into seven relative coordinates plus a zero anchor, reconstruct it during forward passes, and train it with `GaugeAdamW`.
mechanism: Trajectory-preserving third MLP output-column gauge
evidence_used: Preserving the 1,595-parameter initialization trajectory rescued the second `fc2` anchor from 74.07% to 99.97% at 1,594 parameters; applying the same post-initialization split preserves the now-successful 1,594-parameter draw stream.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1593, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring a fourth `fc2` column after the successful initialization draw stream has been generated will reduce the model from 1,593 to 1,592 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the fourth-to-last initialized `fc2` column into seven relative coordinates plus a zero anchor, reconstruct it during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Trajectory-preserving fourth MLP output-column gauge
evidence_used: Post-initialization splitting rescued the second `fc2` anchor to 99.97% at 1,594 parameters, and the identically initialized third anchor then achieved 99.99% at 1,593 parameters; this applies the same exact gauge without changing the successful initialization draw stream.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1592, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring a fifth `fc2` column after generating the successful initialization draw stream will reduce the model from 1,592 to 1,591 parameters while retaining at least 99% accuracy after 21,000 steps.
change: Split the fifth-to-last initialized `fc2` column into seven learned relative coordinates plus a zero anchor, reconstruct it during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Trajectory-preserving fifth MLP output-column gauge
evidence_used: Post-initialization splitting produced 99.97%, 99.99%, and 99.95% accuracy while successively anchoring the second, third, and fourth `fc2` columns; the fifth applies the same exact symmetry and preserves the validated initialization draw stream.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1591, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring a sixth `fc2` column after generating the successful initialization draw stream will reduce the model from 1,591 to 1,590 parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Split the sixth-to-last initialized `fc2` column into seven learned relative coordinates plus a zero anchor, reconstruct it during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Trajectory-preserving sixth MLP output-column gauge
evidence_used: Post-initialization splitting successively retained 99.97%, 99.99%, 99.95%, and 99.93% accuracy while anchoring the second through fifth `fc2` columns; the sixth applies the identical exact symmetry without perturbing the validated initialization draw stream.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7293999999999999, "parameters": 1590, "training_steps": 20999}



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
