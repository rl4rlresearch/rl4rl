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
verified_results: {"accuracy": 0.9948, "parameters": 1594, "training_steps": 4999}
prior_hypothesis: Compressing the sixth attention output-projection column while emulating its full eight-coordinate AdamW dynamics will retain at least 99% accuracy with 1594 parameters, because virtual optimizer geometry rescued both previously sensitive fourth and fifth columns.

## Recent verification evidence

RECENT RESULT
hypothesis: Anchoring the final output coordinate of the fifth attention projection column will reduce the model to 1596 parameters while retaining at least 99% accuracy, because it applies the successful anchored gauge to the first coordinate of the second attention head while leaving the sensitive fourth column fully learned.
change: Compress attention output-projection columns one, two, three, and five to seven learned coordinates each; reconstruct their eighth coordinates as zero and store columns four, six, seven, and eight without constraint.
mechanism: Second-head attention-output common-mode gauge anchor
evidence_used: Anchoring the first three columns achieved 99.98% accuracy at 1597 parameters, while both coordinate anchoring and an orthonormal zero-mean parameterization failed on the fourth column; this indicates column-specific sensitivity and motivates testing the next unconstrained column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7303000000000001, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final output coordinate of the sixth attention projection column will reduce the model to 1596 parameters while maintaining at least 99% accuracy, showing that the second head’s failure is specific to its fifth column rather than all of its coordinates.
change: Compress attention output-projection columns one, two, three, and six to seven learned coordinates each, reconstruct their eighth coordinates as zero, and retain full storage for columns four, five, seven, and eight.
mechanism: Sixth attention-output common-mode gauge anchor
evidence_used: The first three columns tolerated identical anchors with 99.98% accuracy at 1597 parameters, while columns four and five failed; testing column six is the most direct way to distinguish isolated column sensitivity from a second-head-wide limitation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7238, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the final positional row will reduce the model to 1596 parameters while retaining at least 99% accuracy, because the resulting feature-common shift is exactly erased by every downstream LayerNorm.
change: Store seven degrees of freedom for the final positional row by fixing its first feature to zero, while jointly recentering the existing positional and token embedding gauges to preserve the original initialization modulo exact invariances.
mechanism: Final-position residual-stream common-mode anchor
evidence_used: The current three-column attention anchor achieved 99.98% accuracy at 1597 parameters, while further attention-column anchors on columns four through six failed; this motivates testing an independent LayerNorm-invariant common-mode gauge instead of constraining another sensitive attention column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4043, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final `ln2` gain to one will retain at least 99% accuracy with 1596 parameters, because its bias coordinate is identically zero and any learned scaling can be absorbed by the unconstrained final input column of `fc1`.
change: Store seven learned gains in `ReducedBiasLayerNorm` and reconstruct the eighth as one before normalization.
mechanism: Pre-MLP LayerNorm final-gain gauge fixing
evidence_used: The 1597-parameter design achieved 99.98%, while reductions targeting additional attention columns and positional coordinates failed; this tests an independent, initialization-preserving scale gauge at the unconstrained MLP interface.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.726, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring rather than zero-mean centering the third `fc2` output column will achieve at least 99% accuracy with 1596 parameters, because the same coordinate-anchor chart succeeded for three attention output columns.
change: Store seven coordinates for the third `fc2` column, reconstruct its eighth coordinate as zero, and initialize it by subtracting the original eighth coordinate; retain the existing parameterization for the first two columns.
mechanism: Anchored third MLP output-column common-mode gauge
evidence_used: Zero-mean centering the third `fc2` column failed at 74.54%, while final-coordinate anchoring succeeded sequentially for three attention output columns and produced the current 99.98%-accurate 1597-parameter design, making the alternative gauge chart the informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6121, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the fourth attention projection column will achieve at least 99% accuracy with 1596 parameters when its updates emulate full eight-coordinate AdamW dynamics, because the prior anchored and orthonormal failures indicate optimizer-geometry sensitivity rather than loss of model expressivity.
change: Store seven learned coordinates for the fourth projection column and maintain virtual eight-coordinate Adam moments during training so its quotient-space updates match the successful full-column parameterization.
mechanism: Optimizer-preserving fourth-column gauge quotient
evidence_used: The current three-column anchor reached 99.98% at 1597 parameters, while naïve anchoring of the fourth column reached 84.5% and an orthonormal parameterization reached 40.36%; preserving the original AdamW coordinate dynamics directly targets the distinction between those failures and the successful design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9937, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the fifth attention output-projection column while emulating its full eight-coordinate AdamW dynamics will retain at least 99% accuracy with 1595 parameters, because virtual optimizer geometry rescued the analogous fourth-column anchor from 84.5% to 99.37%.
change: Store seven learned coordinates for each of the first five projection columns, reconstruct their eighth coordinates as zero, and apply virtual full-coordinate AdamW updates to both the fourth and fifth compressed columns.
mechanism: Optimizer-preserving fifth-column gauge quotient
evidence_used: Naive anchoring failed for both the fourth and fifth columns, but preserving full-coordinate optimizer dynamics made the fourth-column reduction succeed at 99.37%; applying the same mechanism to the fifth column directly tests whether its prior failure had the same optimizer-geometric cause.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the sixth attention output-projection column while emulating its full eight-coordinate AdamW dynamics will retain at least 99% accuracy with 1594 parameters, because virtual optimizer geometry rescued both previously sensitive fourth and fifth columns.
change: Store seven learned coordinates for each of the first six projection columns, reconstruct their eighth coordinates as zero, and extend virtual full-coordinate AdamW updates to the sixth compressed column.
mechanism: Optimizer-preserving sixth-column gauge quotient
evidence_used: Naive sixth-column anchoring failed at 72.38%, but the same naive failure pattern on the fourth and fifth columns was overcome by virtual full-coordinate optimizer dynamics, reaching 99.37% and 99.98% respectively.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9948, "parameters": 1594, "training_steps": 4999}

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
