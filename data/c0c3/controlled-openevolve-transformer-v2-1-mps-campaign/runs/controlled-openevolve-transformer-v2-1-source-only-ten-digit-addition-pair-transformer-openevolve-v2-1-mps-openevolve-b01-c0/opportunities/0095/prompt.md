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
verified_results: {"accuracy": 0.9998, "parameters": 1595, "training_steps": 4999}
prior_hypothesis: Compressing the fifth attention output-projection column while emulating its full eight-coordinate AdamW dynamics will retain at least 99% accuracy with 1595 parameters, because virtual optimizer geometry rescued the analogous fourth-column anchor from 84.5% to 99.37%.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Representing the fourth attention projection column in an orthonormal zero-mean basis will achieve at least 99% accuracy with 1596 parameters because it removes the same downstream LayerNorm-invariant direction without the poorly conditioned coordinate anchor that scored 84.5%.
change: Compress the fourth projection column to seven learned orthonormal coordinates, reconstruct its zero-mean eight-dimensional column, and retain full storage for columns five through eight.
mechanism: Orthonormal fourth attention-output common-mode gauge
evidence_used: Final-coordinate anchoring succeeded for the first three projection columns at 1597 parameters but failed only when extended to the fourth; this motivates testing whether the fourth column needs a better-conditioned gauge chart rather than its redundant common-mode parameter.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4036, "parameters": 1596, "training_steps": 4999}

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
