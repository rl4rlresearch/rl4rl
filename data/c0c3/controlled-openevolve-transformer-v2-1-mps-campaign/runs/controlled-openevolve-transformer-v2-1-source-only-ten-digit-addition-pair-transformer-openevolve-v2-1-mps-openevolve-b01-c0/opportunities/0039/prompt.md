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
verified_results: {"accuracy": 0.9992, "parameters": 1619, "training_steps": 4999}
prior_hypothesis: Removing one value-projection bias coordinate will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because any resulting token-independent attention output can be represented by the centered output-projection bias up to a LayerNorm-null common offset.

## Recent verification evidence

RECENT RESULT
hypothesis: Centering the sixth positional-embedding feature column will reduce the model from 1624 to 1623 parameters while retaining at least 99% accuracy, because transferring its mean into the matching token-embedding column preserves transformer inputs exactly and changes tied logits only by a class-independent offset.
change: Represent positional features four, five, and six with learned centered contrasts, transferring their initialization means into the corresponding token-embedding columns while preserving the original RNG sequence.
mechanism: Third positional-oriented token–position embedding gauge
evidence_used: Positional-oriented gauges for the fourth and fifth features achieved 99.83% and 99.99% accuracy at 1625 and 1624 parameters; extending the same exact redundancy to one adjacent feature is the most conservative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the seventh positional-embedding feature column will reduce the model from 1623 to 1622 parameters while retaining at least 99% accuracy, because transferring its mean into the matching token-embedding column preserves transformer inputs exactly and changes tied logits only by a class-independent offset.
change: Represent positional features four through seven with learned centered contrasts, transferring their initialization means into the corresponding token-embedding columns while preserving the original initialization RNG sequence.
mechanism: Fourth positional-oriented token–position embedding gauge
evidence_used: The same positional-oriented gauge passed for features four, five, and six, most recently achieving 99.91% accuracy at 1623 parameters; extending it to one adjacent feature is the smallest evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the eighth positional-embedding feature column will reduce the model from 1622 to 1621 parameters while retaining at least 99% accuracy, because transferring its mean into the matching token-embedding column preserves transformer inputs exactly and changes tied logits only by a class-independent offset.
change: Represent positional features four through eight with learned centered contrasts, transferring their initialization means into the corresponding token-embedding columns while preserving the original initialization RNG sequence.
mechanism: Fifth positional-oriented token–position embedding gauge
evidence_used: Positional-oriented gauges passed for features four through seven, most recently achieving 99.94% accuracy at 1622 parameters; extending the same exact redundancy to the final adjacent feature is the smallest evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the first positional feature will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because transferring the eliminated joint token-position mean as an equal offset across all eight residual features leaves every LayerNorm output unchanged and alters tied logits only by a class-independent offset.
change: Represent positional feature one with centered learned contrasts, subtract its eliminated joint mean from the remaining free positional and token columns during initialization, and preserve the original initialization RNG sequence.
mechanism: Global LayerNorm-null embedding-offset gauge
evidence_used: All eight token-position mean redundancies have now been removed successfully, with the fifth positional-oriented gauge reaching 99.97% at 1621 parameters; the same proven positional contrast parameterization can fix the remaining global all-features offset symmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1041, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the attention output-projection bias will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because its removed featurewise-common component passes unchanged through the residual stream, is invisible to `ln2`, and is eliminated by the final LayerNorm.
change: Replace the eight-parameter attention projection bias with seven learned centered contrasts while preserving its zero initialization and the original initialization RNG sequence.
mechanism: LayerNorm-null attention projection bias gauge
evidence_used: Balanced common-bias gauges in both block LayerNorms and the terminal `fc2` path previously passed, whereas the failed 1620 embedding gauge substantially altered embedding optimization; this applies the proven local bias-centering mechanism to a distinct exact LayerNorm-null direction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering one attention output-projection weight column will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because its removed output-coordinate mean contributes only a tokenwise common residual offset that is invisible to subsequent LayerNorms.
change: Represent the first attention projection column with seven learned zero-sum contrasts, retain the other columns unchanged, and reconstruct its original centered initialization without changing the RNG sequence.
mechanism: LayerNorm-null attention projection-column gauge
evidence_used: Centering the attention projection bias achieved 99.89% at 1620 parameters through the same residual-stream gauge; changing only one projection column is conservative given that two analogous `fc2` column gauges passed while a third collapsed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9892, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the second attention output-projection weight column will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because the removed output-coordinate mean contributes only a tokenwise common residual offset eliminated by subsequent LayerNorms.
change: Represent the second attention projection column with seven zero-sum contrasts, retain all other columns unchanged, and reconstruct its centered initialization while preserving the original RNG sequence.
mechanism: Second attention projection-column LayerNorm gauge
evidence_used: Centering the projection bias passed at 99.89%, while centering the first projection column narrowly missed at 98.92%; testing a different column is the smallest informative reduction, and analogous first and second `fc2` column gauges both previously passed despite feature-specific failures appearing later.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7245, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the first attention output-projection column in an orthonormal zero-sum basis will reduce the model to 1619 parameters while achieving at least 99% accuracy, because it removes the same exact LayerNorm-null direction as the narrowly failing 98.92% trial without the anisotropic optimization geometry of anchored contrasts.
change: Replace the first attention projection column with seven learned Helmert-basis coordinates, retain the other seven columns unchanged, and reconstruct the centered original initialization without altering the RNG sequence.
mechanism: Orthonormal LayerNorm-null attention projection gauge
evidence_used: The attention projection-bias gauge reached 99.89% at 1620 parameters, proving this residual common-offset symmetry is usable; the first projection-column attempt reached 98.92%, so improving that exact gauge’s conditioning is more strongly motivated than testing the second column again, which reached only 72.45%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8332999999999999, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one value-projection bias coordinate will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because any resulting token-independent attention output can be represented by the centered output-projection bias up to a LayerNorm-null common offset.
change: Store all eight query biases but only seven value biases, reconstructing the final value bias as zero without changing initialization or the causal attention computation.
mechanism: Value-bias/output-bias redundancy
evidence_used: Centering the attention output-projection bias passed at 99.89%, while modifying output-projection weight columns failed; removing a redundant value-bias coordinate uses the successful bias path and avoids perturbing learned projection weights.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second value-projection bias coordinate will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because the resulting token-independent attention output remains representable by the centered output-projection bias up to a LayerNorm-null common offset.
change: Store all eight query biases but only six value biases, reconstructing the final two value-bias coordinates as zero without changing initialization or causal attention.
mechanism: Second value-bias/output-bias redundancy
evidence_used: Removing the first value-bias coordinate achieved 99.92% accuracy at 1619 parameters, while output-projection weight gauges failed; extending the successful bias redundancy by one coordinate is the smallest evidence-backed reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5327000000000001, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the matching final value-bias coordinate from each attention head will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because both token-independent value offsets remain representable by the centered output-projection bias while preserving equal trainable bias capacity across heads.
change: Store six value-bias coordinates and reconstruct coordinates four and eight as zero, leaving three learned value-bias coordinates in each head.
mechanism: Head-balanced value-bias/output-bias redundancy
evidence_used: Removing value coordinate eight alone achieved 99.92% at 1619 parameters, whereas removing coordinates seven and eight collapsed to 53.27%; distributing the two removed coordinates evenly across the two heads tests whether that failure arose from the second head’s asymmetric loss of bias capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.16940000000000002, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the fourth `fc2` output column will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because its removed output-coordinate mean produces only a tokenwise common residual offset eliminated by the final LayerNorm.
change: Represent the fourth `fc2` column with seven learned zero-sum contrasts, retain the third and remaining columns unchanged, and reconstruct the centered original initialization without altering the RNG sequence.
mechanism: Fourth-hidden-unit MLP output-column LayerNorm gauge
evidence_used: The first two `fc2` column gauges passed, while the third was feature-specifically unsuccessful; testing the untried fourth column is more informative than another second value-bias removal after two such variants collapsed to 53.27% and 16.94%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7240000000000001, "parameters": 1618, "training_steps": 4999}



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
