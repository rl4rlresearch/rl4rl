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
verified_results: {"accuracy": 0.9987, "parameters": 1617, "training_steps": 4999}
prior_hypothesis: Removing a third `ln1` bias degree of freedom will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because its additive effect can be absorbed by the query bias, is softmax-null for keys, and is representable through the value/output bias path.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing the eighth `ln2` scale at its initial value will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because its effect can be absorbed exactly by the corresponding `fc1` input-weight column and bias.
change: Store seven learned scales in `ReducedBiasLayerNorm` and reconstruct the eighth as one during the forward pass.
mechanism: Pre-MLP LayerNorm scale/input-weight gauge
evidence_used: Removing a second value bias and centering the fourth `fc2` column collapsed to 53.27% and 72.40%, respectively, motivating a distinct pre-GELU gauge whose eliminated scale is redundant with the following unconstrained linear layer.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7364, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering one key-projection weight row will reduce the verified model from 1619 to 1618 parameters while retaining at least 99% accuracy, because the eliminated component produces only a position-independent key offset on the LayerNorm affine hyperplane and therefore cannot affect causal attention probabilities.
change: Replace the first eight-coordinate key-weight row with seven learned centered contrasts, retain every other QKV weight unchanged, and reconstruct the centered original initialization without altering the RNG sequence.
mechanism: Softmax-null centered key-projection row
evidence_used: The verified 1619-parameter design already removes every key bias while achieving 99.92%, demonstrating that position-independent key offsets are safely softmax-null; this targets the corresponding key-weight redundancy instead of extending value-bias, output-column, or LayerNorm-scale changes that recently collapsed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7016, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one key-row coordinate to zero will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because LayerNorm confines inputs to an affine hyperplane and the omitted direction changes that key coordinate only by a position-independent offset, which causal softmax removes.
change: Replace the first key-projection row with seven directly represented coordinates and a fixed zero eighth coordinate, reconstructing a functionally equivalent initialization while preserving the original RNG sequence.
mechanism: Coordinate-anchored softmax-null key-row gauge
evidence_used: The centered key-row gauge tested the same exact redundancy but reached only 70.16%; its zero-sum reconstruction couples every learned coordinate to the eighth weight. This coordinate-anchored basis removes that coupling while retaining the verified 1619-parameter design’s successful elimination of all key biases.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7401000000000001, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third `ln2` bias degree of freedom will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because its zero-initialized additive effect is representable by the downstream `fc1` bias without the multiplicative optimization disruption seen when fixing an `ln2` scale.
change: Store five learned `ln2` bias contrasts and reconstruct two anchored coordinates before centering the seven active coordinates; leave `ln1` unchanged.
mechanism: Pre-MLP LayerNorm bias/input-bias redundancy
evidence_used: Balanced LayerNorm common-bias gauges previously passed, whereas fixing the eighth `ln2` scale reached only 73.64%; this extends the successful additive-bias mechanism while preserving the initial function and RNG sequence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fourth `ln2` bias degree of freedom will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because its zero-initialized additive contribution remains representable by the downstream `fc1` bias.
change: Store four learned `ln2` bias contrasts instead of five and reconstruct three anchored coordinates before centering the seven active coordinates.
mechanism: Fourth pre-MLP LayerNorm bias/input-bias redundancy
evidence_used: Removing the third `ln2` bias degree achieved 99.93% accuracy at 1618 parameters, while fixing an `ln2` scale achieved only 73.64%; this supports extending the successful additive-bias reduction by one coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.374, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third `ln1` bias degree of freedom will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because its additive effect can be absorbed by the query bias, is softmax-null for keys, and is representable through the value/output bias path.
change: Store five learned `ln1` bias contrasts instead of six and reconstruct two anchored coordinates before centering the seven active coordinates.
mechanism: Pre-attention LayerNorm bias/projection-bias redundancy
evidence_used: Removing a third `ln2` bias degree achieved 99.93% at 1618 parameters, establishing that an additional additive LayerNorm-bias gauge can train successfully; the fourth `ln2` reduction collapsed, so testing the analogous third reduction in the distinct pre-attention LayerNorm is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1617, "training_steps": 4999}



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
