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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1616, "training_steps": 4999}
prior_hypothesis: Fixing feature-4 of the final token row and transferring its offset into every positional row will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because token-plus-position inputs and output probabilities are unchanged, and the virtual AdamW path reproduces the omitted gradients and updates.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing second-LayerNorm scale coordinate 1 at one, alongside coordinate 4, will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy because the full `fc1` weight can absorb both scale degrees of freedom.
change: Store six second-LayerNorm scale coordinates and reconstruct coordinates 1 and 4 as one, while retaining the successful coordinate-4 bias gauge.
mechanism: Coordinate-1 LayerNorm-scale absorption into the downstream MLP affine map
evidence_used: The additional coordinate-3 scale gauge narrowly missed at 98.67%, showing this quotient remains promising but coordinate-sensitive; analogous value-bias coordinate 1 achieved 99.98%, making coordinate 1 the strongest evidenced alternative.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 1 at zero, alongside coordinate 4, will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy because the full `fc1` bias exactly absorbs both coordinates.
change: Store six second-LayerNorm bias coordinates and reconstruct coordinates 1 and 4 as zero, while retaining the successful scale-coordinate and MLP-output gauges.
mechanism: Coordinate-1 LayerNorm bias absorption into the downstream MLP affine map
evidence_used: The coordinate-4 LayerNorm bias gauge achieved 99.84%, and coordinate 1 is the strongest additional candidate because its LayerNorm scale gauge achieved 99.81% and its analogous value-bias removal achieved 99.98%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.441, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm scale coordinate 0 at one, alongside coordinates 1 and 4, will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy because the full `fc1` weight can absorb all three scale degrees of freedom.
change: Store five second-LayerNorm scale coordinates and reconstruct coordinates 0, 1, and 4 as one, retaining the successful coordinate-4 bias and MLP-output gauges.
mechanism: Coordinate-0 LayerNorm-scale absorption into the downstream MLP affine map
evidence_used: The coordinate-1 scale gauge achieved 99.81%, while coordinate 3 narrowly missed at 98.67%, demonstrating coordinate sensitivity rather than exhaustion of the quotient. Analogous value-bias coordinate 0 later supported 99.94%, making coordinate 0 the strongest evidenced untested scale candidate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 0 at zero alongside coordinate 4 will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy because the full `fc1` bias can absorb both coordinates.
change: Store six second-LayerNorm bias coordinates and reconstruct coordinates 0 and 4 as zero, retaining the successful scale-coordinate and MLP-output gauges.
mechanism: Coordinate-0 LayerNorm-bias absorption into the downstream MLP affine map
evidence_used: The current coordinate-0 LayerNorm-scale gauge achieved 99.83%, and analogous value-bias coordinate 0 previously supported 99.94%; this makes coordinate 0 a better evidenced bias candidate than coordinate 1, whose LayerNorm-bias removal failed at 44.1%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8558, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing key-projection coordinate 1 will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because its omitted weight is an exact softmax-invisible quotient and the existing virtual AdamW path preserves its optimization geometry.
change: Add key row 1 to the five already gauge-fixed QKV rows; the existing dynamic virtual-gradient, clipping, and optimizer logic automatically handles the sixth coordinate.
mechanism: Additional LayerNorm-null key-projection gauge
evidence_used: The current 1617-parameter design achieves 99.83% while simultaneously fixing five coordinates through this same key-projection quotient. Coordinate 1 is the strongest remaining candidate because its analogous second-LayerNorm scale gauge retained 99.81% accuracy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7123999999999999, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm scale coordinate 6 at one, alongside coordinates 0, 1, and 4, will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy because the full `fc1` weight can absorb all four scale degrees of freedom.
change: Store four second-LayerNorm scale coordinates and reconstruct coordinates 0, 1, 4, and 6 as one, retaining the successful coordinate-4 bias and MLP-output gauges.
mechanism: Coordinate-6 LayerNorm-scale absorption into the downstream MLP affine map
evidence_used: LayerNorm scale removals at coordinates 0, 1, and 4 jointly achieved 99.83%; coordinate 6 is the strongest untested scale candidate because the current successful model already fixes the analogous value-bias coordinate and LayerNorm-null key-projection row, without the negative MLP-bias evidence associated with coordinate 7.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6757, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm scale coordinate 2 at one, alongside coordinates 0, 1, and 4, will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy because the full `fc1` affine map can absorb all four scale degrees of freedom.
change: Store four second-LayerNorm scale coordinates and reconstruct coordinates 0, 1, 2, and 4 as one, retaining the successful coordinate-4 LayerNorm-bias and MLP-output gauges.
mechanism: Coordinate-2 LayerNorm-scale absorption into the downstream MLP affine map
evidence_used: The current three-coordinate scale gauge achieved 99.83%. Additional coordinates 3 and 6 were accuracy-sensitive, while the successful current model already removes analogous value-bias coordinate 2; this makes coordinate 2 the strongest remaining untested scale candidate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.369, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the sole remaining value-bias coordinate will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because a constant value shift produces a position-independent attention output that the learned projection bias can absorb.
change: Remove value-bias coordinate 5 and its reconstruction, leaving the value projection unbiased while retaining the existing seven-coordinate output-projection bias.
mechanism: Final value-bias absorption into attention output bias
evidence_used: The verified 1617-parameter model reaches 99.83% accuracy with seven of eight value-bias coordinates already fixed; testing the last coordinate is the most direct extension of that successful quotient, whereas recent additional LayerNorm-scale, LayerNorm-bias, and key-row removals failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.16829999999999998, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing feature-4 of the final token row and transferring its offset into every positional row will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because token-plus-position inputs and output probabilities are unchanged, and the virtual AdamW path reproduces the omitted gradients and updates.
change: Extend the token-embedding quotient by one coordinate, transfer that coordinate into positional embeddings at initialization, and jointly optimize token and positional parameters with both omitted coordinates restored virtually.
mechanism: Joint token–position common-column gauge
evidence_used: The current global embedding quotient reaches 99.83%. Unlike the isolated positional-row quotient that fell to 37.73%, this gauge leaves every token-plus-position input exactly unchanged; feature 4 is also the strongest coordinate-specific choice, having supported successful LayerNorm-scale, LayerNorm-bias, and MLP-output gauges.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing feature 1 of the final token row and transferring its offset into every positional row will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because token-plus-position inputs and output probabilities remain unchanged and the generalized virtual AdamW path restores all omitted gradients and updates.
change: Extend the successful embedding quotient to features 1 and 4, and generalize initialization, virtual-gradient reconstruction, clipping, and optimizer updates for both transferred coordinates.
mechanism: Second joint token–position common-column gauge
evidence_used: The feature-4 token–position gauge achieved 99.93% at 1616 parameters. Feature 1 is the strongest remaining coordinate-specific candidate because its analogous value-bias removal achieved 99.98% and its LayerNorm-scale gauge achieved 99.81%.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Fixing features 1 and 4 of the final token row will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because both offsets are transferred into positional embeddings and the generalized virtual AdamW path preserves the corresponding full-model updates.
change: Extend the embedding quotient to features 1 and 4, then generalize initialization, gradient reconstruction, clipping, and optimizer projection for three omitted token coordinates.
mechanism: Second joint token–position common-column gauge
evidence_used: The feature-4 quotient achieved 99.93% at 1616 parameters; feature 1 is the strongest additional candidate because analogous value-bias and LayerNorm-scale gauges achieved 99.98% and 99.81%. The prior feature-1 implementation was unverified rather than accuracy-rejected.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6638, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing feature 0 alongside feature 4 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because both offsets are transferred into every positional row and the generalized virtual AdamW path reconstructs and updates all three omitted token coordinates.
change: Generalize the successful token–position embedding quotient from feature 4 to features 0 and 4, including initialization transfer, virtual gradients, clipping, and optimizer projection.
mechanism: Second joint token–position common-column gauge
evidence_used: The feature-4 quotient achieved 99.93% at 1616 parameters. After feature 1 failed at 66.38%, feature 0 is the strongest remaining coordinate: its LayerNorm-scale gauge achieved 99.83%, and its analogous value-bias gauge previously supported 99.94%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6001, "parameters": 1615, "training_steps": 4999}



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
