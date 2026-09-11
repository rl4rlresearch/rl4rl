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
verified_results: {"accuracy": 0.9991, "parameters": 1534, "training_steps": 4999}
prior_hypothesis: Constraining `fc1` row 5 through its second input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because the identical coordinate parameterization already succeeded on adjacent row 6.

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its fourth input coordinate will reduce the model from 1,536 to 1,535 parameters while retaining at least 99% accuracy, because changing the constrained coordinate previously rescued row 7 and avoids reusing row 5’s failed third-coordinate constraint.
change: Remove row 5’s fourth input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve the row’s initial map on zero-mean LayerNorm inputs.
mechanism: Coordinate-alternate eleventh MLP quotient
evidence_used: Row 5 constrained on the third coordinate collapsed to 66.51%, while row 7 improved from 40.22% with a final-coordinate constraint to 99.95% with a first-coordinate constraint. This directly motivates testing whether a different distributed coordinate similarly rescues row 5.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.379, "parameters": 1535, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 3 through its fourth input coordinate will reduce the model from 1,536 to 1,535 parameters while retaining at least 99% accuracy, showing that row 5’s failure on this coordinate was neuron-specific.
change: Remove row 3’s fourth input weight, reconstruct it as zero during the forward pass, and canonicalize its fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.
mechanism: Alternate-neuron staircase MLP quotient
evidence_used: Row 5 constrained on its third coordinate failed at 66.51%, while moving that same constraint to row 4 achieved 99.41%; row 5’s fourth-coordinate failure at 37.9% therefore most directly motivates testing the same coordinate on the remaining unconstrained row 3.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1535, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its fifth input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because coordinate choice has shown strong neuron-specific effects and this coordinate remains untested for row 5.
change: Remove row 5’s fifth input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve the row’s initial map on zero-mean LayerNorm inputs.
mechanism: Coordinate-alternate twelfth MLP quotient
evidence_used: Row 5 failed when constrained on its third and fourth coordinates, but moving row 7’s constraint from its final coordinate to its first rescued accuracy from 40.22% to 99.95%. This makes another coordinate-specific row-5 probe the smallest informative extension of the 1,535-parameter design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0993, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its first input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because the first-coordinate parameterization previously rescued row 7 and avoids row 5’s unsuccessful third-, fourth-, and fifth-coordinate choices.
change: Remove row 5’s first input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.
mechanism: First-coordinate twelfth MLP quotient
evidence_used: Row 5 achieved only 66.51%, 37.9%, and 9.93% when constrained through coordinates three, four, and five, while moving row 7’s constraint to the first coordinate raised accuracy from 40.22% to 99.95%; this makes the first coordinate the strongest untested row-5 alternative.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3961, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final `ln2` scale at one will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because its effect remains representable by the learned final input column of `fc1` and fresh initialization is unchanged.
change: Remove the final trainable scale coordinate from the bias-free MLP LayerNorm and reconstruct it as a fixed one during the forward pass.
mechanism: Final-scale/downstream-column gauge fixing
evidence_used: The 1,535-parameter design achieved 99.87% accuracy with seven `fc1` rows already independent of the final input coordinate, while four attempts to constrain the remaining row 5 failed. This motivates preserving row 5 and instead testing the exact upstream-scale/downstream-column gauge, which affects only the five rows retaining that column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7492, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-weight coordinate through a score-preserving orthogonal query/key rotation will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, avoiding the optimization shock caused by fixing all 32 attention-basis gauges simultaneously.
change: Replace the QKV projection with a one-parameter-smaller projection whose first query coordinate is fixed at zero, and orthogonally canonicalize the corresponding query/key rows during fresh initialization while preserving initial attention scores.
mechanism: Single-coordinate orthogonal query/key gauge fixing
evidence_used: The 1,535-parameter design reached 99.87%, whereas removing all 32 query/key basis-gauge parameters at once collapsed to 23.28% despite preserving score functions. Fixing only one gauge coordinate is the smallest test of whether that failure resulted from aggressive simultaneous reparameterization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7392, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing independent seven-dimensional token content vectors with a learned rank-five factorization will reduce the model from 1,535 to 1,360 parameters while retaining at least 99% accuracy, because token identities remain independently learnable and retain independent output offsets while the model learns a shared vocabulary geometry.
change: Factorize the tied mean-free token embedding and output matrix into per-token rank-five coordinates and a learned map into the seven-dimensional centered model space; initialize it with the rank-five SVD of each fresh embedding draw and exempt both factors from weight decay.
mechanism: Learned rank-five tied token manifold
evidence_used: The 1,535-parameter model reached 99.87%, while repeated removal of isolated MLP or attention coordinates caused highly brittle optimization failures. Those attempts assumed the dense per-token representation was load-bearing and targeted small downstream redundancies; this patch instead tests whether the dominant token-parameter block can learn addition through a substantially smaller shared latent vocabulary manifold.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9042, "parameters": 1342, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its second input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because the identical coordinate parameterization already succeeded on adjacent row 6.
change: Remove row 5’s second input weight, reconstruct it as zero during the forward pass, and canonicalize initialization to preserve its initial map on zero-mean LayerNorm inputs.
mechanism: Adjacent-neuron second-coordinate MLP quotient
evidence_used: Row 6 constrained through its second coordinate retained 99.95% accuracy, while row 5’s failures on the first, third, fourth, and fifth coordinates demonstrate strong coordinate-specific optimization effects and leave the successful second-coordinate parameterization as the most informative untested row-5 probe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fifth `ln1` bias coordinate at zero will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because its query and value effects are absorbable by learned Q/V biases and its key effect is softmax-null.
change: Retain only the first three trainable `ln1` bias coordinates and reconstruct the remaining five as zeros.
mechanism: Fifth additive LayerNorm-bias quotient
evidence_used: The current 1,534-parameter model achieves 99.91% accuracy with four `ln1` bias coordinates already removed. Extending that same additive quotient by one coordinate is the smallest targeted reduction and avoids the multiplicative optimization failure observed when fixing an `ln2` scale.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1119, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing `ln1` bias coordinate zero instead of the previously failed coordinate three will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because equivalent quotient constraints have shown strong coordinate-specific optimization behavior.
change: Retain trainable `ln1` bias coordinates one through three and reconstruct coordinate zero plus the final four coordinates as zeros; fresh initialization remains unchanged.
mechanism: First-coordinate alternate LayerNorm-bias quotient
evidence_used: The 1,534-parameter model reached 99.91% with four bias coordinates removed, but removing coordinate three next collapsed to 11.19%. Earlier MLP experiments showed that changing only the constrained coordinate rescued row 7 from 40.22% to 99.95% and row 5 succeeded only on its second coordinate, motivating an alternate coordinate for the same fifth additive quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5624, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate one at zero will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because this coordinate remains untested and the same coordinate constraint succeeds in two adjacent `fc1` rows.
change: Retain trainable `ln1` bias coordinates zero, two, and three, reconstruct coordinate one and the final four coordinates as zeros, and preserve the unchanged zero-bias initialization.
mechanism: Second-coordinate LayerNorm-bias quotient
evidence_used: Removing coordinates three and zero produced 11.19% and 56.24% accuracy, demonstrating strong coordinate-specific optimization effects; meanwhile, second-coordinate constraints on `fc1` rows five and six are both present in the 99.91%-accurate 1,534-parameter design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.755, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate two at zero will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because it is the only untested coordinate among the four currently trainable biases and prior quotient results demonstrate strong coordinate-specific optimization behavior.
change: Retain trainable `ln1` bias coordinates zero, one, and three, while reconstructing coordinate two and the final four coordinates as zeros without changing fresh zero-bias initialization.
mechanism: Third-coordinate LayerNorm-bias quotient
evidence_used: Removing currently trainable `ln1` bias coordinates three, zero, and one yielded 11.19%, 56.24%, and 75.50% accuracy respectively, while coordinate-specific changes previously rescued MLP quotients—including row 5 succeeding only on its second coordinate—making the remaining coordinate-two test the most targeted next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7182, "parameters": 1533, "training_steps": 4999}



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
