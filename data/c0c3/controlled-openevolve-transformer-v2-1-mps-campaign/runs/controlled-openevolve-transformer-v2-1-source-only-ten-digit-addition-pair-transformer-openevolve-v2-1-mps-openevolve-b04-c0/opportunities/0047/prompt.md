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
verified_results: {"accuracy": 0.9995, "parameters": 1605, "training_steps": 4999}
prior_hypothesis: Fixing coordinate 1 of `fc1.bias` at zero will reduce the model from 1606 to 1605 parameters while retaining at least 99% accuracy, because the second remaining learned pre-MLP LayerNorm shift provides another independent degree of freedom for absorbing a hidden-unit bias constraint.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 4 to one will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy, because its scale is absorbable by `fc1.weight` and its distributed embedding anchor previously retained 99.94% accuracy.
change: Remove gain coordinate 4 from the learned vector and reconstruct gains 3, 4, 5, and 7 as fixed ones during the forward pass.
mechanism: Fourth pre-MLP scale-to-weight gauge fixing
evidence_used: Fixing gains 3, 5, and 7 achieved 99.95% accuracy at 1613 parameters; coordinate 4 is the strongest untested distributed-coordinate candidate because its embedding anchor previously achieved 99.94%, while coordinate 1 gain fixing collapsed to 27.21%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 6 to one will reduce the model from 1612 to 1611 parameters while retaining at least 99% accuracy, because its scale is absorbable by column 6 of `fc1.weight` and its distributed embedding anchor previously retained 99.72% accuracy.
change: Remove gain coordinate 6 from the learned vector and reconstruct gains 3 through 7 as fixed ones during the forward pass.
mechanism: Fifth pre-MLP scale-to-weight gauge fixing
evidence_used: Fixing gains 3, 4, 5, and 7 achieved 99.92% accuracy at 1612 parameters. Coordinate 6 is the only remaining untested distributed-anchor gain; its embedding anchor passed at 99.72%, while gain coordinate 1 previously collapsed to 27.21% and coordinate 2 has shown bias sensitivity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 2 to one will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy, because its scale is absorbable by `fc1.weight` while its known-sensitive bias remains learned.
change: Remove gain coordinate 2 from the learned vector and reconstruct gains 2 through 7 as fixed ones.
mechanism: Bias-sensitive coordinate scale gauge fixing
evidence_used: Coordinate 5 bias fixing collapsed to 73.24%, but fixing its gain retained 99.95%, showing that bias sensitivity does not imply gain sensitivity; coordinate 2 can therefore retain its learned shift while testing the same successful scale redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4842, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm bias coordinate 4 at zero will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy, because its shift is absorbable by `fc1.bias` and coordinate 4 has remained viable while its gain was fixed.
change: Remove bias coordinate 4 from the learned vector and reconstruct bias coordinates 0, 1, 3, 4, and 7 as fixed zeros.
mechanism: Fifth pre-MLP bias gauge fixing
evidence_used: The current design achieved 100% accuracy at 1611 parameters with gain coordinate 4 fixed, and coordinate 4’s distributed embedding anchor previously retained 99.94%; this makes it the strongest untested remaining bias coordinate after bias coordinates 2 and 5 caused failures.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm bias coordinate 6 at zero will reduce the model from 1610 to 1609 parameters while retaining at least 99% accuracy, because its shift is absorbable by `fc1.bias`, its gain is already fixed, and unlike remaining bias coordinates 2 and 5 it has not caused a prior failure.
change: Remove bias coordinate 6 from the learned vector and reconstruct bias coordinates 0, 1, 3, 4, 6, and 7 as fixed zeros.
mechanism: Sixth pre-MLP bias gauge fixing
evidence_used: The 1610-parameter design reached 99.91% after fixing bias coordinate 4; coordinates 2 and 5 previously collapsed when fixed, whereas coordinate 6’s distributed embedding anchor retained 99.72% and its gain was fixed in the 100%-accurate 1611-parameter design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 0 to one will reduce the model from 1609 to 1608 parameters while retaining at least 99% accuracy, because its scale is absorbable by column 0 of `fc1.weight` and its bias has remained successfully fixed throughout every passing anchored design.
change: Remove gain coordinate 0 from the learned vector and reconstruct gains 0 and 3 through 7 as fixed ones, leaving the sensitive gains 1 and 2 learned.
mechanism: Final untested pre-MLP scale gauge fixing
evidence_used: Fixing gains 3 through 7 retained 100% accuracy at 1611 parameters, while fixing gains 1 or 2 collapsed to 27.21% and 48.42%; coordinate 0 is the only untested gain and already tolerates a fixed LayerNorm bias in the 99.99%-accurate 1609-parameter design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9903, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-attention LayerNorm bias coordinate 3 at zero will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because its query and value shifts are absorbable by the learned attention biases, while its constant key shift cancels in the softmax.
change: Replace the first block LayerNorm with an equivalent parameterization whose bias coordinate 3 is fixed at zero.
mechanism: Pre-attention shift-to-bias gauge fixing
evidence_used: Coordinate 3 has been the strongest stable anchor: fixing its pre-MLP bias retained 99.93%, and its gain also remained fixed in subsequent passing designs. This tests a distinct exact attention-side redundancy without further constraining the sensitive pre-MLP coordinates 1, 2, or 5.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-attention LayerNorm gain coordinate 3 to one will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because its bias is already zero and its scale is exactly absorbable by column 3 of the learned QKV projection.
change: Remove pre-attention gain coordinate 3 from the learned vector and reconstruct it as a fixed one during the forward pass.
mechanism: Paired pre-attention scale gauge fixing
evidence_used: Fixing pre-attention bias coordinate 3 retained 99.91% accuracy at 1607 parameters, while fixing the corresponding pre-MLP gain retained 99.93%; this tests the same demonstrated scale-to-weight redundancy on the attention side without constraining a new coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9869, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-attention LayerNorm bias coordinate 4 at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because its query and value shifts are absorbable by learned attention biases and its constant key shift cancels under softmax.
change: Remove pre-attention bias coordinate 4 from the learned vector and reconstruct bias coordinates 3 and 4 as fixed zeros.
mechanism: Second pre-attention shift-to-bias gauge fixing
evidence_used: Fixing pre-attention bias coordinate 3 achieved 99.91% at 1607 parameters, whereas fixing its gain narrowly failed at 98.69%; coordinate 4 is the strongest next shift candidate because its pre-MLP bias fixing achieved 99.91% and its distributed embedding anchor achieved 99.94%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.36340000000000006, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 3 of `fc2.bias` at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the sole block is followed immediately by final LayerNorm, which makes any common shift across all MLP output coordinates unobservable.
change: Replace the eight-parameter second MLP bias with seven learned coordinates and reconstruct coordinate 3 as a fixed zero during the forward pass.
mechanism: Final-LayerNorm common-shift gauge fixing
evidence_used: The 1607-parameter design reached 99.91% accuracy, while the next attention-side bias constraint collapsed to 36.34%; this motivates testing a distinct exact null direction, using coordinate 3 because its pre-MLP gain and bias constraints both previously retained at least 99.9% accuracy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.59, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 0 of `fc1.bias` at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because either remaining learned pre-MLP LayerNorm shift can absorb that scalar while the other hidden-unit biases compensate for its distributed effect.
change: Replace the first MLP projection with an initialization-equivalent linear layer whose bias has eleven learned coordinates and a fixed-zero coordinate 0.
mechanism: Reverse pre-MLP shift-to-bias gauge fixing
evidence_used: The 1607-parameter design achieved 99.91% accuracy while pre-MLP bias coordinates 2 and 5 remained learned; attempts to fix those shifts failed, so this patch preserves both sensitive coordinates and tests the reverse redundancy after six other pre-MLP shifts were successfully absorbed by `fc1.bias`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing coordinate 1 of `fc1.bias` at zero will reduce the model from 1606 to 1605 parameters while retaining at least 99% accuracy, because the second remaining learned pre-MLP LayerNorm shift provides another independent degree of freedom for absorbing a hidden-unit bias constraint.
change: Replace the first MLP projection’s eleven learned bias coordinates with ten, reconstructing bias coordinates 0 and 1 as fixed zeros while preserving ordinary linear-layer initialization RNG use.
mechanism: Second reverse pre-MLP shift-to-bias gauge fixing
evidence_used: Fixing `fc1.bias` coordinate 0 achieved 99.96% accuracy at 1606 parameters; hidden units are initialization-symmetric, and two pre-MLP LayerNorm shifts remain learned, motivating the corresponding second constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1605, "training_steps": 4999}



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
