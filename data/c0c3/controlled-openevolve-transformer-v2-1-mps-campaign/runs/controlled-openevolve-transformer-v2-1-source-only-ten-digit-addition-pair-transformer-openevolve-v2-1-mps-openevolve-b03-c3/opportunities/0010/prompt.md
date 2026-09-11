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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9997, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Retaining query and value biases while removing only the key bias will preserve at least 99% accuracy with 1,636 parameters, because a position-independent key bias adds the same scalar to every attention logit in a softmax row.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Removing the key-projection bias and the second LayerNorm bias will retain at least 99% accuracy with 1,628 parameters because key bias cancels under softmax and the LayerNorm offset can be absorbed by the existing MLP input bias.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the sole MLP width from 12 to 8 will retain at least 99% accuracy while reducing parameters from 1,644 to 1,576.
change: Set the default feed-forward width to 8; all training and optimization settings remain unchanged.
mechanism: Narrower nonlinear channel
evidence_used: The current width-12 model achieved 99.96% accuracy, leaving substantial margin above the 99% requirement; removing four MLP units saves 68 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.629, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the width-12 MLP while removing key and value biases will preserve at least 99% accuracy with 1,628 parameters, because key bias cancels in softmax and value bias is representable by the output-projection bias.
change: Replace the fused 24-parameter QKV bias with an 8-parameter learned query-only bias, saving 16 parameters without narrowing the model.
mechanism: Remove functionally redundant attention biases
evidence_used: Reducing MLP width from 12 to 8 lowered accuracy from 99.96% to 62.9%, indicating nonlinear width should be preserved and structurally redundant parameters should be targeted instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1649, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining query and value biases while removing only the key bias will preserve at least 99% accuracy with 1,636 parameters, because a position-independent key bias adds the same scalar to every attention logit in a softmax row.
change: Replace the fused QKV bias with separate learned query and value biases while leaving the key projection unbiased; all other architecture and training settings remain unchanged.
mechanism: Remove softmax-invariant key bias
evidence_used: Removing both key and value biases reduced accuracy to 16.49%, so this narrower ablation preserves the potentially optimization-critical value bias while testing only the algebraically redundant key bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the key-projection bias and the second LayerNorm bias will retain at least 99% accuracy with 1,628 parameters because key bias cancels under softmax and the LayerNorm offset can be absorbed by the existing MLP input bias.
change: Preserve the width-12 MLP and query/value attention biases, while eliminating the redundant key bias and `ln2` bias.
mechanism: Absorbable normalization and softmax-invariant biases
evidence_used: The 1,636-parameter query/value-bias design achieved 99.97% accuracy, while narrowing the MLP failed; this motivates retaining nonlinear width and removing another structurally absorbable 8-parameter bias instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the first LayerNorm bias from the verified 1,628-parameter design will retain at least 99% accuracy with 1,620 parameters, because its query/value effects can be absorbed by the retained biases while its key effect is softmax-invariant.
change: Remove key-projection, `ln1`, and `ln2` biases while retaining learned query/value attention biases and the width-12 MLP.
mechanism: Absorbable pre-attention normalization offset
evidence_used: The 1,628-parameter design without key and `ln2` biases achieved 99.95% accuracy; `ln1` bias is similarly redundant because the following attention projections already provide query/value offsets.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3396, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1,628-parameter design, removing each positional embedding’s all-ones component will preserve at least 99% accuracy while saving one parameter per sequence position, because that component is erased by the pre-block and final LayerNorms under the fixed zero-dropout configuration.
change: Apply the verified key-bias and `ln2`-bias removals, then represent positional embeddings in a learned seven-dimensional orthonormal basis spanning the mean-zero subspace.
mechanism: LayerNorm gauge-fixed positional embeddings
evidence_used: The 1,628-parameter design achieved 99.95% accuracy, whereas removing the full `ln1` bias failed at 33.96%; this motivates retaining optimization-sensitive offsets while eliminating an exact LayerNorm-invariant positional direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7291, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the key bias, `ln2` bias, and attention output-projection bias will retain at least 99% accuracy with 1,620 parameters because the retained value bias supplies the projection’s constant offset while preserving the optimization-critical value-bias pathway.
change: Use unbiased fused QKV weights with learned query/value biases, remove the `ln2` bias, and make the attention output projection unbiased.
mechanism: Value-bias absorption of the attention output offset
evidence_used: The 1,628-parameter design removing key and `ln2` biases achieved 99.95% accuracy, while removing the value bias collapsed accuracy to 16.49%; this motivates retaining value bias and testing the output bias that it can functionally absorb.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0418, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one all-ones gauge degree from the first positional embedding will reduce the verified 1,628-parameter design to 1,627 parameters while retaining at least 99% accuracy, because the removed component is erased throughout the pre-LayerNorm residual stack and the minimal reparameterization avoids the disruption observed when all positional rows were compressed.
change: Apply the verified key-bias and `ln2`-bias removals, then represent only the first positional embedding with seven learned coordinates and one fixed coordinate while preserving its initialization distribution modulo the removed shift.
mechanism: Single-position LayerNorm gauge fixing
evidence_used: The 1,628-parameter design achieved 99.95% accuracy, whereas removing all 23 positional gauge degrees fell to 72.91%; removing only one such degree is the most conservative test of the exact invariance while retaining nearly all original positional optimization geometry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9859, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one coordinate of the first LayerNorm bias from the verified 1,628-parameter design will retain at least 99% accuracy with 1,627 parameters, because its query/value effects can be absorbed by retained projection biases and its key effect is softmax-invariant, while preserving seven of eight optimization-sensitive bias coordinates.
change: Remove the key-projection and `ln2` biases as in the verified 1,628-parameter design, then replace `ln1` with an equivalent LayerNorm whose final bias coordinate is fixed at zero.
mechanism: Single-coordinate redundant normalization-offset removal
evidence_used: The 1,628-parameter design reached 99.95%; removing all eight `ln1` bias coordinates collapsed accuracy to 33.96%, so a one-coordinate ablation tests the same structural redundancy with minimal optimization disruption.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.34159999999999996, "parameters": 1627, "training_steps": 4999}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
