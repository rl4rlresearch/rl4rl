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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1044, "training_steps": 4999}
prior_hypothesis: Removing the five pre-attention LayerNorm bias coefficients will produce a 1,044-parameter transformer with at least 99% accuracy, because their query effect is absorbable by `q_bias`, key shifts cancel in softmax, and value shifts are absorbable by the attention output bias.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9981, "parameters": 1040, "training_steps": 4999}
prior_hypothesis: Reducing the qualified 1,041-parameter model’s pre-MLP LayerNorm bias from two learned directions to one will produce a 1,040-parameter transformer with at least 99% accuracy, because the removed offset through `fc1` remains representable by `fc1.bias`.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1038, "training_steps": 4999}
prior_hypothesis: Fixing the sinusoidal position scale at its initialization value on the qualified affine-free pre-MLP design will produce a 1,038-parameter transformer with at least 99% accuracy, because token and residual-branch weights can learn the required relative scaling.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9994, "parameters": 1039, "training_steps": 4999}
prior_hypothesis: Removing all three pre-MLP LayerNorm bias coefficients will produce a 1,039-parameter transformer with at least 99% accuracy because their constant contribution is exactly representable by `fc1.bias`.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the qualified 1,280-parameter model’s pre-MLP LayerNorm bias from six learned directions to five will produce a 1,279-parameter transformer with at least 99% accuracy, because the removed constant offset remains absorbable by `fc1.bias`.
change: Apply the verified quotient-input MLP and retain five learned zero-mean pre-MLP bias coefficients.
mechanism: Second-direction pre-MLP bias gauge fixing
evidence_used: The six-coefficient quotient-input design achieved 99.95% with 1,280 parameters; the analogous pre-attention LayerNorm retained 99.91% with five coefficients, making a one-direction reduction the narrowest supported next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1279, "training_steps": 4999}

RECENT RESULT
hypothesis: Folding each head’s Q/K and V/output factorizations into unrestricted quotient-space operators, on top of the qualified 1,279-parameter design, will produce a 1,257-parameter transformer with at least 99% accuracy because the direct operators contain every function representable by the removed rank-four factorizations.
change: Adopt the verified quotient-input MLP and five-direction pre-MLP bias, then replace factorized QKV/output attention with learned per-head score and value-output matrices while preserving causal masking, relative biases, dropout, and generic decoding.
mechanism: Gauge-free direct attention-head operators
evidence_used: The 1,279-parameter quotient-input design achieved 99.97% accuracy. Its attention still represents each head’s score as `WqᵀWk` and value/output map as `WoWv`; learning those products directly removes 22 factorization-gauge parameters without narrowing their representable operators.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8902, "parameters": 1257, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the tied token embedding and classifier to six learned coordinates will produce a 1,164-parameter transformer with at least 99% accuracy, because lexical representation need not occupy the full seven-dimensional residual quotient while the successful seven-dimensional attention and MLP computation remains intact.
change: Replace the shared assumption that tokens require every zero-mean residual direction with a six-dimensional learned token/output code, and reduce the aligned final classifier bias accordingly.
mechanism: Six-coordinate tied lexical bottleneck
evidence_used: The 1,325-parameter fixed-token-basis design retained 99.97% accuracy after removing token-subspace tilt, and the current 1,279-parameter model reaches 99.97%; meanwhile, direct attention operators fell to 89.02%, motivating a substantial reduction at the large tied lexical interface while preserving the qualified attention mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1164, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified six-coordinate token embedding and classifier to five learned coordinates will produce a 1,049-parameter transformer with at least 99% accuracy, because the six-coordinate design achieved 99.92% while preserving ample continuous capacity to distinguish the vocabulary.
change: Apply the qualified five-direction pre-MLP bias configuration, reduce the tied token/output code from six coordinates to five, and align the final learned bias with that lexical subspace.
mechanism: Five-coordinate tied lexical bottleneck
evidence_used: The 1,164-parameter six-coordinate design achieved 99.92% accuracy, while the seven-coordinate design achieved 99.97%; the negligible accuracy change motivates testing the next lexical rank before narrowing attention, whose direct-operator reduction failed at 89.02%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1049, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified five-coordinate token embedding and classifier to four coordinates will produce a 934-parameter transformer with at least 99% accuracy, because the five-coordinate model achieved 99.93% and the preceding six-to-five reduction caused no meaningful accuracy loss.
change: Apply the qualified quotient-input MLP and five-direction pre-MLP bias, then reduce the tied token/output code from five coordinates to four and align the final classifier bias with that subspace.
mechanism: Four-coordinate tied lexical bottleneck
evidence_used: The 1,049-parameter five-coordinate design achieved 99.93%, slightly exceeding the 1,164-parameter six-coordinate design’s 99.92%; this supports testing another lexical-rank reduction while preserving the attention architecture whose direct-operator replacement failed at 89.02%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9859, "parameters": 934, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the five pre-attention LayerNorm bias coefficients will produce a 1,044-parameter transformer with at least 99% accuracy, because their query effect is absorbable by `q_bias`, key shifts cancel in softmax, and value shifts are absorbable by the attention output bias.
change: Replace the five-direction pre-attention LayerNorm with an affine-free LayerNorm while preserving the qualified factorized attention architecture.
mechanism: Pre-attention constant-shift absorption
evidence_used: The current 1,049-parameter model achieved 99.93% accuracy, and prior pre-attention gauge evidence established the exact query-bias, key-softmax, and attention-output-bias absorptions used by this reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1044, "training_steps": 4999}

RECENT RESULT
hypothesis: The qualified 1,044-parameter model will retain at least 99% accuracy with 1,043 parameters when the pre-MLP LayerNorm bias is reduced from five directions to four, because the omitted constant contribution is representable by `fc1`’s learned bias.
change: Apply the qualified five-coordinate lexical bottleneck and affine-free pre-attention LayerNorm, then remove one learned pre-MLP bias coefficient.
mechanism: Incremental pre-MLP bias gauge fixing
evidence_used: The five-coordinate model achieved 99.93%, its affine-free pre-attention variant achieved 99.90% with 1,044 parameters, and earlier incremental pre-MLP bias reductions from seven to six and six to five retained 99.95% and 99.97%, respectively.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1043, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified 1,043-parameter model’s pre-MLP LayerNorm bias from four learned directions to three will produce a 1,042-parameter transformer with at least 99% accuracy, because the removed fixed offset through `fc1` remains representable by `fc1.bias`.
change: Apply the qualified five-coordinate lexical bottleneck, affine-free pre-attention LayerNorm, quotient-input MLP, and retain three learned zero-mean pre-MLP bias directions.
mechanism: Incremental pre-MLP bias gauge fixing
evidence_used: The 1,043-parameter four-direction design achieved 99.97% accuracy, while earlier incremental pre-MLP reductions from seven to six, six to five, and five to four all retained at least 99.95%; this makes one further direction the narrowest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1042, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified 1,042-parameter model’s pre-MLP LayerNorm bias from three learned directions to two will produce a 1,041-parameter transformer with at least 99% accuracy, because the omitted constant contribution remains representable by `fc1.bias`.
change: Apply the qualified five-coordinate lexical bottleneck, fixed sinusoidal positions, quotient-space attention and MLP maps, affine-free pre-attention normalization, and retain two learned zero-mean pre-MLP bias directions.
mechanism: Two-direction pre-MLP bias gauge fixing
evidence_used: The 1,042-parameter three-direction design achieved 99.95% accuracy, and every preceding one-direction pre-MLP bias reduction retained at least 99.95%, supporting one further incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1041, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified 1,041-parameter model’s pre-MLP LayerNorm bias from two learned directions to one will produce a 1,040-parameter transformer with at least 99% accuracy, because the removed offset through `fc1` remains representable by `fc1.bias`.
change: Retain one learned zero-mean pre-MLP LayerNorm bias coefficient instead of two while preserving the qualified architecture and training procedure.
mechanism: One-direction pre-MLP bias gauge fixing
evidence_used: The 1,041-parameter two-direction design achieved 99.92% accuracy, and every preceding single-direction pre-MLP bias reduction retained at least 99.95% until that step, making one further incremental reduction the most directly supported test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1040, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all three pre-MLP LayerNorm bias coefficients will produce a 1,039-parameter transformer with at least 99% accuracy because their constant contribution is exactly representable by `fc1.bias`.
change: Replace the three-direction pre-MLP LayerNorm with an affine-free LayerNorm while preserving the qualified architecture and training procedure.
mechanism: Affine-free pre-MLP normalization
evidence_used: The one-direction 1,040-parameter design achieved 99.81% accuracy after every preceding incremental bias reduction qualified; eliminating its final redundant direction is the narrowest remaining parameter reduction, although earlier complete removal showed optimization risk.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1039, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the sinusoidal position scale at its initialization value on the qualified affine-free pre-MLP design will produce a 1,038-parameter transformer with at least 99% accuracy, because token and residual-branch weights can learn the required relative scaling.
change: Remove all four pre-MLP LayerNorm bias coefficients and replace the learned positional-scale scalar with a fixed nonpersistent buffer.
mechanism: Fixed positional-amplitude gauge
evidence_used: The affine-free pre-MLP reference achieved 99.94% accuracy with 1,039 parameters; fixing the remaining positional-scale scalar at its existing initialization is the narrowest next reduction and preserves the successful initial signal balance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1038, "training_steps": 4999}

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
