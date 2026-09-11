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
verified_results: {"accuracy": 0.9978, "parameters": 1035, "training_steps": 4999}
prior_hypothesis: Extending the qualified per-head Q/K gauge fixing to one additional key channel will produce a 1,035-parameter transformer with at least 99% accuracy, because its corresponding learned query channel and query bias retain the reciprocal scaling freedom.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9994, "parameters": 1037, "training_steps": 4999}
prior_hypothesis: Fixing one nonzero key-projection coefficient while leaving its reciprocal query scale learned will reduce the qualified model from 1,038 to 1,037 parameters and retain at least 99% accuracy, because this removes one multiplicative Q/K factorization gauge without replacing the successfully optimized factorized attention.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9965, "parameters": 1034, "training_steps": 4999}
prior_hypothesis: Fixing the second key-channel coefficient in the remaining attention head will produce a 1,034-parameter transformer with at least 99% accuracy because its corresponding learned query channel and bias preserve the reciprocal scaling freedom.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1036, "training_steps": 4999}
prior_hypothesis: Applying the qualified fixed positional amplitude and fixing one key-projection coefficient in each attention head will produce a 1,036-parameter transformer with at least 99% accuracy, because each head has an independent reciprocal query/key scaling gauge and the first such fix already achieved 99.94%.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Replacing separate per-head key/value projections with one learned four-dimensional key/value representation will reduce the qualified 1,038-parameter model to 982 parameters while retaining at least 99% accuracy, because head-specific queries, relative biases, attended summaries, and output mixing still provide distinct routing, while digit identity and position can share a common key/value representation.
change: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then replace the 24-output QKV projection with an eight-dimensional query projection and a shared eight-output key/value projection broadcast across both attention heads.
mechanism: Multi-query causal attention with shared key/value features
evidence_used: The 1,038-parameter design achieved 99.96%, whereas replacing factorized attention with direct quotient-space operators fell to 89.02%; this motivates preserving factorized dot-product attention and head-specific routing while challenging the shared assumption that each head needs independent key and value features. The failed four-coordinate lexical bottleneck also suggests moving compression away from token representation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8368000000000001, "parameters": 982, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero key-projection coefficient while leaving its reciprocal query scale learned will reduce the qualified model from 1,038 to 1,037 parameters and retain at least 99% accuracy, because this removes one multiplicative Q/K factorization gauge without replacing the successfully optimized factorized attention.
change: Add a quotient-input QKV projection whose first key coefficient is a fixed initialization-scale buffer, and use it in causal self-attention.
mechanism: Single-coordinate Q/K scale gauge fixing
evidence_used: The 1,038-parameter factorized-attention design achieved 99.96%, while folding the attention factorizations into direct operators fell to 89.02%; this motivates an incremental gauge fix that preserves the factorized dot-product parameterization and its optimization behavior.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1037, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the qualified fixed positional amplitude and fixing one key-projection coefficient in each attention head will produce a 1,036-parameter transformer with at least 99% accuracy, because each head has an independent reciprocal query/key scaling gauge and the first such fix already achieved 99.94%.
change: Replace the 168 learned QKV coefficients with 166 learned coefficients plus one fixed nonpersistent key coefficient per head, and replace the learned positional-scale scalar with its qualified fixed value.
mechanism: Per-head Q/K scale gauge fixing
evidence_used: The 1,037-parameter design achieved 99.94% after fixing one key coefficient, while the fixed-position 1,038-parameter design achieved 99.96%; extending the same gauge fix to the untouched second head is the narrowest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1036, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified per-head Q/K gauge fixing to one additional key channel will produce a 1,035-parameter transformer with at least 99% accuracy, because its corresponding learned query channel and query bias retain the reciprocal scaling freedom.
change: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then replace three learned QKV coefficients—one per head plus a second channel in the first head—with fixed nonzero initialization-scale coefficients.
mechanism: Third reciprocal Q/K scale gauge fix
evidence_used: The 1,036-parameter design achieved 99.99% accuracy after fixing one key coefficient per head; fixing one additional channel is the narrowest extension of that successful gauge reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 1035, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the second key-channel coefficient in the remaining attention head will produce a 1,034-parameter transformer with at least 99% accuracy because its corresponding learned query channel and bias preserve the reciprocal scaling freedom.
change: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then fix two nonzero key-projection coefficients per attention head.
mechanism: Symmetric fourth reciprocal Q/K scale gauge fix
evidence_used: The 1,036-parameter per-head gauge-fixed design achieved 99.99%, and extending it to three fixed coefficients produced a qualified 1,035-parameter model at 99.78%; symmetrically fixing the second channel of the other head is the narrowest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965, "parameters": 1034, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third key-channel coefficient in the first attention head will reduce the model from 1,034 to 1,033 parameters while retaining at least 99% accuracy, because that channel’s learned query projection and query bias preserve the reciprocal Q/K scaling freedom.
change: Extend the qualified symmetric four-coefficient gauge fixing by fixing one additional nonzero key-projection coefficient in the first attention head.
mechanism: Fifth reciprocal Q/K channel-scale gauge fix
evidence_used: The current model achieved 99.65% accuracy with two fixed key-channel coefficients per head, after the three-coefficient model achieved 99.78%; the next single-channel gauge fix is the narrowest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1033, "training_steps": 4999}



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
