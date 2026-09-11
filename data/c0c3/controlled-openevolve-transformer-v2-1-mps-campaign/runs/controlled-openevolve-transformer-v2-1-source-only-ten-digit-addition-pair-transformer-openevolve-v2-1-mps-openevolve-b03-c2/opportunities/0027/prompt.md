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
verified_results: {"accuracy": 0.9942, "parameters": 1595, "training_steps": 4999}
prior_hypothesis: Removing both block LayerNorm bias vectors will produce a 1595-parameter model with at least 99% accuracy, because their affine offsets can be represented by the existing QKV and `fc1` biases without reducing the learned model’s function class.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9998, "parameters": 1593, "training_steps": 4999}
prior_hypothesis: Extending the qualified value-bias quotient from one to two omitted coordinates will produce a 1593-parameter model with at least 99% accuracy while preserving both omitted coordinates’ clipping and AdamW dynamics.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9982, "parameters": 1612, "training_steps": 4999}
prior_hypothesis: Combining the verified positional quotient with the verified MLP-output quotient, while restoring the attention-output bias, will produce a 1612-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing all eight softmax-invariant key-bias coordinates as one learned scalar will produce a 1637-parameter model with at least 99% accuracy; unlike the failed 1637 design, it removes the asymmetric fixed-zero anchor.
change: Replace the current one-zero/five-shared key-bias parameterization with a single learned value shared across every key-bias coordinate.
mechanism: Symmetric complete sharing of key-projection bias
evidence_used: Six-coordinate sharing achieved 99.96% accuracy at 1638 parameters, while the asymmetric 1637 design with one fixed-zero coordinate and seven shared coordinates collapsed to 83.72%; fully symmetric sharing isolates whether that anchoring caused the optimization discontinuity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1637, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the mean learned query bias across all key-bias coordinates will produce a 1636-parameter model with at least 99% accuracy.
change: Replace the independent shared key-bias scalar with the mean of the learned query bias, while retaining one fused QKV parameter and a symmetric key bias.
mechanism: Cross-projection sharing of the softmax-invariant key bias
evidence_used: Symmetric complete key-bias sharing achieved 99.97% at 1637 parameters, whereas fixed-zero key-bias elimination failed; reusing a learned query-bias statistic removes the last independent key parameter without reintroducing a fixed-zero anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing each MLP output bias with seven relative coordinates and applying the difference of the corresponding eight-coordinate AdamW updates will produce a 1635-parameter model with at least 99% accuracy.
change: Remove the final-LayerNorm-invariant uniform degree of each `fc2` bias, reconstruct a mean-anchored eight-coordinate bias, and preserve full-bias gradient clipping and AdamW quotient dynamics.
mechanism: Mean-anchored quotient bias with AdamW trajectory preservation
evidence_used: The current 1636-parameter model reached 99.92%; the earlier fixed-zero MLP-output gauge reduction collapsed to 71.82%, showing that this exact redundancy is optimization-sensitive and motivating trajectory-preserving removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the feature-uniform degree from both MLP and attention output biases will produce a 1634-parameter model with at least 99% accuracy, because both uniform residual shifts are canceled by downstream LayerNorms and quotient-aware optimization preserves their full-bias AdamW dynamics.
change: Adopt the qualified query-derived key bias, reduce both `fc2.bias` and `attn.proj.bias` to seven relative coordinates, reconstruct their eight-coordinate biases during projection, and train them with quotient-aware clipping and AdamW updates.
mechanism: Dual residual-output quotient biases with AdamW trajectory preservation
evidence_used: The 1635-parameter MLP quotient design achieved 99.85%, demonstrating that trajectory-preserving removal of a LayerNorm-invariant bias degree works; the attention output bias has the same exact uniform-shift invariance, motivating the next one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9942, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing each positional embedding row’s feature-uniform degree while preserving its full-coordinate clipping and AdamW difference dynamics will reduce the model to `1638 - INPUT_LEN` parameters and retain at least 99% accuracy.
change: Store seven relative coordinates per positional embedding row, reconstruct eight-coordinate embeddings during forward passes, and apply row-wise quotient-aware gradient clipping and AdamW updates.
mechanism: Row-wise positional-embedding quotient under pre-LayerNorm invariance
evidence_used: The current 1638-parameter design achieved 99.96% accuracy, while the 1635-parameter MLP quotient design achieved 99.85%, demonstrating that trajectory-preserving removal of feature-uniform coordinates canceled by downstream LayerNorm can retain accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the qualified 1634-parameter dual-output quotient design with the verified 23-parameter positional quotient will produce a 1611-parameter model with at least 99% accuracy.
change: Adopt query-derived key bias and seven-coordinate attention/MLP output biases, retain compressed positional embeddings, and apply quotient-aware clipping and AdamW updates to all three parameter groups.
mechanism: Composed positional and residual-output quotient gauges
evidence_used: The dual-output quotient design achieved 99.42% at 1634 parameters, while positional quotienting independently removed one parameter per position and achieved 99.83%; both preserve full-coordinate clipping and AdamW difference dynamics.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7356999999999999, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified positional quotient with the verified MLP-output quotient, while restoring the attention-output bias, will produce a 1612-parameter model with at least 99% accuracy.
change: Compress every positional embedding row to seven relative coordinates, retain the seven-coordinate MLP output bias, restore the full attention projection bias, and generalize quotient-aware clipping and AdamW updates across vector and row-wise parameters.
mechanism: Positional and MLP quotient gauges with full attention-output bias
evidence_used: Positional quotienting achieved 99.83% at 1615 parameters and MLP-output quotienting achieved 99.85% at 1635, while adding both output quotients to the positional design collapsed to 73.57%; restoring the less independently validated attention quotient isolates that interaction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified positional-plus-MLP quotient design with removal of the tied token embedding’s single globally uniform degree will produce a 1611-parameter model with at least 99% accuracy.
change: Compress each positional row and the tied token embedding into relative coordinates, reconstruct them for embedding and logit computation, and preserve their full-coordinate clipping and AdamW difference dynamics alongside the MLP bias quotient.
mechanism: Global tied-embedding quotient under pre-LayerNorm and logit-shift invariance
evidence_used: The 1612-parameter positional-plus-MLP quotient design achieved 99.82%. The failed 1611 design additionally quotienting the attention-output bias indicates that bias interaction should remain untouched; this patch instead removes a distinct exact gauge while retaining the full attention projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9951000000000001, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing both block LayerNorm bias vectors will produce a 1595-parameter model with at least 99% accuracy, because their affine offsets can be represented by the existing QKV and `fc1` biases without reducing the learned model’s function class.
change: Disable the eight-parameter bias in `ln1` and `ln2` while retaining their learned scale parameters and all downstream projection biases.
mechanism: Downstream projection-bias absorption of pre-LayerNorm offsets
evidence_used: The current 1611-parameter quotient design achieved 99.51% accuracy, and prior successful reductions show that exact redundant degrees can be removed while retaining accuracy. LayerNorm biases initialize to zero, while `qkv` and `fc1` already provide the downstream offsets needed to absorb them.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9942, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1595-parameter design by fixing one `ln2` scale coordinate at unity will produce a 1594-parameter model with at least 99% accuracy, because the omitted common scale can be absorbed by `fc1` while preserving the initialized forward function.
change: Adopt the qualified tied-token quotient and biasless block LayerNorms, then store seven learned `ln2` scale coordinates, reconstruct the eighth as one, and retain quotient-aware optimization for token, positional, and MLP-output parameters.
mechanism: One-coordinate pre-LayerNorm scale gauge
evidence_used: The 1595-parameter design achieved 99.42% accuracy after token quotienting and removal of both block LayerNorm biases; fixing one `ln2` scale coordinate is the smallest next reduction and exploits its exact multiplicative redundancy with the following learned `fc1` matrix.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2896, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1595-parameter design, removing one attention value-bias coordinate and projecting its full-coordinate AdamW update into `attn.proj.bias` will produce a 1594-parameter model with at least 99% accuracy.
change: Adopt the qualified token, positional, MLP-output, key-bias, and biasless-block-LayerNorm reductions; then store seven value-bias coordinates and preserve the omitted eighth coordinate’s gradient clipping and AdamW effect through the downstream attention projection bias.
mechanism: Quotient-aware value-bias absorption into the attention output bias
evidence_used: The 1595-parameter design achieved 99.42% after downstream biases successfully absorbed both block LayerNorm offsets, while trajectory-preserving quotient optimization qualified for positional, token, and MLP-output gauges; an attention value bias is likewise passed unchanged through normalized attention and can be absorbed by the existing projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified value-bias quotient from one to two omitted coordinates will produce a 1593-parameter model with at least 99% accuracy while preserving both omitted coordinates’ clipping and AdamW dynamics.
change: Store six value-bias coordinates, reconstruct two zero-gauge coordinates, and generalize the optimizer and gradient clipping to track and absorb both omitted updates.
mechanism: Multi-coordinate value-bias absorption into the attention projection bias
evidence_used: The current 1594-parameter design achieved 99.73% after one value-bias coordinate was absorbed into `attn.proj.bias`; the second coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1593, "training_steps": 4999}



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
