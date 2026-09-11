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
verified_results: {"accuracy": 0.9995, "parameters": 1430, "training_steps": 4999}
prior_hypothesis: Reparameterizing all key projections on seven zero-mean LayerNorm coordinates will reduce the model from 1,438 to 1,430 parameters while retaining at least 99% accuracy, because key constants are softmax-invisible and virtual full-width AdamW moments preserve the successful optimization geometry.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing the first LayerNorm’s learned scale-and-shift with non-affine normalization plus a learned shift will reduce the model from 1,576 to 1,568 parameters while retaining at least 99% accuracy, because its scale can be absorbed exactly into the full-width QKV projection.
change: Remove only the eight learned `ln1` scale parameters while preserving its learned shift, full eight-coordinate attention input, and all attention bias pathways.
mechanism: Attention pre-normalization scale absorption
evidence_used: Removing both `ln1` affine vectors together with an attention-input coordinate failed at 77.79%, so it did not isolate affine removal; the analogous MLP LayerNorm affine absorption succeeded, motivating a conservative test of only the exactly redundant attention scale.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7548, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing only the MLP output projection from eight to seven learned coordinates will produce a 1,564-parameter model with at least 99% accuracy, because the omitted coordinate can be subtracted from every MLP output coordinate as a LayerNorm-invisible common-mode residual shift.
change: Make `fc2` produce seven coordinates and zero-pad the eighth, while leaving the previously implicated attention output projection and shared value/output bias unchanged.
mechanism: Isolated MLP residual-output gauge fixing
evidence_used: Seven-coordinate positional embeddings and MLP output bias reached 99.82%. The combined attention-and-MLP output reduction failed at 72.44%, but the attention projection bias is coupled to the value bias; isolating the independently gauge-equivalent MLP projection tests the supported invariance without disturbing that load-bearing attention pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9333, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln1` scale coordinate at one while retaining the other seven learned scales will reduce the model from 1,576 to 1,575 parameters and maintain at least 99% accuracy, because that scale can be absorbed into the corresponding QKV input-weight column and LayerNorm bias.
change: Replace the first LayerNorm’s eight learned scales with seven learned scales plus one fixed unit scale, while retaining all eight learned shifts and the full-width attention input.
mechanism: Single-coordinate LayerNorm scale gauge fixing
evidence_used: Removing all eight `ln1` scales fell to 75.48%, showing attention optimization is sensitive to wholesale scale removal; this conservative patch fixes only one analytically absorbable coordinate and preserves seven adaptive scales.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.748, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing the MLP output in an orthonormal seven-dimensional zero-mean basis will produce a 1,564-parameter model with at least 99% accuracy, because it removes only the LayerNorm-invisible common mode while preserving isotropic initialization and well-conditioned gradients.
change: Make `fc2` produce seven coordinates, including its bias, then project them through a fixed orthonormal basis spanning the eight-dimensional zero-mean residual subspace.
mechanism: Orthonormal residual gauge coordinates
evidence_used: The zero-padded seven-coordinate MLP projection reached only 93.33%, despite being function-class equivalent modulo LayerNorm; its induced coordinate metric has an eightfold weak direction. The current 1,576-parameter gauge-fixed model reached 99.82%, motivating the same exact 12-parameter reduction with an orthonormal rather than asymmetric gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1275, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the LayerNorm-invisible common-mode component of the 12 MLP output-weight columns as the 12 hidden biases will reduce parameters from 1,576 to 1,564 while maintaining at least 99% accuracy.
change: Remove the independent `fc1` bias, derive an equally conditioned hidden bias from the normalized row-sum of `fc2.weight`, and center that weight before producing the residual update.
mechanism: Orthogonal reuse of residual common-mode weights
evidence_used: Direct seven-coordinate MLP output projections failed at 93.33% and 12.75% despite common-mode redundancy, suggesting harmful optimization geometry; the successful 1,576-parameter design confirms that residual common modes are unobservable. This patch preserves the full output-weight tensor and repurposes its redundant orthogonal component instead of deleting it.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7172, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only one LayerNorm-invisible common-mode weight from one MLP output column will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because the other 11 columns keep their original optimization geometry.
change: Split `fc2` into an eight-output projection for 11 hidden features and a seven-coordinate projection vector for the final feature, padding its eighth residual coordinate with zero.
mechanism: Single-column residual common-mode gauge fixing
evidence_used: Removing all 12 MLP output common modes at once reached only 93.33% or 12.75%, despite exact functional redundancy, indicating an optimization-geometry problem; the 1,576-parameter model reached 99.82%, motivating the smallest possible isolated gauge reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2712, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating the 12 LayerNorm-invisible common-mode `fc2` weights while initializing from full-width weight differences and emulating the omitted row’s AdamW moments will produce a 1,564-parameter model with at least 99% accuracy.
change: Reduce `fc2` to seven residual coordinates, preserve the successful full-width observable initialization, and train it with quotient-aware AdamW updates and gradient clipping.
mechanism: Optimizer-preserving residual gauge quotient
evidence_used: Direct and orthonormal seven-coordinate `fc2` projections reached only 93.33% and 12.75%, while the full-width model reached 99.82%; this indicates that altered initialization and optimizer geometry, rather than lost function capacity, caused the failures.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the 161-parameter absolute positional table with 46 learned causal distance logits will produce a 1,449-parameter transformer with at least 99% accuracy, because addition’s reusable token alignments can be represented directly by attention offsets rather than reconstructed from absolute position vectors.
change: Remove absolute positional embeddings and give each attention head a learned bias for every causal relative distance, while preserving the existing model’s initialization RNG stream for a clean comparison.
mechanism: Head-specific learned relative-distance attention
evidence_used: Compressing absolute positional embeddings to four dimensions reached only 73.99%, showing that positional information is load-bearing; the successful seven-contrast design retained 99.82%. This patch challenges the shared assumption that those relationships require residual-stream position vectors by moving positional representation into a fully learned, head-specific attention mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1449, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one relative-attention bias per head at zero and training the remaining differences with virtual full-width AdamW moments will reduce the model from 1,449 to 1,447 parameters while retaining at least 99% accuracy.
change: Store 22 relative-distance bias differences per head instead of 23, reconstruct a fixed zero reference bias, and generalize the quotient-aware optimizer to preserve the omitted biases’ gradients and AdamW moments.
mechanism: Optimizer-preserving attention-logit gauge quotient
evidence_used: The 1,449-parameter relative-distance model achieved 99.99%, and quotient-aware gauge optimization previously preserved 99.75% accuracy when direct removal of functionally redundant MLP coordinates failed; this applies the same optimizer-preserving method to an exact per-head softmax-shift invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1447, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the single global additive degree of freedom shared by every tied embedding coordinate will reduce the model from 1,447 to 1,446 parameters while retaining at least 99% accuracy, because it changes input residuals only by LayerNorm-invisible scalar shifts and output logits only by a softmax-invisible common shift.
change: Store all but one flattened embedding coordinate as differences from a fixed reference, reconstruct the tied input/output weight dynamically, preserve the original initialization RNG stream, and include the omitted coordinate in quotient-aware AdamW moments and gradient clipping.
mechanism: Optimizer-preserving tied-embedding common-mode gauge quotient
evidence_used: The 1,447-parameter model reached 99.97%, and optimizer-preserving gauge quotients successfully removed both MLP residual and relative-attention common modes where direct reparameterizations disrupted training; this applies the same proven optimization treatment to another exact model-wide invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1446, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight LayerNorm-invisible common-mode weights from the attention output projection will reduce the model from 1,446 to 1,438 parameters while retaining at least 99% accuracy when full-width initialization and virtual AdamW moments are preserved.
change: Store seven output rows for the attention projection, reconstruct an eighth zero row, retain its full-width shared value/output bias, and train the omitted row through the existing quotient-aware optimizer.
mechanism: Optimizer-preserving attention residual gauge quotient
evidence_used: Quotient-aware optimization let the seven-row MLP projection reach 99.75% after direct reparameterizations failed at 93.33% and 12.75%; the earlier combined attention/MLP reduction lacked this optimizer-preserving treatment, while the current 1,446-parameter model provides 99.93% accuracy headroom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1438, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing all key projections on seven zero-mean LayerNorm coordinates will reduce the model from 1,438 to 1,430 parameters while retaining at least 99% accuracy, because key constants are softmax-invisible and virtual full-width AdamW moments preserve the successful optimization geometry.
change: Replace the full-width key portion of QKV with an eight-by-seven learned projection over non-affine LayerNorm outputs, retain full-width query/value projections and `ln1` affine parameters, and add the omitted key coordinates to quotient-aware optimization.
mechanism: Optimizer-preserving key-projection LayerNorm quotient
evidence_used: Quotient-aware optimization preserved 99.75% for the MLP residual quotient and 99.93% for the attention-output quotient, whereas removing `ln1` scales failed near 75%; this isolates an exact key-only redundancy without constraining the load-bearing query/value affine pathways.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1430, "training_steps": 4999}



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
