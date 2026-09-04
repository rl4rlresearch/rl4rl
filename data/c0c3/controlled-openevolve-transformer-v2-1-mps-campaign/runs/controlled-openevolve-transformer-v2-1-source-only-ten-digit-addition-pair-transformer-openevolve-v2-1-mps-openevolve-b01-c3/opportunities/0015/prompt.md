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
verified_results: {"accuracy": 1.0, "parameters": 1534, "training_steps": 4999}
prior_hypothesis: Fixing one final LayerNorm scale coordinate at 1 while learning the other seven will produce a 1,534-parameter model with at least 99% accuracy, because it removes the prediction-invariant global-scale degree of freedom without imposing the failed mean-one constraint that prevented the learned scale average from moving.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1573, "training_steps": 4999}
prior_hypothesis: Constraining positional embeddings to the seven-dimensional mean-zero subspace, alongside the verified affine-free block norms, will reduce parameters below 1,596 while retaining at least 99% accuracy because position-dependent all-ones components are removed by every downstream LayerNorm and never affect logits.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9963, "parameters": 1535, "training_steps": 4999}
prior_hypothesis: Applying the verified 1,543-parameter gauge reductions and removing only the final LayerNorm bias will produce a 1,535-parameter model with at least 99% accuracy, because the terminal bias supplies only a position-independent class prior while the learned scale and all proven-critical internal biases remain intact.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing only the attention key bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy, because a position-independent key bias adds the same value to every unmasked attention logit for a query and therefore cancels exactly in softmax.
change: Replace the packed QKV bias with learned query and value biases, preserving all functional attention biases, `d_ff=12`, and both residual-output biases.
mechanism: Softmax-invariant key-bias elimination
evidence_used: The 1,644-parameter design reached 99.96% accuracy, while reducing nonlinear width or removing residual-output biases caused large regressions; this patch preserves those demonstrated capacities and removes only a mathematically non-identifiable parameter vector.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the attention value bias in addition to the proven-redundant key bias will reduce parameters from 1,644 to 1,628 while retaining at least 99% accuracy, because attention weights sum to one and the retained output-projection bias can represent the value bias’s constant contribution.
change: Use a bias-free packed QKV projection with only an explicit learned query bias, while retaining both residual-output biases and `d_ff=12`.
mechanism: Softmax-normalized value-bias absorption
evidence_used: Key-bias elimination achieved 99.95% accuracy with 1,636 parameters, while removing residual-output biases reduced accuracy to 75.38%; this patch preserves those critical output biases and removes a different attention bias that is functionally absorbable by the retained projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified bias-minimal attention with affine-free pre-norm LayerNorms will reduce parameters from 1,628 to 1,596 while retaining at least 99% accuracy, because each removed LayerNorm scale can be absorbed into its following learned projection, while its shift is absorbed by the retained query, attention-output, and MLP input biases.
change: Use bias-free packed QKV with only a learned query bias, and disable learned affine parameters in both transformer-block LayerNorms while preserving `d_ff=12` and both residual-output biases.
mechanism: Downstream absorption of pre-normalization affine parameters
evidence_used: Reference Design 2 achieved 99.98% accuracy with 1,628 parameters after removing key and value biases; unlike the failed feed-forward-width and residual-output-bias reductions, this patch preserves those demonstrated capacities and removes downstream-reparameterizable normalization parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the QKV and MLP input projections over the seven-dimensional mean-zero LayerNorm subspace will reduce parameters from 1,596 to 1,560 while retaining at least 99% accuracy, because it preserves their full function classes and isotropic initialization on all inputs they receive.
change: Add a parameter-efficient linear layer that projects affine-free LayerNorm outputs onto an orthonormal basis of their mean-zero subspace, then use it for QKV and MLP input projections.
mechanism: Orthonormal mean-zero input projection
evidence_used: The 1,596-parameter affine-free LayerNorm design achieved 99.81% accuracy, establishing that both projections receive mean-zero normalized inputs; unlike the failed `d_ff` reduction, this removes only unidentifiable input directions while preserving `d_ff=12`.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0022, "parameters": 1560, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining positional embeddings to the seven-dimensional mean-zero subspace, alongside the verified affine-free block norms, will reduce parameters below 1,596 while retaining at least 99% accuracy because position-dependent all-ones components are removed by every downstream LayerNorm and never affect logits.
change: Use an orthonormally parameterized mean-zero positional embedding and disable affine parameters in both block LayerNorms, while preserving the verified attention and MLP capacities.
mechanism: Mean-zero positional gauge fixing
evidence_used: Affine-free block norms achieved 99.81% accuracy with 1,596 parameters; the failed 1,560-parameter projection rewrite motivates leaving QKV and MLP projections unchanged and instead removing only the position-embedding direction that is exactly invisible to the pre-norm residual network and final LayerNorm.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1573, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining both residual-branch output projections to the seven-dimensional mean-zero subspace will reduce the verified 1,573-parameter design to 1,551 parameters while retaining at least 99% accuracy, because every discarded all-ones output component is invisible to subsequent affine-free pre-norms and the final LayerNorm.
change: Reproduce the verified mean-zero positional, query-only attention, affine-free block-norm design, then parameterize the attention and MLP residual outputs in the mean-zero subspace while retaining their seven identifiable bias dimensions.
mechanism: Mean-zero residual-update codomains
evidence_used: The 1,573-parameter design reached 99.93% accuracy. Removing entire residual-output biases failed at 75.38%, so this patch preserves their identifiable mean-zero components and removes only the scalar residual directions that downstream LayerNorms eliminate exactly.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1551, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering tied token embeddings across vocabulary entries will reduce the verified 1,551-parameter model to 1,543 parameters while retaining at least 99% accuracy, because the removed shared embedding vector can be absorbed by the mean-zero positional embeddings up to an all-ones residual shift, while its output-head effect is only a class-independent logit shift.
change: Parameterize the tied token embedding matrix in the vocabulary-mean-zero subspace and use its reconstructed full matrix for both input lookup and output logits.
mechanism: Vocabulary-translation gauge fixing
evidence_used: The current mean-zero positional and residual-output design achieved 99.94% accuracy with 1,551 parameters, demonstrating that featurewise all-ones residual shifts are discarded by downstream normalization; vocabulary-wide translation of tied embeddings adds only a softmax-invariant common output-logit offset.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1543, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing each head’s query/key routing width from 4 to 3 while preserving 4-dimensional values, `d_ff=12`, and all verified gauge reductions will produce a 1,509-parameter model with at least 99% accuracy; the old design assumes routing and transported content need equal width, while addition may require richer value content than attention-addressing geometry.
change: Apply the verified mean-zero residual-output and vocabulary-centered embedding parameterizations, then replace packed QKV attention with separate 3-dimensional-per-head query/key projections and full-width value projections.
mechanism: Narrow-key/query attention with full-width values
evidence_used: The 1,543-parameter gauge-fixed design achieved 99.99% accuracy, while reducing `d_ff` to 8 caused a collapse to 72.25%; this motivates preserving nonlinear and value-stream capacity while testing the previously unchallenged query/key-width assumption.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9818000000000001, "parameters": 1509, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified 1,543-parameter gauge reductions and removing only the final LayerNorm bias will produce a 1,535-parameter model with at least 99% accuracy, because the terminal bias supplies only a position-independent class prior while the learned scale and all proven-critical internal biases remain intact.
change: Add mean-zero positional and residual-output parameterizations, vocabulary-center the tied embeddings, and replace the final LayerNorm with a learned-scale-only normalization.
mechanism: Gauge-fixed embeddings and residuals with bias-free terminal normalization
evidence_used: The fully gauge-fixed 1,543-parameter design achieved 99.99% accuracy; unlike the failed removal of internal residual-output biases, this removes only the terminal normalization shift after all attention and MLP computation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9963, "parameters": 1535, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the final LayerNorm scale to have mean one will reduce the verified 1,535-parameter model to 1,534 parameters while retaining at least 99% accuracy, because any positive global multiplier of its scale only changes logit temperature and not greedy predictions, while all seven relative feature scales remain learnable.
change: Reparameterize the eight-element terminal normalization scale as an all-ones vector plus seven learned mean-zero coordinates.
mechanism: Mean-fixed terminal scale gauge
evidence_used: The current gauge-fixed model reached 99.63% accuracy with 1,535 parameters after removing the final LayerNorm bias; this motivates a minimal follow-up that preserves every relative terminal scale rather than removing the full scale vector.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9887, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one final LayerNorm scale coordinate at 1 while learning the other seven will produce a 1,534-parameter model with at least 99% accuracy, because it removes the prediction-invariant global-scale degree of freedom without imposing the failed mean-one constraint that prevented the learned scale average from moving.
change: Replace the affine final LayerNorm with a bias-free normalization whose eight-element scale consists of seven learned coordinates and one fixed anchor.
mechanism: Anchored terminal-scale gauge
evidence_used: Bias-free terminal normalization achieved 99.63% with 1,535 parameters, while fixing the scale mean at one narrowly failed at 98.87%; anchoring one coordinate tests a less restrictive seven-parameter scale chart that retains variation in the overall scale average.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified 1,534-parameter gauge-fixed design and fixing one additional final LayerNorm scale coordinate at 1 will produce a 1,533-parameter model with at least 99% accuracy, because the 1,534-parameter design achieved 100% accuracy and this removes only one relative terminal-scale degree of freedom while leaving six independently learned scales.
change: Add vocabulary-centered tied embeddings and mean-zero residual projections, then use a bias-free final LayerNorm with six learned scale coordinates and two fixed anchors.
mechanism: Two-coordinate terminal-scale anchoring with verified gauge reductions
evidence_used: The seven-scale, one-anchor design achieved 100% accuracy with 1,534 parameters. The failed mean-one constraint coupled all scale coordinates, whereas this patch makes the smallest independent follow-up ablation by fixing only one additional coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.72, "parameters": 1533, "training_steps": 4999}



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
