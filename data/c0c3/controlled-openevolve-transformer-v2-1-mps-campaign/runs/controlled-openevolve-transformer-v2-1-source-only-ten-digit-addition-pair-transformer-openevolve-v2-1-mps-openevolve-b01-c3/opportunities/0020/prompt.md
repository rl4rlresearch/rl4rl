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
verified_results: {"accuracy": 0.9994, "parameters": 1532, "training_steps": 4999}
prior_hypothesis: Fixing one query-bias coordinate per attention head will reduce the model from 1,534 to 1,532 parameters while retaining at least 99% accuracy, because a headwise orthogonal rotation can move any query-bias vector into the remaining three coordinates while applying the same rotation to queries and keys leaves every attention logit unchanged.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.998, "parameters": 1527, "training_steps": 4999}
prior_hypothesis: Tying the two scalar query biases will reduce the verified 1,528-parameter design to 1,527 parameters while retaining at least 99% accuracy, because each head’s unrestricted query and key projections permit reciprocal rescaling that normalizes nonzero bias amplitudes to one shared learned value without changing attention logits.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9986, "parameters": 1528, "training_steps": 4999}
prior_hypothesis: Reducing each head’s query bias from four coordinates to one scalar, while applying the verified anchored terminal scale, will produce a 1,528-parameter model with at least 99% accuracy because unrestricted query/key projections can rotate any nonzero headwise bias onto a single coordinate without changing attention logits.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing one query-bias coordinate per attention head will reduce the model from 1,534 to 1,532 parameters while retaining at least 99% accuracy, because a headwise orthogonal rotation can move any query-bias vector into the remaining three coordinates while applying the same rotation to queries and keys leaves every attention logit unchanged.
change: Store three learned query-bias coordinates per four-dimensional head, reconstructing the fourth as zero, while preserving full query/key projection widths and the verified terminal normalization.
mechanism: Headwise query-bias rotational gauge fixing
evidence_used: The 1,534-parameter design achieved 100% accuracy, whereas narrowing query/key width reached only 98.18% and adding a second terminal-scale anchor collapsed to 72%; this motivates preserving those capacities and removing only an exact headwise attention-coordinate redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1532, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing each head’s query bias from four coordinates to one scalar, while applying the verified anchored terminal scale, will produce a 1,528-parameter model with at least 99% accuracy because unrestricted query/key projections can rotate any nonzero headwise bias onto a single coordinate without changing attention logits.
change: Use one learned query-bias scalar per head reconstructed with three fixed zeros, and replace the final normalization with the verified seven-scale, one-anchor parameterization.
mechanism: Full headwise query-bias rotational gauge fixing
evidence_used: The 1,532-parameter design achieved 99.94% accuracy after fixing one query-bias coordinate per head, while the one-anchor terminal scale previously achieved 100%; this extends the same headwise rotational gauge without narrowing query/key projections or removing terminal relative-scale capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1528, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the two scalar query biases will reduce the verified 1,528-parameter design to 1,527 parameters while retaining at least 99% accuracy, because each head’s unrestricted query and key projections permit reciprocal rescaling that normalizes nonzero bias amplitudes to one shared learned value without changing attention logits.
change: Apply the verified vocabulary-centered embeddings, mean-zero residual projections, anchored terminal normalization, and scalar query-bias gauge reductions, then share one learned query-bias scalar across both attention heads.
mechanism: Cross-head query-bias scale gauge tying
evidence_used: The independent-scalar query-bias design achieved 99.86% accuracy with 1,528 parameters; sharing only their amplitudes preserves full query/key widths and tests a remaining headwise scaling redundancy without repeating the failed capacity reductions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1527, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the shared scalar query bias to one will reduce the verified 1,527-parameter model to 1,526 parameters while retaining at least 99% accuracy, because reciprocal rescaling of each head’s unrestricted query and key projections can absorb any nonzero shared bias amplitude without changing attention logits.
change: Remove the learned shared query-bias scalar and reconstruct both heads’ query biases with a fixed unit coordinate.
mechanism: Fixed query-bias scale gauge
evidence_used: Sharing the two scalar query biases achieved 99.8% accuracy with 1,527 parameters; its successful cross-head scale tying supports taking the remaining nonzero amplitude as a gauge and anchoring it while preserving full query/key width and all other verified capacities.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3969, "parameters": 1526, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the verified learned query-bias scalar and fixing one redundant QKV input coefficient will produce a 1,526-parameter model with at least 99% accuracy, because affine-free LayerNorm makes the QKV input mean-zero, so a row’s common input-weight component is functionally invisible.
change: Tie the two query-bias scalars while replacing QKV with a bias-free projection whose first output row fixes one input coefficient to zero; all other projection weights and capacities remain unchanged.
mechanism: Single-coordinate QKV input gauge fixing
evidence_used: The shared-bias 1,527-parameter design reached 99.8%, while fixing its bias amplitude collapsed to 39.69%; this retains the successful learned amplitude and removes a different, exact null direction instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2957, "parameters": 1526, "training_steps": 4999}

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
