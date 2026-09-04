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
verified_results: {"accuracy": 1.0, "parameters": 1267, "training_steps": 49999}
prior_hypothesis: Reusing the normalized row-common component of attention projection column 1 as the second learned final-LayerNorm scale will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy; unlike the timed-out raw-mean attempt, the orthonormal scaling gives this scale unit gradient norm and preserves the initialized function exactly.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing a second centered query/key coefficient in the first head will reduce the model from 1,270 to 1,269 parameters while retaining at least 99% accuracy.
change: Rotate the first head’s remaining center-0/center-2 plane at initialization, omit center-2’s first coefficient, and reconstruct it as zero during forward passes.
mechanism: Second function-preserving query/key rotation in the first attention head
evidence_used: One independently initialized rotational gauge per head retained 99.99% accuracy at 1,270 parameters, while imposing all six gauges at once failed; adding one constraint to a single head is the smallest supported continuation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1269, "training_steps": 49999}

RECENT RESULT
hypothesis: Applying the proven second centered rotation gauge to the second head will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy.
change: Rotate the second head’s remaining center-0/center-2 plane at initialization, omit center-2’s first coefficient, and reconstruct it as zero during forward passes.
mechanism: Second function-preserving query/key rotation in the second attention head
evidence_used: The identical second rotation in the first head retained 100% accuracy at 1,269 parameters, while one rotation in each head retained 99.99%; the untouched second-head rotation is the closest supported one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7373999999999999, "parameters": 1268, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing the remaining centered rotational degree of freedom in the first head will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy.
change: Rotate the first head’s center-1/center-2 plane during initialization, omit center-2’s second input coefficient, and reconstruct both fixed coefficients as zeros during forward passes.
mechanism: Third function-preserving query/key rotation in the first attention head
evidence_used: The first head retained 100% accuracy after its second sequential rotation gauge at 1,269 parameters, whereas applying that gauge to the second head fell to 73.74%; therefore the smallest supported next test is the remaining rotation in the head that has tolerated both prior constraints.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3072, "parameters": 1268, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing one rotationally redundant value-projection coefficient will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy, because an orthogonal rotation of two value coordinates within a head can be exactly canceled in the attention output projection.
change: Rotate the first two value rows at initialization, omit the second row’s first coefficient, reconstruct it as zero during forward passes, and counter-rotate the corresponding output-projection columns to preserve the initialized model function exactly.
mechanism: Single value-output rotational gauge
evidence_used: A single function-preserving query/key rotation retained 100% accuracy at 1,271 parameters, but adding further constraints within the increasingly restricted query/key charts failed; this tests one parameter from a separate exact attention symmetry while retaining function-preserving initialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7495999999999999, "parameters": 1268, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing one positional-embedding scalar to zero will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy.
change: Store the learned non-anchor positional table with its final scalar omitted, reconstruct that scalar as zero, and otherwise preserve initialization and training.
mechanism: Single positional-coordinate pruning
evidence_used: The current 1,269-parameter model reached 100%, while substantially stronger positional compression previously reached 97.63%. A one-scalar positional restriction is the smallest alternative to the additional attention gauge constraints that repeatedly disrupted optimization at 1,268 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2335, "parameters": 1268, "training_steps": 49999}

RECENT RESULT
hypothesis: Replacing the independent 8D token table with a well-conditioned learned rank-7 factorization will retain at least 99% accuracy while reducing the verified model from 1,269 to 1,212 parameters.
change: Factor the tied token embedding and output classifier into learned per-token 7D codes and a learned 7×8 basis, initialized as an orthonormal centered subspace.
mechanism: Learned rank-7 token/unembedding geometry
evidence_used: Prior positional compression and even one-scalar positional pruning failed, indicating the learned absolute-position table is load-bearing. This instead challenges the assumption that all 114 tokens require independent 8D geometry; LayerNorm exposes seven centered contextual degrees, while the learned basis can rotate away from its centered initialization if the final classifier needs another orientation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.22219999999999998, "parameters": 1212, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing final-LayerNorm scale coordinate 5 at its unit initialization will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy.
change: Replace the last of three learned final-LayerNorm scales with a fixed one, preserving the fresh model’s initial function exactly.
mechanism: Sixth fixed final-LayerNorm scale
evidence_used: The verified 1,269-parameter model reached 100% accuracy while already fixing five of eight final-LayerNorm scales and seven of eight biases. This tests the smallest continuation in that tolerant component, avoiding the attention and positional restrictions that failed at 1,268 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1673, "parameters": 1268, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing the second head’s second rotationally redundant coefficient at input coordinate 1 will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy, because it preserves the dense initialization exactly while avoiding two gauge constraints on the same input coordinate.
change: Rotate the second head’s center-0/center-2 plane using input coordinate 1, omit center-2’s coordinate-1 coefficient, and reconstruct it as zero during forward passes.
mechanism: Cross-coordinate query/key rotation gauge
evidence_used: Two first-head rotations using coordinate 0 retained 100% accuracy at 1,269 parameters, but imposing the analogous second rotation on the second head fell to 73.74%. Changing only the second pivot tests whether that failure arose from the conditioning of concentrating both constraints on one input coordinate rather than from removing the exact rotational redundancy itself.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.364, "parameters": 1268, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing a LayerNorm-invisible row-common attention-output coefficient as the sixth final-LayerNorm scale will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy, because the reparameterization preserves the full function class and fresh initialization function.
change: Remove one dedicated final-LayerNorm scale and derive it from the mean of the final attention projection’s first column, whose row-common component is otherwise erased by subsequent LayerNorms.
mechanism: Null-direction parameter reuse
evidence_used: Fixing this final-LayerNorm scale at one reached only 16.73%, showing that it must remain trainable; directly removing a row-common projection degree also failed. Reusing that null direction preserves both degrees of functional freedom while the current model’s 1,269-parameter configuration reached 100%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1268, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing a second LayerNorm-invisible attention-output column mean as the remaining learned final-LayerNorm scale will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy, because it preserves both the full function class and the fresh initialization function.
change: Remove the dedicated final-LayerNorm scale for coordinate 2, derive it from the mean of attention projection column 1, and initialize that column mean to one.
mechanism: Second null-direction parameter reuse
evidence_used: The first null-direction reuse retained 100% accuracy at 1,268 parameters, while fixing a final-LayerNorm scale reached only 16.73%; duplicating the successful reuse with an independent null projection direction preserves the required trainability.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reusing a second LayerNorm-invisible attention-output column mean as the learned final-LayerNorm bias will reduce the verified model from 1,268 to 1,267 parameters while retaining at least 99% accuracy, because it preserves both the full function class and the freshly initialized function.
change: Remove the dedicated final-LayerNorm bias, derive it from attention projection column 1, and initialize that column’s mean to zero.
mechanism: Attention null-direction reuse for final normalization bias
evidence_used: Reusing projection column 0’s null-direction mean as a final-LayerNorm scale retained 100% accuracy at 1,268 parameters. Unlike the timed-out attempt to reuse another scale, this tests the same proven symmetry on the independently initialized zero-valued normalization bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7413, "parameters": 1267, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing the normalized row-common component of attention projection column 1 as the second learned final-LayerNorm scale will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy; unlike the timed-out raw-mean attempt, the orthonormal scaling gives this scale unit gradient norm and preserves the initialized function exactly.
change: Remove the dedicated final-LayerNorm scale for coordinate 2, derive it as the normalized mean of projection column 1, and initialize that null component to produce scale one.
mechanism: Orthonormal attention-null parameter reuse
evidence_used: The first projection-null reuse retained 100% accuracy at 1,268 parameters, establishing that row-common attention-output components can carry required normalization degrees of freedom. The prior second-scale attempt produced no accuracy result because training timed out, so testing the same exact symmetry with an orthonormally conditioned coordinate is the closest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1267, "training_steps": 49999}



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
