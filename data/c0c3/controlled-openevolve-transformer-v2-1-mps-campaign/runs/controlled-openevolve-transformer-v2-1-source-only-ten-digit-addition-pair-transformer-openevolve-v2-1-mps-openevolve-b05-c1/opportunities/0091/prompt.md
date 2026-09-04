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
verified_results: {"accuracy": 1.0, "parameters": 1269, "training_steps": 49999}
prior_hypothesis: Fixing a second centered query/key coefficient in the first head will reduce the model from 1,270 to 1,269 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the eight row-common degrees of freedom from the attention output matrix will reduce the model from 1,272 to 1,264 parameters while preserving at least 99% accuracy, because the removed component adds only a position-dependent all-ones residual shift that every downstream LayerNorm eliminates.
change: Replace the attention output projection’s dense 8×8 weight with a learned 7×8 matrix padded by a fixed-zero eighth output row, while retaining its unrestricted eight-coordinate bias and all existing bias sharing.
mechanism: Common-mode-free attention output projection
evidence_used: The 1,272-parameter design retained 100% accuracy after exact embedding gauge reductions, and the current MLP already uses this same seven-output-plus-zero-padding gauge successfully. Unlike the failed rotational query/key chart, this is a direct linear common-mode elimination already represented elsewhere in the verified architecture.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7464, "parameters": 1264, "training_steps": 49999}

RECENT RESULT
hypothesis: A full-rank sinusoidal positional basis plus per-head learned causal-distance biases will reach at least 99% accuracy with 1,226 parameters, because attention can learn operand alignments directly instead of reconstructing them from independently learned absolute vectors.
change: Replace the 154-parameter independent positional table with a learned 8×8 projection of fixed anchored sinusoidal codes and add 44 gauge-fixed per-head relative-attention biases.
mechanism: Sinusoidal absolute positions with learned relative-lag attention
evidence_used: The sinusoidal projection reached 97.63%, showing structured absolute positions retain nearly all required information, while a learned residual correction failed at 30.60%. Injecting the missing positional flexibility directly into attention logits tests a different mechanism tailored to pairwise alignment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6472, "parameters": 1226, "training_steps": 49999}

RECENT RESULT
hypothesis: Replacing the failed oblique zero-row attention projection with an orthonormal mean-zero output basis will preserve at least 99% accuracy while reducing the model from 1,272 to 1,264 parameters.
change: Parameterize the attention projection’s input-dependent output in the seven-dimensional mean-zero subspace using an orthonormal Helmert basis, while retaining its unrestricted eight-coordinate bias and existing bias sharing.
mechanism: Orthonormal quotient chart for attention outputs
evidence_used: Removing the same eight common-mode parameters with a zero-padded output row reached only 74.64%, despite those outputs being LayerNorm-invisible. That chart makes one observable direction eight times less responsive; an orthonormal chart removes the same exact redundancy without this conditioning distortion.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4125, "parameters": 1264, "training_steps": 49999}

RECENT RESULT
hypothesis: Removing only one row-common attention-output degree of freedom will reduce the model from 1,272 to 1,271 parameters while retaining at least 99% accuracy, because the discarded component produces only a LayerNorm-invisible all-ones residual shift, and limiting the quotient reparameterization to one column avoids the global optimization disruption seen when all eight columns were changed.
change: Parameterize the first attention-output weight column in a seven-dimensional orthonormal mean-zero basis while leaving the other seven columns dense and retaining the full bias.
mechanism: Single-column orthonormal attention-output quotient
evidence_used: The current 1,272-parameter model reached 100% accuracy, while removing all eight attention-output common modes reached only 74.64% with an oblique chart and 41.25% with an orthonormal chart. A one-column quotient is the smallest test of whether those failures came from simultaneously reparameterizing the entire projection rather than from removing a single redundant direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7415, "parameters": 1271, "training_steps": 49999}

RECENT RESULT
hypothesis: Removing one rotationally redundant query/key parameter will reduce the verified model from 1,272 to 1,271 parameters while retaining at least 99% accuracy, because the chart preserves the dense initialization’s exact attention function and changes only one centered rotation instead of six simultaneously.
change: Replace the tied query/key projection with a one-parameter-smaller chart for the first head, fixing one centered-basis coefficient while leaving the value projection and second head dense.
mechanism: Single-rotation query/key gauge with function-preserving initialization
evidence_used: Removing all six query/key rotational redundancies fell to 55.34%, while the 1,272-parameter baseline reached 100%; a single well-conditioned gauge with initialization obtained by rotating an ordinary dense sample isolates whether the earlier failure came from imposing all six constraints at once.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1271, "training_steps": 49999}

RECENT RESULT
hypothesis: Applying the proven single-rotation query/key gauge independently to the second head will reduce the verified model from 1,271 to 1,270 parameters while retaining at least 99% accuracy.
change: Replace the second head’s dense tied query/key weights with the same one-parameter-smaller, function-preserving centered-rotation chart already used by the first head.
mechanism: One rotational gauge per attention head
evidence_used: One rotation in the first head retained 100% accuracy at 1,271 parameters, whereas imposing all six rotations simultaneously reached only 55.34%; adding one analogous constraint to the untouched head is the smallest supported continuation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1270, "training_steps": 49999}

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
