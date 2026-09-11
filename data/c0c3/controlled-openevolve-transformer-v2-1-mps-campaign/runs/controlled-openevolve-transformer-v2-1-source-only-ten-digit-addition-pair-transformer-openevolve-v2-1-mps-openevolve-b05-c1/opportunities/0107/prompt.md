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
verified_results: {"accuracy": 1.0, "parameters": 1266, "training_steps": 49999}
prior_hypothesis: Replacing the dedicated final-LayerNorm bias with the normalized row-common component of attention projection column 2 will reduce the model from 1,267 to 1,266 parameters while retaining at least 99% accuracy, because it preserves the initialized function and gives the reused bias unit gradient norm.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Replacing the dedicated final-LayerNorm bias with the normalized row-common component of attention projection column 2 will reduce the model from 1,267 to 1,266 parameters while retaining at least 99% accuracy, because it preserves the initialized function and gives the reused bias unit gradient norm.
change: Remove the final-LayerNorm bias parameter, derive it as `sqrt(d_model)` times projection column 2’s mean, and initialize that null component to zero.
mechanism: Orthonormal attention-null reuse for final normalization bias
evidence_used: Raw-mean reuse for this bias reached only 74.13%, while changing the analogous second-scale reuse from a raw mean to an orthonormal mean produced 100% accuracy at 1,267 parameters; this directly tests whether conditioning caused the bias-reuse failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1266, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing projection column 3’s normalized row-common component as the remaining learned final-LayerNorm scale will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy.
change: Remove the final LayerNorm’s dedicated scale parameter, derive coordinate 0’s scale from the normalized mean of attention projection column 3, and initialize that null component to produce scale one.
mechanism: Third orthonormal attention-null reuse for final normalization scale
evidence_used: Orthonormal null-direction reuse retained 100% accuracy for both a final-LayerNorm scale at 1,267 parameters and its bias at 1,266, directly supporting the same function-preserving reuse for the sole remaining dedicated scale.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reusing projection column 3’s normalized row-common component as the remaining final-LayerNorm scale will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy; batching the four null-component reductions into one operation may avoid the prior attempt’s verification timeout.
change: Remove the dedicated final-LayerNorm scale, initialize projection column 3’s null component to represent scale one, and derive all shared final-normalization values from one batched column-mean reduction.
mechanism: Batched orthonormal attention-null reuse
evidence_used: Orthonormal null reuse achieved 100% accuracy for a final-LayerNorm scale at 1,267 parameters and its bias at 1,266. The identical third reuse previously timed out without producing contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing independent absolute-position embeddings with per-head learned relative-distance attention biases will retain at least 99% accuracy while reducing the model from 1,266 to 1,156 parameters, because aligned addition columns and preceding outputs can be selected by relative distance without encoding every absolute position in the residual stream.
change: Remove the 154-parameter absolute-position table and add a 44-parameter per-head causal relative-bias table, with one softmax-redundant bias fixed per head.
mechanism: Learned relative-distance attention routing
evidence_used: Strong positional compression previously reached 97.63%, showing that the full absolute table is not entirely necessary, while single-scalar pruning was optimization-destructive. This tests a different learned computation: positional routing occurs directly in attention rather than through compressed absolute embeddings.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reusing projection column 3’s normalized row-common component as the remaining final-LayerNorm scale will produce a 1,265-parameter model with at least 99% accuracy; eliminating zero-dropout calls and duplicate token-weight reconstruction will allow the previously timed-out exact reparameterization to finish verification.
change: Remove the dedicated final-LayerNorm scale, derive it from projection column 3, batch the normalized column reductions, and streamline function-preserving operations for the fixed zero-dropout configuration.
mechanism: Runtime-streamlined third orthonormal attention-null reuse
evidence_used: Orthonormal attention-null reuse achieved 100% accuracy at both 1,267 and 1,266 parameters. The same third reuse timed out twice without contrary accuracy evidence, motivating preservation of that exact symmetry while reducing hot-path overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing the remaining rotationally redundant coefficient in the first attention head will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy.
change: Rotate the first head’s center-1/center-2 plane at input coordinate 1, omit center-2’s coordinate-1 coefficient, and reconstruct it as zero during forward passes.
mechanism: Complete first-head centered rotation gauge
evidence_used: Two rotation constraints in the first head retained 100% accuracy, whereas further constraints in the second head failed; completing the exact centered-subspace gauge in the already-tolerant first head tests a distinct 1,265-parameter reduction without the runtime overhead that caused repeated third null-reuse timeouts.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Quotienting the remaining first-head centered rotation and reallocating its redundant parameter slot to the final LayerNorm scale will produce a 1,265-parameter model with at least 99% accuracy; reusing the projection-bias mean, skipping zero-dropout calls, caching token weights, and compacting ignored loss rows will let verification finish within the time limit.
change: Fix the first head’s center-1/center-2 rotational gauge at input coordinate 1, replace the dedicated final-LayerNorm scale with a scalar stored in the freed QKV parameter budget, and reduce function-preserving training overhead.
mechanism: Rotational-gauge parameter transfer with hot-path streamlining
evidence_used: The 1,266-parameter model reached 100% accuracy, and the identical first-head gauge reduction previously timed out without contrary accuracy evidence. Successful orthonormal null reuse at 1,267 and 1,266 shows that transferring a redundant degree of freedom to a required final-normalization parameter can preserve accuracy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Encoding the remaining final-LayerNorm scale as one plus a value/output rotation-gauge coefficient will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy, without adding a costly reduction to the training hot path.
change: Orthogonally rotate the final two value channels during initialization so one coefficient is zero, inverse-rotate the corresponding output-projection columns to preserve the initialized function, remove the dedicated final-LayerNorm scale, and derive that scale from the zero-centered gauge coefficient.
mechanism: Orthogonal value/output gauge reuse for final normalization scale
evidence_used: Orthonormally conditioned null-direction reuse retained 100% accuracy at both 1,267 and 1,266 parameters, whereas reductions adding another column-mean operation or coefficient reconstruction repeatedly timed out. This uses the same unit-gradient, initialization-preserving principle through an independent exact value/output basis symmetry.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Fixing the remaining exact centered-subspace rotation in the first attention head will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy; eliminating zero-dropout dispatch, duplicate reductions, and redundant token-weight reconstruction will help the previously timed-out gauge reduction finish verification.
change: Rotate the first head’s center-1/center-2 plane so the center-2 coefficient at input coordinate 1 is zero, omit that coefficient, and streamline mathematically equivalent zero-dropout, shared-bias, normalization-statistic, token-weight, and masked-loss operations.
mechanism: Complete first-head centered rotation gauge with hot-path streamlining
evidence_used: The 1,266-parameter design reached 100%, and two existing first-head rotation constraints also retained 100%; the same final exact gauge constraint previously timed out without contrary accuracy evidence, while failures from constraining the second head make completing the already-tolerant first-head gauge the better-supported reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing one redundant value/output rotation coefficient will produce a 1,265-parameter model with at least 99% accuracy, while cached tied weights and zero-dropout identities offset reconstruction overhead sufficiently for verification to finish.
change: Canonicalize one first-head value coefficient to zero, inverse-rotate the matching output-projection columns at initialization, store only the remaining value coefficients, and streamline zero-dropout and tied-weight use.
mechanism: Exact value/output rotation-gauge quotient
evidence_used: The 1,266-parameter design achieved 100% accuracy. The prior value/output-gauge reuse was unverifiable because it coupled the gauge coefficient to the final LayerNorm scale; directly quotienting the same exact symmetry while retaining that scale isolates the parameter reduction, and avoids the repeatedly timed-out extra column-mean reduction.
result: training did not finish within the verification time limit



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
