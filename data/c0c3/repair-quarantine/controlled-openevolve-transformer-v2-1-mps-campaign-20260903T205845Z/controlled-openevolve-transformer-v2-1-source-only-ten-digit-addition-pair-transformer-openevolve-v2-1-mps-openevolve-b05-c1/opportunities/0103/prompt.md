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
verified_results: {"accuracy": 1.0, "parameters": 1268, "training_steps": 49999}
prior_hypothesis: Reusing a LayerNorm-invisible row-common attention-output coefficient as the sixth final-LayerNorm scale will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy, because the reparameterization preserves the full function class and fresh initialization function.

## Recent verification evidence

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
hypothesis: Reusing a second LayerNorm-invisible attention-output column mean will produce a 1,267-parameter model with at least 99% accuracy; computing both shared scales in one reduction may avoid the prior attempt’s timeout.
change: Remove final-LayerNorm scale coordinate 2, derive it from attention projection column 1, initialize both shared column means to one, and extract them together.
mechanism: Fused dual null-direction parameter reuse
evidence_used: The first null-direction reuse achieved 100% accuracy at 1,268 parameters. The analogous second reuse only timed out, providing no evidence of an accuracy failure, while fixed-scale pruning failed at 16.73%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1595, "parameters": 1267, "training_steps": 49999}

RECENT RESULT
hypothesis: Explicitly separating the second attention-output column into centered coefficients and a scalar mean will yield a 1,267-parameter model with at least 99% accuracy, because the reused final-LayerNorm scale will no longer be contaminated by AdamW updates to the column’s functional centered component.
change: Reparameterize the second attention-output column with seven orthonormal mean-zero coefficients plus one explicit common scalar, reuse that scalar as final-LayerNorm scale coordinate 2, and retain the successful first-column mean reuse unchanged.
mechanism: Optimizer-decoupled null-direction reuse
evidence_used: Implicitly reusing two dense column means produced only 15.95% accuracy, while one dense-column reuse achieved 100%. The explicit chart preserves the same full function class and fresh initialization but isolates the second null scalar in optimizer coordinates, directly testing whether dense-coordinate coupling caused the dual-reuse failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5440999999999999, "parameters": 1267, "training_steps": 49999}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
