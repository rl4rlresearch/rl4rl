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
verified_results: {"accuracy": 1.0, "parameters": 1579, "training_steps": 4999}
prior_hypothesis: Orthogonally isolating the decay-free token-row-mean output-bias coordinates before removing a seventh final-LayerNorm bias direction will produce a 1,579-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the full value bias while quotienting two final-LayerNorm bias degrees of freedom into otherwise hidden token-row means will produce 1,584 parameters and retain at least 99% accuracy.
change: Restore all eight value-bias coordinates, initialize each token embedding row mean-free, and replace the final LayerNorm bias with six learned mean-free coordinates plus a fixed common component that turns token-row means into a direct output-bias channel.
mechanism: Final-LayerNorm bias absorption through token-row means
evidence_used: The full-value-bias tied-embedding design achieved 99.97% at 1,585 parameters, whereas pruning one value-bias coordinate fell to 97.31%; this keeps the load-bearing value bias and instead removes an exact output-side redundancy while preserving the initial function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9962000000000001, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying one attention output-bias coordinate to a learned value-bias coordinate will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because value bias and projection bias contribute only through their combined position-independent projected offset.
change: Keep all eight learned value-bias coordinates, but reuse the final value-bias scalar as the final mean-free attention projection-bias coordinate instead of learning that coordinate independently.
mechanism: Redundant value/output-bias coordinate tying
evidence_used: Removing even one value-bias coordinate reduced accuracy to 97.31%, while the full-value-bias design reached 99.62% at 1,584 parameters; tying a redundant projection-bias coordinate preserves every value channel and its direct learned gradient path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6277, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one common-direction coordinate from a key-projection row will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because LayerNorm initially produces mean-free features and the removed component changes every key by the same position-independent scalar, which causal-attention softmax cancels.
change: Represent one key-projection row in a seven-dimensional orthonormal mean-free basis, reconstruct the full QKV matrix during attention, and preserve the original full-width initialization draw.
mechanism: LayerNorm-induced key-weight gauge quotient
evidence_used: Eliminating all eight softmax-null key-bias coordinates retained 99.94% accuracy, while tying value and output biases failed at 62.77%; this motivates extending the proven key-side softmax invariance instead of coupling optimization-sensitive value pathways.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4104, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln2` bias coordinate at zero will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because `fc1`’s learned bias exactly absorbs any constant contribution from that LayerNorm coordinate.
change: Add a three-pruned-bias LayerNorm and use it only before the MLP, preserving initialization, attention capacity, and all multiplicative scale parameters.
mechanism: Incremental downstream bias absorption
evidence_used: The 1,584-parameter design achieved 99.62%, while fixing an `ln2` scale coordinate reached only 97.98%; this tests the less disruptive additive redundancy already used successfully for two `ln2` bias coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4061, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one additional final-LayerNorm bias direction will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because token-row means provide a learned output-bias channel while remaining invisible to the input-side LayerNorm.
change: Reduce the learned final-LayerNorm bias basis from six to five coordinates, preserving all attention, MLP, value-bias, and training settings.
mechanism: Incremental final-output bias absorption
evidence_used: The existing two-direction final-LayerNorm quotient achieved 99.62% at 1,584 parameters. Unlike the failed third `ln2` bias removal, this extends the demonstrated output-side absorption mechanism where tied-token row means directly replace omitted logit offsets.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fourth final-LayerNorm bias direction will reduce the model from 1,583 to 1,582 parameters while retaining at least 99% accuracy, because the same token-row-mean output-bias channel supported 99.91% accuracy after the third direction was removed.
change: Reduce the learned final-LayerNorm bias basis from five to four coordinates while preserving all attention, MLP, value-bias, initialization, and training settings.
mechanism: Incremental final-output bias absorption
evidence_used: The immediately preceding three-direction final-LayerNorm quotient achieved 99.91% accuracy with 1,583 parameters, improving on the two-direction quotient’s 99.62%; this makes a one-coordinate extension of that demonstrated mechanism the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9828, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Scaling the fixed common final-LayerNorm bias while removing a fourth learned bias direction will retain at least 99% accuracy with 1,582 parameters, because stronger common activation lets smaller token-row means absorb omitted output biases with less context-dependent interference.
change: Reduce the final-LayerNorm bias basis from five to four learned coordinates and increase its fixed common component from a unit-norm vector to an all-ones vector.
mechanism: Amplified token-row-mean output-bias channel
evidence_used: The three-direction quotient achieved 99.91% at 1,583 parameters, while the otherwise identical four-direction quotient narrowly missed at 98.28%; this tests whether optimization through the demonstrated token-row-mean channel, rather than insufficient attention or MLP capacity, caused the gap.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5445, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fourth final-LayerNorm bias direction while exempting the tied token embedding from weight decay will achieve at least 99% accuracy with 1,582 parameters, because the token-row means that absorb omitted output biases are input-side invisible and should not be penalized toward zero.
change: Reduce the learned final-LayerNorm bias basis from five to four coordinates and use a zero-weight-decay optimizer group for the tied token embedding while retaining configured weight decay elsewhere.
mechanism: Weight-decay-free output-bias channel
evidence_used: The four-direction quotient narrowly missed at 98.28%, whereas the three-direction quotient reached 99.91%; amplifying the fixed bias failed, motivating an optimizer-level change that directly supports the demonstrated token-row-mean absorption channel without changing its activation scale.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing each head’s exact GL(4) query/key basis redundancy will reduce the model from 1,582 to 1,550 parameters while retaining at least 99% accuracy, because it preserves the full learned attention-score family and exactly preserves conventional Q/K initialization logits.
change: Replace unconstrained query and key projections with a gauge-fixed factorization whose query anchor submatrix is fixed and whose learned key factor absorbs the inverse basis transformation; retain independent values, query biases, heads, and all other successful settings.
mechanism: Gauge-fixed headwise Q/K factorization
evidence_used: The 1,582-parameter model reaches 99.95% with full Q/K factors, while deleting one key direction fell to 41.04%; this instead removes only 32 functionally redundant factor-basis coordinates without deleting an addressing direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7898000000000001, "parameters": 1550, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fifth final-LayerNorm bias direction will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because exempting the tied token embedding from weight decay raised the four-direction quotient from 98.28% to 99.95%, demonstrating that its token-row means can reliably learn the replacement output-bias channel.
change: Reduce the final-LayerNorm bias basis from four to three learned coordinates while preserving the successful zero-weight-decay embedding group and all attention, MLP, initialization, and training settings.
mechanism: Incremental decay-free final-output bias absorption
evidence_used: The immediately preceding four-direction quotient achieved 99.95% at 1,582 parameters only after token-embedding weight decay was removed; this strongly motivates testing one additional coordinate through the same demonstrated absorption mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a sixth final-LayerNorm bias direction will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because the zero-weight-decay token-row-mean channel achieved 99.96% accuracy after absorbing five bias directions.
change: Reduce the final-LayerNorm learned bias basis from three to two coordinates while preserving the successful embedding optimizer group and all attention, MLP, initialization, and decoding behavior.
mechanism: Incremental decay-free final-output bias absorption
evidence_used: The immediately preceding five-direction quotient achieved 99.96% accuracy at 1,581 parameters, improving slightly on the four-direction quotient’s 99.95%; this makes one additional coordinate removal the most direct test of the demonstrated absorption mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9913, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Orthogonally isolating the decay-free token-row-mean output-bias coordinates before removing a seventh final-LayerNorm bias direction will produce a 1,579-parameter model with at least 99% accuracy.
change: Reparameterize the unchanged globally mean-free tied embedding into row-centered content and explicit mean-free row offsets, keep both components free of weight decay, and reduce the final-LayerNorm learned bias basis from two coordinates to one.
mechanism: Optimizer-aligned token-row-mean quotient
evidence_used: Removing embedding weight decay raised the four-direction quotient from 98.28% to 99.95%, and the current six-direction quotient still reaches 99.13%; this indicates that optimization of the token-row-mean replacement channel is decisive and motivates aligning that channel with independent Adam coordinates before pruning another bias direction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1579, "training_steps": 4999}



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
