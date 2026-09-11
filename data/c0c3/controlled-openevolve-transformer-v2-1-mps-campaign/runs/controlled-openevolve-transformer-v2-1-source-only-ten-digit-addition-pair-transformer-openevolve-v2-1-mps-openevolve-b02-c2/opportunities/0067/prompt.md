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
verified_results: {"accuracy": 0.9991, "parameters": 1619, "training_steps": 4999}
prior_hypothesis: Applying the verified seven-dimensional key-row quotient to the first key coordinate of both attention heads will produce a 1,619-parameter model with at least 99% accuracy, because each removed direction changes its key coordinate only by a position-independent shift canceled by causal attention softmax.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1616, "training_steps": 4999}
prior_hypothesis: Fixing a second positional-embedding origin coordinate will reduce the verified 1,617-parameter model to 1,616 parameters while retaining at least 99% accuracy, because an equal coordinate-wise translation can be absorbed exactly into every token embedding without changing initialized hidden inputs.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9943000000000001, "parameters": 1627, "training_steps": 4999}
prior_hypothesis: Omitting one head-1 query-bias coordinate while preserving its critical final value-bias scalar will produce a 1,627-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1615, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,616-parameter design to fix positional-embedding origin coordinate 2 will yield 1,615 parameters and at least 99% accuracy, because the same coordinate-wise translation leaves initialized hidden inputs unchanged and shifts every output logit equally.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Gauge-fixing the uniform component of the verified 1,622-parameter design’s effective attention offset in a near-orthonormal basis will yield 1,621 parameters and at least 99% accuracy while retaining the learned value-bias pathway.
change: Reproduce the qualified positional, QKV, and `ln2` compactions, then replace seven independent projection-bias coordinates with six orthonormal coordinates while reusing the retained value-bias scalar and centering their combined effective offset.
mechanism: Orthonormal joint value–projection common-mode quotient
evidence_used: The shared value/projection design achieved 99.92% at 1,622 parameters. The 98.04% orthonormal `fc2` quotient showed that symmetric quotient geometry was substantially better than the 26.13% final-coordinate gauge; this patch applies that evidence to the already-shared value/projection offset and preserves its critical learned scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9956999999999999, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Expressing one key-projection row in a seven-dimensional zero-mean basis scaled by `ln1.weight` will reduce the verified model to 1,620 parameters while retaining at least 99% accuracy, because the removed component produces only a position-independent key shift and cannot change causal attention probabilities.
change: Replace one eight-scalar QKV key-weight row with seven Helmert-basis coordinates, reconstructing it through the learned `ln1` scales while preserving all other QKV weights and compact biases.
mechanism: Orthonormal LayerNorm–key-weight gauge quotient
evidence_used: The current orthonormal attention-offset quotient reached 99.57% at 1,621 parameters, while every qualified reference already removes all key biases without harming accuracy; together these results motivate another well-conditioned attention-invariant quotient targeting constant key offsets rather than a sensitive learned pathway.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified seven-dimensional key-row quotient to the first key coordinate of both attention heads will produce a 1,619-parameter model with at least 99% accuracy, because each removed direction changes its key coordinate only by a position-independent shift canceled by causal attention softmax.
change: Reproduce the qualified zero-mean shared value/projection offset, then reconstruct two QKV key rows from independent seven-dimensional Helmert coordinates scaled by `ln1.weight`.
mechanism: Second orthonormal LayerNorm–key-weight gauge quotient
evidence_used: The same quotient on one key row achieved 99.81% at 1,620 parameters; extending that successful, well-conditioned exact invariance to the corresponding row of the other head is the smallest informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified key-row quotient to the second key coordinate of head 0 will reduce the model to 1,618 parameters while retaining at least 99% accuracy, because the removed direction creates only a position-independent key shift canceled by causal attention softmax.
change: Reproduce the qualified compact bias, projection-offset, and LayerNorm design, while reconstructing three QKV key rows from independent seven-dimensional Helmert coordinates scaled by `ln1.weight`.
mechanism: Third orthonormal LayerNorm–key-weight gauge quotient
evidence_used: The identical quotient applied to one key row achieved 99.81% at 1,620 parameters and to corresponding rows in both heads achieved 99.91% at 1,619 parameters, directly motivating one additional independently invariant key-row reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-row key quotient to the second key coordinate of head 1 will yield 1,617 parameters and at least 99% accuracy because the removed direction contributes only a position-independent key shift canceled by causal attention softmax.
change: Reproduce the qualified 1,618-parameter design and reconstruct four QKV key rows from independent seven-dimensional Helmert coordinates scaled by `ln1.weight`.
mechanism: Fourth orthonormal LayerNorm–key-weight gauge quotient
evidence_used: The identical quotient succeeded for three key rows at 1,618 parameters with 99.96% accuracy; applying it to the symmetric second coordinate of the other head is the smallest directly supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9196, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Expressing one attention output-projection column in a seven-dimensional zero-mean basis will reduce the model to 1,617 parameters while retaining at least 99% accuracy, because the removed component produces only a token-dependent uniform residual shift canceled by `ln2` and `ln_f`.
change: Reparameterize the first attention output-projection column with a Helmert basis while preserving the existing shared value/projection offset quotient.
mechanism: Orthonormal attention-projection residual gauge quotient
evidence_used: The current 1,618-parameter model reached 99.96%, and the orthonormal joint value/projection common-mode quotient reached 99.57%; the failed fourth LayerNorm-coupled key quotient motivates testing the corresponding fixed-basis residual invariance instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8036, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the third key coordinate of head 0, instead of the failed second coordinate of head 1, will produce a 1,617-parameter model with at least 99% accuracy because it removes the same attention-softmax-invariant direction while avoiding the empirically sensitive head-1 coordinate.
change: Reconstruct four key rows from independent seven-dimensional Helmert coordinates: the first three rows of head 0 and the first row of head 1.
mechanism: Alternative fourth LayerNorm–key-weight gauge quotient
evidence_used: Three key-row quotients achieved 99.96% at 1,618 parameters, whereas adding head 1’s second key row fell to 91.96%; testing head 0’s unused third key row isolates whether the failure was coordinate-specific rather than a general four-quotient limit.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9825, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1,618-parameter design, expressing one `fc1` weight row in a seven-dimensional zero-mean basis scaled by `ln2.weight` will yield 1,617 parameters and at least 99% accuracy, because the removed direction affects only a constant preactivation offset that the row’s independent learned bias preserves.
change: Reproduce the qualified three-key-row quotient, then quotient the first MLP input-weight row while retaining its full learned bias.
mechanism: Orthonormal LayerNorm–MLP row gauge quotient
evidence_used: Three key-row quotients achieved 99.96% at 1,618 parameters, while alternative fourth key quotients reached only 91.96% and 98.25%; this motivates applying the successful orthonormal LayerNorm gauge to an MLP row with an independent bias instead of another sensitive key coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified `fc1` row quotient to a second independently biased row will produce a 1,616-parameter model with at least 99% accuracy, because each removed direction contributes only a constant preactivation offset representable by that row’s learned bias.
change: Reproduce the qualified 1,617-parameter design, then reconstruct the first two `fc1` weight rows from independent seven-dimensional Helmert coordinates scaled by `ln2.weight`.
mechanism: Second orthonormal LayerNorm–MLP row gauge quotient
evidence_used: The first LayerNorm–MLP row quotient achieved 99.84% accuracy at 1,617 parameters; repeating the same exact gauge on another row retains an independent bias and avoids the empirically sensitive fourth key-row quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9838, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second positional-embedding origin coordinate will reduce the verified 1,617-parameter model to 1,616 parameters while retaining at least 99% accuracy, because an equal coordinate-wise translation can be absorbed exactly into every token embedding without changing initialized hidden inputs.
change: Absorb `pos_emb[0, 1]` into token-embedding coordinate 1, then omit the first two flattened positional parameters and reconstruct them as zeros.
mechanism: Second token–position embedding translation gauge
evidence_used: The current model achieves 99.84% at 1,617 parameters while already using the identical gauge for coordinate 0; unlike the failed second MLP-row quotient at 98.38%, this extends a qualified embedding reparameterization without constraining a learned computational pathway.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,616-parameter design to fix positional-embedding origin coordinate 2 will yield 1,615 parameters and at least 99% accuracy, because the same coordinate-wise translation leaves initialized hidden inputs unchanged and shifts every output logit equally.
change: Reproduce the qualified first MLP-row quotient and two positional gauges, then absorb `pos_emb[0, 2]` into token-embedding coordinate 2 and omit the first three flattened positional parameters.
mechanism: Third token–position embedding translation gauge
evidence_used: The first LayerNorm–MLP row quotient achieved 99.84% at 1,617 parameters, and adding the second exact token–position translation gauge produced 99.93% at 1,616; extending that qualified embedding symmetry by one coordinate is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1615, "training_steps": 4999}



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
