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
verified_results: {"accuracy": 0.9943000000000001, "parameters": 1627, "training_steps": 4999}
prior_hypothesis: Omitting one head-1 query-bias coordinate while preserving its critical final value-bias scalar will produce a 1,627-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9981, "parameters": 1620, "training_steps": 4999}
prior_hypothesis: Expressing one key-projection row in a seven-dimensional zero-mean basis scaled by `ln1.weight` will reduce the verified model to 1,620 parameters while retaining at least 99% accuracy, because the removed component produces only a position-independent key shift and cannot change causal attention probabilities.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1618, "training_steps": 4999}
prior_hypothesis: Extending the verified key-row quotient to the second key coordinate of head 0 will reduce the model to 1,618 parameters while retaining at least 99% accuracy, because the removed direction creates only a position-independent key shift canceled by causal attention softmax.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1619, "training_steps": 4999}
prior_hypothesis: Applying the verified seven-dimensional key-row quotient to the first key coordinate of both attention heads will produce a 1,619-parameter model with at least 99% accuracy, because each removed direction changes its key coordinate only by a position-independent shift canceled by causal attention softmax.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-fixing the final `fc2` output-bias coordinate will produce a 1,622-parameter model with at least 99% accuracy because any common shift of the residual stream is removed by the final LayerNorm.
change: Reproduce the qualified 1,623-parameter design, then represent `fc2.bias` relative to its final coordinate and reconstruct that coordinate as zero.
mechanism: Residual-stream common-mode bias quotient
evidence_used: The qualified 1,623-parameter design reached 99.67%; unlike the failed second `ln2`-bias and positional removals, this targets an untouched functionally null common-mode direction immediately upstream of the final LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2613, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the retained final head-1 value bias to the corresponding final attention-projection bias will produce a 1,622-parameter model with at least 99% accuracy while preserving both learned bias pathways.
change: Split the compact QKV bias into query and value parameters, then reuse the value scalar as the final projection-bias coordinate instead of learning a separate scalar.
mechanism: Shared value–projection offset quotient
evidence_used: The qualified 1,623-parameter model achieved 99.67% while retaining the critical final head-1 value bias; prior 1,622 designs fixed sensitive coordinates to zero, whereas this sharing removes a redundant constant offset without deleting either pathway.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Building on the verified 1,622-parameter design, tying the penultimate two `ln2` bias coordinates while retaining both pathways will yield 1,621 parameters and at least 99% accuracy.
change: Reproduce the qualified shared head-0 query bias, shared value/projection offset, and final-zero `ln2` bias, then share the two adjacent retained `ln2` bias coordinates instead of fixing the second one to zero.
mechanism: Shared adjacent LayerNorm-offset quotient
evidence_used: The 1,622-parameter value/projection-sharing design achieved 99.92%; independently, sharing two head-0 query biases succeeded at 99.64% where deleting the second pathway missed at 98.68%, motivating sharing rather than the failed second `ln2`-bias deletion.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6233, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the tied embedding’s globally uniform scalar direction from the verified 1,622-parameter design will yield 1,621 parameters while maintaining at least 99% accuracy, because LayerNorm cancels its input effect and softmax cancels its uniform logit shift.
change: Reproduce the qualified shared-query, compact-ln2, and shared value/projection-bias design, then gauge-fix one globally redundant tied-embedding scalar.
mechanism: Tied-embedding global translation gauge quotient
evidence_used: The 1,622-parameter shared value/projection design achieved 99.92%; the prior 1,621 attempt instead constrained sensitive ln2 offsets and fell to 62.33%, motivating an untouched exact invariance outside attention and MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2576, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,622-parameter design with an orthonormal seven-dimensional parameterization of `fc2` bias will produce 1,621 parameters and maintain at least 99% accuracy, because it removes only the uniform residual offset canceled by the final LayerNorm while preserving initialization and well-conditioned quotient geometry.
change: Reproduce the qualified compact positional, shared-query, compact-`ln2`, and shared value/projection-bias design, then express `fc2.bias` in an orthonormal zero-mean basis.
mechanism: Orthonormal residual-bias common-mode quotient
evidence_used: The shared value/projection design achieved 99.92% at 1,622 parameters. A prior final-coordinate `fc2` gauge fix collapsed to 26.13%, motivating an orthonormal quotient that preserves the exact functional symmetry without the asymmetric coordinate parameterization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9804, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the penultimate `ln2` bias with the final `fc1` bias will produce a 1,621-parameter model with at least 99% accuracy, because it retains both learned affine pathways while `fc1`’s remaining biases preserve full effective preactivation-offset capacity.
change: Split the compact `ln2` bias into six independent coordinates and one shared scalar, then reuse that scalar as the final `fc1` bias instead of learning a separate parameter.
mechanism: Coupled LayerNorm–MLP affine-offset sharing
evidence_used: The 1,622-parameter value/projection-offset sharing design reached 99.92%, while fixing a second `ln2` bias reached only 97.73%; applying the successful upstream/downstream sharing pattern avoids deleting the sensitive LayerNorm pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8547, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final MLP bias’s uniform direction while updating its seven coordinates through reconstructed eight-coordinate AdamW dynamics will produce a 1,621-parameter model with at least 99% accuracy.
change: Reproduce the qualified 1,622-parameter design, express the final `fc2` bias in a seven-dimensional zero-mean basis, and emulate the parent model’s full-coordinate AdamW update for that quotient parameter.
mechanism: Full-coordinate Adam on an orthonormal residual-bias quotient
evidence_used: The 1,622-parameter parent reached 99.92%, while the same orthonormal residual-bias quotient with ordinary basis-coordinate AdamW reached 98.04%; this isolates optimizer-coordinate geometry as the most informative remaining issue.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6724, "parameters": 1621, "training_steps": 4999}

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
