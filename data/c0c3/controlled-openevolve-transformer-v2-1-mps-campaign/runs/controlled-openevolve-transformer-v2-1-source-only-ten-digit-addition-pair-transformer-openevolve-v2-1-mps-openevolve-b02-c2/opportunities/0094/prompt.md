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
verified_results: {"accuracy": 0.9954999999999999, "parameters": 1598, "training_steps": 4999}
prior_hypothesis: Helmert-parameterizing the penultimate value row will reduce the verified model from 1,599 to 1,598 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component produces only a position-independent attention output that the learned projection-offset subspace can absorb.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1602, "training_steps": 4999}
prior_hypothesis: Helmert-parameterizing a second query row in the second attention head will reduce the verified model from 1,603 to 1,602 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component is absorbable by that row’s independent learned query bias.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.991, "parameters": 1596, "training_steps": 4999}
prior_hypothesis: Fixing a sixth token/position translation coordinate will reduce the verified 1,597-parameter model to 1,596 parameters while retaining at least 99% accuracy, because the input embeddings remain unchanged and the tied output embedding changes logits only by a vocabulary-uniform offset.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1600, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,601-parameter design and Helmert-parameterizing the first first-head query row will yield 1,600 parameters with at least 99% accuracy, because its shared bias and the paired full query row jointly absorb the removed LayerNorm-uniform component.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the qualified 1,605-parameter design with a fourth compact key-weight row will yield 1,604 parameters and at least 99% accuracy, because centering the LayerNorm-scaled key row changes every key by only an attention-softmax-invariant constant.
change: Reproduce the verified two-coordinate `ln1.bias` compaction, then Helmert-parameterize one additional key row while preserving the full reconstructed QKV projection.
mechanism: Fourth LayerNorm-induced key-row gauge quotient
evidence_used: The two-coordinate `ln1.bias` design achieved 99.54% at 1,605 parameters, whereas fixing a third coordinate fell to 96.07%; the successful designs already contain three instances of the independent key-row gauge, motivating extension of that exact attention invariance instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,604-parameter design and compacting a fifth key-weight row will yield 1,603 parameters with at least 99% accuracy, because centering its LayerNorm-scaled weights changes that key coordinate only by a source-position-independent offset canceled by attention softmax.
change: Apply the verified two-coordinate `ln1.bias` compaction and extend `CompactQKV` from three to five Helmert-parameterized key rows.
mechanism: Fifth LayerNorm-induced key-row gauge quotient
evidence_used: Four compact key rows achieved 99.87% accuracy at 1,604 parameters, while removing a third `ln1.bias` coordinate failed at 96.07%; extending the exact key-row invariance is therefore the strongest supported next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7264, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,604-parameter design by Helmert-parameterizing one second-head query row will yield 1,603 parameters and at least 99% accuracy, because that row’s removed LayerNorm-uniform component is a position-independent query offset absorbable by its independent learned query bias.
change: Reproduce the qualified two-coordinate `ln1.bias`, three-coordinate `ln2.bias`, and four-key-row compactions, then compact the first query row of the second attention head.
mechanism: Bias-absorbed LayerNorm query-row gauge quotient
evidence_used: Four compact key rows achieved 99.87% at 1,604 parameters, while a fifth key row failed at 72.64%; the already-qualified biased `fc1` row demonstrates the independent LayerNorm affine-row quotient used here and motivates testing it on a query row instead of extending the failed axes.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Helmert-parameterizing a second query row in the second attention head will reduce the verified model from 1,603 to 1,602 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component is absorbable by that row’s independent learned query bias.
change: Extend `CompactQKV` to reconstruct the first two second-head query rows from independent seven-dimensional zero-mean coordinates while preserving their learned biases.
mechanism: Second bias-absorbed LayerNorm query-row gauge quotient
evidence_used: Compacting the first second-head query row achieved 99.93% accuracy at 1,603 parameters, whereas extending key-row compaction to a fifth row failed at 72.64%; repeating the successful query-row quotient is the strongest supported next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Helmert-parameterizing the third independently biased second-head query row will reduce the verified 1,602-parameter design to 1,601 parameters while retaining at least 99% accuracy.
change: Extend `CompactQKV` to reconstruct all three independently biased second-head query rows from seven-dimensional zero-mean coordinates.
mechanism: Third bias-absorbed LayerNorm query-row gauge quotient
evidence_used: The first and second query-row quotients achieved 99.93% at 1,603 parameters and 99.90% at 1,602 parameters, respectively, while extending key compaction failed; the remaining second-head query row has the same independent learned-bias structure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,601-parameter design and Helmert-parameterizing the first first-head query row will yield 1,600 parameters with at least 99% accuracy, because its shared bias and the paired full query row jointly absorb the removed LayerNorm-uniform component.
change: Compact four key rows, the three qualified independently biased second-head query rows, and one first-head query row while retaining its shared-bias partner at full width.
mechanism: Shared-bias paired-query LayerNorm gauge quotient
evidence_used: Three successive query-row quotients achieved 99.93%, 99.90%, and 99.92% accuracy down to 1,601 parameters; this extends the successful query-weight axis while retaining a full paired row, whereas fifth-key and additional LayerNorm-bias reductions failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,600-parameter design and Helmert-parameterizing the final value row will yield 1,599 parameters with at least 99% accuracy, because the removed LayerNorm-uniform component produces only a position-independent value offset absorbable by the learned projection-offset subspace.
change: Compact four key rows, four qualified query rows, and the final value row; also reproduce the verified two-coordinate `ln1.bias` and three-coordinate `ln2.bias` compactions.
mechanism: Downstream-projection-absorbed value-row LayerNorm gauge
evidence_used: The four-query-row reference achieved 99.93% accuracy at 1,600 parameters. Its qualified biased query and `fc1` row quotients show that LayerNorm-uniform weight components can be removed when learned downstream offsets retain their constant contribution.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Helmert-parameterizing the penultimate value row will reduce the verified model from 1,599 to 1,598 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component produces only a position-independent attention output that the learned projection-offset subspace can absorb.
change: Extend `CompactQKV` from one to two compact value rows, reconstructing both final value rows from independent seven-dimensional zero-mean coordinates.
mechanism: Second downstream-projection-absorbed value-row LayerNorm gauge
evidence_used: Compacting the final value row achieved 99.97% accuracy at 1,599 parameters; applying the same value-row quotient to the adjacent row is the closest supported reduction, while fifth-key and additional LayerNorm-bias reductions failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9954999999999999, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Helmert-parameterizing a third value row will reduce the qualified 1,598-parameter design to 1,597 parameters while retaining at least 99% accuracy.
change: Extend `CompactQKV` to reconstruct the final three value rows from independent seven-dimensional zero-mean coordinates.
mechanism: Third downstream-projection-absorbed value-row LayerNorm gauge
evidence_used: Compacting one value row achieved 99.97% accuracy at 1,599 parameters, and compacting the adjacent second row achieved 99.55% at 1,598; extending the same successful quotient to the next adjacent row is the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9892, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,598-parameter design by Helmert-parameterizing positional row 3 will produce 1,597 parameters with at least 99% accuracy, because a position-local uniform residual shift is removed by every downstream LayerNorm without changing learned attention or logits.
change: Reproduce the qualified four-query/two-value QKV compaction, then center positional rows 1–3 instead of rows 1–2.
mechanism: Residual-uniform positional-row gauge quotient
evidence_used: The two-value-row reference achieved 99.55% at 1,598 parameters, while a third value-row quotient fell to 98.92%; the current verified designs already compact positional rows 1 and 2, motivating extension along that independent exact gauge instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,597-parameter design and Helmert-parameterizing positional row 4 will yield 1,596 parameters with at least 99% accuracy, because its position-local uniform residual shift is removed by downstream LayerNorms.
change: Apply the qualified four-query/two-value QKV compaction and center positional rows 1–4 instead of rows 1–2.
mechanism: Residual-uniform positional-row gauge quotient
evidence_used: Centering positional row 3 produced 99.82% accuracy at 1,597 parameters, while compacting a third value row fell to 98.92%; extending the successful exact positional gauge is the strongest next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9725, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a sixth token/position translation coordinate will reduce the verified 1,597-parameter model to 1,596 parameters while retaining at least 99% accuracy, because the input embeddings remain unchanged and the tied output embedding changes logits only by a vocabulary-uniform offset.
change: Extend `CompactPositionEmbedding` and its initialization from five to six fixed translation coordinates while preserving the centered positional-origin tail and positional rows 1–3.
mechanism: Tied-embedding token/position translation gauge quotient
evidence_used: The current design achieves 99.82% accuracy with five coordinatewise translation gauges at 1,597 parameters, whereas extending positional-row centering to row 4 fell to 97.25%; testing another instance of the already-used exact translation gauge is therefore the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.991, "parameters": 1596, "training_steps": 4999}



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
