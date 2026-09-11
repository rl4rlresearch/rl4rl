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
verified_results: {"accuracy": 0.9979, "parameters": 1579, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,580-parameter nonadjacent `fc2` design and zero-centering positional row 5 will yield 1,579 parameters with at least 99% accuracy, because a position-specific uniform residual shift is removed by every downstream LayerNorm.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.993, "parameters": 1580, "training_steps": 4999}
prior_hypothesis: Reproducing the verified seven-centered-key, three-centered-`fc2` design and additionally centering the final, nonadjacent `fc2` input column will yield 1,580 parameters with at least 99% accuracy, because its uniform output component is removed by the final LayerNorm while avoiding the adjacent-column layout that previously failed.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9982, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,584-parameter design will retain at least 99% accuracy because the added leading `ln1` scale anchor is absorbed by QKV and centering the separated final row of the first key head removes only a softmax-invariant key offset.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9973000000000001, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,579-parameter design and zero-centering positional row 6 will yield 1,578 parameters with at least 99% accuracy, because its position-specific uniform residual shift is canceled by every downstream LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Reproducing the verified 1,584-parameter design and centering the final key row of the second attention head will yield 1,583 parameters with at least 99% accuracy, because the removed component creates only a position-independent key offset that causal softmax cancels.
change: Apply the qualified first-and-last `ln1` scale anchors and fifth centered key row, then center the separated final key row of the second head and reconstruct all QKV rows in their original order.
mechanism: Nonadjacent second-head key-offset gauge quotient
evidence_used: Centering the final key row of the first head produced 99.98% accuracy at 1,584 parameters; applying the same exact softmax-invariant gauge to the corresponding nonadjacent final row of the other head is the closest supported one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,583-parameter design and centering the remaining third key row of the first attention head will yield 1,582 parameters with at least 99% accuracy, because the removed component contributes only a position-independent key offset canceled by causal softmax.
change: Apply the qualified positional-row and nonadjacent `ln1` scale reductions, reproduce the six centered key rows, then move key row `key_start + 2` into the centered basis.
mechanism: Seventh causal key-offset gauge quotient
evidence_used: Six centered key rows achieved 99.92% accuracy at 1,583 parameters, after the analogous fifth and sixth key-row reductions achieved 99.98% and 99.92%; the remaining first-head row has the same softmax-invariant gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the remaining third key row of the second attention head will reduce the verified 1,582-parameter model to 1,581 parameters while retaining at least 99% accuracy, because the removed component produces only a position-independent key offset canceled by causal softmax.
change: Move key row `second_head_start + 2` from the full QKV weight into the centered key basis and reconstruct all eight key rows in their original order.
mechanism: Eighth causal key-offset gauge quotient
evidence_used: Seven centered key rows achieved 99.91% accuracy at 1,582 parameters, and the fifth, sixth, and seventh key-row reductions all remained above 99%; the sole remaining key row has the same softmax-invariant gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8397, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified seven-centered-key 1,582-parameter design and centering the third `fc2` output column will yield 1,581 parameters with at least 99% accuracy, because its removed uniform residual component is canceled by the final LayerNorm.
change: Add the qualified seventh centered key row, then extend `CompactResidualLinear` from two to three centered weight columns.
mechanism: Third MLP residual-output uniform gauge quotient
evidence_used: Reference Design 3 achieved 99.91% accuracy at 1,582 parameters, and its residual projection already removes the identical uniform direction from two columns; the failed eighth-key reduction motivates testing an independent exact gauge family.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,581-parameter design from three to four centered `fc2` output columns will yield 1,580 parameters with at least 99% accuracy, because each removed column-uniform component is canceled by the final LayerNorm.
change: Represent the first four `fc2` weight columns in the existing seven-dimensional zero-mean basis while retaining full parameters from the fifth column onward.
mechanism: Fourth MLP residual-output uniform gauge quotient
evidence_used: Centering the third `fc2` column reduced the seven-centered-key design from 1,582 to 1,581 parameters and achieved 99.87% accuracy; applying the identical gauge to the next column is the closest supported one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9213, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified seven-centered-key, three-centered-`fc2` design and additionally centering the final, nonadjacent `fc2` input column will yield 1,580 parameters with at least 99% accuracy, because its uniform output component is removed by the final LayerNorm while avoiding the adjacent-column layout that previously failed.
change: Center seven qualified QKV key rows, retain only the remaining second-head key row, and represent `fc2` columns 0–2 plus its final column in the zero-mean output basis while reconstructing their original order.
mechanism: Nonadjacent fourth MLP residual-output uniform gauge quotient
evidence_used: The seven-key, three-column design achieved 99.87% accuracy at 1,581 parameters, whereas extending to the adjacent fourth `fc2` column achieved only 92.13%; earlier nonadjacent LayerNorm anchors succeeded where adjacent extensions failed, motivating the same separated-coordinate test for `fc2`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.993, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,580-parameter design and centering the first attention output-projection column will yield 1,579 parameters with at least 99% accuracy, because the removed component adds only a position-dependent uniform residual shift canceled by the following and final LayerNorms.
change: Apply the qualified seven-key and nonadjacent four-column `fc2` compactions, then represent the first attention projection column in a seven-dimensional zero-mean basis.
mechanism: Attention-projection residual-uniform gauge quotient
evidence_used: The seven-key, nonadjacent four-`fc2`-column design achieved 99.30% accuracy at 1,580 parameters; successful `fc2` column reductions establish the same residual-uniform gauge, while the failed eighth-key and adjacent-fourth-`fc2` experiments motivate testing it in the independent attention projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9891, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final attention output-projection column will yield 1,579 parameters with at least 99% accuracy, because its removed uniform residual component is canceled by subsequent LayerNorms, while the trailing-coordinate layout may avoid the optimization weakness observed when centering the first column.
change: Represent the final `attn.proj` input column in a seven-dimensional zero-mean basis and reconstruct it during the forward pass.
mechanism: Trailing attention-projection residual-uniform gauge quotient
evidence_used: The 1,580-parameter design achieved 99.30%; centering the first attention-projection column narrowly missed at 98.91%, while the analogous trailing, nonadjacent `fc2` column succeeded where an adjacent column failed, motivating a trailing-coordinate test of the same exact gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5528, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,580-parameter nonadjacent `fc2` design and zero-centering positional row 5 will yield 1,579 parameters with at least 99% accuracy, because a position-specific uniform residual shift is removed by every downstream LayerNorm.
change: Center positional rows 1–5 instead of 1–4, and apply the qualified nonadjacent `fc2` compaction to columns 0–2 and the final column.
mechanism: Sixth positional-row LayerNorm-null gauge quotient
evidence_used: The nonadjacent four-column `fc2` design achieved 99.30% at 1,580 parameters, while extending positional centering through row 4 achieved 99.84%; this tests the next exact positional gauge instead of the attention-projection gauges that missed at 98.91% and 55.28%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,579-parameter design and zero-centering positional row 6 will yield 1,578 parameters with at least 99% accuracy, because its position-specific uniform residual shift is canceled by every downstream LayerNorm.
change: Center positional rows 1–6 and apply the qualified nonadjacent `fc2` compaction to columns 0–2 and the final column.
mechanism: Seventh positional-row LayerNorm-null gauge quotient
evidence_used: Centering positional row 5 with the nonadjacent four-column `fc2` design achieved 99.79% accuracy at 1,579 parameters; extending the same exact positional gauge to the next row is the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,578-parameter design and zero-centering positional row 7 will yield 1,577 parameters with at least 99% accuracy, because its uniform residual shift is canceled by every downstream LayerNorm.
change: Apply the qualified seven-key and nonadjacent four-column `fc2` compactions, then center positional rows 1–7 while reconstructing all tensors in their original order.
mechanism: Eighth positional-row LayerNorm-null gauge quotient
evidence_used: Centering positional row 6 produced 99.73% accuracy at 1,578 parameters after row 5 produced 99.79%; extending the same exact positional gauge to the next row is the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9662999999999999, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final positional row instead of the failed adjacent row 7 will reduce the verified 1,578-parameter model to 1,577 parameters while retaining at least 99% accuracy, because its position-specific uniform shift is canceled by downstream LayerNorms.
change: Keep positional rows 1–6 centered, additionally center the final positional row, and reconstruct all intermediate rows in their original order.
mechanism: Nonadjacent trailing positional-row LayerNorm-null gauge quotient
evidence_used: The current six-row-centered design achieved 99.73% accuracy at 1,578 parameters, while adjacent row 7 failed at 96.63%; the analogous nonadjacent final-column `fc2` reduction succeeded at 99.30% after its adjacent alternative failed at 92.13%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9620000000000001, "parameters": 1577, "training_steps": 4999}



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
