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
verified_results: {"accuracy": 0.9992, "parameters": 1583, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,584-parameter design and centering the final key row of the second attention head will yield 1,583 parameters with at least 99% accuracy, because the removed component creates only a position-independent key offset that causal softmax cancels.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Centering the final key row of the first attention head will reduce the verified 1,585-parameter model to 1,584 parameters while retaining at least 99% accuracy, because its removed uniform component produces only a position-independent key offset that causal softmax cancels.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9982, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,584-parameter design will retain at least 99% accuracy because the added leading `ln1` scale anchor is absorbed by QKV and centering the separated final row of the first key head removes only a softmax-invariant key offset.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,583-parameter design and centering the remaining third key row of the first attention head will yield 1,582 parameters with at least 99% accuracy, because the removed component contributes only a position-independent key offset canceled by causal softmax.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinates 0, 1, and 7 will reduce the qualified 1,588-parameter design to 1,587 parameters while retaining at least 99% accuracy, because `fc1` absorbs the scales and coordinate 1 retains an independent trainable LayerNorm bias.
change: Reconstruct three `ln2` scale coordinates as constant one and use the reconstructed full scale in the compact first MLP row.
mechanism: Third downstream LayerNorm scale gauge with trainable-bias leading anchor
evidence_used: The nonadjacent coordinate-0-and-7 anchors achieved 99.99% accuracy at 1,588 parameters, whereas anchoring the two trailing, bias-fixed coordinates achieved only 98.79%; extending the leading side tests a third anchor whose bias remains trainable.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4777, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified first-and-last `ln2` scale anchors will reduce the current model from 1,590 to 1,588 parameters while retaining at least 99% accuracy.
change: Fix the first and last `ln2` scale coordinates at one, reconstruct the full scale in LayerNorm and the compact first MLP row, and preserve all other qualified compactions and training settings.
mechanism: Nonadjacent downstream LayerNorm scale gauge quotient
evidence_used: The identical nonadjacent two-anchor design achieved 99.99% accuracy with 1,588 parameters; adjacent trailing anchors reached only 98.79%, and adding a third anchor collapsed to 47.77%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9915999999999999, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,588-parameter first-and-last `ln2` scale design and zero-centering positional row 4 will yield 1,587 parameters with at least 99% accuracy, because a position-specific uniform residual shift is invisible to every pre-LayerNorm branch and the final LayerNorm.
change: Apply the qualified six-coordinate `ln2` bias reduction and nonadjacent scale anchors, then extend the existing zero-mean positional-row parameterization from rows 1–3 through row 4.
mechanism: Fourth positional-row residual-uniform gauge quotient
evidence_used: The nonadjacent `ln2` scale design achieved 99.99% and 99.16% accuracy at 1,588 parameters, while a third scale anchor failed; the qualified model already removes the same uniform direction from positional rows 1–3, so extending that independent exact gauge to row 4 is the most directly supported new reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending zero-centering from positional rows 1–4 through row 5 will reduce the verified 1,587-parameter model to 1,586 parameters while retaining at least 99% accuracy, because a position-specific uniform residual shift is removed by every pre-LayerNorm branch and the final LayerNorm.
change: Represent positional row 5 in the existing seven-dimensional zero-mean basis and retain full parameters only from row 6 onward.
mechanism: Fifth positional-row residual-uniform gauge quotient
evidence_used: Zero-centering positional row 4 reduced the qualified 1,588-parameter design to 1,587 parameters and achieved 99.84% accuracy; row 5 has the same residual-uniform gauge structure, making it the most direct next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7190000000000001, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Zero-centering positional rows 1–4 and the final positional row will yield 1,586 parameters with at least 99% accuracy, because it preserves the verified row-4 reduction while testing whether the row-5 failure was position-specific.
change: Reproduce the qualified row-4 compaction and represent the nonadjacent final positional row in the same seven-dimensional zero-mean basis.
mechanism: Nonadjacent final-position residual-uniform gauge
evidence_used: Centering rows 1–4 achieved 99.84% accuracy at 1,587 parameters, while extending contiguously through row 5 collapsed to 71.90%; the successful nonadjacent LayerNorm anchors show that avoiding an adjacent constrained coordinate can materially improve optimization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2826, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,587-parameter positional-row design with one fixed trailing `ln1` scale will yield 1,586 parameters and at least 99% accuracy, because all downstream QKV columns can absorb that scale.
change: Zero-center positional rows 1–4, fix the final `ln1` scale coordinate at one, and reconstruct its full scale consistently inside LayerNorm and compact QKV.
mechanism: Attention-input LayerNorm scale gauge quotient
evidence_used: Centering positional row 4 achieved 99.84% accuracy at 1,587 parameters, while the analogous single trailing `ln2` scale anchor achieved 99.96%; this tests the same downstream-linear redundancy on the independently parameterized attention branch.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,586-parameter design and fixing both the first and last `ln1` scale coordinates will yield 1,585 parameters with at least 99% accuracy, because QKV absorbs both scales and the nonadjacent layout preserves a trainable bias on the leading coordinate.
change: Zero-center positional rows 1–4, reconstruct first-and-last anchored scales for both LayerNorm branches, and propagate each reconstructed scale through compact QKV or `fc1`.
mechanism: Nonadjacent attention-input LayerNorm scale gauge quotient
evidence_used: The single trailing `ln1` anchor achieved 99.92% accuracy at 1,586 parameters; analogously, nonadjacent first-and-last `ln2` anchors achieved 99.99%, while adjacent trailing anchors achieved only 98.79%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final key row of the first attention head will reduce the verified 1,585-parameter model to 1,584 parameters while retaining at least 99% accuracy, because its removed uniform component produces only a position-independent key offset that causal softmax cancels.
change: Move key row `second_head_start - 1` from the full QKV weight into the seven-dimensional centered key basis and reconstruct it in its original order.
mechanism: Nonadjacent fifth key-row LayerNorm-null gauge
evidence_used: The current model achieved 99.96% accuracy with four centered key rows. Successful nonadjacent LayerNorm anchors outperformed adjacent extensions, motivating a separated final-row anchor instead of centering the adjacent third key row.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,584-parameter design will retain at least 99% accuracy because the added leading `ln1` scale anchor is absorbed by QKV and centering the separated final row of the first key head removes only a softmax-invariant key offset.
change: Fix the first and last `ln1` scale coordinates and move key row `second_head_start - 1` into the centered key basis, reducing the model from 1,586 to 1,584 learned parameters.
mechanism: Nonadjacent attention-scale anchor plus causal key-offset gauge quotient
evidence_used: Reference Design 1 used this exact combination and achieved 99.98% accuracy with 1,584 parameters; the current design already contains all of its other qualified compactions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1584, "training_steps": 4999}

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
