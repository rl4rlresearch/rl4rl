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
verified_results: {"accuracy": 0.9982, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,584-parameter design will retain at least 99% accuracy because the added leading `ln1` scale anchor is absorbed by QKV and centering the separated final row of the first key head removes only a softmax-invariant key offset.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Centering the final key row of the first attention head will reduce the verified 1,585-parameter model to 1,584 parameters while retaining at least 99% accuracy, because its removed uniform component produces only a position-independent key offset that causal softmax cancels.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987, "parameters": 1581, "training_steps": 4999}
prior_hypothesis: Reproducing the verified seven-centered-key 1,582-parameter design and centering the third `fc2` output column will yield 1,581 parameters with at least 99% accuracy, because its removed uniform residual component is canceled by the final LayerNorm.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,583-parameter design and centering the remaining third key row of the first attention head will yield 1,582 parameters with at least 99% accuracy, because the removed component contributes only a position-independent key offset canceled by causal softmax.

## Recent verification evidence

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
