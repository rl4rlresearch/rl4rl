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
verified_results: {"accuracy": 0.9991, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,583-parameter design and centering the remaining third key row of the first attention head will yield 1,582 parameters with at least 99% accuracy, because the removed component contributes only a position-independent key offset canceled by causal softmax.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1581, "training_steps": 4999}
prior_hypothesis: Centering seven key rows and three `fc2` output columns will reduce the model from 1,584 to 1,581 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9982, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,584-parameter design will retain at least 99% accuracy because the added leading `ln1` scale anchor is absorbed by QKV and centering the separated final row of the first key head removes only a softmax-invariant key offset.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9974, "parameters": 1580, "training_steps": 4999}
prior_hypothesis: Centering a fourth `fc2` output column will reduce the model from 1,581 to 1,580 learned parameters while retaining at least 99% accuracy; three centered columns achieved 0.9999, while the failure only appeared when all twelve were centered simultaneously.

## Recent verification evidence

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
hypothesis: Centering the third `fc2` output column will reduce the model from 1,582 to 1,581 learned parameters while retaining at least 99% accuracy.
change: Extend `CompactResidualLinear` from two to three centered weight columns.
mechanism: Final-LayerNorm residual-column gauge fixing
evidence_used: Reference Design 3 verified this exact 1,581-parameter construction at 0.9987 accuracy after 4,999 training steps.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Centering seven key rows and three `fc2` output columns will reduce the model from 1,584 to 1,581 parameters while retaining at least 99% accuracy.
change: Reproduce the verified 1,581-parameter construction by adding two centered key rows and one centered residual-output column.
mechanism: Causal-softmax key-offset and final-LayerNorm residual gauges
evidence_used: Reference Design 2 achieved 0.9987 accuracy with 1,581 parameters after 4,999 training steps using this exact construction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Adopting the verified seven-key compaction and centering every `fc2` output column will reduce the model from 1,584 to 1,572 learned parameters while retaining at least 99% accuracy, because each removed column component produces only a tokenwise uniform residual shift erased by the final LayerNorm.
change: Center seven QKV key rows as verified, then represent all twelve `fc2` columns in the zero-mean output basis instead of only two.
mechanism: Exhaustive final-LayerNorm residual-column gauge fixing
evidence_used: Reference Design 1 achieved 0.9999 accuracy with 1,581 parameters after 4,999 steps while centering seven key rows and three `fc2` columns; the same final-LayerNorm invariance applies independently to every remaining `fc2` column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9764, "parameters": 1572, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering a fourth `fc2` output column will reduce the model from 1,581 to 1,580 learned parameters while retaining at least 99% accuracy; three centered columns achieved 0.9999, while the failure only appeared when all twelve were centered simultaneously.
change: Extend `CompactResidualLinear` from three to four centered weight columns.
mechanism: Incremental final-LayerNorm residual-column gauge fixing
evidence_used: The verified three-column design reached 0.9999 accuracy with 1,581 parameters, whereas centering all twelve reduced accuracy to 0.9764; testing one additional column is the smallest evidence-grounded step toward the capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9974, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering a fifth `fc2` output column will reduce the model to 1,579 learned parameters while retaining at least 99% accuracy.
change: Extend `CompactResidualLinear` from three to five centered columns, incorporating the verified fourth-column constraint and testing one additional column.
mechanism: Incremental final-LayerNorm residual-column gauge fixing
evidence_used: The four-column design achieved 0.9974 accuracy with 1,580 parameters, while centering all twelve fell to 0.9764; adding only the fifth column is the smallest informative step toward the capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7834, "parameters": 1579, "training_steps": 4999}



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
