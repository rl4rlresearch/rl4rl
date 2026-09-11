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
verified_results: {"accuracy": 0.9991, "parameters": 1605, "training_steps": 4999}
prior_hypothesis: Fixing a second featurewise token-position transfer coordinate will reduce the model from 1606 to 1605 learned parameters while retaining at least 99% accuracy, because it is symmetry-equivalent to the already-verified transfer gauge and the generalized virtual optimizer preserves full-parameter AdamW dynamics.

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
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Fixing the remaining key row `d_model + 3` will reduce the model from 1607 to 1606 learned parameters while preserving at least 99% accuracy, because it has the same softmax-invisible LayerNorm-null direction as the seven already-fixed key rows.
change: Remove one additional QKV parameter by extending the existing virtual-optimizer gauge to the sole unfixed key row.
mechanism: Complete LayerNorm-null key-row gauge fixing
evidence_used: The current seven-row gauge-fix design verified at 0.9997 accuracy with 1607 parameters, supporting another symmetry-equivalent key-row removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified token-position quotient across all seven independent feature transfers will reduce parameters from 1606 to 1600 while retaining at least 99% accuracy.
change: Fix the entire final token-embedding row using seven token-position transfer gauges plus the existing global-shift gauge, and generalize virtual AdamW and gradient clipping accordingly.
mechanism: Full token-anchor embedding gauge fixing
evidence_used: The 1606-parameter design achieved 1.0 accuracy while already fixing one token-position transfer coordinate; the remaining featurewise transfers have the same exact input-preserving and softmax-invisible symmetry.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Fixing a second featurewise token-position transfer coordinate will reduce the model from 1606 to 1605 learned parameters while retaining at least 99% accuracy, because it is symmetry-equivalent to the already-verified transfer gauge and the generalized virtual optimizer preserves full-parameter AdamW dynamics.
change: Extend the embedding quotient from feature 4 to features 4 and 5, and generalize initialization, virtual AdamW, and gradient clipping to recover both omitted transfer coordinates.
mechanism: Incremental token-position embedding gauge fixing
evidence_used: The single-transfer design achieved 1.0 accuracy with 1606 parameters. The attempted all-feature extension could not be verified, so adding one coordinate is the most informative incremental test of the same exact symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1605, "training_steps": 4999}



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
