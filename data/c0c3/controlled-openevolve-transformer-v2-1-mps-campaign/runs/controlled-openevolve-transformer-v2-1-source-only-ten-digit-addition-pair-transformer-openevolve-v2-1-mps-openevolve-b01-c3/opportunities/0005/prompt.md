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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Removing only the attention key bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy, because a position-independent key bias adds the same value to every unmasked attention logit for a query and therefore cancels exactly in softmax.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9998, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Removing the attention value bias in addition to the proven-redundant key bias will reduce parameters from 1,644 to 1,628 while retaining at least 99% accuracy, because attention weights sum to one and the retained output-projection bias can represent the value bias’s constant contribution.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will lower deduplicated parameters from 1,644 to 1,576 while retaining at least 99% accuracy because attention width, embeddings, and training budget remain unchanged.
change: Set the single transformer block’s feed-forward width equal to its model width.
mechanism: Width-matched feed-forward bottleneck
evidence_used: The current `d_ff=12` design reached 99.96% accuracy with 1,644 parameters, providing margin for a targeted 68-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7225, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the two 8-element output biases will reduce parameters from 1,644 to 1,628 while retaining at least 99% accuracy, because both biases enter residual streams immediately before downstream normalization and the proven-critical `d_ff=12` capacity remains intact.
change: Disable biases in the attention output projection and MLP output projection.
mechanism: Residual-output bias pruning
evidence_used: Reducing `d_ff` from 12 to 8 cut accuracy to 72.25%, showing that nonlinear width should be preserved; the original `d_ff=12` model achieved 99.96%, leaving margin for a smaller, targeted reduction in comparatively redundant output biases.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7537999999999999, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the attention key bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy, because a position-independent key bias adds the same value to every unmasked attention logit for a query and therefore cancels exactly in softmax.
change: Replace the packed QKV bias with learned query and value biases, preserving all functional attention biases, `d_ff=12`, and both residual-output biases.
mechanism: Softmax-invariant key-bias elimination
evidence_used: The 1,644-parameter design reached 99.96% accuracy, while reducing nonlinear width or removing residual-output biases caused large regressions; this patch preserves those demonstrated capacities and removes only a mathematically non-identifiable parameter vector.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the attention value bias in addition to the proven-redundant key bias will reduce parameters from 1,644 to 1,628 while retaining at least 99% accuracy, because attention weights sum to one and the retained output-projection bias can represent the value bias’s constant contribution.
change: Use a bias-free packed QKV projection with only an explicit learned query bias, while retaining both residual-output biases and `d_ff=12`.
mechanism: Softmax-normalized value-bias absorption
evidence_used: Key-bias elimination achieved 99.95% accuracy with 1,636 parameters, while removing residual-output biases reduced accuracy to 75.38%; this patch preserves those critical output biases and removes a different attention bias that is functionally absorbable by the retained projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1628, "training_steps": 4999}



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
