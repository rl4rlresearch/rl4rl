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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_model` from 8 to 6 will lower deduplicated parameters from 1644 to 1188 while retaining at least 99% accuracy because the verified 99.96% baseline has substantial accuracy margin and the two attention heads plus 12-wide MLP remain intact.
change: Change the default residual width to 6 while preserving the existing architecture, head count, training schedule, and decoding interface.
mechanism: Narrow residual stream with preserved two-head attention
evidence_used: The current 8-wide model achieves 0.9996 accuracy with 1644 parameters after 4999 steps, motivating a direct width reduction to locate the capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1188, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all linear-layer biases will reduce parameters from 1644 to 1592 while retaining at least 99% accuracy because it preserves the verified 8-wide residual stream, two-head attention, and 12-wide MLP.
change: Disable biases in the attention and MLP linear projections without changing model width, training, or decoding.
mechanism: Bias-free learned projections
evidence_used: The 8-wide baseline achieved 0.9996 accuracy, whereas reducing `d_model` to 6 collapsed accuracy to 0.0001; this motivates preserving residual capacity and removing less-essential projection biases instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.039900000000000005, "parameters": 1592, "training_steps": 4999}



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
