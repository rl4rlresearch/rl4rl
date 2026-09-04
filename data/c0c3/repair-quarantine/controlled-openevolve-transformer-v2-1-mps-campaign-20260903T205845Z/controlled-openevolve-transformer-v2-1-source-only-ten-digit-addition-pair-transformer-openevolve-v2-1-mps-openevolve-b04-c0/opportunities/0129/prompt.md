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
verified_results: {"accuracy": 0.9987, "parameters": 1566, "training_steps": 4999}
prior_hypothesis: Fixing `proj.weight[0,6]` alongside `[0,0]`, `[0,2]`, and `[0,4]` will produce a 1566-parameter model with at least 99% accuracy because it is the second-head counterpart of the newly verified component-two anchor.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `proj.weight[0,4]` alongside the verified `proj.weight[0,0]` anchor will produce a 1568-parameter model with at least 99% accuracy, because column 4 is the component-zero counterpart in the second attention head and its removal adds only a feature-uniform residual shift eliminated by downstream LayerNorms.
change: Store 62 learned attention-output projection weights, reconstruct `proj.weight[0,0]` and `proj.weight[0,4]` as zeros, and canonicalize both columns at initialization.
mechanism: Cross-head attention-output residual-shift gauge
evidence_used: The component-zero projection anchor at `proj.weight[0,0]` achieved 99.93% at 1569 parameters, while the adjacent component-one anchor at `[0,1]` collapsed to 15.51%; prior QKV evidence also showed component-zero anchors succeeding across both heads while component one failed, motivating the cross-head component-zero counterpart.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9963, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `proj.weight[0,2]` alongside the successful component-zero anchors will produce a 1567-parameter model with at least 99% accuracy, because the removed projection coefficient contributes only a feature-uniform residual shift, and component 2 previously tolerated a QKV gauge reduction while component 1 did not.
change: Store 61 learned attention-output projection weights, reconstruct `proj.weight[0,0]`, `[0,2]`, and `[0,4]` as zeros, and canonicalize all three columns at initialization.
mechanism: Third attention-output residual-shift gauge
evidence_used: Projection anchors `[0,0]` and `[0,4]` achieved 99.63% at 1568 parameters, whereas `[0,1]` collapsed to 15.51%; separately, the component-2 QKV shear achieved 99.59%, making projection component 2 the most supported untested column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1567, "training_steps": 4999}

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
hypothesis: Fixing `proj.weight[0,6]` alongside `[0,0]`, `[0,2]`, and `[0,4]` will produce a 1566-parameter model with at least 99% accuracy because it is the second-head counterpart of the newly verified component-two anchor.
change: Store 60 learned attention-output projection weights, reconstruct `proj.weight[0,0]`, `[0,2]`, `[0,4]`, and `[0,6]` as zeros, and canonicalize all four columns at initialization.
mechanism: Cross-head component-two attention-output residual-shift gauge
evidence_used: Adding projection anchor `[0,2]` improved the verified model from 1568 to 1567 parameters while reaching 99.89% accuracy; `[0,6]` applies the same component-two gauge to the other attention head, analogous to the successful cross-head component-zero pair `[0,0]` and `[0,4]`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1566, "training_steps": 4999}



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
