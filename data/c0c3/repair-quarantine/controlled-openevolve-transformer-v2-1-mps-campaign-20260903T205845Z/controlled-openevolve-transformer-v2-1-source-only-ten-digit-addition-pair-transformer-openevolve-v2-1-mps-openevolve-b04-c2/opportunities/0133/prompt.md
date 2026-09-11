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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1562, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified 1,563-parameter design and fixing a second bias-free terminal `ln2.weight` coordinate at one will produce a 1,562-parameter model with at least 99% accuracy, because its effective scaling remains representable by the corresponding unrestricted `fc1` column.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1567, "training_steps": 4999}
prior_hypothesis: Removing one additional `ln2` bias direction will reduce the verified 1,568-parameter model to 1,567 parameters while maintaining at least 99% accuracy, because any constant shift it induces before `fc1` is exactly absorbable by the unrestricted `fc1` bias.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9974, "parameters": 1563, "training_steps": 4999}
prior_hypothesis: Fixing the bias-free final coordinate of `ln2.weight` at one and leaving its effective scaling to the unrestricted final `fc1` column will produce a 1,563-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9953, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,583-parameter design by quotienting positional row 6 will produce a 1,582-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first and final LayerNorms.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing the bias-free final coordinate of `ln2.weight` at one and leaving its effective scaling to the unrestricted final `fc1` column will produce a 1,563-parameter model with at least 99% accuracy.
change: Store seven learned `ln2` scale coordinates and reconstruct the eighth as a constant one during the forward pass.
mechanism: Downstream-absorbed terminal LayerNorm scale gauge
evidence_used: The current 1,564-parameter design achieved 99.86% accuracy, while deeper `ln1` and `ln2` bias restrictions failed. This tests an independent exact affine redundancy: the current `ln2` bias basis makes its final channel identically bias-free, and the corresponding `fc1` column is unrestricted.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9974, "parameters": 1563, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1,563-parameter design and fixing a second bias-free terminal `ln2.weight` coordinate at one will produce a 1,562-parameter model with at least 99% accuracy, because its effective scaling remains representable by the corresponding unrestricted `fc1` column.
change: Use the qualified six-coordinate `ln1.bias` and four-coordinate `ln2.bias`, while reconstructing the final two `ln2` scales as constants instead of learned parameters.
mechanism: Progressive downstream-absorbed LayerNorm scale quotient
evidence_used: The first terminal `ln2` scale quotient achieved 99.74% accuracy at 1,563 parameters; this extends that successful independent mechanism by one coordinate while avoiding the failed deeper `ln1.bias` and `ln2.bias` restrictions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1562, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1,562-parameter design and fixing a third bias-free terminal `ln2.weight` coordinate at one will produce a 1,561-parameter model with at least 99% accuracy, because its effective scaling remains representable by the corresponding unrestricted `fc1` column.
change: Use six learned `ln1.bias` coordinates and four learned `ln2.bias` coordinates, then store only the first five `ln2` scales and reconstruct the final three as constants.
mechanism: Progressive downstream-absorbed LayerNorm scale quotient
evidence_used: Fixing the first terminal `ln2` scale achieved 99.74% accuracy at 1,563 parameters, and fixing the second improved this to 99.93% at 1,562; extending the same successful independent redundancy by one coordinate is the most informative next reduction.
result: the implementation could not be verified

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
