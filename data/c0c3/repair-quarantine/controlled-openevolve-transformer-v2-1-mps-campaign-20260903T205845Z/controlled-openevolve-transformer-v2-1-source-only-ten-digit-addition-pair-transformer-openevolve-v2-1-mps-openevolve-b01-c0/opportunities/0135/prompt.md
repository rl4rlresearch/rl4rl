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
verified_results: {"accuracy": 0.9940000000000001, "parameters": 1588, "training_steps": 4999}
prior_hypothesis: Anchoring the eleventh `fc2` column at its second output coordinate will reduce the model to 1588 parameters while retaining at least 99% accuracy, demonstrating that its prior failures were sensitive to the chosen gauge representative.

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
hypothesis: Compressing the third `fc2` column from 8 to 7 parameters while retaining virtual full-coordinate AdamW updates will reduce the model from 1588 to 1587 parameters and preserve at least 99% accuracy.
change: Anchor the third `fc2` column at its second output coordinate, reconstruct it during inference, initialize an equivalent gauge representative, and optimize it with virtual full-column moments.
mechanism: Second-coordinate gauge anchoring of a redundant MLP output column
evidence_used: The current second-coordinate anchoring of the eleventh `fc2` column achieved 0.994 accuracy at 1588 parameters, showing both sufficient accuracy margin and that this gauge representative can preserve training quality.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1945, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the seventh attention output-projection column and training it with virtual full-column AdamW moments will reduce the model from 1588 to 1587 parameters while retaining at least 99% accuracy.
change: Replace the first column of the remaining full attention projection matrix with a seven-parameter, final-coordinate-anchored column and include it in virtual-moment optimization.
mechanism: Seventh-column attention output gauge anchoring
evidence_used: The current 1588-parameter model reached 0.994 accuracy with six attention projection columns already gauge-anchored; extending that proven attention mechanism is better supported than repeating the third `fc2` compression that fell to 0.1945 accuracy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5336, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the third `fc2` column as seven ordinary-AdamW coordinates with a reconstructed zero-sum eighth coordinate will reduce the model from 1588 to 1587 parameters while retaining at least 99% accuracy.
change: Replace the failed coordinate-anchored/virtual-moment approach with the centered parameterization already used successfully by the first two `fc2` columns.
mechanism: Centered third MLP output-column gauge
evidence_used: The 1588-parameter model achieved 0.994 accuracy while training its first and second `fc2` columns in this centered seven-parameter form; the third-column second-coordinate anchoring with virtual moments reached only 0.1945, motivating a change in optimization geometry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.249, "parameters": 1587, "training_steps": 4999}



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
