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
verified_results: {"accuracy": 0.9992, "parameters": 1576, "training_steps": 4999}
prior_hypothesis: Fixing `ln1` bias coordinate 5 at zero will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because the following QKV and attention-output affine transformations can absorb this bias and initialization remains unchanged.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing one key-projection parameter with a gauge that tracks the learned `ln1` scales will produce a 1,575-parameter model with at least 99% accuracy, because it preserves the key-shift invariance throughout training rather than only at initialization.
change: Compact the first key-projection row to seven coordinates, dynamically reconstruct its eighth coordinate orthogonal to the inverse LayerNorm scale, and initialize it from the same full 192-weight draw projected onto that gauge.
mechanism: Scale-aware key-row gauge fixing
evidence_used: The previous fixed-coordinate key-row reduction reached only 53.22% and relied on `ln1` outputs being zero-mean at initialization; learned unequal scales invalidate that premise. The current 1,576-parameter model reaches 99.92%, so correcting that specific gauge mismatch is an informative one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6990999999999999, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fourth `fc2` output-projection column to zero mean will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because its removed all-ones output component is invisible to downstream LayerNorm and initialization remains functionally and RNG aligned.
change: Extend `OutputAnchoredLinear` from three to four zero-mean weight columns, projecting the additional column from the same 94-scalar initialization draw used by the verified design.
mechanism: Fourth MLP-output common-mode gauge
evidence_used: The 1,576-parameter design achieved 99.92% with three `fc2` columns already using this gauge; unlike the failed fifth attention-output constraint, this extends the established projection immediately upstream of the final LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7392, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln1` bias coordinate 4 to learned bias coordinate 1 will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because it preserves an adaptable offset for coordinate 4 and follows the only successful partner mapping previously found for its LayerNorm scale.
change: Store four learned `AnchoredLayerNorm` biases and reconstruct bias coordinate 4 from coordinate 1, while keeping coordinates 5–7 fixed at zero and preserving all verified scale mappings.
mechanism: Scale-aligned pre-attention LayerNorm bias tying
evidence_used: Fixing bias coordinate 4 at zero failed at 69.18%, showing that coordinate needs adaptability; independently, tying scale coordinate 4 to coordinate 1 achieved 99.73%, whereas tying it to coordinates 0, 2, or 3 failed, making bias coordinate 1 the best-supported partner.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.732, "parameters": 1575, "training_steps": 4999}

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
