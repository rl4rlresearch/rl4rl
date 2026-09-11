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
verified_results: {"accuracy": 0.9991, "parameters": 1575, "training_steps": 4999}
prior_hypothesis: Constraining the final `fc2` output-projection column to zero mean will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because this exact downstream-LayerNorm gauge avoids the activation coordinate implicated by the failed fourth-column extension.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate 5 at zero will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because the following QKV and attention-output affine transformations can absorb this bias and initialization remains unchanged.
change: Store five learned `AnchoredLayerNorm` biases and fix the final three biases at zero, while preserving all verified scale mappings.
mechanism: Adjacent pre-attention LayerNorm bias absorption
evidence_used: The 1,577-parameter design achieves 99.73% with two adjacent `ln1` biases already fixed at zero; extending that established bias constraint is better isolated than the failed reductions involving an additional scale tie, positional gauge, or attention-output column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate 4 at zero will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because the verified adjacent constraint on coordinate 5 preserved 99.92% accuracy and the following learned attention projections can adapt to the additional fixed offset.
change: Store four learned `AnchoredLayerNorm` biases and reconstruct the final four bias coordinates as zero, while preserving all verified scale mappings and initialization behavior.
mechanism: Adjacent pre-attention LayerNorm bias absorption
evidence_used: The immediately preceding reduction fixed `ln1` bias coordinate 5 and achieved 99.92% accuracy at 1,576 parameters; extending the same successful adjacent bias constraint is the closest controlled test after reductions involving scales, positional gauges, attention-output columns, and `ln2` scales failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6918000000000001, "parameters": 1575, "training_steps": 4999}

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
hypothesis: Fixing `ln1` bias coordinate 3 at zero will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because coordinates 0–2 remain independently adaptable and coordinate 4—whose removal or tying failed—remains learned.
change: Store `ln1` biases for coordinates 0–2 and 4, reconstruct coordinate 3 and coordinates 5–7 as zero, and preserve all verified scale mappings and initialization behavior.
mechanism: Redundant first-quartet pre-attention bias anchoring
evidence_used: Fixing biases 5–7 achieved 99.92% at 1,576 parameters, whereas fixing or tying coordinate 4 failed; this isolates a different reduction while retaining the evidently essential independent coordinate-4 bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2811, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the final `fc2` output-projection column to zero mean will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because this exact downstream-LayerNorm gauge avoids the activation coordinate implicated by the failed fourth-column extension.
change: Preserve the three verified leading `fc2` gauges, additionally represent only the final column in the zero-mean basis, and initialize it by projecting the same 94-scalar baseline draw as the 1,576-parameter design.
mechanism: Nonadjacent MLP-output common-mode gauge
evidence_used: Three gauged `fc2` columns contributed to the 99.92%-accurate 1,576-parameter model, while extending the gauge specifically to the adjacent fourth column achieved only 73.92%; testing the nonadjacent final column distinguishes coordinate-specific optimization sensitivity from failure of the established gauge mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the final attention-output projection column to zero mean will reduce the model to 1,574 parameters while retaining at least 99% accuracy, because this exact downstream-LayerNorm gauge avoids the fifth-column optimization sensitivity.
change: Preserve the four verified leading attention-output gauges, additionally represent only the final column in the zero-mean basis, and initialize it by projecting an RNG-aligned full-column draw.
mechanism: Nonadjacent attention-output common-mode gauge
evidence_used: The analogous nonadjacent final-column gauge in `fc2` achieved 99.91% after its adjacent fourth-column extension failed; this directly motivates testing the final attention column after the adjacent fifth-column gauge failed at 70.06%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0105, "parameters": 1574, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the penultimate `fc2` output-projection column to zero mean will reduce the model to 1,574 parameters while retaining at least 99% accuracy, because it extends the exact downstream-LayerNorm gauge from the successful nonadjacent final column while preserving initialization and RNG consumption.
change: Preserve the existing gauges on the first three and final `fc2` columns, additionally represent the penultimate column in the zero-mean basis, and reconstruct both trailing columns during the forward pass.
mechanism: Penultimate MLP-output common-mode gauge
evidence_used: Gauging the final `fc2` column produced the current 99.91%-accurate 1,575-parameter design after gauging the adjacent fourth column failed; the penultimate column is the closest controlled test of whether the successful trailing-column region supports another exact gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1788, "parameters": 1574, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc2` column 6 (the seventh column) to zero mean will reduce the model from 1,575 to 1,574 parameters while retaining at least 99% accuracy, because its removed all-ones component is invisible to the downstream final LayerNorm and the central coordinate avoids the optimization-sensitive adjacent fourth and penultimate columns.
change: Preserve the verified gauges on the first three and final `fc2` columns, additionally represent central column 6 in the zero-mean basis, and retain the same 94-scalar initialization draw.
mechanism: Central MLP-output common-mode gauge
evidence_used: The nonadjacent final-column `fc2` gauge achieved 99.91% at 1,575 parameters, while adding gauges at the adjacent fourth or penultimate columns failed; testing an isolated central column is the most direct next probe of coordinate-specific optimization sensitivity within the established exact gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4765, "parameters": 1574, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauging `fc2` column 6 while updating its seven coefficients through reconstructed eight-coordinate AdamW dynamics will reduce the model to 1,574 parameters and retain at least 99% accuracy.
change: Represent central `fc2` column 6 in the existing zero-mean basis, exclude those seven coefficients from ordinary AdamW, and apply projected updates computed from full eight-coordinate Adam moments.
mechanism: Virtual-coordinate Adam for a central MLP output gauge
evidence_used: The same central-column gauge reached only 47.65% with Adam applied directly in rotated coordinates, while the ungauged 1,575-parameter design reached 99.91%; reconstructing full-coordinate Adam isolates optimizer-coordinate sensitivity without changing the represented function class.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7443000000000001, "parameters": 1574, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the last output coordinate of `fc2` column 6 will reduce the model from 1,575 to 1,574 parameters while retaining at least 99% accuracy, because it preserves the exact downstream-LayerNorm function class while letting AdamW optimize seven original-axis weight differences instead of rotated zero-mean coordinates.
change: Represent `fc2` column 6 by seven differences from its omitted final output weight, reconstruct that weight as zero, and preserve the verified design’s 94-scalar initialization draw.
mechanism: Axis-aligned output-column gauge
evidence_used: The zero-mean column-6 gauge achieved 47.65% and virtual-coordinate Adam reached 74.43%, indicating optimizer-coordinate sensitivity rather than loss of expressivity; the 99.91%-accurate current model already succeeds with axis-aligned anchoring in `fc2` bias and `fc1`.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7332, "parameters": 1574, "training_steps": 4999}



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
