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
verified_results: {"accuracy": 0.9994, "parameters": 1608, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,609-parameter design to fix a second `ln2.bias` coordinate will yield 1,608 parameters and at least 99% accuracy, because both removed LayerNorm offsets are exactly absorbable by the retained full `fc1.bias`.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9936, "parameters": 1609, "training_steps": 4999}
prior_hypothesis: Zero-centering positional row 2 will reduce the verified 1,610-parameter model to 1,609 parameters while retaining at least 99% accuracy, because its removed uniform component is canceled locally by pre-attention LayerNorm and ultimately by the final LayerNorm.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9992, "parameters": 1611, "training_steps": 4999}
prior_hypothesis: Combining the verified 1,612-parameter residual-bias design with a zero-mean parameterization of positional row 0’s three unfixed coordinates will yield 1,611 parameters and at least 99% accuracy, because the removed direction is a position-local uniform residual shift canceled by every downstream LayerNorm.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9989, "parameters": 1612, "training_steps": 4999}
prior_hypothesis: Applying a seven-dimensional zero-mean parameterization to `fc2.bias` on the verified five-position-gauge baseline will produce a 1,612-parameter model with at least 99% accuracy, because the removed uniform residual shift is canceled by the final LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing a second positional-embedding origin coordinate will reduce the verified 1,617-parameter model to 1,616 parameters while retaining at least 99% accuracy, because an equal coordinate-wise translation can be absorbed exactly into every token embedding without changing initialized hidden inputs.
change: Absorb `pos_emb[0, 1]` into token-embedding coordinate 1, then omit the first two flattened positional parameters and reconstruct them as zeros.
mechanism: Second token–position embedding translation gauge
evidence_used: The current model achieves 99.84% at 1,617 parameters while already using the identical gauge for coordinate 0; unlike the failed second MLP-row quotient at 98.38%, this extends a qualified embedding reparameterization without constraining a learned computational pathway.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,616-parameter design to fix positional-embedding origin coordinate 2 will yield 1,615 parameters and at least 99% accuracy, because the same coordinate-wise translation leaves initialized hidden inputs unchanged and shifts every output logit equally.
change: Reproduce the qualified first MLP-row quotient and two positional gauges, then absorb `pos_emb[0, 2]` into token-embedding coordinate 2 and omit the first three flattened positional parameters.
mechanism: Third token–position embedding translation gauge
evidence_used: The first LayerNorm–MLP row quotient achieved 99.84% at 1,617 parameters, and adding the second exact token–position translation gauge produced 99.93% at 1,616; extending that qualified embedding symmetry by one coordinate is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,615-parameter design to fix positional-origin coordinate 3 will yield 1,614 parameters and at least 99% accuracy, because the transformation preserves initialized hidden inputs and adds only a uniform shift to tied-head logits.
change: Reproduce the verified three-key-row and first-MLP-row quotients, then absorb the first four positional-origin coordinates into the corresponding token-embedding coordinates and omit those four positional parameters.
mechanism: Fourth token–position embedding translation gauge
evidence_used: The three-gauge 1,615-parameter design achieved 99.84% accuracy; the second and third positional gauges also independently preserved at least 99%, directly supporting one more application of the same symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9959, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing positional-origin coordinate 4 in the verified 1,614-parameter design will yield 1,613 parameters and at least 99% accuracy, because the exact token–position translation preserves initialized hidden inputs and changes tied-head logits only by a vocabulary-uniform shift.
change: Reproduce the qualified three-key-row, shared-projection, LayerNorm-bias, and first-MLP-row compactions, then absorb the first five positional-origin coordinates into token embeddings and omit those five positional parameters.
mechanism: Fifth token–position embedding translation gauge
evidence_used: The four-gauge 1,614-parameter design achieved 99.59% accuracy, while the preceding second and third applications of the same embedding symmetry achieved 99.93% and 99.84%; this directly supports testing one additional coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing positional-origin coordinate 5 will reduce the verified model to 1,612 parameters while retaining at least 99% accuracy, because the token–position translation preserves hidden inputs and changes tied-head logits only by a vocabulary-uniform shift.
change: Absorb `pos_emb[0, 5]` into token-embedding coordinate 5, then omit the first six flattened positional parameters and reconstruct them as zeros.
mechanism: Sixth token–position embedding translation gauge
evidence_used: The same exact gauge succeeded for coordinate 4 at 1,613 parameters with 99.94% accuracy, following successful applications at coordinates 1–3.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9483, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying a seven-dimensional zero-mean parameterization to `fc2.bias` on the verified five-position-gauge baseline will produce a 1,612-parameter model with at least 99% accuracy, because the removed uniform residual shift is canceled by the final LayerNorm.
change: Reproduce the verified fifth token–position gauge, then replace the eight-parameter `fc2` bias with seven Helmert coordinates that reconstruct a zero-mean bias.
mechanism: Orthonormal MLP residual-bias gauge quotient
evidence_used: The five-position-gauge design achieved 99.94% accuracy at 1,613 parameters, while a sixth positional gauge fell to 94.83%; the qualified designs already exploit the analogous zero-mean residual-bias invariance in the attention projection, motivating this independent exact quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified 1,612-parameter residual-bias design with a zero-mean parameterization of positional row 0’s three unfixed coordinates will yield 1,611 parameters and at least 99% accuracy, because the removed direction is a position-local uniform residual shift canceled by every downstream LayerNorm.
change: Reproduce the five verified token-position gauges and compact `fc2` bias, then represent the remaining positional-origin tail with two Helmert coordinates.
mechanism: Positional-origin residual common-mode quotient
evidence_used: The five-position-gauge plus compact-`fc2`-bias design achieved 99.89% at 1,612 parameters, while a sixth token-position gauge failed at 94.83%; this tests a distinct exact positional residual symmetry on the strongest qualified baseline.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,611-parameter design by zero-centering positional row 1 will yield 1,610 parameters and at least 99% accuracy, because that row’s removed uniform component survives only as a position-local residual shift canceled by every downstream LayerNorm.
change: Reproduce the verified five translation gauges, positional-origin quotient, and compact `fc2` bias, then represent positional row 1 with seven Helmert coordinates.
mechanism: Per-position residual common-mode quotient
evidence_used: The positional-origin common-mode quotient achieved 99.92% at 1,611 parameters; applying the same residual-uniform invariance to a separate full positional row is directly supported and avoids the failed sixth coordinate-wise translation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.996, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Zero-centering positional row 2 will reduce the verified 1,610-parameter model to 1,609 parameters while retaining at least 99% accuracy, because its removed uniform component is canceled locally by pre-attention LayerNorm and ultimately by the final LayerNorm.
change: Represent positional rows 1 and 2 with independent seven-dimensional Helmert coordinates, reconstruct both as zero-mean vectors, and retain full parameters only from row 3 onward.
mechanism: Third per-position residual common-mode quotient
evidence_used: Zero-centering positional row 1 produced the current qualified 1,610-parameter model at 99.6% accuracy, while the analogous positional-origin common-mode quotient achieved 99.92%; applying the same verified residual invariance to the next positional row is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9936, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Zero-centering positional row 3 on the verified 1,609-parameter design will yield 1,608 parameters and at least 99% accuracy, because its removed uniform component is invisible to pre-attention LayerNorm and final LayerNorm.
change: Reproduce the qualified zero-mean parameterization of positional rows 1 and 2, then apply the same seven-dimensional Helmert parameterization to row 3.
mechanism: Fourth per-position residual common-mode quotient
evidence_used: Zero-centering row 2 achieved 99.36% accuracy at 1,609 parameters after row 1 achieved 99.6%; row 3 tests the next independent instance of the same exact residual-shift symmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2938, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,609-parameter positional design and quotienting a second `fc1` row with dynamic LayerNorm-bias compensation will produce a 1,608-parameter model with at least 99% accuracy.
change: Compact positional rows 0–2 as in the qualified 1,609-parameter design, then represent the first two `fc1` rows with seven-dimensional Helmert coordinates while decoupling the second row’s effective bias from `ln2.bias`.
mechanism: Bias-decoupled second LayerNorm–MLP row gauge quotient
evidence_used: Positional-row compaction achieved 99.36% at 1,609 parameters. The earlier second `fc1` quotient narrowly missed at 98.38% while using a bias coupled to the changing LayerNorm offset; explicitly compensating that offset tests the same exact gauge with better-conditioned optimization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7301000000000001, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,609-parameter design to fix a second `ln2.bias` coordinate will yield 1,608 parameters and at least 99% accuracy, because both removed LayerNorm offsets are exactly absorbable by the retained full `fc1.bias`.
change: Reproduce the qualified positional-row and `fc2.bias` compactions, then retain six rather than seven `ln2.bias` coordinates and reconstruct its final two coordinates as zeros.
mechanism: Second pre-MLP LayerNorm-bias absorption gauge
evidence_used: The 1,609-parameter positional design achieved 99.36% accuracy while already fixing one `ln2.bias` coordinate. The failed 1,608 attempts instead constrained positional row 3 or a second `fc1` weight row, motivating this independent continuation of the already-qualified downstream-bias redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1608, "training_steps": 4999}



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
