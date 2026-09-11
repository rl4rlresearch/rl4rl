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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1123, "training_steps": 4999}
prior_hypothesis: Fixing one attention-output projection coefficient and absorbing its scale into the corresponding learned value feature will reduce the qualified model from 1,124 to 1,123 parameters while retaining at least 99% accuracy, because this removes an exact factorization symmetry without tying or deleting any attention feature.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1117, "training_steps": 4999}
prior_hypothesis: Applying a second norm-preserving Givens rotation within the two second-head features whose first-row coefficients are already zero will reduce the qualified 1,118-parameter model to 1,117 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1127, "training_steps": 4999}
prior_hypothesis: Extending the qualified four-way final-bias tie to five channels will reduce the model from 1,128 to 1,127 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1118, "training_steps": 4999}
prior_hypothesis: Canonicalizing the two unconstrained second-head features with an orthogonal rotation will reduce the model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because it removes an exact value/projection basis gauge without the inverse scaling or unbounded shears that caused prior 1,118-parameter failures.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one attention-projection coefficient to zero through a triangular value-basis shear will reduce the qualified model from 1,122 to 1,121 parameters while retaining at least 99% accuracy, because it preserves the learned attention function and avoids dividing by the unstable third coefficient that caused both three-scale-anchor failures.
change: Preserve the two qualified value-scale anchors, remove the third projection coefficient, and absorb it into a same-head shear between value features zero and two during initialization and optimizer updates.
mechanism: Within-head value-basis shear quotient
evidence_used: The two-feature scale quotient achieved 99.98% accuracy at 1,122 parameters, while third scale anchors achieved only 0.01% and 42.48%; the proposed shear instead uses the already-fixed nonzero first anchor as its stable denominator and removes an exact within-head factorization gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1121, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-scale, one-shear 1,121-parameter design with a second shear anchored by the fixed nonzero first projection coefficient will reduce it to 1,120 parameters while retaining at least 99% accuracy, because the transformation exactly preserves the learned attention function without dividing by another unstable coefficient.
change: Adopt the qualified two projection-scale anchors and first within-head shear, then eliminate the fourth projection coefficient by shearing value feature three into feature zero during initialization and every projection optimizer step.
mechanism: Second stable within-head value-basis shear quotient
evidence_used: The first within-head shear achieved 99.98% accuracy with 1,121 parameters, while attempts to use the third coefficient as a scale anchor failed at 0.01% and 42.48%; another shear through the already-fixed 0.02 anchor tests the same successful, well-conditioned symmetry removal.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Extending the qualified two-scale, one-shear design with a second shear through the fixed 0.02 projection anchor will reduce the model from 1,121 to 1,120 parameters while retaining at least 99% accuracy.
change: Adopt the qualified fully zero-biased final LayerNorm, preserve both projection-scale anchors and the first shear, then eliminate the fourth projection coefficient by shearing value feature three into feature zero during initialization and every projection update.
mechanism: Second stable within-head value-basis shear quotient
evidence_used: The first stable within-head shear achieved 99.98% accuracy with 1,121 parameters, while unstable third scale anchors failed; the previous second-shear implementation could not be verified, leaving a fully specified extension of the successful shear mechanism as the most informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1120, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-scale, one-shear 1,121-parameter design with a second shear anchored through the distinct fixed second projection coefficient will reduce it to 1,120 parameters while retaining at least 99% accuracy, because it preserves the attention function without coupling both shears through value feature zero.
change: Adopt the qualified fully zero-biased final LayerNorm and two projection-scale anchors, preserve the successful feature-2-to-feature-0 shear, and eliminate the fourth projection coefficient using a feature-3-to-feature-1 shear during initialization and every projection update.
mechanism: Independent-anchor within-head value-basis shear quotient
evidence_used: The first stable shear achieved 99.98% accuracy at 1,121 parameters, while the second shear through the same first anchor collapsed; using the independently fixed second 0.02 anchor isolates whether shared-anchor conditioning caused that failure.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Eliminating the fourth projection coefficient through a feature-3-to-feature-1 shear will reduce the qualified 1,121-parameter model to 1,120 parameters while retaining at least 99% accuracy, because it uses the distinct fixed second anchor and exactly preserves the learned attention function.
change: Extend the two-scale, one-shear projection chart with a second zero anchor, compensate through value feature one at initialization and after projection updates, and reconstruct the additional omitted projection gradient.
mechanism: Independent-anchor within-head value-basis shear quotient
evidence_used: The first stable shear achieved 99.98% accuracy at 1,121 parameters, while a second shear sharing value feature zero collapsed; the independent-anchor variant was not tested because its patch search matched multiple locations.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1120, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the tied token codebook to six learned coordinates while retaining the eight-dimensional attention/MLP workspace will reduce the 1,123-parameter model by `VOCAB_SIZE` parameters and still achieve at least 99% accuracy.
change: Fix a second embedding coordinate to zero, initialize the reduced codebook from the existing distribution, and optimize it directly while leaving the learned attention, 12-unit nonlinear bank, biases, decoding, and checkpoint flow unchanged.
mechanism: Six-coordinate learned token bottleneck
evidence_used: The current internal computation reaches 99.96%, whereas reducing the 12-unit MLP collapsed to 0.01% and constraining attention or residual biases also failed. This challenges the untested assumption that the token input/output interface needs all seven quotient coordinates without disturbing those demonstrated load-bearing components.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.35229999999999995, "parameters": 1009, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying one shear to the previously untouched second attention head, with its largest-magnitude initialization coefficient as the pivot and smallest as the eliminated coordinate, will reduce the qualified 1,121-parameter design to 1,120 parameters while retaining at least 99% accuracy.
change: Adopt the qualified zero final-LayerNorm bias, two projection-scale anchors, and first-head shear, then eliminate one second-head projection coefficient and compensate through the corresponding learned value features during initialization and every optimizer step.
mechanism: Pivot-conditioned second-head value-basis shear
evidence_used: The two-scale, one-shear design achieved 99.98% accuracy at 1,121 parameters. Additional shears coupled to the already constrained first head collapsed, motivating an isolated shear in the unconstrained second head with a conditioned pivot that bounds the initial shear magnitude.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1120, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the largest-magnitude second-head projection pivot at 0.02 and absorbing its scale into the corresponding value feature will reduce the verified 1,120-parameter model to 1,119 parameters while retaining at least 99% accuracy, because this removes an exact multiplicative gauge through an initialization-conditioned pivot rather than adding another destabilizing shear.
change: Remove the learned second-head pivot coefficient, reconstruct it as a fixed 0.02 anchor, and preserve the attention function by inversely rescaling its value feature during initialization and every projection update.
mechanism: Conditioned second-head pivot scale quotient
evidence_used: The current pivot-conditioned second-head shear achieved 100% accuracy with 1,120 parameters, and two fixed projection-scale anchors previously achieved 99.98%; unlike the failed third scale anchor, this anchor is selected as the largest-magnitude coefficient in its head, limiting the initial rescaling condition number.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1119, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the dynamically selected second-largest second-head projection coefficient at 0.02 will reduce the verified model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because its multiplicative scale can be absorbed exactly into the corresponding value feature and its rank-conditioned initialization avoids the unstable positional anchors that previously failed.
change: Add a second fixed scale anchor in the permuted second attention head, omit that projection coordinate from learned storage, inversely rescale its value feature during initialization and optimizer recanonicalization, and update full-gradient reconstruction.
mechanism: Rank-conditioned second-head value/projection scale quotient
evidence_used: The largest-magnitude second-head scale quotient achieved 99.99% accuracy at 1,119 parameters after the rank-conditioned second-head shear achieved 100%; unlike the failed third positional scale anchors, the proposed coordinate is explicitly selected as the second-largest magnitude in its head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0017000000000000001, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Shearing the second-smallest second-head projection coordinate through the verified largest-magnitude 0.02 pivot will reduce the qualified 1,119-parameter model to 1,118 parameters while retaining at least 99% accuracy, because it exactly preserves the learned attention function without the destabilizing inverse rescaling used by the failed second-largest scale quotient.
change: Adopt the qualified three-scale, two-shear projection chart, then dynamically place the two smallest second-head coordinates beside its largest pivot and eliminate both through bounded value-basis shears.
mechanism: Rank-conditioned dual shear in the second attention head
evidence_used: The largest-magnitude second-head pivot scale quotient reached 99.99% accuracy at 1,119 parameters, whereas scaling the second-largest coordinate collapsed to 0.17%; a shear through the already conditioned pivot tests another exact gauge without independently normalizing that coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Canonicalizing the two unconstrained second-head features with an orthogonal rotation will reduce the model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because it removes an exact value/projection basis gauge without the inverse scaling or unbounded shears that caused prior 1,118-parameter failures.
change: Omit one additional second-head projection coordinate, initialize and maintain it at zero with a Givens rotation, and apply the inverse orthogonal transform to the corresponding value features.
mechanism: Norm-preserving Givens quotient in the remaining second-head value subspace
evidence_used: The conditioned second-head pivot scale quotient achieved 99.99% at 1,119 parameters, while a second scale anchor and an additional shear collapsed at 1,118; a bounded norm-preserving rotation isolates conditioning as the failure source while leaving the verified pivot and shear unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying a second norm-preserving Givens rotation within the two second-head features whose first-row coefficients are already zero will reduce the qualified 1,118-parameter model to 1,117 parameters while retaining at least 99% accuracy.
change: Adopt the qualified second-head scale, shear, and rotation chart, then omit one additional projection coefficient by rotating features five and six so their row-one coefficient is zero, with matching value-feature transformations during initialization and optimization.
mechanism: Nested orthogonal quotient in the second-head zero subspace
evidence_used: The first orthogonal second-head quotient achieved 99.96% accuracy at 1,118 parameters, whereas additional scale and shear quotients collapsed; another bounded rotation inside its preserved zero-anchor subspace tests a distinct exact gauge without inverse scaling or unbounded shearing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1117, "training_steps": 4999}



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
