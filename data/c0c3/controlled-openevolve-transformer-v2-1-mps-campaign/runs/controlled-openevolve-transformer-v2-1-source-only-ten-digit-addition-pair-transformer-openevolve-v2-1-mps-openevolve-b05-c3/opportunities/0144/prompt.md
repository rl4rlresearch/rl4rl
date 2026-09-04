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
verified_results: {"accuracy": 1.0, "parameters": 1116, "training_steps": 4999}
prior_hypothesis: Tying a fifth terminal relative-kernel logit while adopting the verified nested Givens quotient will reduce the model from 1,118 to 1,116 parameters and retain at least 99% accuracy, because it preserves the successful bounded projection chart and removes capacity from the independent lag kernel rather than adding another unstable projection scale or shear.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing the positive norm coefficient produced by the qualified nested Givens rotation and absorbing its scale into the matching value feature will reduce the model from 1,117 to 1,116 parameters while retaining at least 99% accuracy.
change: Adopt the qualified two-scale, two-shear, two-rotation projection chart, then omit row-one’s remaining rotated coefficient by fixing it at 0.02 and inversely rescaling its value feature during initialization and every projection update.
mechanism: Rotation-conditioned nested value-scale quotient
evidence_used: The nested orthogonal quotient reached 99.99% accuracy at 1,117 parameters, while a second-largest coordinate scale failed; the proposed anchor acts on a rotation-produced Euclidean norm, which is nonnegative and at least as well conditioned as either source coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1116, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the first head’s raw scalar pivot chart with an orthogonal alignment of features zero and three, followed by the existing 0.02 scale anchor, will reduce the model from 1,117 to 1,116 parameters while retaining at least 99% accuracy because the scale denominator becomes a well-conditioned pair norm.
change: Omit the first row’s fourth projection coefficient, maintain it at zero with a Givens rotation at initialization and after every projection update, and apply the matching rotation to the learned value features before the existing scale and shear transformations.
mechanism: Norm-conditioned first-head Givens quotient
evidence_used: Successive norm-preserving rotations achieved 99.96% at 1,118 parameters and 99.99% at 1,117, while the added rotation-conditioned scale quotient collapsed at 1,116; this tests another orthogonal quotient while retaining the already-qualified scale anchors.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1116, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying a fifth terminal relative-kernel logit while adopting the verified nested Givens quotient will reduce the model from 1,118 to 1,116 parameters and retain at least 99% accuracy, because it preserves the successful bounded projection chart and removes capacity from the independent lag kernel rather than adding another unstable projection scale or shear.
change: Add the qualified second nested rotation and omit its projection coordinate, then shorten the learned relative-bias vector by one while reconstructing five tied terminal logits.
mechanism: Five-terminal kernel tie atop the qualified nested orthogonal projection quotient
evidence_used: The nested orthogonal quotient achieved 99.99% accuracy at 1,117 parameters, while both subsequent 1,116 projection modifications collapsed; this motivates preserving that projection geometry and testing one additional tie in the already four-way-tied relative kernel.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1116, "training_steps": 4999}

RECENT RESULT
hypothesis: Adopting the verified 1,116-parameter nested-rotation/five-terminal-kernel design and fixing its remaining shared normalization scale at its unit initialization will produce a 1,115-parameter model with at least 99% accuracy; the learned codebook and attention/MLP maps can express confidence without a global learned temperature.
change: Replace the current projection with the qualified scale/shear/nested-Givens chart, tie five terminal relative-kernel logits, and replace the learned final/attention normalization scalar with a fixed unit scale.
mechanism: Fixed-temperature calibration atop nested orthogonal attention quotients
evidence_used: Reference Design 3 achieved 100% accuracy at 1,116 parameters. All qualified designs retained the adaptive scalar, so its necessity is an untested load-bearing assumption; unlike the failed six-coordinate token bottleneck, this preserves all seven learned codebook coordinates and every learned nonlinear and attention feature.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Extending the verified 1,116-parameter nested-rotation/five-terminal-kernel design to six tied terminal logits will reduce the model to 1,115 parameters while retaining at least 99% accuracy, because it preserves the qualified projection geometry and removes only one additional relative-kernel degree of freedom.
change: Adopt the verified nested scale/shear/Givens projection chart and fully tied final LayerNorm bias, then shorten the learned relative-bias vector by one and reconstruct six tied terminal logits.
mechanism: Six-terminal kernel tie atop nested orthogonal attention quotients
evidence_used: The nested orthogonal quotient achieved 99.99% at 1,117 parameters, and adding the fifth terminal kernel tie improved the resulting 1,116-parameter design to 100%; projection-side attempts at 1,116 collapsed, so extending the successful kernel tie is the most informative next reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Extending the qualified five-terminal tie by one will reduce the model from 1,116 to 1,115 parameters while retaining at least 99% accuracy; reducing validation frequency will allow all 4,999 training updates to finish within the verification limit.
change: Shorten the learned relative-bias vector by one, reconstruct six tied terminal logits, and evaluate every 1,000 steps while retaining mandatory final evaluation and checkpointing.
mechanism: Six-terminal relative-kernel tie with sparse validation
evidence_used: The five-terminal design achieved 100% accuracy at 1,116 parameters, whereas projection-side 1,116-parameter reductions collapsed. The earlier six-terminal attempt timed out without producing contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Tying six terminal kernel logits will reduce the verified nested-rotation design from 1,117 to 1,115 parameters while retaining at least 99% accuracy, and eliminating interim validation will allow all 4,999 updates to finish within the verification limit.
change: Reconstruct six tied terminal relative-bias logits instead of four and evaluate only after the final training update, which still writes both required checkpoints.
mechanism: Six-terminal relative-kernel tie with final-only validation
evidence_used: The five-terminal design achieved 100% accuracy at 1,116 parameters; six-terminal attempts timed out rather than showing an accuracy failure, including with validation every 1,000 steps, so removing all nonessential interim evaluations directly tests the remaining runtime confound.
result: training did not finish within the verification time limit



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
