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
verified_results: {"accuracy": 1.0, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Removing key and value biases while retaining the query bias will reduce parameters from 1644 to 1628 and preserve at least 99% accuracy because, with zero dropout, key bias cancels in the attention softmax and value bias is representable by the output-projection bias.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9997, "parameters": 1509, "training_steps": 4999}
prior_hypothesis: The verified 1,567-parameter quotient-residual, `d_ff=11` transformer will retain at least 99% accuracy when its full-rank tied token matrix is replaced by a learned rank-seven factorization, reducing the model to 1,509 parameters; this challenges the prior assumption that token representation and decoding require all eight hidden dimensions.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1567, "training_steps": 4999}
prior_hypothesis: Reducing `d_ff` from 12 to 11 will lower parameters from 1583 to 1567 while retaining at least 99% accuracy, because it preserves nearly all nonlinear capacity of the verified 99.90%-accurate design.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will remove 68 learned parameters (1644 → 1576) while retaining at least 99% accuracy because the current 99.96% result provides substantial error-margin evidence that the MLP is overprovisioned.
change: Change the default feed-forward width from 12 to 8.
mechanism: Narrower nonlinear residual bottleneck
evidence_used: The current one-layer, width-8 model achieved 0.9996 accuracy with 1644 parameters; this motivates isolating a moderate 33% reduction in its MLP width before changing attention or embedding capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7149, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing key and value biases while retaining the query bias will reduce parameters from 1644 to 1628 and preserve at least 99% accuracy because, with zero dropout, key bias cancels in the attention softmax and value bias is representable by the output-projection bias.
change: Replace the combined QKV bias with a learned query-only bias.
mechanism: Eliminate redundant attention biases
evidence_used: Narrowing `d_ff` from 12 to 8 reduced accuracy to 71.49%, showing that nonlinear width is capacity-critical; the original `d_ff=12` model reached 99.96%, motivating an exact attention reparameterization instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all QKV biases will reduce parameters from 1644 to 1620 while retaining at least 99% accuracy, because the remaining query bias is representable through the pre-attention LayerNorm bias and query projection; induced key and value offsets are redundant.
change: Make the combined QKV projection bias-free.
mechanism: LayerNorm-absorbed query offset
evidence_used: The query-only-bias reference achieved 100% accuracy with 1628 parameters, establishing that key and value biases are unnecessary and motivating the isolated removal of the remaining 8-parameter query bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7742, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the proven query-only attention bias while making the pre-MLP LayerNorm non-affine will reduce parameters from 1628 to 1612 and maintain at least 99% accuracy, because its scale and bias can be learned equivalently by the immediately following `fc1` weight and bias.
change: Use bias-free QKV with an explicit learned query bias, and remove the redundant 16-parameter affine transform from `ln2`.
mechanism: Downstream absorption of pre-MLP normalization affine parameters
evidence_used: Query-only attention bias achieved 100% accuracy with 1628 parameters, while narrowing `d_ff` caused accuracy to fall to 71.49%; this motivates preserving MLP width and removing parameters that are functionally absorbable by its first projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5917, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: The query-only attention design with `ln2` bias removed will use 1620 parameters and retain at least 99% accuracy because `fc1.bias` can absorb the LayerNorm offset while the learned LayerNorm scale remains available.
change: Apply the proven query-only QKV bias parameterization and remove only the eight-parameter bias from the pre-MLP LayerNorm.
mechanism: Downstream absorption of the pre-MLP normalization bias
evidence_used: Query-only attention reached 100% accuracy with 1628 parameters. Removing all `ln2` affine parameters failed at 59.17%, motivating the narrower test that preserves its potentially capacity-critical learned scale.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0855, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the proven query bias across both attention heads will reduce parameters from 1628 to 1624 while retaining at least 99% accuracy, because both heads process the same digit-addition structure and still receive a learned four-dimensional query offset.
change: Remove the combined QKV bias and introduce one head-dimension query bias broadcast across both attention heads.
mechanism: Head-shared learned query offset
evidence_used: Query-only bias achieved 100% accuracy with 1628 parameters, whereas removing it entirely fell to 77.42%; this motivates preserving its function while testing cross-head parameter sharing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6960999999999999, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the proven query-only-bias design, restricting each positional vector to the zero-mean feature subspace will remove `INPUT_LEN` additional parameters while maintaining at least 99% accuracy, because with zero dropout any omitted all-ones positional component is invisible through the pre-normalized residual stack and final LayerNorm.
change: Apply query-only attention bias and represent positional embeddings with `d_model - 1` learned orthonormal-basis coefficients per position.
mechanism: LayerNorm-quotient positional embeddings
evidence_used: Query-only attention achieved 100% accuracy with 1628 parameters. Unlike the failed removals of query bias, LayerNorm affine terms, or MLP width, this change removes only a LayerNorm-invariant positional degree of freedom while preserving the model’s function class.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified query-only attention and quotient positional embeddings with zero-mean attention and MLP residual outputs will reduce parameters from 1605 to 1583 while retaining at least 99% accuracy, because each removed all-ones output component is erased by subsequent LayerNorms when dropout is zero.
change: Apply the 1605-parameter reference design, then parameterize the attention output projection and MLP second projection in the seven-dimensional zero-mean feature subspace.
mechanism: LayerNorm-quotient residual projections
evidence_used: The quotient-position design achieved 99.91% accuracy with 1605 parameters, while reductions affecting query bias, LayerNorm capacity, or MLP width failed; this motivates removing only additional LayerNorm-invariant degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower parameters from 1583 to 1567 while retaining at least 99% accuracy, because it preserves nearly all nonlinear capacity of the verified 99.90%-accurate design.
change: Change the default feed-forward width from 12 to 11.
mechanism: Single-neuron feed-forward capacity reduction
evidence_used: The current `d_ff=12` quotient-residual design achieved 99.90% accuracy, while the larger reduction to `d_ff=8` failed at 71.49%; testing the smallest possible width reduction is the most informative next capacity-boundary experiment.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1567, "training_steps": 4999}

RECENT RESULT
hypothesis: The verified 1,567-parameter quotient-residual, `d_ff=11` transformer will retain at least 99% accuracy when its full-rank tied token matrix is replaced by a learned rank-seven factorization, reducing the model to 1,509 parameters; this challenges the prior assumption that token representation and decoding require all eight hidden dimensions.
change: Apply the verified quotient residual projections and `d_ff=11` configuration, then replace the tied 114×8 embedding/head with learned 114×7 token codes and a learned 7×8 projection used for both input embedding and output logits.
mechanism: Rank-seven learned lexical subspace
evidence_used: Reference Design 3 achieved 99.95% accuracy with 1,567 parameters, establishing that the quotient-residual architecture and eleven-unit MLP are viable. Its final normalized state varies on a seven-dimensional manifold, making a learned rank-seven lexical subspace a plausible, clean test of whether the full-rank tied vocabulary map is the remaining overprovisioned component.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1509, "training_steps": 4999}



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
