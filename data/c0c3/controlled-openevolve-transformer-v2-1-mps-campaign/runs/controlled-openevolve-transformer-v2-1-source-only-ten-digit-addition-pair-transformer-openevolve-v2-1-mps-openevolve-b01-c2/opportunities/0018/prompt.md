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
verified_results: {"accuracy": 0.9997, "parameters": 1609, "training_steps": 4999}
prior_hypothesis: Removing one `fc1` input-weight coordinate from the verified distributed-anchor design will yield 1,609 parameters and at least 99% accuracy, because bias-free LayerNorm outputs have zero mean and the omitted coefficient is exactly absorbable by the other coefficients in that output row; gauge-adjusted initialization preserves the original model’s initial function and RNG stream.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1611, "training_steps": 4999}
prior_hypothesis: Fixing one `ln2` scale coordinate at its initial value while retaining the other seven learned scales will produce a 1,611-parameter model with at least 99% accuracy, because the following MLP input weight can absorb that coordinate’s scale without removing the adaptive normalization pathway.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9997, "parameters": 1610, "training_steps": 4999}
prior_hypothesis: Fixing one scale coordinate in each pre-sublayer LayerNorm will produce a 1,610-parameter model with at least 99% accuracy, because each normalization retains seven adaptive scales and its downstream learned projection can absorb the fixed coordinate.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the eight softmax-invariant key-bias parameters while retaining the baseline’s constructor RNG stream and single fused linear bias addition will achieve at least 99% accuracy in 5,000 steps with 1,636 parameters.
change: Preserve fused QKV construction, replace its bias with learned query/value components, synthesize the zero key component, and pass the complete bias through one fused `F.linear` call.
mechanism: Fused zero-key-bias reparameterization
evidence_used: The 1,644-parameter baseline achieved 99.96%, while the prior 1,636-parameter implementation reached 97.79% after applying biases separately; preserving the baseline fused bias-add computation targets that remaining numerical difference.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing both the softmax-invariant key bias and the value bias will reduce the model to 1,628 parameters while retaining at least 99% accuracy, because attention weights sum to one, making the value bias a position-independent offset that the learned output-projection bias can represent.
change: Preserve the baseline fused QKV construction and single fused linear call, but retain only the query-bias parameters and synthesize zero key and value bias components.
mechanism: Output-bias absorption of constant value bias
evidence_used: The initialization-preserving fused zero-key-bias design achieved 99.89% accuracy with 1,636 parameters; extending the same successful fused reparameterization to the value bias targets another mathematically redundant eight-parameter component without narrowing the failed MLP bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining only the effective query bias and removing both pre-sublayer LayerNorm biases will reduce the model to 1,612 parameters while maintaining at least 99% accuracy, because those normalization offsets are representable by the downstream query, attention-output, and MLP biases.
change: Apply the verified query-only fused-QKV reparameterization and disable the biases in `ln1` and `ln2`.
mechanism: Pre-normalization bias absorption
evidence_used: The 1,628-parameter query-only-bias design achieved 99.97% accuracy; its margin motivates removing 16 additional parameters that are initialized to zero and functionally absorbable by existing downstream biases.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing both pre-sublayer LayerNorm scale vectors will reduce the model from 1,612 to 1,596 parameters while maintaining at least 99% accuracy, because each scale can be represented exactly by the following QKV or MLP input-projection weights.
change: Make `ln1` and `ln2` entirely parameter-free while retaining their normalization behavior.
mechanism: Downstream absorption of pre-normalization scales
evidence_used: The 1,612-parameter design reached 99.95% accuracy after removing both pre-sublayer normalization biases; the remaining scales are more directly redundant because they multiply normalized features immediately before learned linear projections.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7963, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only `ln1`’s scale while retaining `ln2`’s scale will reduce the model to 1,604 parameters and maintain at least 99% accuracy, because the attention QKV weights can absorb `ln1`’s multiplicative scale while the MLP retains its normalization scale.
change: Disable all affine parameters in `ln1` and remove only the bias from `ln2`.
mechanism: Isolated attention-side normalization scale removal
evidence_used: Removing both LayerNorm biases achieved 99.95% with 1,612 parameters, while removing both scales simultaneously fell to 79.63%; isolating one scale tests whether the failure came from eliminating both optimization pathways together.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2043, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate at its initial value while retaining the other seven learned scales will produce a 1,611-parameter model with at least 99% accuracy, because the following MLP input weight can absorb that coordinate’s scale without removing the adaptive normalization pathway.
change: Apply the verified query-only QKV bias and bias-free pre-LayerNorm design, then replace `ln2` with a bias-free LayerNorm having seven learned scales and one fixed unit scale.
mechanism: Single-coordinate LayerNorm scale gauge fixing
evidence_used: The 1,612-parameter query-only, bias-free LayerNorm design achieved 99.95%; removing all scales failed, so anchoring only one MLP-side scale is a conservative exact reparameterization that preserves seven adaptive scales.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second `ln2` scale coordinate at one will reduce the model from 1,611 to 1,610 learned parameters while retaining at least 99% accuracy, because the successful first anchor achieved 100% accuracy and the following MLP input weights can absorb both fixed coordinate scales.
change: Retain six learned `ln2` scales and synthesize two fixed unit scales.
mechanism: Second-coordinate LayerNorm scale gauge fixing
evidence_used: The 1,611-parameter model with one fixed `ln2` scale coordinate achieved 100% accuracy, while removing all normalization scales failed; extending the successful reduction by exactly one coordinate is the most conservative capacity probe.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7340000000000001, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified single-coordinate `ln2` scale anchor with a fixed zero coordinate in the attention output bias will produce a 1,610-parameter model with at least 99% accuracy after 5,000 steps.
change: Retain seven learned `ln2` scales, retain seven learned attention projection-bias coordinates, and synthesize fixed unit-scale and zero-bias coordinates during forward passes.
mechanism: Residual common-mode bias gauge fixing
evidence_used: The one-coordinate `ln2` anchor achieved 100% accuracy with 1,611 parameters, while a second scale anchor fell to 73.4%; removing an orthogonal additive gauge is therefore better motivated. A uniform shift in the attention projection bias is invisible to `ln2` and can be absorbed exactly by the MLP output bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9777, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one scale coordinate in each pre-sublayer LayerNorm will produce a 1,610-parameter model with at least 99% accuracy, because each normalization retains seven adaptive scales and its downstream learned projection can absorb the fixed coordinate.
change: Replace both biased LayerNorms with bias-free LayerNorms that learn seven scale coordinates and synthesize one fixed unit scale.
mechanism: Distributed single-coordinate LayerNorm scale anchoring
evidence_used: A single anchored `ln2` coordinate achieved 100% accuracy with 1,611 parameters, while anchoring a second `ln2` coordinate failed at 73.4%; distributing the second anchor to `ln1` preserves seven adaptive coordinates in both pathways. Removing all `ln1` scales failed, motivating this one-coordinate reduction instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the MLP output bias at zero will reduce the verified 1,610-parameter model to 1,609 parameters while retaining at least 99% accuracy, because any uniform component of that bias is erased by the final LayerNorm and the remaining seven coordinates still span all functionally relevant bias differences.
change: Retain seven learned `fc2` bias coordinates, synthesize one fixed zero coordinate, and preserve the existing linear weight shape and initialization path.
mechanism: Final-residual common-mode bias gauge fixing
evidence_used: The distributed LayerNorm anchoring design achieved 99.97% accuracy with 1,610 parameters, while removing a second normalization scale caused a large failure; this motivates removing one exact additive gauge from the final residual branch instead of further reducing normalization capacity. The earlier attention-output bias anchor reached 97.77%, so targeting the later MLP output bias avoids perturbing activations consumed by another learned sublayer.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6988, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified distributed LayerNorm anchors with one fixed-zero query-bias coordinate will yield a 1,609-parameter model with at least 99% accuracy, because an invertible query/key basis change within the affected attention head can eliminate that bias coordinate without changing attention logits.
change: Anchor one scale coordinate in `ln1` as in the qualified 1,610-parameter design, then retain seven learned query-bias coordinates and synthesize the eighth as zero.
mechanism: Single-coordinate query-bias head-basis gauge fixing
evidence_used: Distributed LayerNorm anchoring achieved 99.97% with 1,610 parameters. The failed 1,609-parameter `fc2` bias anchor targeted the residual pathway; this instead removes an independently redundant attention-head basis coordinate while preserving its zero-initialized value.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6092, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one `fc1` input-weight coordinate from the verified distributed-anchor design will yield 1,609 parameters and at least 99% accuracy, because bias-free LayerNorm outputs have zero mean and the omitted coefficient is exactly absorbable by the other coefficients in that output row; gauge-adjusted initialization preserves the original model’s initial function and RNG stream.
change: Anchor one scale coordinate in each pre-sublayer LayerNorm, then replace `fc1` with a linear layer that learns all but one weight, synthesizes the omitted weight as zero, and initializes the retained row coefficients to an equivalent gauge.
mechanism: LayerNorm-nullspace weight gauge fixing
evidence_used: The distributed LayerNorm-anchor design achieved 99.97% with 1,610 parameters. Both prior 1,609-parameter attempts altered sensitive additive pathways and failed, motivating a different one-parameter reduction based on the exact zero-mean nullspace immediately downstream of `ln2`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1609, "training_steps": 4999}



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
