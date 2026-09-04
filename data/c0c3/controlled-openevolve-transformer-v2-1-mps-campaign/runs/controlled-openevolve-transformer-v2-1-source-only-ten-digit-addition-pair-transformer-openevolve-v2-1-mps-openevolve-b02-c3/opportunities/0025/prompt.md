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
verified_results: {"accuracy": 0.9997, "parameters": 1325, "training_steps": 4999}
prior_hypothesis: Removing the seven token-subspace tilt parameters from the qualified relative-position design will yield a 1,325-parameter transformer with at least 99% accuracy, because every residual and positional contribution is zero-mean, LayerNorm removes the token embedding’s all-ones component, and the tied classifier receives a zero-mean state.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1332, "training_steps": 4999}
prior_hypothesis: Fixing the pre-attention LayerNorm scale to one will reduce the qualified model from 1,340 to 1,332 parameters while retaining at least 99% accuracy, because its learned scale can be absorbed into the unrestricted QKV input columns without reducing the attention function class.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9959, "parameters": 1293, "training_steps": 4999}
prior_hypothesis: Replacing the attention QKV map’s eight-dimensional input with its seven-dimensional zero-mean quotient will reduce the verified model from 1,317 to 1,293 parameters while retaining at least 99% accuracy, because `GaugeFixedLayerNorm` always produces zero-mean vectors and the removed QKV weight direction therefore receives neither activation nor gradient.

## Recent verification evidence

RECENT RESULT
hypothesis: Representing the rank-seven token codes in a fixed zero-sum vocabulary basis will reduce the qualified 1,460-parameter model to 1,453 parameters while retaining at least 99% accuracy, because it removes the same seven-dimensional translation redundancy as anchoring one token but keeps every vocabulary entry symmetrically trainable.
change: Apply the qualified gauge-fixed rank-seven lexical factorization, then parameterize its token-code matrix with 113 orthonormal zero-sum vocabulary coefficients instead of 114 unconstrained rows.
mechanism: Symmetric vocabulary-translation gauge fixing
evidence_used: The gauge-fixed rank-seven design achieved 99.96% accuracy with 1,460 parameters. Anchoring token zero reduced accuracy to 96.83%, motivating a symmetric gauge choice that removes the redundant shared translation without freezing or privileging any token.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4953, "parameters": 1453, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified 1,460-parameter gauge-fixed rank-seven model from `d_ff=11` to `d_ff=10` will produce a 1,444-parameter transformer while retaining at least 99% accuracy, because the immediately preceding one-unit reduction preserved 99.95% accuracy and this tests the next feed-forward capacity boundary without altering attention or lexical rank.
change: Apply the qualified quotient-position, quotient-residual, gauge-fixed rank-seven token architecture and reduce the default feed-forward width to ten.
mechanism: Gauge-fixed lexical factorization with a ten-unit quotient MLP
evidence_used: The 1,460-parameter `d_ff=11` design reached 99.96% accuracy, while the earlier `d_ff=12` to `d_ff=11` reduction also retained 99.95%; the failure at `d_ff=8` shows a lower boundary exists and makes the single-step `d_ff=10` reduction the most informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1439, "parameters": 1444, "training_steps": 4999}

RECENT RESULT
hypothesis: Restricting the final LayerNorm bias to the fixed seven-dimensional basis will reduce parameters from 1,460 to 1,459 while retaining at least 99% accuracy, because the rank-seven lexical projection cannot observe its remaining bias direction.
change: Make the final LayerNorm bias-free and add a seven-parameter post-normalization bias expressed in the lexical basis.
mechanism: Final LayerNorm null-direction quotient bias
evidence_used: The current gauge-fixed rank-seven model achieved 99.96% accuracy. Unlike the failed removal of `ln2` bias, this preserves every output-relevant bias degree of freedom because the lexical projection satisfies `WᵀB = I`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1459, "training_steps": 4999}

RECENT RESULT
hypothesis: The qualified 1,459-parameter transformer will retain at least 99% accuracy with 1,458 parameters because the all-ones component of the pre-attention LayerNorm bias is functionally absorbable by the learned query and attention-output biases.
change: Apply the qualified gauge-fixed rank-seven lexical design, then restrict the pre-attention LayerNorm bias to the seven-dimensional zero-mean basis while retaining its learned scale.
mechanism: Single-direction pre-attention LayerNorm bias gauge fixing
evidence_used: The 1,459-parameter final-LayerNorm quotient achieved 99.93% accuracy; unlike the failed removal of all pre-MLP LayerNorm bias parameters, this removes only one redundant direction and preserves seven learned pre-attention bias degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1458, "training_steps": 4999}

RECENT RESULT
hypothesis: Restricting the qualified 1,458-parameter model’s pre-attention LayerNorm bias from seven basis coefficients to six will yield 1,457 parameters while retaining at least 99% accuracy, because the omitted constant input shift is absorbable through query bias, key-softmax invariance, and attention-output bias.
change: Apply the verified gauge-fixed rank-seven lexical, quotient-position, quotient-residual, `d_ff=11` architecture, while retaining only six learned pre-attention LayerNorm bias directions.
mechanism: Second-direction pre-attention LayerNorm bias gauge fixing
evidence_used: The 1,458-parameter design achieved 99.94% after removing one pre-attention LayerNorm bias direction. Removing one additional direction is the narrowest continuation of that successful mechanism, while preserving substantially more bias capacity than the failed complete LayerNorm-bias removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1457, "training_steps": 4999}

RECENT RESULT
hypothesis: Restricting the verified 1,457-parameter model’s pre-attention LayerNorm bias from six basis coefficients to five will yield 1,456 parameters while retaining at least 99% accuracy, because the removed constant attention-input shift can be absorbed by query bias, key-softmax invariance, and attention-output bias.
change: Remove one additional learned pre-attention LayerNorm bias direction while preserving five learned zero-mean bias coefficients.
mechanism: Third-direction pre-attention LayerNorm bias gauge fixing
evidence_used: Consecutive one-direction reductions from 1,459 to 1,458 and then 1,457 parameters retained 99.94% and 99.95% accuracy; another single-direction reduction is the narrowest test of the same successful mechanism, while the failed complete bias removal argues against a larger step.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1456, "training_steps": 4999}

RECENT RESULT
hypothesis: Restricting the verified 1,456-parameter model’s pre-attention LayerNorm bias from five learned directions to four will produce a 1,455-parameter transformer while retaining at least 99% accuracy.
change: Parameterize the pre-attention LayerNorm bias with four basis coefficients instead of the current seven, advancing one parameter beyond the qualified five-coefficient design.
mechanism: Fourth-direction pre-attention LayerNorm bias reduction
evidence_used: Successive reductions to seven, six, and five pre-attention bias coefficients achieved 99.94%, 99.95%, and 99.91% accuracy respectively; reducing one additional direction is the narrowest continuation of this demonstrated trend.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.705, "parameters": 1455, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the seven-dimensional absolute-position lookup with a fixed sinusoidal code and learned per-head relative-distance attention biases will reduce the qualified 1,456-parameter architecture to 1,340 parameters while retaining at least 99% accuracy, because addition’s reusable positional routing can be learned by relative offsets rather than independent vectors at every position.
change: Use the qualified five-coefficient pre-attention LayerNorm, replace the 161-parameter positional embedding with a one-parameter-scaled sinusoidal encoding, and add 44 learned relative-distance attention biases with the softmax-invariant zero-distance coefficient fixed to zero.
mechanism: Gauge-fixed relative-distance attention routing
evidence_used: The 1,456-parameter design achieved 99.91% accuracy, while reducing feed-forward width failed sharply; all qualified designs still assume a full 23-by-7 learned positional table. This patch preserves the verified nonlinear and lexical capacities while directly challenging that shared positional-representation assumption.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1340, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the pre-attention LayerNorm scale to one will reduce the qualified model from 1,340 to 1,332 parameters while retaining at least 99% accuracy, because its learned scale can be absorbed into the unrestricted QKV input columns without reducing the attention function class.
change: Disable the pre-attention LayerNorm’s eight learned scale parameters while preserving all five external bias coefficients.
mechanism: Pre-attention LayerNorm scale gauge fixing
evidence_used: The current 1,340-parameter model achieved 100% accuracy, and prior successful pre-attention gauge reductions established that query bias, key-softmax invariance, and attention-output bias absorb constant effects; fixing the scale also preserves its initial value and leaves QKV free to learn the equivalent column scaling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1332, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the seven token-subspace tilt parameters from the qualified relative-position design will yield a 1,325-parameter transformer with at least 99% accuracy, because every residual and positional contribution is zero-mean, LayerNorm removes the token embedding’s all-ones component, and the tied classifier receives a zero-mean state.
change: Apply the verified 1,332-parameter sinusoidal/relative-distance design, then fix the token projection to its seven-dimensional zero-mean basis instead of learning seven functionally unobservable tilt parameters.
mechanism: Quotient-null token-subspace elimination
evidence_used: Reference Design 2 achieved 99.96% accuracy with 1,332 parameters. Its token tilt starts at zero, while its fixed zero-mean positional encoding, quotient residual projections, and final LayerNorm make the tilt’s all-ones component unobservable throughout the model.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1325, "training_steps": 4999}

RECENT RESULT
hypothesis: The qualified 1,325-parameter relative-position transformer will retain at least 99% accuracy with 1,317 parameters because the pre-MLP LayerNorm scale can be absorbed exactly into `fc1`’s unrestricted input columns, while retaining the LayerNorm bias that prior evidence indicates is optimization-critical.
change: Apply the qualified 1,325-parameter fixed-token-basis, sinusoidal/relative-distance architecture and replace `ln2` with a scale-fixed LayerNorm that retains all eight learned bias parameters.
mechanism: Pre-MLP LayerNorm scale absorption
evidence_used: The 1,325-parameter reference achieved 99.97% accuracy, and fixing the analogous pre-attention LayerNorm scale retained 99.96% accuracy. Unlike the unsuccessful removal of `ln2` bias, this patch preserves that bias and removes only a scale exactly reparameterizable through the following learned affine layer.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1317, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the attention QKV map’s eight-dimensional input with its seven-dimensional zero-mean quotient will reduce the verified model from 1,317 to 1,293 parameters while retaining at least 99% accuracy, because `GaugeFixedLayerNorm` always produces zero-mean vectors and the removed QKV weight direction therefore receives neither activation nor gradient.
change: Add a quotient-input linear layer and use it for the learned QKV projection, removing 24 functionally inactive weights without changing attention behavior or decoding.
mechanism: Zero-mean attention-input quotient
evidence_used: The current 1,317-parameter design achieved 100% accuracy after fixing both pre-attention and pre-MLP LayerNorm scales. Its pre-attention bias is also expressed entirely in a zero-mean basis, making the QKV input’s all-ones direction exactly unobservable; the orthonormal reparameterization preserves the initialized function distribution.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9959, "parameters": 1293, "training_steps": 4999}



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
