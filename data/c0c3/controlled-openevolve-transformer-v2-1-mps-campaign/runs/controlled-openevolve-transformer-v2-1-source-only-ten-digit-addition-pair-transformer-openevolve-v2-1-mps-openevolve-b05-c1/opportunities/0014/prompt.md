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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1596, "training_steps": 4999}
prior_hypothesis: Removing both block LayerNorm scale vectors will reduce parameters from 1,612 to 1,596 while retaining at least 99% accuracy, because each scale is immediately followed by a learned linear map that can absorb the same per-channel scaling.

## Recent verification evidence

RECENT RESULT
hypothesis: Setting `d_ff` to 11 will reduce the model from 1,644 to 1,627 learned parameters while preserving at least 99% accuracy.
change: Change only the default feed-forward width from 12 to 11, preserving the successful architecture and training configuration otherwise.
mechanism: Feed-forward width boundary search
evidence_used: The 1,644-parameter `d_ff=12` design achieved 99.96% accuracy, while `d_ff=10` collapsed to 11.92%; testing the intermediate width directly probes the apparent capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7336, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing two redundant LayerNorm biases and replacing the attention QKV bias with only a query bias will reduce parameters from 1,644 to 1,612 while retaining at least 99% accuracy.
change: Keep `d_ff=12`, remove both block LayerNorm biases, and retain only the functionally relevant query component of the QKV bias; constant key offsets cancel in softmax and constant value offsets are representable by the attention output bias.
mechanism: Function-preserving affine redundancy removal
evidence_used: The 1,644-parameter `d_ff=12` model achieved 99.96%, whereas widths 11 and 10 failed; this motivates preserving the successful hidden width and reducing affine parameters that do not expand the attention function class.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing both block LayerNorm scale vectors will reduce parameters from 1,612 to 1,596 while retaining at least 99% accuracy, because each scale is immediately followed by a learned linear map that can absorb the same per-channel scaling.
change: Make both block LayerNorms non-affine while preserving the successful `d_ff=12` architecture and training configuration.
mechanism: Pre-linear LayerNorm scale absorption
evidence_used: The 1,612-parameter affine-redundancy design achieved 99.96% accuracy, while narrowing `d_ff` failed; this motivates another function-class-preserving affine reduction instead of removing hidden capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the QKV and MLP-input linear maps on the seven-dimensional mean-zero subspace produced by non-affine LayerNorm will reduce parameters from 1,596 to 1,560 while retaining at least 99% accuracy.
change: Add a linear layer whose fixed orthonormal basis removes the unobservable all-ones input-weight direction, then use it for QKV and `fc1`.
mechanism: LayerNorm nullspace parameterization
evidence_used: The 1,596-parameter design reached 99.99% accuracy after making both preceding LayerNorms non-affine; their outputs have zero channel mean, so 24 QKV and 12 `fc1` weight directions are functionally inactive and can be removed without narrowing either learned output space.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9776, "parameters": 1560, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final LayerNorm bias will reduce parameters from 1,596 to 1,588 while retaining at least 99% accuracy.
change: Keep the successful architecture and training schedule unchanged, but disable only the eight-element bias in `ln_f`.
mechanism: Bias-free final normalization
evidence_used: The 1,596-parameter model reached 99.99% after removing affine parameters from both block LayerNorms, while the larger 36-parameter nullspace reparameterization fell to 97.76%; this motivates a conservative normalization-affine reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9336, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the position-zero embedding will reduce the model from 1,596 to 1,588 parameters while retaining at least 99% accuracy, because a shared shift between token and positional embeddings leaves hidden states unchanged and changes tied-head logits only by a class-independent constant.
change: Replace the positional embedding table with a table whose zero-position vector is fixed at zero, and initialize the remaining embeddings with a gauge-equivalent transformation.
mechanism: Additive embedding gauge fixing
evidence_used: The 1,596-parameter model achieved 99.99%, while deleting the capacity-bearing final LayerNorm bias failed at 93.36%; this motivates removing an exact eight-parameter embedding redundancy instead of another affine ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7803, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the attention-output and MLP-output biases to their seven-dimensional mean-zero subspaces will reduce the model from 1,596 to 1,594 parameters while retaining at least 99% accuracy, because their discarded all-ones components are removed by subsequent LayerNorm operations.
change: Add an orthonormally parameterized mean-zero-bias linear layer and use it for the attention projection and MLP output projection.
mechanism: Mean-zero residual-bias gauge fixing
evidence_used: The 1,596-parameter design achieved 99.99% accuracy; unlike the failed 36-parameter LayerNorm-nullspace reduction, this patch removes only two independently unobservable scalar bias directions while preserving initialization and all output dimensions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1333, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the MLP output bias at zero will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because any removed bias value is equivalent to a uniform residual shift that the final LayerNorm cancels.
change: Replace `fc2` with a linear layer that learns seven bias coordinates and fixes the eighth at zero; leave all weights, widths, initialization, training, and decoding unchanged.
mechanism: Final-residual bias gauge anchoring
evidence_used: The 1,596-parameter design achieved 99.99% accuracy, while jointly reparameterizing two residual biases fell to 13.33%; isolating the simplest single scalar gauge tests the same redundancy with minimal optimization disruption.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9251999999999999, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing independent position vectors with a fixed sinusoidal basis and eight learned channel scales will reduce parameters by `8 * (max_seq_len - 1)` while retaining at least 99% accuracy, because addition’s position relationships are regular and can be represented compositionally without narrowing the successful attention or MLP.
change: Remove the learned positional embedding table; encode absolute positions with generic sinusoidal features whose channel amplitudes are learned from fresh initialization.
mechanism: Shared sinusoidal absolute-position manifold
evidence_used: The 1,596-parameter `d_model=8, d_ff=12` model reached 99.99%, while narrower MLPs and small affine gauge changes failed. This challenges the shared assumption that every position needs an independent learned vector while preserving the demonstrated load-bearing feature capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1420, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing only the MLP-input map onto the seven-dimensional mean-zero LayerNorm subspace will reduce parameters from 1,596 to 1,584 while retaining at least 99% accuracy.
change: Add an orthonormal mean-zero input projection and use it only for `fc1`, preserving its bias, output width, initialization scale, and full function class on normalized inputs.
mechanism: Isolated LayerNorm-nullspace compression
evidence_used: Jointly compressing QKV and `fc1` reached 97.76%, close to the threshold, while the unchanged 1,596-parameter model reached 99.99%; isolating the smaller 12-parameter `fc1` reduction tests the exact redundancy with less optimization disruption.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8024, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate at zero will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because an orthogonal rotation of that head’s query and key coordinates can place any query-bias vector in the resulting three-dimensional subspace without changing attention logits.
change: Learn seven query-bias coordinates and pad the final coordinate with zero during attention; leave initialization, widths, training, and decoding unchanged.
mechanism: Single-head query-basis gauge anchoring
evidence_used: The 1,596-parameter design achieved 99.99% accuracy. Although anchoring one MLP-output bias coordinate failed, this tests a distinct exact symmetry internal to one attention head and preserves the current zero initialization with only a one-scalar reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2903, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing only one QKV input-weight row will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because non-affine LayerNorm makes one weight direction unobservable and gauge-equivalent initialization limits optimization disruption to a single row.
change: Replace the QKV linear map with an equivalent learned map whose first output row uses seven anchored weights while its other 23 rows remain unchanged.
mechanism: Single-row LayerNorm nullspace gauge fixing
evidence_used: Compressing all 36 nullspace directions reached 97.76%, close to the threshold, while the 1,596-parameter model reached 99.99%; testing one of those exact redundancies is the smallest informative step toward locating the safe compression boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1328, "parameters": 1595, "training_steps": 4999}



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
