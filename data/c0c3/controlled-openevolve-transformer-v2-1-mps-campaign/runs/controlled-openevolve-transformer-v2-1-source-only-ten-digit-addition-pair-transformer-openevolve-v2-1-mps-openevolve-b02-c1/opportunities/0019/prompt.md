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
verified_results: {"accuracy": 0.9982, "parameters": 1576, "training_steps": 4999}
prior_hypothesis: Removing the unobservable all-ones component from every positional embedding and from the MLP output bias will reduce the model from 1,600 to 1,576 parameters while preserving at least 99% accuracy, because all subsequent pre-LayerNorm computations and the final LayerNorm are invariant to per-position scalar residual shifts.

## Recent verification evidence

RECENT RESULT
hypothesis: Because non-affine LayerNorm outputs sum to zero across the eight residual dimensions, the MLP input projection needs only seven coordinates; reducing `fc1` by 12 parameters to produce a 1,600-parameter model will preserve at least 99% accuracy.
change: Replace the MLP’s eight-input projection with a seven-input projection and omit the linearly dependent final normalized coordinate.
mechanism: Zero-mean LayerNorm coordinate elimination
evidence_used: Making the MLP-preceding LayerNorm non-affine retained 99.92% accuracy at 1,612 parameters, supporting a further exact reparameterization at the same interface rather than compressing positional or token representations, which failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Making the attention-preceding LayerNorm non-affine and omitting its linearly dependent final normalized coordinate from QKV projection will reduce the model from 1,600 to 1,560 parameters while preserving at least 99% accuracy.
change: Remove the first LayerNorm’s 16 affine parameters and reduce the QKV projection from eight to seven input coordinates, retaining the learned causal attention computation and residual width.
mechanism: Attention pre-normalization affine absorption and zero-mean coordinate elimination
evidence_used: The analogous non-affine MLP LayerNorm retained 99.92% accuracy, and eliminating its dependent normalized coordinate then reached 99.99% at 1,600 parameters; this motivates applying the same exact zero-mean reparameterization to the attention input.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7779, "parameters": 1560, "training_steps": 4999}

RECENT RESULT
hypothesis: Preserving the learned first LayerNorm affine transform while omitting one dependent attention-input coordinate and adding an independent value bias will reduce the model from 1,600 to 1,584 parameters while retaining at least 99% accuracy.
change: Reduce QKV input width from eight to seven coordinates and replace the shared value/output bias role with a dedicated learned value bias, while retaining the projection’s output bias.
mechanism: Affine-hyperplane attention coordinate elimination
evidence_used: Seven-coordinate projection after non-affine LayerNorm failed at 77.79%, but prior results showed that losing the value-bias pathway harms accuracy and that restoring it recovers 99.99%; retaining the LayerNorm affine parameters and explicitly preserving both bias pathways isolates the analytically redundant coordinate without repeating those removals.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7306999999999999, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the 12-channel GELU MLP with a 7-channel learned GLU will retain at least 99% accuracy while reducing the model from 1,600 to 1,576 parameters, because multiplicative gating can model conditional digit interactions more efficiently than independent GELU features.
change: Replace the additive GELU MLP with a narrower sigmoid-gated linear unit and set its hidden width to seven, while preserving the successful embeddings, positional representation, normalization, and attention paths.
mechanism: Multiplicatively gated nonlinear mixer
evidence_used: The 1,600-parameter model reached 99.99%, while compressing token, positional, or attention representations failed badly. This challenges the shared assumption that a conventional 12-feature GELU basis is the most parameter-efficient nonlinear computation without disturbing those load-bearing representations.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.386, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the unobservable all-ones component from every positional embedding and from the MLP output bias will reduce the model from 1,600 to 1,576 parameters while preserving at least 99% accuracy, because all subsequent pre-LayerNorm computations and the final LayerNorm are invariant to per-position scalar residual shifts.
change: Store seven learned coordinates per positional embedding and seven learned coordinates for the MLP output bias, padding each with a fixed zero eighth coordinate in the residual stream.
mechanism: Residual-stream common-mode gauge fixing
evidence_used: The seven-coordinate non-affine MLP input reached 99.99%, while reducing positional embeddings to four dimensions failed at 73.99%; this patch retains all seven observable positional contrast dimensions and removes only exact LayerNorm-invariant common-mode degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the attention and MLP output projections to seven learned residual coordinates will reduce parameters from 1,576 to 1,556 while retaining at least 99% accuracy, because their omitted eighth-coordinate contributions are per-position scalar residual shifts removed by subsequent LayerNorms.
change: Make both residual-branch output projections produce seven coordinates and zero-pad the eighth, while retaining the attention projection bias as a full-width shared value/output bias.
mechanism: Residual-output common-mode gauge fixing
evidence_used: Seven-coordinate positional embeddings and MLP output bias already achieved 99.82% accuracy at 1,576 parameters by removing the same LayerNorm-invariant common mode; this extends that successful exact gauge fixing to the projections generating those residual updates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7243999999999999, "parameters": 1556, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the first LayerNorm’s learned scale-and-shift with non-affine normalization plus a learned shift will reduce the model from 1,576 to 1,568 parameters while retaining at least 99% accuracy, because its scale can be absorbed exactly into the full-width QKV projection.
change: Remove only the eight learned `ln1` scale parameters while preserving its learned shift, full eight-coordinate attention input, and all attention bias pathways.
mechanism: Attention pre-normalization scale absorption
evidence_used: Removing both `ln1` affine vectors together with an attention-input coordinate failed at 77.79%, so it did not isolate affine removal; the analogous MLP LayerNorm affine absorption succeeded, motivating a conservative test of only the exactly redundant attention scale.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7548, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing only the MLP output projection from eight to seven learned coordinates will produce a 1,564-parameter model with at least 99% accuracy, because the omitted coordinate can be subtracted from every MLP output coordinate as a LayerNorm-invisible common-mode residual shift.
change: Make `fc2` produce seven coordinates and zero-pad the eighth, while leaving the previously implicated attention output projection and shared value/output bias unchanged.
mechanism: Isolated MLP residual-output gauge fixing
evidence_used: Seven-coordinate positional embeddings and MLP output bias reached 99.82%. The combined attention-and-MLP output reduction failed at 72.44%, but the attention projection bias is coupled to the value bias; isolating the independently gauge-equivalent MLP projection tests the supported invariance without disturbing that load-bearing attention pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9333, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln1` scale coordinate at one while retaining the other seven learned scales will reduce the model from 1,576 to 1,575 parameters and maintain at least 99% accuracy, because that scale can be absorbed into the corresponding QKV input-weight column and LayerNorm bias.
change: Replace the first LayerNorm’s eight learned scales with seven learned scales plus one fixed unit scale, while retaining all eight learned shifts and the full-width attention input.
mechanism: Single-coordinate LayerNorm scale gauge fixing
evidence_used: Removing all eight `ln1` scales fell to 75.48%, showing attention optimization is sensitive to wholesale scale removal; this conservative patch fixes only one analytically absorbable coordinate and preserves seven adaptive scales.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.748, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing the MLP output in an orthonormal seven-dimensional zero-mean basis will produce a 1,564-parameter model with at least 99% accuracy, because it removes only the LayerNorm-invisible common mode while preserving isotropic initialization and well-conditioned gradients.
change: Make `fc2` produce seven coordinates, including its bias, then project them through a fixed orthonormal basis spanning the eight-dimensional zero-mean residual subspace.
mechanism: Orthonormal residual gauge coordinates
evidence_used: The zero-padded seven-coordinate MLP projection reached only 93.33%, despite being function-class equivalent modulo LayerNorm; its induced coordinate metric has an eightfold weak direction. The current 1,576-parameter gauge-fixed model reached 99.82%, motivating the same exact 12-parameter reduction with an orthonormal rather than asymmetric gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1275, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the LayerNorm-invisible common-mode component of the 12 MLP output-weight columns as the 12 hidden biases will reduce parameters from 1,576 to 1,564 while maintaining at least 99% accuracy.
change: Remove the independent `fc1` bias, derive an equally conditioned hidden bias from the normalized row-sum of `fc2.weight`, and center that weight before producing the residual update.
mechanism: Orthogonal reuse of residual common-mode weights
evidence_used: Direct seven-coordinate MLP output projections failed at 93.33% and 12.75% despite common-mode redundancy, suggesting harmful optimization geometry; the successful 1,576-parameter design confirms that residual common modes are unobservable. This patch preserves the full output-weight tensor and repurposes its redundant orthogonal component instead of deleting it.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7172, "parameters": 1564, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only one LayerNorm-invisible common-mode weight from one MLP output column will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because the other 11 columns keep their original optimization geometry.
change: Split `fc2` into an eight-output projection for 11 hidden features and a seven-coordinate projection vector for the final feature, padding its eighth residual coordinate with zero.
mechanism: Single-column residual common-mode gauge fixing
evidence_used: Removing all 12 MLP output common modes at once reached only 93.33% or 12.75%, despite exact functional redundancy, indicating an optimization-geometry problem; the 1,576-parameter model reached 99.82%, motivating the smallest possible isolated gauge reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2712, "parameters": 1575, "training_steps": 4999}



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
