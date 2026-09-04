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
verified_results: {"accuracy": 0.9984000000000001, "parameters": 845, "training_steps": 4999}
prior_hypothesis: Restoring the unbiased head’s first-row scale coordinate while fixing both heads’ farthest-lag biases at zero will produce a 845-parameter model with at least 99% accuracy, because it restores the better-conditioned 847-parameter Q/K optimization geometry and removes only two biases governing the single maximum-distance attention edge.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing the twelve-unit GELU MLP with a seven-unit learned GLU will reduce the model from 866 to 847 parameters while retaining at least 99% accuracy, because seven independent output directions span the entire LayerNorm-visible residual quotient and multiplicative gates provide greater nonlinear efficiency per direction.
change: Add a generic gauge-fixed output projection, replace the GELU MLP with seven learned value/gate pairs, update ambient optimization for its columns, and set the default gated width to seven.
mechanism: Rank-complete sigmoid-gated MLP bottleneck
evidence_used: Tying the MLP readout collapsed accuracy to 37.01%, showing that independent output directions are load-bearing; this design preserves seven independent directions—the maximum observable rank after common-mode quotienting—while challenging the assumption that twelve separate GELU features are required.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 847, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero weight coordinate in each of the seven GLU value rows will reduce the model from 847 to 840 learned parameters while retaining at least 99% accuracy, because independently scaling a value row and inversely scaling its output-projection column leaves the learned MLP function unchanged.
change: Gauge-fix the seven GLU value-row scales at their largest initialized coordinates, reconstruct those coordinates from checkpointed buffers, and optimize the remaining weights in ambient space.
mechanism: GLU value-channel scale quotient
evidence_used: The seven-unit GLU achieved 99.94% accuracy at 847 parameters while preserving seven independent output directions; the proposed change retains every direction and removes only seven independent continuous scaling redundancies between value rows and their corresponding output columns.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4512, "parameters": 840, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero coordinate of a value-projection row will reduce the model from 847 to 846 learned parameters while retaining at least 99% accuracy, because inverse scaling of the corresponding attention output-projection column preserves the learned function exactly.
change: Anchor the largest initialized coordinate of the first value row, omit it from trainable coordinates, reconstruct it during forward passes, and optimize the remaining row coordinates in ambient space.
mechanism: Single attention value/output scale quotient
evidence_used: The analogous anchored Q/K scale gauges retained 99.96% and 99.93% accuracy; applying one value/output gauge isolates a single exact redundancy, whereas simultaneously gauge-fixing all seven GLU channels collapsed accuracy to 45.12%.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing one nonzero coefficient of the unbiased head’s first query row will reduce the model from 847 to 846 learned parameters while retaining at least 99% accuracy, because scaling that query channel while inversely scaling its key channel preserves every attention logit.
change: Anchor the largest surviving coefficient of the unbiased head’s first query row, omit it from trainable coordinates, reconstruct it from a checkpointed buffer, and reuse the existing ambient gauge optimizer.
mechanism: Unbiased-head diagonal Q/K scale gauge
evidence_used: The analogous two biased-head query-scale gauges retained 99.96% and 99.93% accuracy; unlike the value/output-scale attempt that timed out, this adds no optimizer entries and extends the already-successful Q/K gauge parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9913, "parameters": 846, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the largest coefficient of the unbiased head’s second query row will reduce the model from 846 to 845 learned parameters while retaining at least 99% accuracy, because inverse scaling of the corresponding key channel preserves every attention logit.
change: Reuse the unbiased-head shear pivot as a fixed query-row scale anchor, omit that coordinate from training, reconstruct it from a checkpointed buffer, and include it in ambient gauge optimization.
mechanism: Unbiased-head second-channel diagonal Q/K scale gauge
evidence_used: The analogous biased-head second-channel scale quotient reduced 867 parameters to 866 while achieving 99.93% accuracy; the current unbiased head retains the same unused diagonal Q/K scaling symmetry after its successful rotation, shear, and first-channel scale gauges.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9841, "parameters": 845, "training_steps": 4999}

RECENT RESULT
hypothesis: Restoring the unbiased head’s first-row scale coordinate while fixing both heads’ farthest-lag biases at zero will produce a 845-parameter model with at least 99% accuracy, because it restores the better-conditioned 847-parameter Q/K optimization geometry and removes only two biases governing the single maximum-distance attention edge.
change: Undo the unbiased-head scale anchor that reduced training margin, then shorten each head’s learned relative-bias vector by one coordinate and reconstruct the unused maximum-lag entries as zeros.
mechanism: Optimization-aware Q/K gauge rollback with farthest-lag edge pruning
evidence_used: The seven-unit GLU reached 99.94% at 847 parameters, whereas adding the first unbiased-head scale gauge reduced accuracy to 99.13% and adding the second fell to 98.41%; this motivates restoring the demonstrated optimizer geometry and seeking the net reduction in the sparsest-used relative-attention coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 845, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the unbiased head’s second-farthest-lag bias at zero will reduce the model from 845 to 844 parameters while retaining at least 99% accuracy, because the successful maximum-lag pruning achieved 99.84% and this removes only one additional sparsely used positional-bias coordinate.
change: Flatten the learned relative-bias coordinates, omit the unbiased head’s final learned coordinate, and reconstruct that second-farthest-lag bias as zero before attention.
mechanism: Asymmetric next-edge relative-bias pruning
evidence_used: Fixing both heads’ maximum-lag biases at zero retained 99.84% accuracy at 845 parameters, indicating substantial margin for testing the immediately adjacent sparse attention edge one head at a time.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing the biased head’s second-farthest-lag bias at zero will reduce the model from 845 to 844 parameters while retaining at least 99% accuracy, because both maximum-lag biases were removed at 99.84% accuracy and this prunes only one additional sparsely used attention edge.
change: Flatten the relative-bias parameters, omit the biased head’s final learned lag coordinate, and reconstruct that coordinate as zero without adding per-step Python loops.
mechanism: Biased-head second-farthest-lag pruning
evidence_used: The 845-parameter model achieved 99.84% after pruning both maximum-lag biases. The complementary unbiased-head next-edge experiment timed out rather than producing negative accuracy evidence, motivating an asymmetric test on the other head with a compact reconstruction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Tying both heads’ second-farthest-lag biases will reduce the model from 845 to 844 parameters while retaining at least 99% accuracy, because it preserves a learned bias for those sparse edges while removing only head-specificity.
change: Shorten each head’s independent relative-bias vector by one coordinate and append one shared learned coordinate before reconstructing the fixed endpoint biases.
mechanism: Cross-head sharing of the second-farthest relative-attention bias
evidence_used: Fixing both maximum-lag biases yielded 99.84% accuracy at 845 parameters. The two second-farthest single-head pruning attempts timed out rather than showing an accuracy failure, so sharing that coordinate is a conservative test that retains more capacity than either zero-pruning proposal.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Combining the previously successful unbiased-head first-channel scale gauge with the current farthest-lag pruning will reduce the model from 845 to 844 parameters while retaining at least 99% accuracy, because each reduction independently exceeded the threshold and this gauge preserves the attention function class exactly.
change: Fix the largest surviving coefficient of the unbiased head’s first query row at initialization, omit it from learned coordinates, reconstruct it during forward passes, and retain ambient-space gauge optimization.
mechanism: Unbiased-head diagonal Q/K scale quotient
evidence_used: The same unbiased-head scale anchor achieved 99.13% at 846 parameters, while the current two-edge relative-bias pruning achieved 99.84% at 845 parameters; composing them tests a known exact gauge without the reconstruction overhead associated with the three timed-out relative-bias experiments.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the seventh independent gate with a variance-preserving learned consensus of the other six gates will reduce the model from 845 to 837 parameters while retaining at least 99% accuracy, because all seven independent value and output directions remain available while only one gating hyperplane is shared.
change: Challenge the old assumption that every observable MLP direction needs an independently parameterized gate; retain seven value/output directions but synthesize the seventh gate from six learned gates, while preserving the verified conceptual initialization stream.
mechanism: Consensus-gated seven-direction MLP
evidence_used: The seven-unit GLU achieved 99.94%, while tying the MLP readout collapsed accuracy to 37.01%; this specifically identifies independent output directions as load-bearing, but provides no evidence that all seven gate projections must also be independent.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Tying only the sixth and seventh GLU gate biases will reduce the verified model from 845 to 844 parameters while retaining at least 99% accuracy, because all seven gate weight vectors and all seven value/output directions remain independently learned.
change: Store thirteen rather than fourteen `fc1` biases and reuse the final stored gate bias for the seventh gate, while preserving the original constructor random draw.
mechanism: Pairwise GLU gate-intercept sharing
evidence_used: The 845-parameter seven-direction GLU reached 99.84%; the more aggressive 837-parameter consensus-gate experiment timed out rather than demonstrating an accuracy failure, so sharing one scalar gate intercept is a conservative test of whether full gate-specific biasing is necessary.
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
