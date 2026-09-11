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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 821, "training_steps": 4999}
prior_hypothesis: An 821-parameter transformer will retain at least 99% accuracy because fixing the already-shared maximum-distance bias at zero changes only the single sparsest causal query-key pair per head while preserving independent learned biases at every nearer distance.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9978, "parameters": 811, "training_steps": 4999}
prior_hypothesis: An 811-parameter transformer will retain at least 99% accuracy because fixing one sigmoid-gate bias at its zero initialization preserves all four learned gated features, their input-dependent gates, and the qualified model’s attention and lexical capacity.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9994, "parameters": 822, "training_steps": 4999}
prior_hypothesis: An 822-parameter transformer will retain at least 99% accuracy because it preserves the qualified 823-parameter value path while sharing only the learned maximum-distance bias, which affects a single causal query-key pair per head at full sequence length.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9983, "parameters": 823, "training_steps": 4999}
prior_hypothesis: The 823-parameter transformer will retain at least 99% accuracy because zeroing the second value row’s third input coefficient removes a remaining within-pivot shear degree while leaving its diagonal scale learnable.

## Recent verification evidence

RECENT RESULT
hypothesis: A 717-parameter transformer will retain at least 99% accuracy because rank five is preserved while only four lexical coordinates require independent per-token learning; a generic fixed fifth coordinate with learned global amplitude can supply the missing classifier-visible direction.
change: Replace each five-coordinate learned token code with four learned coordinates plus one normalized sinusoidal token-identity coordinate, retaining the existing five-dimensional projection, attention computation, MLP, and terminal calibration.
mechanism: Hybrid learned–sinusoidal lexical rank
evidence_used: The 829-parameter rank-five design achieved 99.88%, while the rank-four bottleneck failed; this establishes that a fifth lexical direction is load-bearing but does not establish that all 110 values along it need independent parameters. The current 826-parameter design’s 99.98% provides margin for testing this alternative full-rank representation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9512, "parameters": 713, "training_steps": 4999}

RECENT RESULT
hypothesis: A 823-parameter transformer will retain at least 99% accuracy because fixing a third off-axis coefficient in the tolerant first-head value matrix removes another V/output-projection shear degree without altering attention routing or residual width.
change: Reconstruct the qualified 824-parameter QKV constraints, then additionally fix input coefficient three of the first value channel at zero while leaving the sensitive second head unconstrained.
mechanism: Third first-head value-space shear gauge fixing
evidence_used: Successive first-head constraints achieved 99.99% at 825 parameters and 99.54% at 824, whereas constraining the second head collapsed; the narrowest informative next reduction is therefore another adjacent shear constraint within the demonstrated tolerant first head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.14859999999999998, "parameters": 823, "training_steps": 4999}

RECENT RESULT
hypothesis: A 823-parameter transformer will retain at least 99% accuracy because fixing the second coefficient of the second first-head value channel completes another entry of an invertible 3×3 value-basis pivot, using a V/output-projection gauge degree without constraining non-pivot value features.
change: Reconstruct the qualified 824-parameter first-head value constraints and additionally fix the second value row’s second input coefficient at 0.02.
mechanism: Structured first-head value-basis pivot fixing
evidence_used: The 824-parameter design achieved 99.54%, while the failed 823-parameter design fixed a fourth-column coefficient outside the three-column value-basis pivot; this tests a structurally distinct, gauge-aligned reduction within the tolerant first head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.069, "parameters": 823, "training_steps": 4999}

RECENT RESULT
hypothesis: The 823-parameter transformer will retain at least 99% accuracy because zeroing the second value row’s third input coefficient removes a remaining within-pivot shear degree while leaving its diagonal scale learnable.
change: Add one zero-fixed off-diagonal coefficient to the first head’s value matrix, reducing learned parameters from 824 to 823.
mechanism: Lower-triangular first-head value-basis gauge fixing
evidence_used: The 824-parameter design reached 99.54% with two first-row shears fixed; prior 823-parameter attempts either constrained a non-pivot coefficient or fixed a diagonal coefficient at 0.02, motivating an untested within-pivot shear constraint that preserves the diagonal scale.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 823, "training_steps": 4999}

RECENT RESULT
hypothesis: An 822-parameter transformer will retain at least 99% accuracy because zeroing the remaining off-diagonal coefficient in the first head’s 3×3 value pivot removes another V/output-projection shear degree while keeping both remaining diagonal pivots learnable.
change: Reconstruct the qualified 823-parameter value constraints and additionally fix the third value row’s second input coefficient at zero.
mechanism: First-head value-basis shear completion
evidence_used: The 823-parameter lower-triangular value-basis design achieved 99.83%, while fixing a diagonal coefficient failed; this motivates eliminating the remaining off-diagonal pivot entry without constraining another diagonal or the sensitive second head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9777, "parameters": 822, "training_steps": 4999}

RECENT RESULT
hypothesis: An 822-parameter transformer will retain at least 99% accuracy because fixing one first-head output-projection coefficient uses a residual value/output basis gauge without adding another fragile constraint to the qualified value pivot.
change: Reconstruct the qualified 823-parameter QKV constraints, then fix the first coefficient of the attention output projection at zero, removing one learned parameter.
mechanism: Residual first-head value/output shear gauge fixing
evidence_used: The 823-parameter lower-triangular value design achieved 99.83%, while extending its value-matrix constraints reached only 97.77%; moving the next gauge constraint to the coupled output side tests the remaining symmetry without further restricting that value pivot.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 822, "training_steps": 4999}

RECENT RESULT
hypothesis: An 822-parameter transformer will retain at least 99% accuracy because it preserves the qualified 823-parameter value path while sharing only the learned maximum-distance bias, which affects a single causal query-key pair per head at full sequence length.
change: Reconstruct the qualified 823-parameter QKV constraints, then replace the two head-specific farthest-distance biases with one shared learned scalar, removing one parameter.
mechanism: Cross-head sharing of the sparsest relative-distance bias
evidence_used: The 823-parameter lower-triangular value design achieved 99.83%; both failed 822-parameter attempts further constrained the value/output path, motivating an orthogonal reduction at the sparsest relative-attention distance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 822, "training_steps": 4999}

RECENT RESULT
hypothesis: An 821-parameter transformer will retain at least 99% accuracy because extending the successful maximum-distance sharing to the adjacent distance preserves a learned bias at every causal distance while tying only the two sparsest query-key diagonals across heads.
change: Reduce the head-specific relative-bias table by one column and replace the shared scalar tail with two learned biases shared across both attention heads.
mechanism: Cross-head sharing of the two sparsest relative-distance biases
evidence_used: The current 822-parameter model achieved 99.94% after sharing the farthest-distance bias, while further value/output constraints failed; extending the same successful, orthogonal sharing mechanism to the next-sparsest distance is the narrowest adjacent reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0546, "parameters": 821, "training_steps": 4999}

RECENT RESULT
hypothesis: An 821-parameter transformer will retain at least 99% accuracy because sharing the maximum-distance bias preserves the qualified 822-parameter design, while fixing the remaining first-head key-channel scale uses a reciprocal Q/K basis degree without tying the functionally important adjacent-distance biases.
change: Reconstruct the qualified shared maximum-distance relative bias and fix the third first-head key channel’s leading coefficient at 0.02, reducing the current 823-parameter model by two parameters.
mechanism: First-head key-scale gauge completion with maximum-distance bias sharing
evidence_used: Maximum-distance sharing achieved 99.94% at 822 parameters, whereas extending sharing to the adjacent distance collapsed to 5.46%; the successful 99.98% fixing of all three first-head value-channel scales motivates testing the analogous remaining scale constraint on the first-head key basis instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0557, "parameters": 821, "training_steps": 4999}

RECENT RESULT
hypothesis: An 821-parameter transformer will retain at least 99% accuracy because fixing the already-shared maximum-distance bias at zero changes only the single sparsest causal query-key pair per head while preserving independent learned biases at every nearer distance.
change: Reconstruct the qualified 822-parameter value-path constraints and replace its shared learned farthest-distance bias with a fixed zero.
mechanism: Fixed maximum-distance attention bias
evidence_used: The 822-parameter design reached 99.94% when only the farthest-distance bias was shared, whereas sharing the adjacent distance collapsed to 5.46%; this isolates the next reduction to the successful maximum-distance mechanism without constraining the important neighboring bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 821, "training_steps": 4999}

RECENT RESULT
hypothesis: A 812-parameter transformer will retain at least 99% accuracy because four multiplicatively gated MLP features can replace seven independent GELU features while preserving the qualified 821-parameter model’s load-bearing lexical rank and attention routing.
change: Reconstruct the qualified 821-parameter attention design, then replace its seven-unit additive GELU MLP with a four-unit sigmoid-gated MLP, reducing the nonlinear sublayer by nine parameters.
mechanism: Compact sigmoid-gated nonlinear carry state
evidence_used: The 821-parameter reference achieved 99.96%; lexical-rank reduction reached only 95.12% and adjacent-distance sharing collapsed to 5.46%, so this patch preserves both mechanisms and instead challenges the untested assumption that carry-related nonlinear computation requires seven independent additive features.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9929000000000001, "parameters": 812, "training_steps": 4999}

RECENT RESULT
hypothesis: An 811-parameter transformer will retain at least 99% accuracy because fixing one sigmoid-gate bias at its zero initialization preserves all four learned gated features, their input-dependent gates, and the qualified model’s attention and lexical capacity.
change: Replace the gated MLP input projection’s final learned bias—the fourth gate intercept—with a fixed zero, removing one parameter while leaving every projection weight and the other seven biases learned.
mechanism: Single fixed gate intercept
evidence_used: The 812-parameter four-feature gated MLP achieved 99.29% accuracy; this motivates testing a one-parameter reduction inside that successful mechanism instead of reducing feature count, lexical rank, or additional relative-attention biases, which prior evidence identifies as fragile.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 811, "training_steps": 4999}



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
