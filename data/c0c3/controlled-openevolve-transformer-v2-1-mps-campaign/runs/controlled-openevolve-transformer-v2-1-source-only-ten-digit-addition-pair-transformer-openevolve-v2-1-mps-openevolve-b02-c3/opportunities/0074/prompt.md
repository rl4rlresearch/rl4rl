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
verified_results: {"accuracy": 0.9998, "parameters": 826, "training_steps": 4999}
prior_hypothesis: A 826-parameter transformer will retain at least 99% accuracy because fixing the third value-channel coefficient in the already-qualified first attention head removes another reciprocal V/output-projection scale degree without altering attention routing or model width.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9991, "parameters": 827, "training_steps": 4999}
prior_hypothesis: A 827-parameter transformer will retain at least 99% accuracy because fixing a second value-channel coefficient within the already-qualified first attention head removes another reciprocal V/output-projection scale degree without constraining attention routing.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9997, "parameters": 828, "training_steps": 4999}
prior_hypothesis: An 828-parameter transformer will retain at least 99% accuracy because fixing one value-channel coefficient removes a reciprocal V/output-projection scale degree without changing attention routing or reducing the qualified lexical, MLP, and terminal calibration capacity.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9983, "parameters": 823, "training_steps": 4999}
prior_hypothesis: The 823-parameter transformer will retain at least 99% accuracy because zeroing the second value row’s third input coefficient removes a remaining within-pivot shear degree while leaving its diagonal scale learnable.

## Recent verification evidence

RECENT RESULT
hypothesis: A 827-parameter transformer will retain at least 99% accuracy because the verified 829-parameter design reached 99.88%, and fixing the remaining key-channel scale in each head removes two reciprocal Q/K gauge degrees without reducing attention expressivity.
change: Reconstruct the verified six-dimensional, rank-five lexical design and fix all three key-channel scale gauges per head instead of two.
mechanism: Full per-head Q/K diagonal gauge fixing
evidence_used: The 829-parameter six-dimensional design achieved 99.88%, while reducing its terminal scales failed; extending the already-qualified Q/K gauge fixing targets exact attention-score reparameterizations while preserving the load-bearing five-dimensional token representation, seven-unit MLP, and three terminal scales.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9623999999999999, "parameters": 827, "training_steps": 4999}

RECENT RESULT
hypothesis: An 827-parameter transformer will retain at least 99% accuracy because fixing one off-axis key coefficient per head removes Q/K channel-basis shear degrees while preserving the verified rank-five lexical state, seven-unit MLP, three terminal scales, and unconstrained final key-channel scales.
change: Reconstruct the verified 829-parameter six-dimensional design, then fix one additional key-matrix coefficient at zero in each head, removing two learned parameters.
mechanism: Per-head Q/K shear-gauge fixing
evidence_used: The 829-parameter design achieved 99.88% accuracy; the 827-parameter full key-scale fixing reached only 96.24%, motivating a different gauge reduction that leaves the remaining key-channel scales learnable.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0554, "parameters": 827, "training_steps": 4999}

RECENT RESULT
hypothesis: An 828-parameter transformer will retain at least 99% accuracy because fixing one value-channel coefficient removes a reciprocal V/output-projection scale degree without changing attention routing or reducing the qualified lexical, MLP, and terminal calibration capacity.
change: Reconstruct the verified 829-parameter six-dimensional design, then fix the first coefficient of one value channel at its initialization scale, removing one learned parameter whose magnitude can be absorbed by the corresponding output-projection column.
mechanism: Single-channel value/output projection scale gauge
evidence_used: The six-dimensional 829-parameter design achieved 99.88%; terminal-scale removal collapsed to 8.21% and fixing both remaining Q/K scales reached 96.24%, motivating a one-parameter gauge reduction in the value path that preserves both terminal calibration and attention scores.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 828, "training_steps": 4999}

RECENT RESULT
hypothesis: A 827-parameter transformer will retain at least 99% accuracy because fixing the first value-channel coefficient in the second attention head extends the value/output reciprocal scale gauge that achieved 99.97% at 828 parameters without changing attention routing or representational width.
change: Fix the first coefficient of one value channel in each attention head, removing one additional learned parameter from the verified 828-parameter design.
mechanism: Per-head value/output scale gauge fixing
evidence_used: The current 828-parameter model achieved 99.97% after one value-channel coefficient was fixed, while terminal-scale and additional Q/K reductions failed; this makes the corresponding value-scale gauge in the other head the narrowest evidence-backed next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0521, "parameters": 827, "training_steps": 4999}

RECENT RESULT
hypothesis: A 827-parameter transformer will retain at least 99% accuracy because fixing a second value-channel coefficient within the already-qualified first attention head removes another reciprocal V/output-projection scale degree without constraining attention routing.
change: Extend the successful 828-parameter value-path gauge fixing to the first two value channels of the first head, while leaving the second head unconstrained.
mechanism: Intra-head value/output scale gauge fixing
evidence_used: Fixing one value coefficient in the first head achieved 99.97%, whereas applying the corresponding constraint to the second head collapsed to 5.21%; this motivates testing whether gauge-fixing tolerance is head-specific by keeping both reductions within the successful head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 827, "training_steps": 4999}

RECENT RESULT
hypothesis: A 826-parameter transformer will retain at least 99% accuracy because fixing the third value-channel coefficient in the already-qualified first attention head removes another reciprocal V/output-projection scale degree without altering attention routing or model width.
change: Reconstruct the qualified six-dimensional rank-five design and fix all three first-head value-channel scales, extending the successful two-channel constraint by one parameter.
mechanism: Intra-head value/output scale gauge completion
evidence_used: The 827-parameter design fixing two value coefficients within the first head achieved 99.91%, while fixing a coefficient in the second head collapsed to 5.21%; this supports an adjacent reduction confined to the tolerant first head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 826, "training_steps": 4999}

RECENT RESULT
hypothesis: An 825-parameter transformer will retain at least 99% accuracy because fixing one off-axis coefficient in the already-qualified first-head value matrix removes a value/output-projection shear degree while preserving attention routing and representational width.
change: Reconstruct the qualified 826-parameter six-dimensional design, then fix the second input coefficient of its first value channel at zero in addition to the three established first-head value scales.
mechanism: First-head value-space shear gauge fixing
evidence_used: The 826-parameter design achieved 99.98% accuracy after fixing all three first-head value-channel scales, whereas constraining the second head collapsed; an additional within-first-head shear constraint is the narrowest reduction confined to the demonstrated tolerant head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 825, "training_steps": 4999}

RECENT RESULT
hypothesis: An 824-parameter transformer will retain at least 99% accuracy because fixing a second off-axis coefficient in the first value channel removes another V/output-projection shear degree without reducing attention routing or representational width.
change: Fix the third input coefficient of the first value channel at zero, extending the qualified first-head value gauge while leaving the sensitive second head unchanged.
mechanism: Second first-head value-space shear gauge fixing
evidence_used: The current 825-parameter design achieved 99.99% accuracy after fixing the adjacent second input coefficient of the same first-head value channel; prior constraints involving the second head collapsed, so the next informative reduction remains within the demonstrated tolerant first head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9954000000000001, "parameters": 824, "training_steps": 4999}

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
