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
verified_results: {"accuracy": 0.9954000000000001, "parameters": 824, "training_steps": 4999}
prior_hypothesis: An 824-parameter transformer will retain at least 99% accuracy because fixing a second off-axis coefficient in the first value channel removes another V/output-projection shear degree without reducing attention routing or representational width.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing a second classifier-visible terminal LayerNorm scale at one will produce a 968-parameter transformer with at least 99% accuracy, because four learned visible scales and the three-coordinate final latent bias retain substantial output calibration capacity.
change: Replace the eight-parameter terminal LayerNorm with four learned scales and four fixed unit scales, preserving the verified seven-unit MLP and all attention parameters.
mechanism: Second classifier-visible terminal-scale fixing
evidence_used: The adjacent 969-parameter design achieved 99.93% accuracy after fixing one classifier-visible scale, while removing another final-bias coordinate collapsed; extending the successful terminal-scale compression by one parameter is the narrowest informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 968, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third classifier-visible terminal LayerNorm scale at one will produce a 967-parameter transformer with at least 99% accuracy, because three learned visible scales and the retained three-coordinate final latent bias preserve sufficient output calibration.
change: Reconstruct the verified seven-unit MLP and replace the terminal LayerNorm’s eight learned scales with three learned scales followed by five fixed unit scales.
mechanism: Third classifier-visible terminal-scale fixing
evidence_used: The adjacent 968-parameter design achieved 99.98% accuracy after fixing two classifier-visible scales, improving on the successful 969-parameter design’s 99.93%; extending that same compression by one scale is the narrowest informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 967, "training_steps": 4999}

RECENT RESULT
hypothesis: A 829-parameter transformer will retain at least 99% accuracy because the five-dimensional lexical representation and seven nonlinear features are load-bearing, while the two classifier-orthogonal residual dimensions are not necessary for learned attention-based addition.
change: Reconstruct the qualified 967-parameter design, reduce `d_model` from eight to six, and expand the token basis from `d_model - 3` to the full five-dimensional zero-mean subspace; preserve two attention heads, the seven-unit MLP, three learned final scales, and all three final-bias coordinates.
mechanism: Full-lexical six-dimensional residual state
evidence_used: The 967-parameter design achieved 99.77%; the rank-four lexical bottleneck and six-unit MLP failed, so this patch retains rank five and seven nonlinear units while testing the previously shared assumption that computation requires two additional classifier-invisible scratch dimensions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 829, "training_steps": 4999}

RECENT RESULT
hypothesis: The verified 829-parameter transformer will retain at least 99% accuracy with 828 parameters because two learned terminal scales plus the retained three-coordinate final bias should preserve adequate output calibration.
change: Fix one additional classifier-visible terminal LayerNorm scale at one, reducing its learned scales from three to two.
mechanism: Fourth terminal-scale fixing
evidence_used: The current six-dimensional design achieved 99.88% accuracy, and three consecutive terminal-scale reductions in the eight-dimensional architecture remained above 99%; this adjacent one-parameter reduction extends the most consistently successful compression mechanism without altering the load-bearing five-dimensional lexical representation, seven-unit MLP, or attention routing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0821, "parameters": 828, "training_steps": 4999}

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

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
