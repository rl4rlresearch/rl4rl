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
verified_results: {"accuracy": 0.9998, "parameters": 968, "training_steps": 4999}
prior_hypothesis: Fixing a second classifier-visible terminal LayerNorm scale at one will produce a 968-parameter transformer with at least 99% accuracy, because four learned visible scales and the three-coordinate final latent bias retain substantial output calibration capacity.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9976999999999999, "parameters": 967, "training_steps": 4999}
prior_hypothesis: Fixing a third classifier-visible terminal LayerNorm scale at one will produce a 967-parameter transformer with at least 99% accuracy, because three learned visible scales and the retained three-coordinate final latent bias preserve sufficient output calibration.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987999999999999, "parameters": 829, "training_steps": 4999}
prior_hypothesis: A 829-parameter transformer will retain at least 99% accuracy because the five-dimensional lexical representation and seven nonlinear features are load-bearing, while the two classifier-orthogonal residual dimensions are not necessary for learned attention-based addition.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 969, "training_steps": 4999}
prior_hypothesis: Fixing one classifier-visible terminal LayerNorm scale at one will reduce the model to 969 parameters while retaining at least 99% accuracy, because the remaining scales and final latent bias can represent the original terminal vector up to a positive global logit scale, which does not change greedy decoding.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the qualified 10-unit GELU MLP to nine units will produce a 1,002-parameter transformer with at least 99% accuracy, because the 10-unit model achieved 99.96% accuracy and retained substantial margin above the threshold.
change: Reconstruct the qualified four-key-fix, two-coordinate final-bias reduction, then reduce `d_ff` from 10 to 9, removing 15 additional MLP parameters.
mechanism: Nine-feature nonlinear carry bottleneck
evidence_used: The qualified 1,017-parameter model with `d_ff=10` achieved 99.96% accuracy; testing the adjacent width of nine is the narrowest informative continuation of the most successful large parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1002, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified nine-unit GELU MLP to eight units will produce a 987-parameter transformer with at least 99% accuracy, because the nine-unit model achieved 99.94% accuracy and this adjacent reduction preserves its two specialized attention heads and five-dimensional token representation.
change: Reconstruct the qualified four-key-fix and two-coordinate final-bias reductions, then reduce `d_ff` from nine to eight, removing 15 additional MLP parameters.
mechanism: Eight-feature nonlinear carry bottleneck
evidence_used: The 1,002-parameter design with `d_ff=9` achieved 99.94% accuracy, while the preceding `d_ff=10` design achieved 99.96%; the adjacent width reduction is the narrowest informative continuation of the successful MLP compression path.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 987, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified eight-unit GELU MLP to seven units will produce a 972-parameter transformer with at least 99% accuracy, because the eight-unit model achieved 99.78% while preserving both specialized attention heads and the five-dimensional token representation.
change: Reduce `d_ff` from eight to seven, removing 15 learned MLP parameters while leaving the qualified attention, embedding, and output-bias design unchanged.
mechanism: Seven-feature nonlinear carry bottleneck
evidence_used: Successive reductions from 10 to nine and then eight MLP units achieved 99.96%, 99.94%, and 99.78% accuracy respectively; testing the adjacent width of seven is the narrowest continuation of this consistently successful compression path.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965999999999999, "parameters": 972, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified seven-unit GELU MLP to six units will produce a 957-parameter transformer with at least 99% accuracy, because the seven-unit model achieved 99.66% while preserving both specialized attention heads and the five-dimensional token representation.
change: Reduce `d_ff` from nine to six, reconstructing the qualified architecture at the next untested width below the verified seven-unit design and removing 45 parameters from the current implementation.
mechanism: Six-feature nonlinear carry bottleneck
evidence_used: Successive widths of ten, nine, eight, and seven achieved 99.96%, 99.94%, 99.78%, and 99.66% accuracy; six units are therefore the narrowest informative continuation of the consistently successful MLP-compression path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2419, "parameters": 957, "training_steps": 4999}

RECENT RESULT
hypothesis: The qualified seven-unit MLP will retain at least 99% accuracy with 970 parameters because the tied rank-five classifier ignores the terminal LayerNorm’s final two output coordinates, so fixing their scales at one removes two functionally irrelevant learned parameters.
change: Restore the verified `d_ff=7` architecture and replace its eight-parameter terminal LayerNorm scale with six learned classifier-visible scales plus two fixed scales.
mechanism: Classifier-aware terminal normalization pruning
evidence_used: The seven-unit design achieved 99.66% accuracy with 972 parameters, while six units collapsed to 24.19%; this motivates preserving the seven nonlinear features and instead removing two terminal scales whose coordinates are outside the five-dimensional classifier subspace.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9961, "parameters": 970, "training_steps": 4999}

RECENT RESULT
hypothesis: The verified seven-unit, classifier-aware 970-parameter design will retain at least 99% accuracy with 969 parameters when a third zero-initialized final latent-bias coordinate is fixed at zero, because the first two such reductions retained 99.77% and 99.92% accuracy.
change: Reconstruct the qualified `d_ff=7` architecture with six learned terminal LayerNorm scales, then shorten the final latent bias from three learned coordinates to two and reconstruct the remaining three as fixed zeros.
mechanism: Third final-classifier latent-bias coordinate fixing
evidence_used: The seven-unit classifier-aware design achieved 99.61% accuracy with 970 parameters, while successive final-bias reductions previously retained 99.77% and 99.92%; a one-coordinate extension of that successful output-path compression is the narrowest informative change below 970 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4545, "parameters": 969, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one classifier-visible terminal LayerNorm scale at one will reduce the model to 969 parameters while retaining at least 99% accuracy, because the remaining scales and final latent bias can represent the original terminal vector up to a positive global logit scale, which does not change greedy decoding.
change: Learn five terminal LayerNorm scales and fix the sixth classifier-visible scale alongside the two classifier-ignored scales.
mechanism: Argmax-invariant terminal scale gauge
evidence_used: The 970-parameter design achieved 99.61% accuracy, while removing a third final-bias coordinate collapsed to 45.45%; this preserves all successful output-bias capacity and instead removes a terminal scale degree associated with global logit magnitude.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 969, "training_steps": 4999}

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
