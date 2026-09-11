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
verified_results: {"accuracy": 0.9995, "parameters": 1535, "training_steps": 4999}
prior_hypothesis: A 1,535-parameter model will retain at least 99% accuracy because the fifth `ln2` scale is exactly absorbable into its `fc1` column and the verified ambient-coordinate AdamW factorization preserves its separate optimization dynamics.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1536, "training_steps": 4999}
prior_hypothesis: Extending the verified three-scale optimizer-side factorization to a fourth `ln2` scale will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, because the additional scale is exactly absorbable into its `fc1` column and its separate AdamW dynamics remain preserved outside the deduplicated model.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1539, "training_steps": 4999}
prior_hypothesis: Fixing one `ln2` scale to one will reduce the model from 1,540 to 1,539 parameters while retaining at least 99% accuracy, because the following `fc1` weight column can absorb that coordinate’s learned scaling without reducing functional capacity.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one `ln1` scale to one will reduce the qualified model from 1,539 to 1,538 parameters while retaining at least 99% accuracy, because the corresponding columns of the learned query, key, and value projections can independently absorb that scale.
change: Reuse the qualified seven-scale bias-free LayerNorm for `ln1`, while preserving the current attention, MLP, gauges, optimizer, and decoding behavior.
mechanism: Single-coordinate first-pre-norm scale absorption
evidence_used: Fixing one `ln2` scale achieved 99.93% accuracy at 1,539 parameters, while removing all eight scales failed; applying the same minimal one-coordinate constraint to the distinct `ln1` scale vector is the most controlled untested reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7165, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,538-parameter model will retain at least 99% accuracy because the two omitted `ln2` scales are exactly absorbed into their `fc1` columns, while optimizer-side factorization reproduces the full model’s separate AdamW updates instead of the optimization geometry that caused the naive two-scale design to reach only 80.28%.
change: Reproduce the qualified attention projection bias/first-column gauges, replace `ln2` with a six-scale LayerNorm, and train the two absorbed scale/weight factorizations in ambient coordinates while continuously materializing only their products in the learned model.
mechanism: Ambient AdamW for two multiplicatively absorbed LayerNorm scales
evidence_used: The full-scale 1,540-parameter attention-gauged design achieved 99.99%, whereas directly fixing a second `ln2` scale fell to 80.28%; since each scale is functionally absorbable into the following `fc1` column, this isolates and restores the lost optimizer dynamics while removing both parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-scale optimizer-side factorization to a third `ln2` scale will produce a 1,537-parameter model with at least 99% accuracy, because the third scale is exactly absorbable into its corresponding `fc1` column and its separate AdamW dynamics are preserved outside the deduplicated learned model.
change: Reproduce the qualified 1,538-parameter shared-key/value, bias-free-value, bias-free-`ln1`, attention-gauged, four-terminal-gauge design, then absorb three rather than two `ln2` scales into `fc1` with ambient-coordinate moments and clipping.
mechanism: Third multiplicatively absorbed LayerNorm scale with ambient AdamW
evidence_used: The two-scale ambient-factorization design achieved 99.94% accuracy at 1,538 parameters, whereas directly fixing two scales achieved only 80.28%; this isolates optimizer geometry as the failure cause and motivates extending the verified factorization by one coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1537, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-scale optimizer-side factorization to a fourth `ln2` scale will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, because the additional scale is exactly absorbable into its `fc1` column and its separate AdamW dynamics remain preserved outside the deduplicated model.
change: Absorb four rather than three trailing `ln2` scales into the corresponding `fc1` columns, expanding the existing ambient-coordinate moments, clipping correction, and materialization update accordingly.
mechanism: Fourth multiplicatively absorbed LayerNorm scale with ambient AdamW
evidence_used: The three-scale ambient-factorization design achieved 99.97% accuracy at 1,537 parameters, improving on the already-qualified two-scale design; this makes one additional coordinate along the same successful exact-factorization axis the smallest controlled next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1536, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,535-parameter model will retain at least 99% accuracy because the fifth `ln2` scale is exactly absorbable into its `fc1` column and the verified ambient-coordinate AdamW factorization preserves its separate optimization dynamics.
change: Expand the current two-scale absorption to five trailing `ln2` scales, including the virtual scales, optimizer moments, clipping correction, and effective-column materialization.
mechanism: Fifth multiplicatively absorbed LayerNorm scale with ambient AdamW
evidence_used: Four-scale ambient factorization achieved 100% accuracy at 1,536 parameters, after three scales achieved 99.97%; extending the same successful exact-factorization axis by one coordinate is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1535, "training_steps": 4999}

RECENT RESULT
hypothesis: Absorbing a sixth `ln2` scale into its corresponding `fc1` column will reduce the model to 1,534 parameters while retaining at least 99% accuracy, because the scale is functionally redundant and its separate AdamW dynamics are preserved outside the deduplicated model.
change: Replace the one-fixed-scale LayerNorm with a six-fixed-scale variant and train all six absorbed scale/weight factorizations using ambient-coordinate moments, clipping, and effective-column materialization.
mechanism: Sixth multiplicatively absorbed LayerNorm scale with ambient AdamW
evidence_used: Five-scale ambient factorization achieved 99.95% accuracy at 1,535 parameters, following four-scale factorization’s 100%; extending that same verified axis by one coordinate is the smallest controlled reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7232, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing independent eight-dimensional position vectors with learned four-dimensional position codes and a shared learned basis will reduce parameters by 99 while retaining at least 99% accuracy; the qualified four-dimensional shared key/value stream suggests four positional routing coordinates are sufficient, whereas the failed two-dimensional routing design identifies a plausible lower boundary.
change: Challenge the assumption that every position needs an independent full-width residual vector by factorizing the positional table into four-dimensional learned codes and a shared 4×8 basis, then train both factors with ordinary AdamW while preserving all qualified attention, MLP, gauge, and five-scale-factorization behavior.
mechanism: Four-dimensional learned positional subspace
evidence_used: The current four-dimensional shared key/value mechanism achieved 99.95% accuracy at 1,535 parameters, but reducing routing to two dimensions achieved only 81.88%; a four-dimensional positional subspace therefore tests a substantially smaller representation without repeating the demonstrated two-dimensional bottleneck.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9706, "parameters": 1476, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,534-parameter model will retain at least 99% accuracy because the qualified five-scale design already achieved 99.95%, while the scalar component removed from a second positional vector is independently erased by the pre-norm and final LayerNorm operations and its full AdamW dynamics are preserved in ambient coordinates.
change: Extend the qualified five-scale `ln2` absorption to the current design and gauge-fix the second positional embedding row, including its ambient optimizer moments, gradients, clipping contribution, and update.
mechanism: Second positional-row shift gauge atop five-scale factorization
evidence_used: Reference Design 2 achieved 99.95% accuracy with 1,535 parameters using five absorbed `ln2` scales, and every qualified reduced design already uses the identical positional shift gauge for the first row; applying that exact invariance to one additional row is the smallest new reduction atop the best qualified design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.986, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,523-parameter model will retain at least 99% accuracy because five-scale absorption already achieved 99.95%, while each `fc1` row has one exactly redundant coefficient on LayerNorm’s mean-zero subspace; jointly updating the full ambient scales and weights preserves their AdamW dynamics before materializing the deduplicated model.
change: Reproduce five absorbed `ln2` scales and remove the final input coefficient from every `fc1` row, training the corresponding full eight-scale and eight-column representation solely as optimizer-coordinate state.
mechanism: Coupled LayerNorm-scale absorption with mean-zero input-weight gauge
evidence_used: The qualified five-scale factorization reached 99.95% at 1,535 parameters, whereas the sixth-scale extension failed; this patch keeps the verified five fixed scales and targets a distinct exact 12-parameter redundancy while avoiding the unstable quotient reconstruction by directly maintaining full ambient weights and scales.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9786, "parameters": 1523, "training_steps": 4999}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: The 1,534-parameter model will retain at least 99% accuracy because the removed `ln1` scale is exactly absorbable into the corresponding query, key, and value columns, while ambient-coordinate AdamW preserves the optimizer dynamics whose absence caused the naive `ln1` reduction to fail.
change: Use five-scale `ln2` absorption from the qualified 1,535-parameter design, additionally absorb one `ln1` scale across all three attention input projections, and train both factorizations with ambient moments, clipping, and product materialization.
mechanism: Ambient AdamW factorization of one `ln1` scale atop five absorbed `ln2` scales
evidence_used: Five-scale `ln2` factorization achieved 99.95% at 1,535 parameters; more importantly, naive two-scale `ln2` fixing failed at 80.28% before ambient factorization restored 99.94%, directly motivating the same optimizer-side remedy for the naive one-scale `ln1` result of 71.65%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7347, "parameters": 1534, "training_steps": 4999}



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
