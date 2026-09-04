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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1531, "training_steps": 4999}
prior_hypothesis: A 1,531-parameter model will retain at least 99% accuracy because the verified six-terminal-column design achieved 99.91%, while removing a second attention-projection column’s common-output component is an exact downstream-LayerNorm symmetry whose full AdamW dynamics are preserved in ambient coordinates.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9989, "parameters": 1533, "training_steps": 4999}
prior_hypothesis: A 1,533-parameter model will retain at least 99% accuracy because the qualified 1,534-parameter tied-embedding/five-scale design already achieved 99.90%, while the fifth `fc2` column’s common-output component is exactly erased by the final LayerNorm and its full AdamW dynamics remain preserved.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1532, "training_steps": 4999}
prior_hypothesis: A 1,532-parameter model will retain at least 99% accuracy because the qualified 1,533-parameter design reached 99.89%, while a sixth `fc2` column’s common-output component is also exactly erased by the final LayerNorm and its full AdamW dynamics can be preserved.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing the tied embedding’s single global scalar-shift degree of freedom will produce a 1,534-parameter model with at least 99% accuracy because LayerNorm erases its input-side effect, softmax erases its common-logit output effect, and ambient-coordinate AdamW preserves the qualified optimization dynamics.
change: Replace the full tied token table with an exactly gauge-fixed table containing one fewer learned scalar, retain the generic learned output head, preserve initialization RNG order, and extend the existing ambient moments, clipping, and updates to the omitted embedding coordinate.
mechanism: Global tied-token-embedding shift gauge with ambient AdamW
evidence_used: The current five-scale design achieved 99.95% accuracy at 1,535 parameters while already succeeding with an ambient additive positional gauge; the second positional-row gauge reached only 98.60%, motivating this distinct global symmetry whose input and output effects are both exactly unobservable.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,533-parameter model will retain at least 99% accuracy because the qualified 1,534-parameter tied-embedding/five-scale design already achieved 99.90%, while the fifth `fc2` column’s common-output component is exactly erased by the final LayerNorm and its full AdamW dynamics remain preserved.
change: Reproduce the qualified tied-token gauge and five absorbed `ln2` scales, then remove the common-output component from a fifth terminal projection column and include it in the existing ambient gauge optimizer.
mechanism: Fifth terminal-column output-shift gauge with ambient AdamW
evidence_used: Reference Design 2 achieved 99.90% accuracy at 1,534 parameters using the tied-token gauge, five-scale absorption, and four terminal-column gauges; extending that already-qualified exact terminal symmetry by one column is the smallest controlled reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,532-parameter model will retain at least 99% accuracy because the qualified 1,533-parameter design reached 99.89%, while a sixth `fc2` column’s common-output component is also exactly erased by the final LayerNorm and its full AdamW dynamics can be preserved.
change: Reproduce the qualified tied-token gauge, five absorbed `ln2` scales, and five terminal-column gauges, then gauge-fix a sixth terminal column using the existing ambient-coordinate optimizer.
mechanism: Sixth terminal-column output-shift gauge with ambient AdamW
evidence_used: The five-terminal-column 1,533-parameter design achieved 99.89% accuracy; extending the same exact symmetry by one column is the smallest controlled reduction beyond the best qualified design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1532, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,531-parameter model will retain at least 99% accuracy because the verified 1,532-parameter design reached 99.91%, while a seventh `fc2` column’s common-output component is likewise erased by the final LayerNorm and its full AdamW dynamics remain preserved.
change: Extend the terminal projection gauge from six to seven weight columns, retaining the existing ambient-coordinate gradient, clipping, moment, and update machinery.
mechanism: Seventh terminal-column output-shift gauge with ambient AdamW
evidence_used: Extending the terminal-column gauge from four to five columns achieved 99.89%, and extending it to six achieved 99.91%; applying the same exact symmetry to one additional column is the smallest controlled reduction beyond the current qualified design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6848000000000001, "parameters": 1531, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,334-parameter model will retain at least 99% accuracy because a fixed eight-dimensional Chebyshev position basis preserves full residual width and lets learned projections express structured forward or reversed positional relationships, avoiding the four-dimensional bottleneck that limited the learned positional-subspace design to 97.06%.
change: Challenge the assumption that every sequence position needs an independent learned vector; replace the gauge-fixed positional table with generic fixed polynomial features and a learned 8×8 mixer, while training that mixer through ordinary AdamW and retaining the qualified attention, MLP, and gauge machinery.
mechanism: Full-width orthogonal-polynomial positional representation
evidence_used: The learned four-dimensional positional subspace reached 97.06%, showing that structured positional compression retained most task behavior but that four routing dimensions were insufficient; this patch restores all eight positional dimensions while using only 64 learned positional parameters instead of an independent full-width table.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9419, "parameters": 1414, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,531-parameter model will retain at least 99% accuracy because the verified six-terminal-column design achieved 99.91%, while removing a second attention-projection column’s common-output component is an exact downstream-LayerNorm symmetry whose full AdamW dynamics are preserved in ambient coordinates.
change: Extend the qualified terminal projection gauge from four to six columns and independently gauge-fix a second attention output-projection column, updating initialization, gradient clipping, and ambient AdamW bookkeeping accordingly.
mechanism: Second attention-output column gauge atop six terminal-column gauges
evidence_used: Reference Design 3 achieved 99.91% accuracy with 1,532 parameters and six terminal-column gauges. The seventh terminal gauge failed at 68.48%, motivating a distinct reduction axis; the current qualified designs already demonstrate that the first attention-projection output gauge is compatible with high accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1531, "training_steps": 4999}



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
