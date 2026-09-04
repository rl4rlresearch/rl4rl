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
verified_results: {"accuracy": 0.9997, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Fixing a seventh `ln2` bias coordinate will reduce the verified 1,578-parameter model to 1,577 parameters while retaining at least 99% accuracy, because its effect can be represented by the downstream learned `fc1` bias.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9981, "parameters": 1577, "training_steps": 4999}
prior_hypothesis: Centering positional rows 1–6 and fixing a seventh `ln2` bias coordinate will yield 1,577 parameters with at least 99% accuracy, because each reduction independently exceeded 99.7% accuracy at 1,578 parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9982, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,584-parameter design will retain at least 99% accuracy because the added leading `ln1` scale anchor is absorbed by QKV and centering the separated final row of the first key head removes only a softmax-invariant key offset.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9973000000000001, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,579-parameter design and zero-centering positional row 6 will yield 1,578 parameters with at least 99% accuracy, because its position-specific uniform residual shift is canceled by every downstream LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Reproducing the verified seven-centered-key, three-centered-`fc2` design and additionally centering the final, nonadjacent `fc2` input column will yield 1,580 parameters with at least 99% accuracy, because its uniform output component is removed by the final LayerNorm while avoiding the adjacent-column layout that previously failed.
change: Center seven qualified QKV key rows, retain only the remaining second-head key row, and represent `fc2` columns 0–2 plus its final column in the zero-mean output basis while reconstructing their original order.
mechanism: Nonadjacent fourth MLP residual-output uniform gauge quotient
evidence_used: The seven-key, three-column design achieved 99.87% accuracy at 1,581 parameters, whereas extending to the adjacent fourth `fc2` column achieved only 92.13%; earlier nonadjacent LayerNorm anchors succeeded where adjacent extensions failed, motivating the same separated-coordinate test for `fc2`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.993, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,580-parameter design and centering the first attention output-projection column will yield 1,579 parameters with at least 99% accuracy, because the removed component adds only a position-dependent uniform residual shift canceled by the following and final LayerNorms.
change: Apply the qualified seven-key and nonadjacent four-column `fc2` compactions, then represent the first attention projection column in a seven-dimensional zero-mean basis.
mechanism: Attention-projection residual-uniform gauge quotient
evidence_used: The seven-key, nonadjacent four-`fc2`-column design achieved 99.30% accuracy at 1,580 parameters; successful `fc2` column reductions establish the same residual-uniform gauge, while the failed eighth-key and adjacent-fourth-`fc2` experiments motivate testing it in the independent attention projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9891, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final attention output-projection column will yield 1,579 parameters with at least 99% accuracy, because its removed uniform residual component is canceled by subsequent LayerNorms, while the trailing-coordinate layout may avoid the optimization weakness observed when centering the first column.
change: Represent the final `attn.proj` input column in a seven-dimensional zero-mean basis and reconstruct it during the forward pass.
mechanism: Trailing attention-projection residual-uniform gauge quotient
evidence_used: The 1,580-parameter design achieved 99.30%; centering the first attention-projection column narrowly missed at 98.91%, while the analogous trailing, nonadjacent `fc2` column succeeded where an adjacent column failed, motivating a trailing-coordinate test of the same exact gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5528, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,580-parameter nonadjacent `fc2` design and zero-centering positional row 5 will yield 1,579 parameters with at least 99% accuracy, because a position-specific uniform residual shift is removed by every downstream LayerNorm.
change: Center positional rows 1–5 instead of 1–4, and apply the qualified nonadjacent `fc2` compaction to columns 0–2 and the final column.
mechanism: Sixth positional-row LayerNorm-null gauge quotient
evidence_used: The nonadjacent four-column `fc2` design achieved 99.30% at 1,580 parameters, while extending positional centering through row 4 achieved 99.84%; this tests the next exact positional gauge instead of the attention-projection gauges that missed at 98.91% and 55.28%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,579-parameter design and zero-centering positional row 6 will yield 1,578 parameters with at least 99% accuracy, because its position-specific uniform residual shift is canceled by every downstream LayerNorm.
change: Center positional rows 1–6 and apply the qualified nonadjacent `fc2` compaction to columns 0–2 and the final column.
mechanism: Seventh positional-row LayerNorm-null gauge quotient
evidence_used: Centering positional row 5 with the nonadjacent four-column `fc2` design achieved 99.79% accuracy at 1,579 parameters; extending the same exact positional gauge to the next row is the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,578-parameter design and zero-centering positional row 7 will yield 1,577 parameters with at least 99% accuracy, because its uniform residual shift is canceled by every downstream LayerNorm.
change: Apply the qualified seven-key and nonadjacent four-column `fc2` compactions, then center positional rows 1–7 while reconstructing all tensors in their original order.
mechanism: Eighth positional-row LayerNorm-null gauge quotient
evidence_used: Centering positional row 6 produced 99.73% accuracy at 1,578 parameters after row 5 produced 99.79%; extending the same exact positional gauge to the next row is the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9662999999999999, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final positional row instead of the failed adjacent row 7 will reduce the verified 1,578-parameter model to 1,577 parameters while retaining at least 99% accuracy, because its position-specific uniform shift is canceled by downstream LayerNorms.
change: Keep positional rows 1–6 centered, additionally center the final positional row, and reconstruct all intermediate rows in their original order.
mechanism: Nonadjacent trailing positional-row LayerNorm-null gauge quotient
evidence_used: The current six-row-centered design achieved 99.73% accuracy at 1,578 parameters, while adjacent row 7 failed at 96.63%; the analogous nonadjacent final-column `fc2` reduction succeeded at 99.30% after its adjacent alternative failed at 92.13%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9620000000000001, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a seventh `ln2` bias coordinate will reduce the verified 1,578-parameter model to 1,577 parameters while retaining at least 99% accuracy, because its effect can be represented by the downstream learned `fc1` bias.
change: Increase `ln2`’s fixed bias coordinates from six to seven, leaving the architecture, training schedule, and protected decoding interface unchanged.
mechanism: Seventh pre-MLP LayerNorm-bias absorption
evidence_used: The current six-coordinate `ln2` compaction achieved 99.73% accuracy at 1,578 parameters; this tests the next coordinate in that existing gauge family after additional positional-row gauges failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering positional rows 1–6 and fixing a seventh `ln2` bias coordinate will yield 1,577 parameters with at least 99% accuracy, because each reduction independently exceeded 99.7% accuracy at 1,578 parameters.
change: Extend positional centering from rows 1–4 to rows 1–6 and increase `ln2`’s fixed bias coordinates from six to seven.
mechanism: Combined positional LayerNorm-null gauge and pre-MLP bias absorption
evidence_used: Reference Design 3 centered positional rows 1–6 and achieved 99.73% accuracy at 1,578 parameters, while Reference Design 2 fixed seven `ln2` bias coordinates and achieved 99.97%; combining these independently qualified gauge reductions is the closest supported path to 1,577 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,577-parameter design and fixing the remaining `ln2` bias coordinate will yield 1,576 parameters with at least 99% accuracy, because every `ln2` bias contribution can be represented by the learned downstream `fc1` bias.
change: Apply the qualified positional, seven-key, and nonadjacent four-column `fc2` compactions, then remove the final trainable `ln2` bias coordinate without creating a zero-length parameter.
mechanism: Complete pre-MLP LayerNorm-bias absorption
evidence_used: The combined 1,577-parameter design achieved 99.81% accuracy, while independently increasing `ln2` bias fixation from six to seven coordinates achieved 99.97%; the remaining coordinate has the same downstream-bias absorption mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing the final `ln2` bias parameter will yield 1,576 parameters with at least 99% accuracy because the downstream learned `fc1` bias exactly absorbs every `ln2` bias contribution; reducing periodic validation frequency should allow the previously timed-out design to finish.
change: Support a parameter-free LayerNorm bias when all coordinates are fixed, set `ln2` to fix all eight bias coordinates, and reduce validation overhead without changing the 5,000-step optimization trajectory.
mechanism: Complete pre-MLP LayerNorm-bias absorption
evidence_used: The 1,577-parameter current design achieved 99.81%, and fixing the seventh `ln2` bias coordinate independently achieved 99.97%; the prior complete-absorption attempt timed out rather than failing accuracy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing all `ln2` bias parameters will produce a 1,576-parameter model with at least 99% accuracy, while collision-loop and validation-overhead reductions will allow the previously timed-out design to complete.
change: Represent fully fixed LayerNorm bias without a zero-length parameter, fix all eight `ln2` bias coordinates, accelerate exact holdout rejection, and validate only the final trained checkpoint.
mechanism: Complete pre-MLP LayerNorm-bias absorption with lower-overhead training
evidence_used: The 1,577-parameter design reached 99.81% accuracy after fixing seven `ln2` bias coordinates; both full-absorption attempts timed out rather than failing accuracy, motivating the same exact gauge reduction with stronger runtime savings.
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
