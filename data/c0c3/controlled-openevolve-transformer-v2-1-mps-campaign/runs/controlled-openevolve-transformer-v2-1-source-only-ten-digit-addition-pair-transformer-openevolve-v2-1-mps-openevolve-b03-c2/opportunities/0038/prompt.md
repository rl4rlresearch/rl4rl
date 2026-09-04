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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1558, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1570-parameter design by removing the feature-uniform output coordinate from each of the 12 `fc2.weight` columns will produce a 1558-parameter model with at least 99% accuracy, because those components add only tokenwise uniform residual shifts canceled by the final LayerNorm.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9989, "parameters": 1571, "training_steps": 4999}
prior_hypothesis: Folding all eight `ln1` scales into `qkv.weight` while reproducing their AdamW and clipping dynamics will reduce the verified 1579-parameter model to 1571 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998, "parameters": 1570, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1571-parameter folded-LayerNorm design by removing the feature-uniform coordinate of `attn.proj.bias` will produce a 1570-parameter model with at least 99% accuracy, because that coordinate passes through the residual stream and is canceled by downstream LayerNorms.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the qualified value-bias quotient from one to two omitted coordinates will produce a 1593-parameter model with at least 99% accuracy while preserving both omitted coordinates’ clipping and AdamW dynamics.
change: Store six value-bias coordinates, reconstruct two zero-gauge coordinates, and generalize the optimizer and gradient clipping to track and absorb both omitted updates.
mechanism: Multi-coordinate value-bias absorption into the attention projection bias
evidence_used: The current 1594-parameter design achieved 99.73% after one value-bias coordinate was absorbed into `attn.proj.bias`; the second coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified two-coordinate value-bias quotient to three omitted coordinates will produce a 1592-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the newly omitted coordinate.
change: Store five learned value-bias coordinates, reconstruct three zero-gauge coordinates, and absorb their independently tracked AdamW updates into `attn.proj.bias`.
mechanism: Three-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting one value-bias coordinate achieved 99.73% at 1594 parameters, and omitting two achieved 99.98% at 1593; the third coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-coordinate value-bias quotient to four omitted coordinates will produce a 1591-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Adopt the qualified token, positional, MLP-output, biasless-LayerNorm, and key-bias reductions; store four fewer value-bias coordinates and absorb their tracked updates into `attn.proj.bias`.
mechanism: Four-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting one, two, and three value-bias coordinates achieved 99.73%, 99.98%, and 100% accuracy respectively; the fourth coordinate has the same attention-invariant role as the three verified omissions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified four-coordinate value-bias quotient to five omitted coordinates will produce a 1590-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Store three learned value-bias coordinates, reconstruct five zero-gauge coordinates, and absorb all five independently tracked AdamW updates into `attn.proj.bias`.
mechanism: Five-coordinate value-bias absorption into the attention projection bias
evidence_used: Successive omission of one, two, three, and four value-bias coordinates achieved 99.73%, 99.98%, 100%, and 99.92% accuracy respectively; the fifth coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified five-coordinate value-bias quotient to six omitted coordinates will produce a 1589-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Store two learned value-bias coordinates, reconstruct six zero-gauge coordinates, and absorb all six independently tracked updates into `attn.proj.bias`.
mechanism: Six-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting five value-bias coordinates achieved 99.95% accuracy at 1590 parameters, after one through four omissions also exceeded 99%; the sixth coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified six-coordinate value-bias quotient to seven omitted coordinates will produce a 1588-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Retain one learned value-bias coordinate, reconstruct seven zero-gauge coordinates, and absorb all seven independently tracked updates into `attn.proj.bias`.
mechanism: Seven-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting six value-bias coordinates achieved 99.73% accuracy at 1589 parameters, after omissions of one through five also exceeded 99%; the seventh coordinate has the same attention-invariant computational role.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7326, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining value-bias coordinate 1 while absorbing coordinates 0 and 2–7 will produce a 1588-parameter model with at least 99% accuracy, showing whether the failed prior reduction was caused by discarding coordinate 1 rather than by the seven-coordinate quotient itself.
change: Store only the second value-bias coordinate, reconstruct the other seven as zero, and generalize quotient-aware clipping and AdamW absorption to noncontiguous omitted coordinates.
mechanism: Retained-coordinate rotation in the value-bias quotient
evidence_used: The 1589-parameter model retaining value coordinates 0 and 1 achieved 99.73%, whereas the 1588-parameter model retaining only coordinate 0 collapsed to 73.26%; retaining coordinate 1 instead directly tests the coordinate implicated by that regression.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting the final retained value-bias degree will produce a 1587-parameter model with at least 99% accuracy because all eight value-bias coordinates have the same projection-bias-absorbable role and the existing optimizer preserves their full-coordinate updates.
change: Store only the eight query-bias parameters, reconstruct the entire value bias as zero, and let the existing quotient-aware clipping and AdamW logic infer and absorb all eight omitted value coordinates.
mechanism: Complete value-bias absorption into the attention projection bias
evidence_used: Retaining only value coordinate 1 achieved 99.81% at 1588 parameters, while six-coordinate omission achieved 99.73%; this directly motivates testing the complete value-bias quotient after the retained-coordinate rotation restored accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1587-parameter complete value-bias quotient, folding all eight `ln2` scales into `fc1.weight` while reproducing their full AdamW and clipping dynamics will yield a 1579-parameter model with at least 99% accuracy.
change: Remove the complete attention value bias and all `ln2` affine parameters, then train the effective `fc1.weight` through optimizer-maintained weight/scale factors whose product is stored in the model.
mechanism: Trajectory-preserving LayerNorm-scale folding into the MLP input projection
evidence_used: Complete value-bias absorption achieved 99.94% at 1587 parameters. Fixing one `ln2` scale without preserving its optimization trajectory collapsed to 28.96%, motivating an exact folded-factor update instead of a fixed-scale optimization change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Folding all eight `ln1` scales into `qkv.weight` while reproducing their AdamW and clipping dynamics will reduce the verified 1579-parameter model to 1571 parameters while retaining at least 99% accuracy.
change: Make `ln1` non-affine and add each attention QKV weight to the optimizer’s existing folded-factor path alongside `fc1.weight`.
mechanism: Trajectory-preserving LayerNorm-scale folding into attention QKV weights
evidence_used: Folding all eight `ln2` scales into `fc1.weight` with factor-aware AdamW and clipping achieved 99.75% accuracy at 1579 parameters; `ln1` has the same columnwise multiplicative redundancy with its downstream QKV projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1571, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1571-parameter folded-LayerNorm design by removing the feature-uniform coordinate of `attn.proj.bias` will produce a 1570-parameter model with at least 99% accuracy, because that coordinate passes through the residual stream and is canceled by downstream LayerNorms.
change: Fold both block LayerNorm scales into their downstream weights using factor-aware AdamW, then store seven relative attention-projection bias coordinates and preserve its full-coordinate optimization and absorbed value-bias updates.
mechanism: Attention-output residual-bias quotient
evidence_used: Reference Design 2 achieved 99.89% accuracy with 1571 parameters after both LayerNorm-scale folds. The current 1587-parameter design achieved 99.94% while applying the identical final-LayerNorm-canceled quotient to `fc2.bias`, motivating the same one-coordinate reduction for the attention residual bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1570, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1570-parameter design by removing the feature-uniform output coordinate from each of the 12 `fc2.weight` columns will produce a 1558-parameter model with at least 99% accuracy, because those components add only tokenwise uniform residual shifts canceled by the final LayerNorm.
change: Adopt the qualified complete value-bias absorption, folded `ln1`/`ln2` scales, and relative attention-projection bias; additionally store `fc2.weight` as twelve rows of seven relative output coordinates and preserve its full-coordinate AdamW and clipping trajectory.
mechanism: MLP residual-output matrix quotient with trajectory-preserving LayerNorm folding
evidence_used: Reference Design 3 achieved 99.98% accuracy at 1570 parameters after removing the uniform coordinate of both residual-output biases. The same invariance applies independently to every `fc2.weight` column, since each omitted coordinate contributes only a feature-uniform residual component before the final LayerNorm.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1558, "training_steps": 4999}



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
