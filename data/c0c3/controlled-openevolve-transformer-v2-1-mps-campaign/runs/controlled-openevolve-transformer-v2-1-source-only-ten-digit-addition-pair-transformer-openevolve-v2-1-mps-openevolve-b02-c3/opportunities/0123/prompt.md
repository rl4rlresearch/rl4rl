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
verified_results: {"accuracy": 0.9952, "parameters": 676, "training_steps": 4999}
prior_hypothesis: A 676-parameter transformer will retain at least 99% accuracy because it preserves the qualified three-way longest-bias sharing while independently tying the next two shorter second-head biases, avoiding the destructive four-way merge.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9971, "parameters": 673, "training_steps": 4999}
prior_hypothesis: A 673-parameter transformer will retain at least 99% accuracy because it preserves the qualified 674-parameter architecture while fixing only one additional zero-initialized MLP output-bias coordinate.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9997, "parameters": 677, "training_steps": 4999}
prior_hypothesis: A 677-parameter transformer will retain at least 99% accuracy because extending the qualified second head’s two-way adjacent bias sharing to three adjacent long-distance biases preserves a learned routing signal while leaving the first head and all normalization scales unchanged.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9952, "parameters": 672, "training_steps": 4999}
prior_hypothesis: A 672-parameter transformer will retain at least 99% accuracy because it preserves the qualified 673-parameter architecture while fixing only one additional zero-initialized MLP output-bias coordinate.

## Recent verification evidence

RECENT RESULT
hypothesis: A 676-parameter transformer will retain at least 99% accuracy because the qualified three-way relative-bias sharing is preserved while one second-head value-channel scale is fixed at 0.02 and remains compensable by the learned output projection.
change: Reconstruct the qualified 677-parameter three-way second-head bias sharing, then remove one learned QKV coefficient by applying the existing first-head value-scale gauge convention to one second-head value channel.
mechanism: Second-head value/output scale-gauge fixing
evidence_used: The 677-parameter three-way-sharing design achieved 99.97%; the source already successfully fixes the corresponding value scales for all first-head channels, while the failed 673-parameter trial changed the entire value transport, motivating this isolated one-coordinate gauge reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.19390000000000002, "parameters": 676, "training_steps": 4999}

RECENT RESULT
hypothesis: A 676-parameter transformer will retain at least 99% accuracy because it preserves the qualified three-way longest-bias sharing while independently tying the next two shorter second-head biases, avoiding the destructive four-way merge.
change: Reconstruct the qualified 677-parameter routing layout, then remove one parameter by adding a separate two-way adjacent bias tie in the tolerant second head.
mechanism: Disjoint adjacent tolerant-head bias sharing
evidence_used: The 677-parameter three-way-sharing design achieved 99.97%, while extending that same group to four positions collapsed to 50.28%; this motivates preserving the successful triple and testing a disjoint adjacent pair.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9952, "parameters": 676, "training_steps": 4999}

RECENT RESULT
hypothesis: A 675-parameter transformer will retain at least 99% accuracy because it preserves the successful three-way longest-bias group and existing disjoint pair while tying a separate pair of shorter second-head biases.
change: Remove one second-head relative-bias parameter and reconstruct two disjoint adjacent pairs before the qualified three-way group.
mechanism: Second disjoint adjacent tolerant-head bias sharing
evidence_used: The 676-parameter disjoint-pair design achieved 99.52%, whereas enlarging the successful three-way group to four collapsed to 50.28%; adding another separate pair tests further sharing without repeating that destructive merge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9878, "parameters": 675, "training_steps": 4999}

RECENT RESULT
hypothesis: A 675-parameter transformer will retain at least 99% accuracy because it preserves the qualified longest-bias triple while sharing a separate learned coefficient across the three immediately preceding second-head positions, avoiding the destructive four-way merge.
change: Remove two parameters by reconstructing the second head’s six longest learned bias positions from two independent three-way shared coefficients.
mechanism: Dual three-way tolerant-head relative-bias sharing
evidence_used: The 677-parameter three-way design achieved 99.97%, and the 676-parameter design with a separate adjacent pair achieved 99.52%; extending the longest group to four collapsed to 50.28%, while a different 675-parameter two-pair layout narrowly missed at 98.78%, motivating this alternative 675-parameter topology.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.12359999999999999, "parameters": 675, "training_steps": 4999}

RECENT RESULT
hypothesis: A 675-parameter transformer will retain at least 99% accuracy because it preserves the qualified 676-parameter attention-routing layout while fixing only one zero-initialized MLP output-bias coordinate.
change: Reconstruct the qualified longest-bias triple and disjoint adjacent pair, then remove one learned parameter by fixing the final quotient-space bias coefficient of the MLP output projection at zero.
mechanism: Single-coordinate MLP output-bias fixing
evidence_used: The 676-parameter routing layout achieved 99.52%, while further relative-bias sharing at 675 parameters missed at 98.78% and QKV/value-path reductions collapsed; this motivates preserving qualified routing and testing an isolated residual-bias reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 675, "training_steps": 4999}

RECENT RESULT
hypothesis: A 674-parameter transformer will retain at least 99% accuracy because it preserves the qualified 675-parameter attention-routing layout while fixing only one additional zero-initialized MLP output-bias coordinate.
change: Reconstruct the qualified longest-bias triple and disjoint adjacent pair, then fix the final two quotient-space bias coefficients of the MLP output projection at zero.
mechanism: Two-coordinate MLP output-bias fixing
evidence_used: The 675-parameter design achieved 99.97% after fixing one MLP output-bias coordinate while preserving the 676-parameter routing layout, directly motivating the next isolated bias-coordinate reduction without further constraining attention or value transport.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 674, "training_steps": 4999}

RECENT RESULT
hypothesis: A 673-parameter transformer will retain at least 99% accuracy because it preserves the qualified 674-parameter architecture while fixing only one additional zero-initialized MLP output-bias coordinate.
change: Increase the MLP output projection’s fixed bias coefficients from two to three, removing exactly one learned parameter.
mechanism: Third-coordinate MLP output-bias fixing
evidence_used: The current 674-parameter design achieved 99.89% after the preceding 675-parameter design achieved 99.97%; both reductions fixed successive MLP output-bias coordinates without altering the qualified attention-routing layout.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9971, "parameters": 673, "training_steps": 4999}

RECENT RESULT
hypothesis: A 672-parameter transformer will retain at least 99% accuracy because it preserves the qualified 673-parameter architecture while fixing only one additional zero-initialized MLP output-bias coordinate.
change: Increase the MLP output projection’s fixed bias coefficients from one to four, reconstructing the qualified 673-parameter design and removing its fourth learned bias coordinate.
mechanism: Fourth-coordinate MLP output-bias fixing
evidence_used: Successive MLP output-bias reductions achieved 99.97% at 675 parameters, 99.89% at 674, and 99.71% at 673 without altering the qualified attention-routing layout, directly motivating the next one-coordinate reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9952, "parameters": 672, "training_steps": 4999}

RECENT RESULT
hypothesis: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter attention and weight architecture while fixing only the final remaining zero-initialized MLP output-bias coordinate.
change: Add quotient-output bias-coordinate fixing, reconstruct the qualified four-coordinate fixing, and remove the sole remaining learned MLP output bias.
mechanism: Fifth-coordinate MLP output-bias fixing
evidence_used: Successive MLP output-bias reductions achieved 99.97% at 675 parameters, 99.89% at 674, 99.71% at 673, and 99.52% at 672; this directly motivates testing the final bias coordinate without altering attention routing or learned weight matrices.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3333, "parameters": 671, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing each four-scalar token code with a three-scalar code expanded into five features by two normalized quadratic interactions will retain at least 99% accuracy while removing one learned parameter per vocabulary token beyond the qualified 672-parameter backbone.
change: Preserve the qualified attention routing and four-coordinate MLP bias fixing. Challenge the prior assumption that five lexical features require four independent token coordinates by generating those features from three learned coordinates and two input/classifier-specific quadratic lifts.
mechanism: Three-coordinate dual-quadratic lexical manifold
evidence_used: The 672-parameter architecture achieved 99.52%, while changes to attention routing and value transport collapsed accuracy. This motivates preserving those load-bearing mechanisms and testing the previously unchallenged lexical-code dimensionality with a full-width nonlinear lift.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.39640000000000003, "parameters": 558, "training_steps": 4999}

RECENT RESULT
hypothesis: A 671-parameter transformer will retain at least 99% accuracy because sharing one direct input-code coordinate between two tokens preserves four-coordinate token codes, both nonlinear lifts, and the qualified 672-parameter attention/MLP architecture.
change: Replace the token embedding table with a one-scalar-smaller parameterization in which the first coordinate of the first two token codes shares one learned coefficient.
mechanism: Single-coordinate cross-token lexical sharing
evidence_used: The 672-parameter design achieved 99.52%, while collapsing every token to three coordinates fell to 39.64% and removing the remaining MLP output bias fell to 33.33%; this motivates an isolated one-scalar lexical tie that retains the full four-coordinate representation and leaves load-bearing attention and MLP paths unchanged.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5663, "parameters": 671, "training_steps": 4999}

RECENT RESULT
hypothesis: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter attention, lexical, and MLP architecture while fixing only one final LayerNorm scale at its unit initialization.
change: Reconstruct the qualified four-coordinate MLP bias fixing, then remove one learned final-normalization scale by extending the unit-valued fixed-scale suffix from three features to four.
mechanism: Final-normalization scale fixing
evidence_used: The 672-parameter design achieved 99.52%, while removing its remaining MLP output bias collapsed to 33.33% and tying a lexical coordinate collapsed to 56.63%; this motivates preserving those components and testing an isolated reduction in the previously untested final normalization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1441, "parameters": 671, "training_steps": 4999}



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
